# 多线程加速、屏蔽日志
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import random
import numpy as np
import torch
import sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "platform"))
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
# 独立全局环境
from portfolio_env_global import PortfolioEnvGlobal
from portfolio_metrics import calc_portfolio_metrics

# ===================== 固定随机种子 =====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["SB3_ALLOW_GYM_V0"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"推演设备：{device}")

# ===================== 路径配置 =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "clean", "train_feature_filtered.csv")
PRICE_PATH = os.path.join(SCRIPT_DIR, "clean", "etf_price_clean.csv")
# 全局训练保存的最优模型（你之前跑global.py生成）
PRETRAIN_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "global_ppo", "global_best.zip")
# 输出结果路径
RESULT_SAVE_DIR = os.path.join(SCRIPT_DIR, "results", "rollout_backtest")
os.makedirs(RESULT_SAVE_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(RESULT_SAVE_DIR, "rollout_net.csv")
EXPORT_CSV = r"E:\农行杯\2026\clean\ppo_rollout_net.csv"

# ===================== 超参（和全局训练保持完全一致） =====================
TRAIN_WINDOW = 1000
VAL_WINDOW = 150
PREDICT_WINDOW = 150
RL_WINDOW = 40
REWARD_COEF = (15.0, 0.01, 0.05, 0.005)
MAX_WEIGHT = 0.55
MIN_WEIGHT = 0.01
TRADE_COST = 0.0003

# ===================== 数据加载&标准化工具 =====================
def load_full_data():
    feat = pd.read_csv(DATA_PATH, parse_dates=["date"], encoding="gbk")
    price = pd.read_csv(PRICE_PATH, parse_dates=["date"], encoding="gbk")
    feat = feat.ffill().fillna(0)
    price = price.ffill().fillna(0)
    common_date = set(feat["date"]) & set(price["date"])
    feat = feat[feat["date"].isin(common_date)].reset_index(drop=True)
    price = price[price["date"].isin(common_date)].reset_index(drop=True)
    price = price.iloc[1:].reset_index(drop=True)
    date_col = feat["date"].copy()
    feat_only = feat.drop(columns=["date"])
    return feat_only, date_col, price

def get_train_scaler(train_df):
    mean = train_df.mean()
    std = train_df.std()
    std = std.where(std > 1e-8, 1.0)
    return mean, std

def apply_norm(df, mean, std):
    return (df - mean) / std

# ===================== 滚动纯推演主逻辑（无训练，仅预测） =====================
def rolling_rollout():
    # 加载预训练全局模型，全程冻结权重不更新
    model = PPO.load(PRETRAIN_MODEL_PATH, device=device)
    print(f"成功加载预训练全局模型：{PRETRAIN_MODEL_PATH}")

    feat_full, date_all, price_full = load_full_data()
    total_len = len(feat_full)
    cur_start = 0
    all_records = []

    print("\n========== 开始滚动窗口推演（仅预测、不训练） ==========")
    while cur_start + TRAIN_WINDOW + VAL_WINDOW + PREDICT_WINDOW <= total_len:
        train_start = cur_start
        train_end = cur_start + TRAIN_WINDOW
        val_start = train_end
        val_end = train_end + VAL_WINDOW
        pred_start = val_end
        pred_end = val_end + PREDICT_WINDOW

        print(f"\n窗口 | 训练:{train_start}~{train_end} 验证:{val_start}~{val_end} 预测:{pred_start}~{pred_end}")

        # 仅用当前窗口训练段做标准化，严格时序无未来泄露
        train_raw = feat_full.iloc[train_start:train_end]
        mean, std = get_train_scaler(train_raw)
        feat_norm_all = apply_norm(feat_full, mean, std)

        # 构建预测环境
        pred_env = PortfolioEnvGlobal(
            feature_df=feat_norm_all,
            price_df=price_full,
            start_idx=pred_start,
            end_idx=pred_end,
            window=RL_WINDOW,
            reward_coef=REWARD_COEF,
            min_w=MIN_WEIGHT,
            max_w=MAX_WEIGHT,
            trade_cost=TRADE_COST
        )
        pred_env = Monitor(pred_env)

        obs, _ = pred_env.reset()
        done = False
        step_idx = 0
        while not done:
            # deterministic=True 固定策略预测，不随机探索
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = pred_env.step(action)
            done = terminated or truncated
            record_date = date_all.iloc[pred_start + step_idx]
            all_records.append({
                "date": record_date,
                "net_value": pred_env.unwrapped.net_value,
                "turnover": info["turnover"],
                "sharpe": info["sharpe"]
            })
            step_idx += 1

        cur_start += PREDICT_WINDOW

    # 整合全周期预测结果
    df_result = pd.DataFrame(all_records)
    df_result = df_result.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)
    df_result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    df_result[["date", "net_value"]].to_csv(EXPORT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n推演完成，完整结果保存：{OUTPUT_CSV}")
    print(f"回测净值导出：{EXPORT_CSV}")

    # 输出整体指标
    metrics = calc_portfolio_metrics(df_result["net_value"])
    print("\n===== 滚动推演全局指标 =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 换手率分析
    avg_turn = df_result["turnover"].mean()
    annual_cost = avg_turn * 240 * 0.0003
    print("\n========== 调仓统计 ==========")
    print(f"日均换手率：{avg_turn:.4f} 年化摩擦成本：{annual_cost:.2%}")

if __name__ == "__main__":
    rolling_rollout()
