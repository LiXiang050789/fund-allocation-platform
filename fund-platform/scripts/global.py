# 屏蔽TF日志 + CPU多线程加速
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import random
import numpy as np
import torch
import torch.nn as nn
import sys
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
# 导入全新独立环境，不碰旧文件
from portfolio_env_global import PortfolioEnvGlobal
from portfolio_metrics import calc_portfolio_metrics

# ===================== 全局随机种子 =====================
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
print(f"训练设备：{device}")

# ===================== 路径配置 =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "platform"))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "clean", "train_feature_filtered_env.csv")
PRICE_PATH = os.path.join(SCRIPT_DIR, "..", "data", "clean", "etf_price_clean.csv")

MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, "models", "global_ppo")
RESULT_SAVE_DIR = os.path.join(SCRIPT_DIR, "results", "global_ppo")
LOG_SAVE_DIR = os.path.join(SCRIPT_DIR, "logs", "global_ppo_log")
for folder in [MODEL_SAVE_DIR, RESULT_SAVE_DIR, LOG_SAVE_DIR]:
    os.makedirs(folder, exist_ok=True)
for folder in [MODEL_SAVE_DIR, RESULT_SAVE_DIR, LOG_SAVE_DIR]:
    os.makedirs(folder, exist_ok=True)

# ===================== 全局超参（激进进攻配置） =====================
RL_WINDOW = 40
# REWARD_COEF=(15.0,0.01,0.05,0.005)
# 调整第三项回撤惩罚 0.05 → 0.08
REWARD_COEF = (15.0, 0.01, 0.08, 0.005)

EPOCH_STEP = 3000
TRAIN_ITER_NUM = 40
EARLY_STOP_PATIENCE = 15
MIN_ITER = 15

# PPO超参，高探索
PPO_KWARGS = {
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 128,
    "gamma": 0.998,
    "clip_range": 0.25,
    "ent_coef": 0.07,
    "max_grad_norm": 0.5,
    "verbose": 0,
    "device": device,
    "seed": SEED,
    "policy_kwargs": {
        "net_arch": [128, 64],
    }
}

MAX_WEIGHT = 0.55
MIN_WEIGHT = 0.01
TRADE_COST = 0.0003

# ===================== 数据加载 & 全局划分 =====================
def load_full_data():
    feat = pd.read_csv(DATA_PATH, parse_dates=["date"], encoding="utf-8-sig")
    price = pd.read_csv(PRICE_PATH, parse_dates=["date"], encoding="utf-8-sig")
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

# ===================== 全局一次性训练主逻辑 =====================
def global_train():
    feat_full, date_all, price_full = load_full_data()
    total_len = len(feat_full)
    split_idx = int(total_len * 0.8)
    train_feat_raw = feat_full.iloc[:split_idx].copy()
    val_feat_raw = feat_full.iloc[split_idx:].copy()

    train_mean, train_std = get_train_scaler(train_feat_raw)
    feat_norm_all = apply_norm(feat_full, train_mean, train_std)

    # 训练环境 前80%
    train_env = PortfolioEnvGlobal(
        feature_df=feat_norm_all,
        price_df=price_full,
        start_idx=0,
        end_idx=split_idx,
        window=RL_WINDOW,
        reward_coef=REWARD_COEF,
        min_w=MIN_WEIGHT,
        max_w=MAX_WEIGHT,
        trade_cost=TRADE_COST
    )
    train_env = Monitor(train_env)

    # 验证环境 后20%
    val_env = PortfolioEnvGlobal(
        feature_df=feat_norm_all,
        price_df=price_full,
        start_idx=split_idx,
        end_idx=total_len - 1,
        window=RL_WINDOW,
        reward_coef=REWARD_COEF,
        min_w=MIN_WEIGHT,
        max_w=MAX_WEIGHT,
        trade_cost=TRADE_COST
    )
    val_env = Monitor(val_env)

    model = PPO("MlpPolicy", train_env, **PPO_KWARGS)
    best_sharpe = -np.inf
    patience = 0
    log_records = []
    best_model_path = os.path.join(MODEL_SAVE_DIR, "global_best.zip")

    for iter_idx in range(TRAIN_ITER_NUM):
        model.learn(total_timesteps=EPOCH_STEP, reset_num_timesteps=False)
        obs, _ = val_env.reset()
        done = False
        net_list = [1.0]
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = val_env.step(action)
            done = terminated or truncated
            net_list.append(val_env.unwrapped.net_value)
        metrics = calc_portfolio_metrics(pd.Series(net_list))
        current_sharpe = metrics["夏普比率"]
        log_records.append({
            "iter": iter_idx,
            "sharpe": current_sharpe,
            "cum_ret": metrics["累计收益"],
            "vol": metrics["年化波动"],
            "mdd": metrics["最大回撤"]
        })
        print(f"全局迭代{iter_idx} | 验证夏普:{current_sharpe:.3f} 年化收益:{metrics['年化收益']:.3f} 最大回撤:{metrics['最大回撤']:.3f}")

        if current_sharpe > best_sharpe:
            best_sharpe = current_sharpe
            model.save(best_model_path)
            patience = 0
        else:
            patience += 1
        if patience >= EARLY_STOP_PATIENCE and iter_idx >= MIN_ITER:
            print(f"🛑 全局训练早停，最优验证夏普={best_sharpe:.3f}")
            break

    log_df = pd.DataFrame(log_records)
    log_df.to_csv(os.path.join(LOG_SAVE_DIR, "train_log.csv"), index=False, encoding="utf-8-sig")
    print(f"全局训练完成，最优模型已保存至 {best_model_path}")

    # 全量预测输出净值
    best_model = PPO.load(best_model_path, device=device)
    pred_env = PortfolioEnvGlobal(
        feature_df=feat_norm_all,
        price_df=price_full,
        start_idx=0,
        end_idx=total_len - 1,
        window=RL_WINDOW,
        reward_coef=REWARD_COEF,
        min_w=MIN_WEIGHT,
        max_w=MAX_WEIGHT,
        trade_cost=TRADE_COST
    )
    obs, _ = pred_env.reset()
    done = False
    pred_records = []
    step_idx = 0
    while not done:
        action, _ = best_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = pred_env.step(action)
        done = terminated or truncated
        pred_records.append({
            "date": date_all.iloc[step_idx],
            "net_value": pred_env.unwrapped.net_value,
            "turnover": info["turnover"]
        })
        step_idx += 1
    pred_df = pd.DataFrame(pred_records)
    pred_df.to_csv(os.path.join(RESULT_SAVE_DIR, "global_ppo_net.csv"), index=False, encoding="utf-8-sig")
    pred_df[["date", "net_value"]].to_csv(r"E:\农行杯\2026\clean\ppo_net_global.csv", index=False, encoding="utf-8-sig")
    print("✅ 全局PPO净值文件导出完成：clean/ppo_net_global.csv")

    full_metrics = calc_portfolio_metrics(pred_df["net_value"])
    print("\n===== 全局训练完整样本外指标 =====")
    for k, v in full_metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    global_train()
