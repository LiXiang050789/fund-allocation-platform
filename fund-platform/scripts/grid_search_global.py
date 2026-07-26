# 多线程加速、屏蔽TF日志
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
from itertools import product
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "platform"))
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
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
DATA_PATH = os.path.join(SCRIPT_DIR, "clean", "train_feature_filtered.csv")
PRICE_PATH = os.path.join(SCRIPT_DIR, "clean", "etf_price_clean.csv")

MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, "models", "grid_search_ppo")
RESULT_SAVE_DIR = os.path.join(SCRIPT_DIR, "results", "grid_search_ppo")
LOG_SAVE_DIR = os.path.join(SCRIPT_DIR, "logs", "grid_search_log")
for folder in [MODEL_SAVE_DIR, RESULT_SAVE_DIR, LOG_SAVE_DIR]:
    os.makedirs(folder, exist_ok=True)

# ===================== 固定超参（全程不变） =====================
RL_WINDOW = 40
EPOCH_STEP = 3000
TRAIN_ITER_NUM = 40
EARLY_STOP_PATIENCE = 15
MIN_ITER = 15
MAX_WEIGHT = 0.55
MIN_WEIGHT = 0.01
TRADE_COST = 0.0003
TURN_PENALTY = 0.005

# ===================== 网格搜索参数范围 =====================
# 1. REWARD_COEF (gain, vol, dd, turn)
gain_list = [12.0, 14.0, 16.0]
vol_list = [0.01, 0.02]
dd_list = [0.05, 0.08]
# 2. 中长期收益权重
lt_weight_list = [0.4, 0.5, 0.6, 0.7]
# 3. PPO 熵系数
ent_list = [0.06, 0.08, 0.10]
# 4. 周期波动惩罚系数
cycle_vol_k_list = [0.02, 0.05]

# 生成全部参数组合
param_combinations = list(product(gain_list, vol_list, dd_list, lt_weight_list, ent_list, cycle_vol_k_list))
total_cases = len(param_combinations)
print(f"网格搜索总组合数量：{total_cases}")

# ===================== 数据加载工具 =====================
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

# ===================== 单组参数训练函数 =====================
def train_single_case(case_idx, gain, vol, dd, lt_w, ent, cycle_k):
    print(f"\n===== 组合 {case_idx}/{total_cases} 开始训练 =====")
    print(f"参数：gain={gain}, vol={vol}, dd={dd}, lt_weight={lt_w}, ent_coef={ent}, cycle_k={cycle_k}")
    reward_coef = (gain, vol, dd, TURN_PENALTY)

    # PPO 参数
    ppo_kwargs = {
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 128,
        "gamma": 0.998,
        "clip_range": 0.25,
        "ent_coef": ent,
        "max_grad_norm": 0.5,
        "verbose": 0,
        "device": device,
        "seed": SEED,
        "policy_kwargs": {"net_arch": [128, 64]}
    }

    # 修改环境全局参数文件的临时配置（运行时生效）
    # 直接传参给环境，无需修改源文件
    feat_full, date_all, price_full = load_full_data()
    total_len = len(feat_full)
    split_idx = int(total_len * 0.8)
    train_feat_raw = feat_full.iloc[:split_idx].copy()
    train_mean, train_std = get_train_scaler(train_feat_raw)
    feat_norm_all = apply_norm(feat_full, train_mean, train_std)

    train_env = PortfolioEnvGlobal(
        feature_df=feat_norm_all,
        price_df=price_full,
        start_idx=0,
        end_idx=split_idx,
        window=RL_WINDOW,
        reward_coef=reward_coef,
        min_w=MIN_WEIGHT,
        max_w=MAX_WEIGHT,
        trade_cost=TRADE_COST
    )
    train_env = Monitor(train_env)
    val_env = PortfolioEnvGlobal(
        feature_df=feat_norm_all,
        price_df=price_full,
        start_idx=split_idx,
        end_idx=total_len - 1,
        window=RL_WINDOW,
        reward_coef=reward_coef,
        min_w=MIN_WEIGHT,
        max_w=MAX_WEIGHT,
        trade_cost=TRADE_COST
    )
    val_env = Monitor(val_env)

    model = PPO("MlpPolicy", train_env, **ppo_kwargs)
    best_sharpe = -np.inf
    patience = 0
    best_case_path = os.path.join(MODEL_SAVE_DIR, f"case_{case_idx}_best.zip")

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

        if current_sharpe > best_sharpe:
            best_sharpe = current_sharpe
            model.save(best_case_path)
            patience = 0
        else:
            patience += 1
        if patience >= EARLY_STOP_PATIENCE and iter_idx >= MIN_ITER:
            print(f"🛑 组合{case_idx}早停，最优验证夏普={best_sharpe:.3f}")
            break

    # 全量预测评估
    best_model = PPO.load(best_case_path, device=device)
    pred_env = PortfolioEnvGlobal(
        feature_df=feat_norm_all,
        price_df=price_full,
        start_idx=0,
        end_idx=total_len - 1,
        window=RL_WINDOW,
        reward_coef=reward_coef,
        min_w=MIN_WEIGHT,
        max_w=MAX_WEIGHT,
        trade_cost=TRADE_COST
    )
    obs, _ = pred_env.reset()
    done = False
    net_list = [1.0]
    while not done:
        action, _ = best_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = pred_env.step(action)
        done = terminated or truncated
        net_list.append(pred_env.unwrapped.net_value)
    full_metrics = calc_portfolio_metrics(pd.Series(net_list))

    result_row = {
        "case_id": case_idx,
        "gain": gain,
        "vol_punish": vol,
        "dd_punish": dd,
        "lt_weight": lt_w,
        "ent_coef": ent,
        "cycle_k": cycle_k,
        "val_best_sharpe": best_sharpe,
        "full_annual": full_metrics["年化收益"],
        "full_sharpe": full_metrics["夏普比率"],
        "full_mdd": full_metrics["最大回撤"],
        "full_vol": full_metrics["年化波动"],
        "model_path": best_case_path
    }
    print(f"组合{case_idx} 全局最终夏普={full_metrics['夏普比率']:.4f} 年化={full_metrics['年化收益']:.4f} 最大回撤={full_metrics['最大回撤']:.4f}")
    return result_row

# ===================== 批量执行网格搜索 =====================
def grid_search_main():
    log_all = []
    for idx, params in enumerate(param_combinations, start=1):
        g, v, d, lt, e, ck = params
        row = train_single_case(idx, g, v, d, lt, e, ck)
        log_all.append(row)
        pd.DataFrame(log_all).to_csv(os.path.join(LOG_SAVE_DIR, "grid_search_log.csv"), index=False, encoding="utf-8-sig")

    # 选出全局最优参数（按全量样本外夏普排序）
    log_df = pd.DataFrame(log_all)
    log_df = log_df.sort_values("full_sharpe", ascending=False).reset_index(drop=True)
    best_row = log_df.iloc[0]
    print("\n==================== 网格搜索全部完成 ====================")
    print("最优参数组合：")
    print(f"case_id: {best_row['case_id']}")
    print(f"收益权重gain: {best_row['gain']}")
    print(f"波动惩罚vol: {best_row['vol_punish']}")
    print(f"回撤惩罚dd: {best_row['dd_punish']}")
    print(f"中长期收益权重lt_weight: {best_row['lt_weight']}")
    print(f"PPO熵ent_coef: {best_row['ent_coef']}")
    print(f"周期波动惩罚cycle_k: {best_row['cycle_k']}")
    print(f"全局样本外夏普: {best_row['full_sharpe']:.4f}")
    print(f"年化收益: {best_row['full_annual']:.4f}")
    print(f"最大回撤: {best_row['full_mdd']:.4f}")
    print(f"最优模型路径: {best_row['model_path']}")
    log_df.to_csv(os.path.join(RESULT_SAVE_DIR, "grid_final_sort.csv"), index=False, encoding="utf-8-sig")
    print("全部日志已保存至 logs/grid_search_log.csv")

if __name__ == "__main__":
    grid_search_main()
