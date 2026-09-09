"""
推理PPO模型，生成每日持仓权重（用于统一回测）
必须与训练脚本 global.py 的参数完全一致
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# 导入环境（需与训练相同）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "platform"))
from portfolio_env_global import PortfolioEnvGlobal

# ---------- 配置参数（与训练脚本 global.py 完全一致） ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(BASE, "data", "clean", "train_feature_filtered_ppo.csv")
PRICE_PATH = os.path.join(BASE, "data", "clean", "etf_price_clean.csv")
MODEL_PATH = os.path.join(BASE, "models", "global_best.zip")
WEIGHT_OUTPUT = os.path.join(BASE, "results", "ppo_weight.csv")

os.makedirs(os.path.dirname(WEIGHT_OUTPUT), exist_ok=True)

# 环境超参（必须与训练时相同）
RL_WINDOW = 40
REWARD_COEF = (3.0, 0.18, 1.3, 0.65)   # 与训练一致
MAX_WEIGHT = 0.25                       # 与训练一致
MIN_WEIGHT = 0.01
TRADE_COST = 0.0003
CONSTRAINT_COEF = 10.0

# 消融实验开关（与训练一致）
EXPERIMENT_CFG = {
    "rebalance_monthly": False,
    "enable_bond_regime_reward": False,
    "enable_bond_loss_couple": True,
    "enable_ladder_dd": True,
    "enable_min_bond_hard_constraint": True,
    "enable_reward_norm": True,
    "enable_lgb_pred_obs": False
}

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 数据加载（复用训练脚本的归一化逻辑） ----------
def load_full_data():
    feat = pd.read_csv(DATA_PATH, parse_dates=["date"], encoding="utf-8-sig")
    price = pd.read_csv(PRICE_PATH, parse_dates=["date"], encoding="gbk")
    # 仅保留价格列
    price = price[[c for c in price.columns if c == "date" or c != "date"]]
    feat = feat.ffill().fillna(0)
    price = price.ffill().fillna(0)
    common_date = set(feat["date"]) & set(price["date"])
    feat = feat[feat["date"].isin(common_date)].reset_index(drop=True)
    price = price[price["date"].isin(common_date)].reset_index(drop=True)
    feat = feat.iloc[1:].reset_index(drop=True)   # 去掉第一行（对齐价格）
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

# ---------- 构建环境（推理时覆盖全量日期） ----------
def build_env(start_idx, end_idx, feat_norm, price_df):
    env = PortfolioEnvGlobal(
        feature_df=feat_norm,
        price_df=price_df,
        start_idx=start_idx,
        end_idx=end_idx,
        window=RL_WINDOW,
        reward_coef=REWARD_COEF,
        min_w=MIN_WEIGHT,
        max_w=MAX_WEIGHT,
        trade_cost=TRADE_COST,
        constraint_coef=CONSTRAINT_COEF,
        **EXPERIMENT_CFG
    )
    env.reset(seed=SEED)
    return Monitor(env)

# ---------- 主流程 ----------
def generate_weights():
    # 1. 加载数据
    feat_full, date_all, price_full = load_full_data()
    total_len = len(feat_full)

    # 2. 使用训练集（前60%）计算归一化参数（与训练脚本完全一致）
    train_end = int(total_len * 0.6)
    train_feat_raw = feat_full.iloc[:train_end].copy()
    train_mean, train_std = get_train_scaler(train_feat_raw)
    feat_norm_all = apply_norm(feat_full, train_mean, train_std)

    # 3. 构建全量环境（从0到total_len-1）
    env = build_env(0, total_len - 1, feat_norm_all, price_full)

    # 4. 加载最优模型
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在：{MODEL_PATH}")
    model = PPO.load(MODEL_PATH, device=device)

    # 5. 逐日推理
    obs, _ = env.reset()
    done = False
    weight_records = []
    step_idx = 0

    while not done:
        # 当前日期
        current_date = date_all.iloc[step_idx]
        action, _ = model.predict(obs, deterministic=True)
        # 执行一步，获取权重
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # 从环境中获取当前权重（last_weight）
        w = env.unwrapped.last_weight.copy()
        weight_records.append({
            "date": current_date,
            "hs300": w[0],
            "zz500": w[1],
            "kc50": w[2],
            "consume": w[3],
            "chip": w[4],
            "gold": w[5],
            "bond10": w[6]
        })
        step_idx += 1

    # 6. 保存权重文件
    weight_df = pd.DataFrame(weight_records)
    # 确保日期为datetime类型
    weight_df["date"] = pd.to_datetime(weight_df["date"])
    # 按日期排序
    weight_df = weight_df.sort_values("date").reset_index(drop=True)
    weight_df.to_csv(WEIGHT_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"✅ PPO每日权重已保存至：{WEIGHT_OUTPUT}")
    print(f"   日期范围：{weight_df['date'].min()} ~ {weight_df['date'].max()}")
    print(f"   总天数：{len(weight_df)}")

if __name__ == "__main__":
    generate_weights()
