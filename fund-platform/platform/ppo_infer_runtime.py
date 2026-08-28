"""PPO runtime inference helper for the Streamlit AI assistant.

This script runs in a subprocess so native crashes from torch/sb3 model loading
cannot take down the Streamlit server process.
"""
import importlib.util
import json
import os
import sys

import pandas as pd


ETF_CODES = ["hs300", "zz500", "kc50", "consume", "chip", "gold", "bond10"]
EXPECTED_FEATURE_DIM = 44
RL_WINDOW = 40
REWARD_COEF = (3.0, 0.18, 1.3, 0.65)
EXPERIMENT_CFG = {
    "rebalance_monthly": False,
    "enable_bond_regime_reward": False,
    "enable_bond_loss_couple": True,
    "enable_ladder_dd": True,
    "enable_min_bond_hard_constraint": True,
    "enable_reward_norm": True,
    "enable_lgb_pred_obs": False,
}

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BASE)
MODEL_PATH = os.path.join(BASE, "models", "global_best.zip")
FEATURE_CANDIDATES = [
    os.path.join(BASE, "data", "clean", "train_feature_filtered_ppo.csv"),
    os.path.join(BASE, "data", "clean", "train_feature_filtered.csv"),
    os.path.join(BASE, "data", "clean", "train_feature_filtered_env.csv"),
]
PRICE_PATH = os.path.join(BASE, "data", "clean", "etf_price_clean.csv")
ENV_PATH = os.path.join(REPO_ROOT, "portfolio_env_global.py")


def load_env_class():
    spec = importlib.util.spec_from_file_location("ppo_portfolio_env_global", ENV_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load PPO env: {ENV_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PortfolioEnvGlobal


def load_data():
    feat = None
    checked = []
    for path in FEATURE_CANDIDATES:
        if not os.path.exists(path):
            checked.append(f"{os.path.basename(path)}: missing")
            continue
        candidate = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"])
        feature_dim = candidate.drop(columns=["date"]).select_dtypes(include="number").shape[1]
        checked.append(f"{os.path.basename(path)}: {feature_dim} features")
        if feature_dim == EXPECTED_FEATURE_DIM:
            feat = candidate
            break
    if feat is None:
        raise ValueError(
            "PPO实时推理需要训练时44特征文件，当前未找到。已检查："
            + "; ".join(checked)
        )

    price = pd.read_csv(PRICE_PATH, encoding="utf-8-sig", parse_dates=["date"])
    feat = feat.ffill().fillna(0)
    price = price.ffill().fillna(0)
    common_dates = set(feat["date"]) & set(price["date"])
    feat = feat[feat["date"].isin(common_dates)].reset_index(drop=True)
    price = price[price["date"].isin(common_dates)].reset_index(drop=True)
    feat = feat.iloc[1:].reset_index(drop=True)
    price = price.iloc[1:].reset_index(drop=True)
    date_col = feat["date"].copy()
    feat_only = feat.drop(columns=["date"])
    train_end = int(len(feat_only) * 0.6)
    mean = feat_only.iloc[:train_end].mean()
    std = feat_only.iloc[:train_end].std()
    std = std.where(std > 1e-8, 1.0)
    return (feat_only - mean) / std, date_col, price


def infer(as_of_date):
    from stable_baselines3 import PPO

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    PortfolioEnvGlobal = load_env_class()
    feat_norm, date_all, price_full = load_data()
    env = PortfolioEnvGlobal(
        feature_df=feat_norm,
        price_df=price_full,
        start_idx=0,
        end_idx=len(feat_norm) - 1,
        window=RL_WINDOW,
        reward_coef=REWARD_COEF,
        min_w=0.01,
        max_w=0.25,
        trade_cost=0.0003,
        constraint_coef=10.0,
        **EXPERIMENT_CFG,
    )
    model = PPO.load(MODEL_PATH, device="cpu")
    obs, _ = env.reset(seed=42)
    if obs.shape != model.observation_space.shape:
        raise ValueError(
            f"PPO观测维度不匹配：env obs={obs.shape}, model obs={model.observation_space.shape}。"
            f"请确认44特征文件列顺序与训练时完全一致。"
        )
    target_date = pd.Timestamp(as_of_date)
    done = False
    step_idx = 0
    last_date = None
    last_weight = None

    while not done:
        current_date = pd.Timestamp(date_all.iloc[step_idx])
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        last_date = current_date
        last_weight = env.last_weight.copy()
        if current_date >= target_date:
            break
        done = terminated or truncated
        step_idx += 1

    if last_weight is None:
        raise RuntimeError("No PPO weight was produced")

    return {
        "date": str(last_date.date()),
        "weights": {code: float(last_weight[i]) for i, code in enumerate(ETF_CODES)},
    }


if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else str(pd.Timestamp.today().date())
    print(json.dumps(infer(date_arg), ensure_ascii=False))
