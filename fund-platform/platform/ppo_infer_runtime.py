"""PPO / SASF 推理 runtime（供 Streamlit 子进程调用）"""
import importlib.util
import json
import os
import sys
import pandas as pd
import numpy as np

ETF_CODES = ["hs300", "zz500", "kc50", "consume", "chip", "gold", "bond10"]
EXPECTED_FEATURE_DIM = 48
RL_WINDOW = 40
REWARD_COEF = (3.0, 0.18, 1.3, 0.65)
EXPERIMENT_CFG = {
    "rebalance_monthly": False,
    "enable_bond_regime_reward": False,
    "enable_bond_loss_couple": False,
    "enable_ladder_dd": False,
    "enable_min_bond_hard_constraint": True,
    "enable_reward_norm": False,
    "enable_lgb_pred_obs": False,
    "enable_equity_only": True,
    "residual_scale": 0.15,
}

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BASE)
MODEL_PATH = os.path.join(BASE, "models", "H_equity_residual_best.zip")
FEATURE_CANDIDATES = [
    os.path.join(BASE, "data", "clean", "train_feature_filtered_ppo.csv"),
    os.path.join(BASE, "data", "clean", "train_feature_filtered.csv"),
]
PRICE_PATH = os.path.join(BASE, "data", "clean", "etf_price_clean.csv")
MARKET_SCORE_PATH = os.path.join(BASE, "data", "clean", "market_score_daily.csv")
ENV_PATH = os.path.join(BASE, "platform", "portfolio_env_global1.py")
DYN_WEIGHT_PATH = os.path.join(BASE, "results", "dynamic_score_monthly_weights.csv")


def load_env_class():
    spec = importlib.util.spec_from_file_location("ppo_portfolio_env_global", ENV_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PortfolioEnvGlobal


def load_data():
    """加载特征与价格，复现训练时的预处理（加 4 个 missing 列 + ffill + median 填充）"""
    MISSING_COLS = ["north_net", "north_20d_sum", "hs300_pe_quantile", "社融存量增速(近似)"]

    feat = None
    checked = []
    for path in FEATURE_CANDIDATES:
        if not os.path.exists(path):
            checked.append(f"{os.path.basename(path)}: missing")
            continue
        candidate = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"])

        # 与训练一致：先加 missing 指示列
        for col in MISSING_COLS:
            if col in candidate.columns and f"{col}_missing" not in candidate.columns:
                candidate[f"{col}_missing"] = candidate[col].isna().astype(np.float32)

        dim = candidate.drop(columns=["date"]).select_dtypes(include="number").shape[1]
        checked.append(f"{os.path.basename(path)}: {dim} features")
        if dim == EXPECTED_FEATURE_DIM:
            feat = candidate
            break

    if feat is None:
        raise ValueError(
            f"PPO 推理需要 {EXPECTED_FEATURE_DIM} 维特征文件，未找到。已检查：" + "; ".join(checked)
        )

    price = pd.read_csv(PRICE_PATH, encoding="utf-8-sig", parse_dates=["date"])

    # 与训练一致：ffill 后逐列 median 填充
    feat = feat.ffill()
    for col in feat.columns:
        if col == "date":
            continue
        med = feat[col].median()
        feat[col] = feat[col].fillna(med)

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
    std = feat_only.iloc[:train_end].std().where(lambda s: s > 1e-8, 1.0)
    return (feat_only - mean) / std, date_col, price


def get_market_state(as_of_date):
    if not os.path.exists(MARKET_SCORE_PATH):
        return "震荡"
    score = pd.read_csv(MARKET_SCORE_PATH, encoding="utf-8-sig", parse_dates=["date"])
    score = score[score["date"] <= pd.Timestamp(as_of_date)]
    if score.empty:
        return "震荡"
    return score.iloc[-1]["market_state"]


def get_dynamic_score_weight(as_of_date):
    if not os.path.exists(DYN_WEIGHT_PATH):
        return None
    dyn = pd.read_csv(DYN_WEIGHT_PATH, encoding="utf-8-sig", parse_dates=["date"])
    dyn = dyn[dyn["date"] <= pd.Timestamp(as_of_date)]
    if dyn.empty:
        return None
    row = dyn.iloc[-1]
    return {code: float(row[code]) for code in ETF_CODES}


def infer_ppo(as_of_date):
    """PPO 原生日频权重"""
    from stable_baselines3 import PPO
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
        raise ValueError(f"观测维度不匹配：env={obs.shape}, model={model.observation_space.shape}")

    target_date = pd.Timestamp(as_of_date)
    done = False
    step_idx = 0
    last_date, last_weight = None, None

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

    return {
        "date": str(last_date.date()),
        "weights": {code: float(last_weight[i]) for i, code in enumerate(ETF_CODES)},
        "n_features": int(feat_norm.shape[1]),
        "model_obs_dim": int(model.observation_space.shape[0]),
        "steps": int(step_idx + 1),
        "source": "ppo",
        "market_state": get_market_state(as_of_date),
    }


def infer_sasf(as_of_date):
    """SASF 硬切换：下行/极寒 → 动态评分，其余 → PPO"""
    state = get_market_state(as_of_date)
    if state in ("下行", "极寒"):
        dw = get_dynamic_score_weight(as_of_date)
        if dw is not None:
            return {
                "date": str(pd.Timestamp(as_of_date).date()),
                "weights": dw,
                "source": "dynamic_score",
                "market_state": state,
                "message": f"市场状态={state} → 切换到动态评分权重",
            }
    # 其余状态用 PPO
    result = infer_ppo(as_of_date)
    result["market_state"] = state
    result["message"] = f"市场状态={state} → 使用 PPO 权重"
    return result


if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else str(pd.Timestamp.today().date())
    mode = sys.argv[2] if len(sys.argv) > 2 else "ppo"  # "ppo" or "sasf"
    if mode == "sasf":
        result = infer_sasf(date_arg)
    else:
        result = infer_ppo(date_arg)
    print(json.dumps(result, ensure_ascii=False))