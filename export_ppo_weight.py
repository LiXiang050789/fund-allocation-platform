# export_ppo_weight.py
"""从 H_ensemble 的 5 个 PPO 模型导出全时段权重，供回测对比.py 使用"""
import os, sys, inspect
import numpy as np
import pandas as pd
import torch, torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "platform"))
try:
    import gymnasium
    sys.modules.setdefault("gym", gymnasium)
except Exception:
    pass

from stable_baselines3 import PPO
from portfolio_env_global1 import PortfolioEnvGlobal, ASSET_NAMES, BOND_IDX, N_ASSET

# ============ 配置 ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "clean", "train_feature_filtered.csv")
PRICE_PATH = os.path.join(SCRIPT_DIR, "clean", "etf_price_clean.csv")
MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, "models", "global_ppo")
OUT_PATH = os.path.join(SCRIPT_DIR, "weight", "ppo_weight1.csv")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

MODEL_NAMES = [
    "H_equity_residual_best",
    "H_equity_residual_43_best",
    "H_equity_residual_44_best",
    "H_equity_residual_45_best",
    "H_equity_residual_46_best",
]

# 与训练时完全一致
ENV_CFG = {
    "window": 40,
    "reward_coef": (10.0, 0.5, 0.0, 1.0),
    "temp": None,
    "min_w": 0.01,
    "max_w": 0.25,
    "trade_cost": 0.0003,
    "constraint_coef": 10.0,
    "rebalance_monthly": False,
    "enable_bond_regime_reward": False,
    "enable_bond_loss_couple": False,
    "enable_ladder_dd": False,
    "enable_min_bond_hard_constraint": True,
    "enable_reward_norm": False,
    "enable_lgb_pred_obs": False,
    "reward_mode": "excess_rule",
    "min_bond_weight": 0.05,
    "action_bound": 5.0,
    "enable_residual": False,
    "residual_scale": 0.15,
    "enable_equity_only": True,
}

# 训练时用的切分（用于反标准化）
TRAIN_END_FRAC = 0.75
SEED = 42


def load_full_data():
    feat = pd.read_csv(DATA_PATH, parse_dates=["date"], encoding="utf-8-sig")
    price = pd.read_csv(PRICE_PATH, parse_dates=["date"], encoding="gbk")
    price = price[[c for c in price.columns if c == "date" or c in ASSET_NAMES]]

    MISSING_COLS = ["north_net", "north_20d_sum", "hs300_pe_quantile", "社融存量增速(近似)"]
    for col in MISSING_COLS:
        if col in feat.columns:
            feat[f"{col}_missing"] = feat[col].isna().astype(np.float32)
    feat = feat.ffill()
    for col in feat.columns:
        if col == "date":
            continue
        feat[col] = feat[col].fillna(feat[col].median())

    price = price.ffill().fillna(0)
    common = set(feat["date"]) & set(price["date"])
    feat = feat[feat["date"].isin(common)].reset_index(drop=True)
    price = price[price["date"].isin(common)].reset_index(drop=True)
    feat = feat.iloc[1:].reset_index(drop=True)
    price = price.iloc[1:].reset_index(drop=True)
    return feat.drop(columns=["date"]), feat["date"].copy(), price


def build_env(start_idx, end_idx, feat_norm, price_df):
    sig = inspect.signature(PortfolioEnvGlobal.__init__)
    valid = {p for p in sig.parameters if p not in ("self", "args", "kwargs")}
    cfg = {k: v for k, v in ENV_CFG.items() if k in valid}
    env = PortfolioEnvGlobal(feature_df=feat_norm, price_df=price_df,
                             start_idx=start_idx, end_idx=end_idx, **cfg)
    env.reset(seed=SEED)
    return env


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    feat_full, date_all, price_full = load_full_data()
    total_len = len(feat_full)
    train_end = int(total_len * TRAIN_END_FRAC)

    # 反标准化参数（必须用训练集统计）
    mean = feat_full.iloc[:train_end].mean()
    std = feat_full.iloc[:train_end].std().where(lambda s: s > 1e-8, 1.0)
    feat_norm_all = (feat_full - mean) / std

    # 全时段（从 0 到末尾），让 PPO 在整段历史上生成权重
    # 起点设为 1，避免 idx=0 时 _is_rebalance_day 报错
    env = build_env(1, total_len - 1, feat_norm_all.reset_index(drop=True), price_full)
    print(f"[Env] 全时段: idx 1 ~ {total_len - 1}, 共 {total_len - 2} 天")

    # 加载 5 个模型
    models = []
    for name in MODEL_NAMES:
        path = os.path.join(MODEL_SAVE_DIR, name)
        if not os.path.exists(path + ".zip"):
            print(f"⚠️  缺失: {path}.zip")
            continue
        models.append(PPO.load(path, device=device))
        print(f"✅ 加载: {name}")

    if not models:
        print("❌ 无模型，退出")
        return
    print(f"\n共 {len(models)} 个模型参与集成\n")

    # 逐日推理
    obs, _ = env.reset(seed=SEED)
    done = False
    rows = []
    idx = 1

    while not done:
        # 5 个模型 action 平均
        acts = np.stack([m.predict(obs, deterministic=True)[0] for m in models], axis=0)
        ens_action = acts.mean(axis=0)

        # 记录当前日期的权重（用 env 内部投影后的真实权重）
        obs, reward, terminated, truncated, info = env.step(ens_action)
        done = terminated or truncated

        # 权重 = env 当前 target
        w = env.last_target_weight.copy()
        date = date_all.iloc[idx] if idx < len(date_all) else date_all.iloc[-1]

        row = {"date": date}
        for j, n in enumerate(ASSET_NAMES):
            row[n] = float(w[j])
        rows.append(row)
        idx += 1

    # 保存
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n✅ 权重文件已导出: {OUT_PATH}")
    print(f"   共 {len(df)} 天")
    print(f"   起止: {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
    print(f"\n   前 3 行预览:")
    print(df.head(3).to_string(index=False))
    print(f"\n   全时段平均权重:")
    for n in ASSET_NAMES:
        print(f"     {n:8s}: {df[n].mean():.4f}")

    # 检查关键区间
    print(f"\n   分区间平均权重（关键验证）:")
    for tag, start, end in [
        ("训练段 2015-2023", "2015-01-06", "2023-08-16"),
        ("验证段 2023-2025", "2023-08-17", "2025-05-16"),
        ("测试段 2025-2026", "2025-05-19", "2026-07-10"),
    ]:
        sub = df[(df["date"] >= start) & (df["date"] <= end)]
        if len(sub) == 0:
            continue
        print(f"   [{tag}] n={len(sub)}")
        for n in ASSET_NAMES:
            print(f"     {n:8s}: {sub[n].mean():.4f}")


if __name__ == "__main__":
    main()