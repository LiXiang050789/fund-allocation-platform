# ensemble_eval.py
import os
import sys
import inspect
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "platform"))
try:
    import gymnasium
    sys.modules.setdefault("gym", gymnasium)
except Exception:
    pass

from stable_baselines3 import PPO
from portfolio_env_global1 import PortfolioEnvGlobal, ASSET_NAMES, BOND_IDX
from portfolio_metrics import calc_portfolio_metrics

# ============ 路径 ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "clean", "train_feature_filtered.csv")
PRICE_PATH = os.path.join(SCRIPT_DIR, "clean", "etf_price_clean.csv")
MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, "models", "global_ppo")
RESULT_SAVE_DIR = os.path.join(SCRIPT_DIR, "results", "global_ppo")
EXPORT_CSV_PATH = os.path.join(SCRIPT_DIR, "clean", "ppo_net_ensemble.csv")
os.makedirs(RESULT_SAVE_DIR, exist_ok=True)

# ============ 要集成的 5 个模型 ============
# 对应 seed 42, 43, 44, 45, 46
MODEL_NAMES = [
    "H_equity_residual_best",       # seed 42
    "H_equity_residual_43_best",    # seed 43
    "H_equity_residual_44_best",    # seed 44
    "H_equity_residual_45_best",    # seed 45
    "H_equity_residual_46_best",    # seed 46
]

# ============ 数据切分（必须和训练一致）============
TRAIN_END_FRAC = 0.75
VAL_END_FRAC = 0.9
SEED = 42   # 环境 reset 用，不影响集成

# ============ 环境配置（必须和训练一致）============
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

# ===================== 数据加载 =====================
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
    common_date = set(feat["date"]) & set(price["date"])
    feat = feat[feat["date"].isin(common_date)].reset_index(drop=True)
    price = price[price["date"].isin(common_date)].reset_index(drop=True)
    feat = feat.iloc[1:].reset_index(drop=True)
    price = price.iloc[1:].reset_index(drop=True)
    return feat.drop(columns=["date"]), feat["date"].copy(), price


def build_env(start_idx, end_idx, feat_norm, price_df, seed=SEED):
    sig = inspect.signature(PortfolioEnvGlobal.__init__)
    valid = {p for p in sig.parameters if p not in ("self", "args", "kwargs")}
    cfg = {k: v for k, v in ENV_CFG.items() if k in valid}
    env = PortfolioEnvGlobal(
        feature_df=feat_norm, price_df=price_df,
        start_idx=start_idx, end_idx=end_idx, **cfg,
    )
    env.reset(seed=seed)
    return env


# ===================== 主流程 =====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 1. 数据加载
    feat_full, date_all, price_full = load_full_data()
    total_len = len(feat_full)
    train_end = int(total_len * TRAIN_END_FRAC)
    val_end = int(total_len * VAL_END_FRAC)

    train_mean = feat_full.iloc[:train_end].mean()
    train_std = feat_full.iloc[:train_end].std().where(lambda s: s > 1e-8, 1.0)
    feat_norm_all = (feat_full - train_mean) / train_std

    # 2. 构建测试环境
    test_feat_norm = feat_norm_all.iloc[val_end - 1:].reset_index(drop=True)
    test_price_slice = price_full.iloc[val_end - 1:].reset_index(drop=True)
    test_env = build_env(1, len(test_feat_norm) - 1, test_feat_norm, test_price_slice)

    print(f"[TestPeriod] {date_all.iloc[val_end].date()} ~ {date_all.iloc[-1].date()}")

    # 3. 加载所有模型
    models = []
    for name in MODEL_NAMES:
        path = os.path.join(MODEL_SAVE_DIR, name)
        if not os.path.exists(path + ".zip"):
            print(f"⚠️  跳过缺失模型: {path}.zip")
            continue
        m = PPO.load(path, device=device)
        models.append(m)
        print(f"✅ 加载模型: {name}")

    if len(models) == 0:
        print("❌ 没有可用模型，退出")
        return
    print(f"\n共 {len(models)} 个模型参与集成\n")

    # 4. 集成测试
    obs, _ = test_env.reset(seed=SEED)
    done = False
    net_list = [1.0]
    step_idx = 0
    records = []
    individual_actions = {i: [] for i in range(len(models))}
    ensemble_actions = []

    while not done:
        # 每个模型给出自己的 action
        actions_each = []
        for m in models:
            a, _ = m.predict(obs, deterministic=True)
            actions_each.append(np.asarray(a, dtype=np.float32))
        actions_each = np.stack(actions_each, axis=0)  # (n_models, N_ASSET)

        # 集成：对 action 取平均
        ensemble_action = actions_each.mean(axis=0)
        ensemble_actions.append(ensemble_action)

        for i, a in enumerate(actions_each):
            individual_actions[i].append(a)

        obs, reward, terminated, truncated, info = test_env.step(ensemble_action)
        done = terminated or truncated
        net_value = test_env.unwrapped.net_value
        net_list.append(net_value)

        date_idx = val_end + step_idx + 1
        if date_idx < len(date_all):
            records.append({
                "date": date_all.iloc[date_idx],
                "net_value": net_value,
                "turnover": info["turnover"],
                "bond_weight": info["bond_weight"],
            })
        step_idx += 1

    # 5. 输出指标
    test_metrics = calc_portfolio_metrics(pd.Series(net_list))
    print("===== 集成测试集指标 =====")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    # 6. 对比每个单模型 vs 集成
    print("\n===== 单模型 vs 集成（测试集夏普） =====")
    for i, m in enumerate(models):
        name = MODEL_NAMES[i]
        acts = np.stack(individual_actions[i])
        # 用单模型 action 重跑一遍，太慢，改为对比 action 分散度
        print(f"  {name}: act_std={acts.std(axis=0).mean():.4f}")

    ens = np.stack(ensemble_actions)
    print(f"  集成:    act_std={ens.std(axis=0).mean():.4f}")

    # 测试集各资产平均权重
    obs, _ = test_env.reset(seed=SEED)
    done = False
    weights_hist = []
    while not done:
        actions_each = [m.predict(obs, deterministic=True)[0] for m in models]
        ensemble_action = np.mean(actions_each, axis=0)
        obs, _, term, trunc, _ = test_env.step(ensemble_action)
        done = term or trunc
        weights_hist.append(test_env.unwrapped.last_target_weight.copy())

    avg_w = np.stack(weights_hist).mean(axis=0)
    print("\n集成平均权重：")
    for n, w in zip(ASSET_NAMES, avg_w):
        print(f"  {n}: {w:.4f}")

    # 7. 保存结果
    out_dir = os.path.join(RESULT_SAVE_DIR, "H_ensemble")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(records).to_csv(
        os.path.join(out_dir, "test_net.csv"),
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(records)[["date", "net_value"]].to_csv(
        EXPORT_CSV_PATH, index=False, encoding="utf-8-sig"
    )
    print(f"\n✅ 结果已保存至 {out_dir}")
    print(f"✅ 净值已导出至 {EXPORT_CSV_PATH}")

    # 8. 与基线对比
    print("\n===== 对照（测试集夏普） =====")
    print("  equal         = 1.8011")
    print("  rule(PE)      = 1.7735")
    print("  H(seed=42)    = 1.8116")
    print("  H(seed=43)    = 1.6952")
    print(f"  H_ensemble    = {test_metrics['夏普比率']:.4f}")


if __name__ == "__main__":
    main()