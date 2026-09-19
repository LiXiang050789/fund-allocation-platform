# 屏蔽TF日志 + CPU多线程加速（必须放在所有导入最顶部）
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_PROC_BIND"] = "close"
os.environ["OMP_PLACES"] = "cores"
import inspect
import warnings
import logging
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("gym").setLevel(logging.ERROR)
import random
import numpy as np
import torch
import torch.nn as nn
import sys
import pandas as pd
import gc
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "platform"))

try:
    import gymnasium
    sys.modules.setdefault("gym", gymnasium)
except Exception:
    pass

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor, load_results
from env_16 import PortfolioEnvGlobal, ASSET_NAMES, BOND_IDX, N_ASSET
from portfolio_metrics import calc_portfolio_metrics

# ===================== 全局随机种子 =====================
SEED = 46
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
EXPORT_CSV_PATH = os.path.join(SCRIPT_DIR, "clean", "ppo_net_global.csv")
MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, "models", "global_ppo")
RESULT_SAVE_DIR = os.path.join(SCRIPT_DIR, "results", "global_ppo")
LOG_SAVE_DIR = os.path.join(SCRIPT_DIR, "logs", "global_ppo_log")
for folder in [MODEL_SAVE_DIR, RESULT_SAVE_DIR, LOG_SAVE_DIR]:
    os.makedirs(folder, exist_ok=True)

# ===================== 单变量实验配置 =====================
# 用法：每次只改一个字段，把 name 也改掉，方便对比。
# 若要跑 baseline，全部保持默认即可。
RUN_CFG = {
    "name": "J_bond_only",
    "min_bond_weight": 0.05,
    "action_bound": 5.0,
    "min_w": 0.01,               # ← 补
    "max_w": 0.25,               # ← 补
    "trade_cost": 0.0003,        # ← 补
    "reward_mode": "excess_rule",
    "reward_coef": (10.0, 0.5, 0.0, 1.0),
    "train_start_offset": 0,
    "train_end_frac": 0.75,
    "val_end_frac": 0.9,
    "ent_coef": 0.12,
    "learning_rate": 5e-05,
    "n_steps": 1024,
    "batch_size": 128,
    "gamma": 0.99,
    "clip_range": 0.2,
    "max_grad_norm": 0.5,
    "net_arch": [128, 64],
    "rebalance_monthly": False,
    "enable_min_bond_hard_constraint": True,
    "enable_reward_norm": False,
    "enable_bond_regime_reward": False,
    "enable_bond_loss_couple": False,
    "enable_ladder_dd": False,
    "enable_lgb_pred_obs": False,
    "enable_residual": False,
    "enable_equity_only": True,
    "residual_scale": 0.20,
}

# 训练循环
RL_WINDOW = 40
EPOCH_STEP = 6000
EARLY_STOP_PATIENCE = 15
MIN_ITER = 20
TRAIN_ITER_NUM = 60

# 打印当前配置
print("\n===== 当前实验配置 =====")
for k, v in RUN_CFG.items():
    print(f"  {k} = {v}")
print("========================\n")

EXP_NAME = RUN_CFG["name"]
ENABLE_TB_LOG = False


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
        med = feat[col].median()
        feat[col] = feat[col].fillna(med)

    price = price.ffill().fillna(0)

    common_date = set(feat["date"]) & set(price["date"])
    feat = feat[feat["date"].isin(common_date)].reset_index(drop=True)
    price = price[price["date"].isin(common_date)].reset_index(drop=True)
    feat = feat.iloc[1:].reset_index(drop=True)
    price = price.iloc[1:].reset_index(drop=True)

    date_col = feat["date"].copy()
    feat_only = feat.drop(columns=["date"])

    if RUN_CFG["enable_lgb_pred_obs"]:
        for col in [f"lgb_pred_{n}" for n in ASSET_NAMES]:
            assert col in feat_only.columns, f"开启LGB观测缺失特征列:{col}"
    return feat_only, date_col, price


def get_train_scaler(train_df):
    mean = train_df.mean()
    std = train_df.std().where(lambda s: s > 1e-8, 1.0)
    return mean, std


def apply_norm(df, mean, std):
    return (df - mean) / std


# ===================== 价格诊断 =====================
def diagnose_prices(price_slice, tag):
    print(f"\n-- 价格诊断 [{tag}] (len={len(price_slice)}) --")
    arr = price_slice[ASSET_NAMES].values.astype(np.float64)
    for i, name in enumerate(ASSET_NAMES):
        col = arr[:, i]
        n_zero = int((col < 1e-6).sum())
        pos = col[col > 1e-6]
        pos_min = float(pos.min()) if len(pos) > 0 else float("nan")
        line = f"  {name:8s} | zero={n_zero:5d} | min>0={pos_min:.4f}"
        if n_zero > 0 and n_zero < len(col):
            zero_idx = np.where(col < 1e-6)[0]
            head_dates = price_slice.iloc[zero_idx[:2]]["date"].dt.date.tolist()
            tail_dates = price_slice.iloc[zero_idx[-2:]]["date"].dt.date.tolist()
            line += f" | zero@head={head_dates} zero@tail={tail_dates}"
        print(line)


# ===================== 环境工厂 =====================
def build_env(start_idx, end_idx, feat_norm, price_df,
              seed=SEED, monitor_filename=None):
    sig = inspect.signature(PortfolioEnvGlobal.__init__)
    valid = {p for p in sig.parameters if p not in ("self", "args", "kwargs")}

    cfg = {
        "window": RL_WINDOW,
        "reward_coef": RUN_CFG["reward_coef"],
        "temp": None,   # 用默认温度 1.0
        "min_w": RUN_CFG["min_w"],
        "max_w": RUN_CFG["max_w"],
        "trade_cost": RUN_CFG["trade_cost"],
        "constraint_coef": 10.0,
        "rebalance_monthly": RUN_CFG["rebalance_monthly"],
        "enable_bond_regime_reward": RUN_CFG["enable_bond_regime_reward"],
        "enable_bond_loss_couple": RUN_CFG["enable_bond_loss_couple"],
        "enable_ladder_dd": RUN_CFG["enable_ladder_dd"],
        "enable_min_bond_hard_constraint": RUN_CFG["enable_min_bond_hard_constraint"],
        "enable_reward_norm": RUN_CFG["enable_reward_norm"],
        "enable_lgb_pred_obs": RUN_CFG["enable_lgb_pred_obs"],
        "reward_mode": RUN_CFG["reward_mode"],
        "min_bond_weight": RUN_CFG["min_bond_weight"],
        "action_bound": RUN_CFG["action_bound"],
        "enable_residual": RUN_CFG.get("enable_residual", False),
        "residual_scale": RUN_CFG.get("residual_scale", 0.30),
        "enable_equity_only": RUN_CFG.get("enable_equity_only", False)
    }
    cfg = {k: v for k, v in cfg.items() if k in valid}
    print(f"[EnvCfg] {cfg}")  # 临时调试

    env = PortfolioEnvGlobal(
        feature_df=feat_norm,
        price_df=price_df,
        start_idx=start_idx,
        end_idx=end_idx,
        **cfg,
    )
    env.reset(seed=seed)
    return Monitor(env, filename=monitor_filename) if monitor_filename else Monitor(env)


# ===================== 基线评估 =====================
def eval_static_baseline(price_df, weights):
    prices = price_df[ASSET_NAMES].values.astype(np.float64)
    prices = np.where(prices < 1e-6, np.nan, prices)
    prices_df = pd.DataFrame(prices).ffill()
    prices = prices_df.values

    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()

    rets = np.zeros_like(prices)
    valid = ~np.isnan(prices[:-1]) & ~np.isnan(prices[1:])
    raw = np.zeros_like(prices[:-1])
    raw[valid] = (prices[1:][valid] - prices[:-1][valid]) / (prices[:-1][valid] + 1e-12)
    raw = np.clip(raw, -0.20, 0.20)
    rets[:-1] = raw

    port_ret = rets @ w
    net = np.cumprod(1.0 + port_ret)
    net = np.concatenate([[1.0], net[:-1]])
    return net


def run_and_print_baselines(price_slice, tag):
    equal_w = np.array([1.0 / 6] * 6 + [0.0])
    b6040_w = np.array([0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4])
    heavy_bond_w = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.70])
    hs300_only = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    results = {}
    for name, w in [("equal", equal_w), ("60/40", b6040_w),
                    ("heavy_bond", heavy_bond_w), ("hs300", hs300_only)]:
        net = eval_static_baseline(price_slice, w)
        m = calc_portfolio_metrics(pd.Series(net))
        results[name] = m
        print(f"  [{tag}] {name:12s} | "
              f"Cum={m['累计收益']:+.4f} | Ann={m['年化收益']:+.4f} | "
              f"Sharpe={m['夏普比率']:+.4f} | MDD={m['最大回撤']:+.4f}")
    return results
def run_rule_baseline(feat_df, price_df, date_all, tag):
    """
    用 PE 分位数当股债比例的"简单规则"，跳过 PPO。
    规则：PE 分位数越高 → 债券越多
         bond_w = clip(0.10 + 0.60 * pe_quantile, 0.05, 0.70)
         权益 6 只等权分配 (1 - bond_w)
    """
    # 对齐 PE 列
    if "hs300_pe_quantile" not in feat_df.columns:
        print(f"  [{tag}] rule baseline: 缺少 hs300_pe_quantile 列，跳过")
        return None

    pe = feat_df["hs300_pe_quantile"].values  # 已经是归一化后的值
    # 反归一化 → 若归一化了，用均值0.5近似；这里直接用 raw 特征
    # 简化：直接用原始特征文件里的 PE 分位
    # 如果你的 feat_df 已经标准化，请传入未经标准化的原始特征
    pe = np.clip(pe, 0.0, 1.0)   # 假定已经在 0~1
    bond_w = np.clip(0.10 + 0.60 * pe, 0.05, 0.70)

    prices = price_df[ASSET_NAMES].values.astype(np.float64)
    prices = np.where(prices < 1e-6, np.nan, prices)
    prices = pd.DataFrame(prices).ffill().values

    rets = np.zeros_like(prices)
    valid = ~np.isnan(prices[:-1]) & ~np.isnan(prices[1:])
    raw = np.zeros_like(prices[:-1])
    raw[valid] = (prices[1:][valid] - prices[:-1][valid]) / (prices[:-1][valid] + 1e-12)
    raw = np.clip(raw, -0.20, 0.20)
    rets[:-1] = raw

    # 每天按当天 bond_w 加权
    T = rets.shape[0]
    port_ret = np.zeros(T)
    for t in range(T):
        bw = bond_w[t] if t < len(bond_w) else 0.30
        ew = 1.0 - bw
        w = np.array([ew / 6] * 6 + [bw])
        port_ret[t] = rets[t] @ w

    net = np.cumprod(1.0 + port_ret)
    net = np.concatenate([[1.0], net[:-1]])
    m = calc_portfolio_metrics(pd.Series(net))
    print(f"  [{tag}] rule(PE)    | "
          f"Cum={m['累计收益']:+.4f} | Ann={m['年化收益']:+.4f} | "
          f"Sharpe={m['夏普比率']:+.4f} | MDD={m['最大回撤']:+.4f}")
    return m


# ===================== 训练主逻辑 =====================
def global_train():
    feat_full, date_all, price_full = load_full_data()
    total_len = len(feat_full)

    train_start = int(RUN_CFG["train_start_offset"])
    train_end = int(total_len * RUN_CFG["train_end_frac"])
    val_end = int(total_len * RUN_CFG["val_end_frac"])
    if train_start >= train_end:
        raise ValueError(f"train_start={train_start} >= train_end={train_end}")

    train_mean, train_std = get_train_scaler(
        feat_full.iloc[train_start:train_end]
    )
    feat_norm_all = apply_norm(feat_full, train_mean, train_std)

    # --- 切片 ---
    train_feat_slice = feat_norm_all.iloc[train_start:train_end].reset_index(drop=True)
    train_price_slice = price_full.iloc[train_start:train_end + 1].reset_index(drop=True)

    val_feat_norm = feat_norm_all.iloc[train_end - 1:val_end].reset_index(drop=True)
    val_price_slice = price_full.iloc[train_end - 1:val_end].reset_index(drop=True)

    test_feat_norm = feat_norm_all.iloc[val_end - 1:].reset_index(drop=True)
    test_price_slice = price_full.iloc[val_end - 1:].reset_index(drop=True)

    # --- 数据切分信息 ---
    print(f"\n[DataSplit] total={total_len}, "
          f"train=[{train_start}, {train_end}), "
          f"val=[{train_end}, {val_end}), "
          f"test=[{val_end}, {total_len})")
    print(f"[TrainPeriod] {date_all.iloc[train_start].date()} ~ "
          f"{date_all.iloc[train_end - 1].date()}")
    print(f"[ValPeriod]   {date_all.iloc[train_end].date()} ~ "
          f"{date_all.iloc[val_end - 1].date()}")
    print(f"[TestPeriod]  {date_all.iloc[val_end].date()} ~ "
          f"{date_all.iloc[-1].date()}")

    # --- 价格诊断 ---
    diagnose_prices(train_price_slice, "train")
    diagnose_prices(val_price_slice,   "val")
    diagnose_prices(test_price_slice,  "test")

    # --- 基线 ---
    print("\n===== 基线对照（价格直接计算） =====")
    run_and_print_baselines(train_price_slice, "train")
    run_and_print_baselines(val_price_slice,   "val")
    run_and_print_baselines(test_price_slice,  "test")
    print("-- 规则基线（PE 分位数定股债） --")
    run_rule_baseline(feat_full.iloc[:train_end], price_full.iloc[:train_end + 1],
                      date_all.iloc[:train_end], "train")
    run_rule_baseline(feat_full.iloc[train_end - 1:val_end], price_full.iloc[train_end - 1:val_end],
                      date_all.iloc[train_end:val_end], "val")
    run_rule_baseline(feat_full.iloc[val_end - 1:], price_full.iloc[val_end - 1:],
                      date_all.iloc[val_end:], "test")

    # --- 构建环境 ---
    train_env = build_env(
        0, train_end - train_start, train_feat_slice, train_price_slice,
        monitor_filename=os.path.join(LOG_SAVE_DIR, f"monitor_{EXP_NAME}_train.csv"),
    )
    val_env = build_env(1, len(val_feat_norm) - 1, val_feat_norm, val_price_slice)
    test_env = build_env(1, len(test_feat_norm) - 1, test_feat_norm, test_price_slice)
    # 临时检查规则
    rule_w = train_env.unwrapped._rule_baseline_weight()
    print(f"[RuleCheck] rule weight = {dict(zip(ASSET_NAMES, rule_w.round(3)))}")
    # --- 训练 ---
    ppo_kwargs = {
        "learning_rate": RUN_CFG["learning_rate"],
        "n_steps": RUN_CFG["n_steps"],
        "batch_size": RUN_CFG["batch_size"],
        "gamma": RUN_CFG["gamma"],
        "clip_range": RUN_CFG["clip_range"],
        "ent_coef": RUN_CFG["ent_coef"],
        "max_grad_norm": RUN_CFG["max_grad_norm"],
        "verbose": 0,
        "device": device,
        "seed": SEED,
        "policy_kwargs": {
            "net_arch": RUN_CFG["net_arch"],
            "activation_fn": nn.Tanh,
        },
    }
    model = PPO("MlpPolicy", train_env, **ppo_kwargs)

    best_sharpe = -np.inf
    patience = 0
    best_model_path = os.path.join(MODEL_SAVE_DIR, f"{EXP_NAME}_best")
    history = []

    for iter_idx in range(TRAIN_ITER_NUM):
        model.learn(total_timesteps=EPOCH_STEP)
        train_env.reset()

        try:
            mon_df = load_results(LOG_SAVE_DIR)
            if len(mon_df) > 0:
                recent = mon_df["r"].tail(min(5, len(mon_df)))
                print(f"[TrainStat] iter {iter_idx} "
                      f"recent mean_reward={recent.mean():.4f} "
                      f"(episodes={len(mon_df)})")
        except Exception as e:
            print(f"[TrainStat] skip: {e}")

        # ---------- 验证 ----------
        obs, _ = val_env.reset(seed=SEED)
        done = False
        net_list = [1.0]
        actions, turnovers, bond_ws, equity_ws, weights_hist = [], [], [], [], []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            actions.append(np.asarray(action, dtype=np.float32))
            obs, reward, terminated, truncated, info = val_env.step(action)
            done = terminated or truncated
            net_list.append(val_env.unwrapped.net_value)
            turnovers.append(info["turnover"])
            bond_ws.append(info["bond_weight"])
            tgt_w = val_env.unwrapped.last_target_weight
            equity_ws.append(float(np.sum(tgt_w[:BOND_IDX])))
            weights_hist.append(tgt_w.copy())

        metrics = calc_portfolio_metrics(pd.Series(net_list))
        current_sharpe = metrics["夏普比率"]

        actions = np.stack(actions, axis=0)
        act_std = float(actions.std(axis=0).mean())
        act_sat = float((np.abs(actions) >= RUN_CFG["action_bound"] * 0.98).mean())
        equity_ws = np.asarray(equity_ws)
        weights_hist = np.stack(weights_hist, axis=0)

        history.append({
            "iter": iter_idx,
            "val_sharpe": current_sharpe,
            "val_net": net_list[-1],
            "val_mdd": metrics["最大回撤"],
            "turn_mean": float(np.mean(turnovers)),
            "equity_mean": float(equity_ws.mean()),
            "equity_min": float(equity_ws.min()),
            "equity_max": float(equity_ws.max()),
            "act_std": act_std,
            "act_sat": act_sat,
        })

        print(f"Iter {iter_idx:>3d} | "
              f"Sharpe={current_sharpe:+.4f} | "
              f"Net[last]={net_list[-1]:.4f} | "
              f"MDD={metrics['最大回撤']:+.4f} | "
              f"Turn={np.mean(turnovers):.4f} | "
              f"EqW={equity_ws.mean():.3f}[{equity_ws.min():.3f},{equity_ws.max():.3f}] | "
              f"ActStd={act_std:.3f} | ActSat={act_sat:.3f}")
        if iter_idx % 5 == 0 or iter_idx == TRAIN_ITER_NUM - 1:
            avg_w_str = ", ".join(
                f"{n}={weights_hist[:, i].mean():.3f}"
                for i, n in enumerate(ASSET_NAMES))
            print(f"        AvgW: {avg_w_str}")

        if current_sharpe > best_sharpe:
            best_sharpe = current_sharpe
            model.save(best_model_path)
            patience = 0
            print(f"✅ 更新最优模型，Val Sharpe={best_sharpe:.4f}")
        else:
            if iter_idx >= MIN_ITER:
                patience += 1

        if patience >= EARLY_STOP_PATIENCE and iter_idx >= MIN_ITER:
            print(f"🛑 早停，最优验证夏普={best_sharpe:.4f}")
            break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------- 测试集最终评估 ----------
    best_model = PPO.load(best_model_path, device=device)
    obs, _ = test_env.reset(seed=SEED)
    done = False
    net_list_test = [1.0]
    step_idx = 0
    pred_records = []
    test_actions, test_weights, test_equity_ws = [], [], []
    while not done:
        action, _ = best_model.predict(obs, deterministic=True)
        test_actions.append(np.asarray(action, dtype=np.float32))
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        net_value = test_env.unwrapped.net_value
        net_list_test.append(net_value)
        tgt_w = test_env.unwrapped.last_target_weight
        test_weights.append(tgt_w.copy())
        test_equity_ws.append(float(np.sum(tgt_w[:BOND_IDX])))

        date_idx = val_end + step_idx + 1
        if date_idx < len(date_all):
            pred_records.append({
                "date": date_all.iloc[date_idx],
                "net_value": net_value,
                "turnover": info["turnover"],
                "equity_weight": test_equity_ws[-1],
                "bond_weight": info["bond_weight"],
            })
        step_idx += 1

    test_metrics = calc_portfolio_metrics(pd.Series(net_list_test))
    print("\n===== 测试集最终指标 =====")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    test_actions = np.stack(test_actions, axis=0)
    test_weights = np.stack(test_weights, axis=0)
    test_equity_ws = np.asarray(test_equity_ws)
    print(f"测试集动作：std={test_actions.std(axis=0).mean():.4f}, "
          f"range=[{test_actions.min():+.3f}, {test_actions.max():+.3f}], "
          f"sat={float((np.abs(test_actions) >= RUN_CFG['action_bound'] * 0.98).mean()):.3f}")
    print(f"测试集权益仓位：mean={test_equity_ws.mean():.3f}, "
          f"range=[{test_equity_ws.min():.3f}, {test_equity_ws.max():.3f}]")
    avg_w_str = ", ".join(
        f"{n}={test_weights[:, i].mean():.3f}"
        for i, n in enumerate(ASSET_NAMES))
    print(f"测试集平均权重：{avg_w_str}")

    # ---------- 保存结果 ----------
    out_dir = os.path.join(RESULT_SAVE_DIR, EXP_NAME)
    os.makedirs(out_dir, exist_ok=True)

    pred_df = pd.DataFrame(pred_records)
    pred_df.to_csv(os.path.join(out_dir, "test_net.csv"),
                   index=False, encoding="utf-8-sig")
    pred_df[["date", "net_value"]].to_csv(EXPORT_CSV_PATH,
                                          index=False, encoding="utf-8-sig")

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(out_dir, "train_history.csv"),
                   index=False, encoding="utf-8-sig")

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in RUN_CFG.items()}, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 实验 [{EXP_NAME}] 结果已保存到 {out_dir}")
    print(f"   最优 Val Sharpe = {best_sharpe:.4f}")
    print(f"   测试集净值已导出至 {EXPORT_CSV_PATH}")


if __name__ == "__main__":
    global_train()