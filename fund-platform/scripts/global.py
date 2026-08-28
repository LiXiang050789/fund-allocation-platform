# 屏蔽TF日志 + CPU多线程加速（必须放在所有导入最顶部）
import os
# 多线程配置置顶
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # 新增：关闭oneDNN浮点提示
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_PROC_BIND"] = "close"
os.environ["OMP_PLACES"] = "cores"
import inspect
# 消除Gym弃用警告
import warnings
import logging
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("gym").setLevel(logging.ERROR)

import warnings
import logging
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# 屏蔽gym库顶层提示
logging.getLogger("gym").setLevel(logging.ERROR)

import random
import numpy as np
import torch
import torch.nn as nn
import sys
import pandas as pd
import gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "platform"))
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from portfolio_env_global import PortfolioEnvGlobal
from portfolio_metrics import calc_portfolio_metrics


# ===================== 全局随机种子（补全全部随机源） =====================
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

# ===================== 路径配置（统一拼接，消除硬编码路径） =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(BASE, "data", "clean", "train_feature_filtered.csv")
PRICE_PATH = os.path.join(BASE, "data", "clean", "etf_price_clean.csv")
EXPORT_CSV_PATH = os.path.join(BASE, "results", "ppo_net_global.csv")

MODEL_SAVE_DIR = os.path.join(BASE, "models", "global_ppo")
RESULT_SAVE_DIR = os.path.join(BASE, "results", "global_ppo")
LOG_SAVE_DIR = os.path.join(BASE, "logs", "global_ppo_log")
for folder in [MODEL_SAVE_DIR, RESULT_SAVE_DIR, LOG_SAVE_DIR]:
    os.makedirs(folder, exist_ok=True)

# ===================== 消融实验开关（移除enable_tb_log，单独拆分） =====================
EXPERIMENT_CFG = {
    "rebalance_monthly": False,
    "enable_bond_regime_reward": False,
    "enable_bond_loss_couple": True,
    "enable_ladder_dd": True,
    "enable_min_bond_hard_constraint": True,
    "enable_reward_norm": True,
    "enable_lgb_pred_obs": False
}
# Tensorboard独立开关，不传入环境
ENABLE_TB_LOG = False

# ===================== 全局超参 =====================
RL_WINDOW = 40

REWARD_COEF = (3.0, 0.18, 1.3, 0.65)

EPOCH_STEP = 3000
EARLY_STOP_PATIENCE = 30
MIN_ITER = 10
TRAIN_ITER_NUM = 40

# PPO超参：熵系数下调降低激进调仓
PPO_KWARGS = {
    "learning_rate": lambda f: 1e-4 * (1 - f * 0.9),   # 初始1e-4，后期1e-5
    "n_steps": 2048,
    "batch_size": 256,
    "gamma": 0.98,
    "clip_range": 0.12,
    "ent_coef": 0.08,                                 # 提高探索
    "max_grad_norm": 0.5,
    "verbose": 0,
    "device": device,
    "seed": SEED,
    "policy_kwargs": {
        "net_arch": [64, 32],                         # 降维
        "activation_fn": nn.Tanh
    }
}


MAX_WEIGHT = 0.25
MIN_WEIGHT = 0.01
TRADE_COST = 0.0003

# 冗余函数注释，未启用market_regime可直接删除
# def regime_one_hot(regime):
#     vec = np.zeros(4, dtype=np.float32)
#     r = min(max(int(regime), 0), 3)
#     vec[r] = 1.0
#     return vec

# ===================== 数据加载 =====================
def load_full_data():
    feat = pd.read_csv(DATA_PATH, parse_dates=["date"], encoding="utf-8-sig")
    price = pd.read_csv(PRICE_PATH, parse_dates=["date"], encoding="gbk")
    # 仅保留价格列，减少内存占用
    price = price[[c for c in price.columns if c == "date" or c != "date"]]
    feat = feat.ffill().fillna(0)
    price = price.ffill().fillna(0)
    common_date = set(feat["date"]) & set(price["date"])
    feat = feat[feat["date"].isin(common_date)].reset_index(drop=True)
    price = price[price["date"].isin(common_date)].reset_index(drop=True)

    feat = feat.iloc[1:].reset_index(drop=True)
    price = price.iloc[1:].reset_index(drop=True)

    date_col = feat["date"].copy()
    feat_only = feat.drop(columns=["date"])

    if EXPERIMENT_CFG["enable_lgb_pred_obs"]:
        lgb_cols = [f"lgb_pred_{name}" for name in ["hs300", "zz500", "kc50", "consume", "chip", "gold", "bond10"]]
        for col in lgb_cols:
            assert col in feat_only.columns, f"开启LGB观测缺失特征列:{col}"
    return feat_only, date_col, price

def get_train_scaler(train_df):
    mean = train_df.mean()
    std = train_df.std()
    std = std.where(std > 1e-8, 1.0)
    return mean, std

def apply_norm(df, mean, std):
    return (df - mean) / std

# ===================== 环境工厂函数 =====================


def build_env(start_idx, end_idx, feat_norm, price_df, seed=SEED):
    # 获取环境 __init__ 的参数名
    sig = inspect.signature(PortfolioEnvGlobal.__init__)
    valid_params = [p for p in sig.parameters.keys() if p not in ('self', 'args', 'kwargs')]
    filtered_cfg = {k: v for k, v in EXPERIMENT_CFG.items() if k in valid_params}

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
        constraint_coef=10.0,
        **filtered_cfg
    )
    env.reset(seed=seed)
    return Monitor(env)


# ===================== 全局一次性训练主逻辑 =====================
def global_train():
    feat_full, date_all, price_full = load_full_data()
    total_len = len(feat_full)
    # 三分法
    train_end = int(total_len * 0.6)
    val_end = int(total_len * 0.8)   # test 为 [val_end, total_len)

    train_feat_raw = feat_full.iloc[:train_end].copy()
    train_mean, train_std = get_train_scaler(train_feat_raw)
    feat_norm_all = apply_norm(feat_full, train_mean, train_std)

    # 训练环境
    train_env = build_env(0, train_end, feat_norm_all.iloc[:train_end].reset_index(drop=True), price_full)

    # 验证环境（使用对应切片）
    val_feat_norm = feat_norm_all.iloc[train_end:val_end].reset_index(drop=True)
    val_price_slice = price_full.iloc[train_end:val_end].reset_index(drop=True)
    val_env = build_env(0, len(val_feat_norm)-1, val_feat_norm, val_price_slice)

    # 测试环境（最后20%）
    test_feat_norm = feat_norm_all.iloc[val_end:].reset_index(drop=True)
    test_price_slice = price_full.iloc[val_end:].reset_index(drop=True)
    test_env = build_env(0, len(test_feat_norm)-1, test_feat_norm, test_price_slice)

    model = PPO("MlpPolicy", train_env, **PPO_KWARGS)
    best_sharpe = -np.inf
    patience = 0
    best_model_path = os.path.join(MODEL_SAVE_DIR, "global_best.zip")
    log_records = []

    for iter_idx in range(TRAIN_ITER_NUM):
        model.learn(total_timesteps=EPOCH_STEP, reset_num_timesteps=True, tb_log_name=f"iter_{iter_idx}")

        # 验证集评估
        obs, _ = val_env.reset(seed=SEED)
        done = False
        net_list = [1.0]
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = val_env.step(action)
            done = terminated or truncated
            net_list.append(val_env.unwrapped.net_value)

        metrics = calc_portfolio_metrics(pd.Series(net_list))
        current_sharpe = metrics["夏普比率"]
        print(f"Iter {iter_idx} | Val Sharpe: {current_sharpe:.4f}")

        # 早停只基于夏普
        if current_sharpe > best_sharpe:
            best_sharpe = current_sharpe
            model.save(best_model_path)
            patience = 0
            print(f"✅ 更新最优模型，Val Sharpe={best_sharpe:.4f}")
        else:
            patience += 1

        if patience >= EARLY_STOP_PATIENCE and iter_idx >= MIN_ITER:
            print(f"🛑 早停，最优验证夏普={best_sharpe:.4f}")
            break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 加载最优模型，在测试集上最终评估
    best_model = PPO.load(best_model_path, device=device)
    obs, _ = test_env.reset()
    done = False
    net_list_test = [1.0]
    step_idx = 0
    pred_records = []
    while not done:
        action, _ = best_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        net_list_test.append(test_env.unwrapped.net_value)
        pred_records.append({
            "date": date_all.iloc[val_end + step_idx],  # 索引需偏移到测试集起始
            "net_value": test_env.unwrapped.net_value,
            "turnover": info["turnover"]
        })
        step_idx += 1

    test_metrics = calc_portfolio_metrics(pd.Series(net_list_test))
    print("\n===== 测试集最终指标（真实泛化性能） =====")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    # 保存测试结果
    pred_df = pd.DataFrame(pred_records)
    pred_df.to_csv(os.path.join(RESULT_SAVE_DIR, "global_ppo_test_net.csv"), index=False, encoding="utf-8-sig")
    pred_df[["date", "net_value"]].to_csv(EXPORT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ 测试集净值已导出至 {EXPORT_CSV_PATH}")

if __name__ == "__main__":
    global_train()
