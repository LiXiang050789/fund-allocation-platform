# 最顶部：屏蔽TF日志 + CPU多线程加速（提速核心）
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "platform"))
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# 注释断点回调，减少IO
# from stable_baselines3.common.callbacks import CheckpointCallback
# 使用优化后的夏普增量奖励环境
from portfolio_env import PortfolioEnv
from portfolio_metrics import calc_portfolio_metrics

# ===================== 全局配置 & 随机种子 =====================
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

# ===================== 轻量化时序LSTM特征提取器 =====================
class CustomLSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.input_dim = observation_space.shape[0]
        self.lstm_hidden = 96
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.lstm_hidden, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(self.lstm_hidden, 96),
            nn.ReLU(),
            nn.Linear(96, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        # 扩充时序维度 (batch, seq_len=1, feature_dim)
        x = observations.unsqueeze(1)
        # 每次前向动态初始化隐状态，不缓存，适配任意batch
        h0 = torch.zeros(1, batch_size, self.lstm_hidden, device=x.device)
        c0 = torch.zeros(1, batch_size, self.lstm_hidden, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        # 取单步时序输出
        last_hidden = out[:, -1, :]
        feat = self.proj(last_hidden)
        return feat



# ===================== 路径 & 全局超参（只保留一份PPO_KWARGS，轻量化网络） =====================
# 脚本所在目录 E:\农行杯\2026\1
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 你的csv直接在 1/clean/ 下，没有data文件夹
DATA_PATH = os.path.join(SCRIPT_DIR, "clean", "train_feature_filtered.csv")
PRICE_PATH = os.path.join(SCRIPT_DIR, "clean", "etf_price_clean.csv")

# 模型、结果、日志输出目录，放在脚本同级
MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, "models", "multi_model")
RESULT_SAVE_DIR = os.path.join(SCRIPT_DIR, "results", "multi_model")
LOG_SAVE_DIR = os.path.join(SCRIPT_DIR, "logs", "train_log")

# 自动创建输出文件夹
for p in [MODEL_SAVE_DIR, RESULT_SAVE_DIR, LOG_SAVE_DIR]:
    os.makedirs(p, exist_ok=True)

# 窗口参数
TRAIN_WINDOW = 800
VAL_WINDOW = 200
PREDICT_WINDOW = 200
EARLY_STOP_PATIENCE = 20   # 缩短早停等待，减少无效迭代
MIN_ITER = 15              # 保底迭代减少，更快出结果
EPOCH_STEP = 3000          # 单次迭代步数从4000→3000，提速25%
RL_WINDOW = 60             # 拉长观测窗口，匹配中长期奖励逻辑
REWARD_COEF = (6.0, 0.08, 0.4, 0.005)

# PPO轻量化提速参数
PPO_KWARGS = {
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 128,
    "gamma": 0.995,
    "clip_range": 0.2,
    "ent_coef": 0.02,
    "max_grad_norm": 0.5,
    "verbose": 0,
    "device": device,
    "seed": SEED,
    # 启用自定义LSTM特征提取器
    "policy_kwargs": {
        "features_extractor_class": CustomLSTMFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 64},
        "net_arch": [64, 32],
    }
}



FINE_TUNE_LR = 8e-5

# ===================== 数据加载（保留原有防泄露逻辑） =====================
def load_data():
    feat = pd.read_csv(DATA_PATH, parse_dates=["date"], encoding="gbk")
    price = pd.read_csv(PRICE_PATH, parse_dates=["date"], encoding="gbk")
    feat = feat.ffill().fillna(0)
    price = price.ffill().fillna(0)
    common_dates = set(feat["date"]) & set(price["date"])
    feat = feat[feat["date"].isin(common_dates)].reset_index(drop=True)
    price = price[price["date"].isin(feat["date"])].reset_index(drop=True)
    feat = feat.dropna().reset_index(drop=True)
    price = price.iloc[1:].reset_index(drop=True)
    date_col = feat["date"].copy()
    feat_only = feat.drop(columns=["date"])
    return feat_only, date_col, price

# ===================== Z-Score工具 =====================
def get_train_scaler(train_feat_df):
    mean = train_feat_df.mean()
    std = train_feat_df.std()
    std = std.where(std > 1e-8, 1.0)
    return mean, std

def apply_zscore(feat_df, mean, std):
    return (feat_df - mean) / std

# ===================== 单轮训练 =====================
def train_single_round(model_cls, model_kwargs, feat_only, price_df,
                       train_start, train_end, val_start, val_end,
                       round_id, model_name, prev_model_path=None):
    train_feat_raw = feat_only.iloc[train_start:train_end].copy()
    train_mean, train_std = get_train_scaler(train_feat_raw)
    feat_norm = feat_only.copy()
    feat_norm = apply_zscore(feat_norm, train_mean, train_std)

    raw_train_env = PortfolioEnv(
        feature_df=feat_norm,
        price_df=price_df,
        start_idx=train_start,
        end_idx=train_end,
        window=RL_WINDOW,
        reward_coef=REWARD_COEF
    )
    raw_val_env = PortfolioEnv(
        feature_df=feat_norm,
        price_df=price_df,
        start_idx=val_start,
        end_idx=val_end,
        window=RL_WINDOW,
        reward_coef=REWARD_COEF
    )
    train_env = Monitor(raw_train_env)
    val_env = Monitor(raw_val_env)

    # 增量加载模型，自动降低学习率
    if prev_model_path is not None and os.path.exists(prev_model_path):
        print(f"  增量加载历史模型，微调学习率={FINE_TUNE_LR}")
        model = model_cls.load(prev_model_path, env=train_env, device=device)
        model.learning_rate = FINE_TUNE_LR
    else:
        model = model_cls("MlpPolicy", train_env, **model_kwargs)

    best_sharpe = -np.inf
    patience = 0
    best_model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_round{round_id}_best.zip")
    log_path = os.path.join(LOG_SAVE_DIR, f"{model_name}_round{round_id}_train_log.csv")
    log_records = []

    # 移除多余初始learn，无断点回调
    for train_iter in range(30):
        model.learn(total_timesteps=EPOCH_STEP, reset_num_timesteps=False)
        # 验证集回测
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
            "train_iter": train_iter,
            "sharpe": current_sharpe,
            "cum_ret": metrics["累计收益"],
            "vol": metrics["年化波动"],
            "mdd": metrics["最大回撤"]
        })
        # 【提速优化】注释单轮迭代打印，仅保留关键轮次输出
        # print(f"轮次{round_id} 迭代{train_iter} | 夏普:{current_sharpe:.2f} 最大回撤:{metrics['最大回撤']:.2%}")
        if current_sharpe > best_sharpe:
            best_sharpe = current_sharpe
            model.save(best_model_path)
            patience = 0
        else:
            patience += 1
        if patience >= EARLY_STOP_PATIENCE and train_iter >= MIN_ITER:
            print(f"🛑 第{round_id}轮早停触发，连续{EARLY_STOP_PATIENCE}轮无提升，最优夏普={best_sharpe:.2f}")
            break
    pd.DataFrame(log_records).to_csv(log_path, index=False, encoding="utf-8-sig")
    best_model = model_cls.load(best_model_path, device=device)
    return best_model, train_mean, train_std

# ===================== 预测、调仓分析函数 =====================
def predict_round(model, feat_only, train_mean, train_std, date_col, price_df, pred_start, pred_end):
    feat_norm = apply_zscore(feat_only, train_mean, train_std)
    pred_env = PortfolioEnv(
        feature_df=feat_norm,
        price_df=price_df,
        start_idx=pred_start,
        end_idx=pred_end,
        window=RL_WINDOW,
        reward_coef=REWARD_COEF
    )
    obs, _ = pred_env.reset()
    done = False
    records = []
    step_idx = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = pred_env.step(action)
        done = terminated or truncated
        current_date = date_col.iloc[pred_start + step_idx]
        records.append({
            "date": current_date,
            "net_value": pred_env.unwrapped.net_value,
            "turnover": info.get("turnover", 0.0),
            "total_score": info.get("total_market_score", 0.0),
            "val_score": info.get("val_score", 0.0),
            "macro_score": info.get("macro_score", 0.0),
            "sent_score": info.get("sent_score", 0.0),
            "trend_score": info.get("trend_score", 0.0),
        })
        step_idx += 1
    return pd.DataFrame(records)

def analyze_turnover(turnover_series, trade_cost=0.0005):
    avg_turnover = turnover_series.mean()
    median_turnover = turnover_series.median()
    max_turnover = turnover_series.max()
    zero_turnover_ratio = (turnover_series == 0).mean()
    daily_cost = avg_turnover * trade_cost
    annual_cost = daily_cost * 240
    print("========== 调仓频率深度分析 ==========")
    print(f"平均每日换手率    : {avg_turnover:.4f} ({avg_turnover * 100:.2f}%)")
    print(f"换手率中位数      : {median_turnover:.4f}")
    print(f"最大单日换手率    : {max_turnover:.4f}")
    print(f"零调仓天数占比    : {zero_turnover_ratio * 100:.2f}%")
    print(f"预估年化摩擦成本  : {annual_cost:.2%}")
    if avg_turnover > 0.5:
        print("风格：高频轮动，摩擦成本高")
    elif avg_turnover > 0.15:
        print("风格：中频择时")
    else:
        print("风格：低频持有")
    return {"avg": avg_turnover, "median": median_turnover, "zero_ratio": zero_turnover_ratio, "annual_cost": annual_cost}

# ===================== 滚动主流程 =====================
def run_rolling_for_model(model_name, model_cls, model_kwargs):
    feat_only, date_col, price_df = load_data()
    total_len = len(feat_only)
    cur_start = 0
    all_res = []
    round_idx = 0
    print(f"\n========== 开始 {model_name} 滚动训练（种子{SEED}）==========")
    prev_model_path = None
    while cur_start + TRAIN_WINDOW + VAL_WINDOW + PREDICT_WINDOW <= total_len:
        round_idx += 1
        # 新增定义，解决未解析引用
        train_start = cur_start
        train_end = cur_start + TRAIN_WINDOW
        val_start = train_end
        val_end = val_start + VAL_WINDOW
        pred_start = val_end
        pred_end = pred_start + PREDICT_WINDOW
        print(
            f"\n第{round_idx}轮 | 训练:{train_start}~{train_end} 验证:{val_start}~{val_end} 预测:{pred_start}~{pred_end}")
        model, train_mean, train_std = train_single_round(
            model_cls=model_cls,
            model_kwargs=model_kwargs,
            feat_only=feat_only,
            price_df=price_df,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            round_id=round_idx,
            model_name=model_name,
            prev_model_path=prev_model_path
        )

        prev_model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_round{round_idx}_best.zip")
        pred_df = predict_round(model, feat_only, train_mean, train_std, date_col, price_df, pred_start, pred_end)
        all_res.append(pred_df)
        cur_start += PREDICT_WINDOW
    if len(all_res) == 0:
        print(f"⚠️ 无完整滚动窗口")
        return pd.DataFrame(), {}
    full_df = pd.concat(all_res, ignore_index=True).drop_duplicates("date", keep="last")
    if not full_df.empty and "turnover" in full_df.columns:
        analyze_turnover(full_df["turnover"])
        save_path = os.path.join(RESULT_SAVE_DIR, f"{model_name}_rolling_detailed.csv")
        full_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"预测明细保存至：{save_path}")
    metrics = calc_portfolio_metrics(full_df["net_value"])
    print(f"\n【滚动外推最终泛化指标】")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    # 导出PPO每日净值，给回测对比脚本读取
    export_df = full_df[["date", "net_value"]].copy()
    export_df.to_csv(r"E:\农行杯\2026\clean\ppo_net_value.csv", index=False, encoding="utf-8-sig")
    print("✅ PPO净值文件已导出至 clean/ppo_net_value.csv")
    return full_df, metrics

def main():
    run_rolling_for_model("PPO_LSTM", PPO, PPO_KWARGS)
    print("\n✅ PPO滚动训练完成！")

if __name__ == "__main__":
    main()
