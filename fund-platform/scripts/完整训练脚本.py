import os
os.environ["SB3_ALLOW_GYM_V0"] = "1"
import pandas as pd
import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from portfolio_env import PortfolioEnv
from portfolio_metrics import calc_portfolio_metrics

# ===================== 自定义特征提取器（扩容） =====================
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )
    def forward(self, observations):
        return self.mlp(observations)

# ===================== 路径与全局超参 =====================
DATA_PATH = "clean/train_feature_filtered.csv"
PRICE_PATH = "clean/etf_price_clean.csv"
MODEL_SAVE_DIR = "model/multi_model/"
RESULT_SAVE_DIR = "result/multi_model/"
LOG_SAVE_DIR = "log/train_log/"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULT_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_SAVE_DIR, exist_ok=True)

TRAIN_WINDOW = 500
VAL_WINDOW = 100    # 拉长验证窗口降低指标噪声
PREDICT_WINDOW = 60
EARLY_STOP_PATIENCE = 20
MIN_ITER = 10       # 保底最少迭代轮数
EPOCH_STEP = 2000
RL_WINDOW = 40
REWARD_COEF = (6.0, 0.4, 2.5, 0.005)

PPO_KWARGS = {
    "learning_rate": 7e-5,
    "n_steps": 1024,
    "batch_size": 64,
    "gamma": 0.99,
    "clip_range": 0.2,
    "ent_coef": 0.10,
    "max_grad_norm": 0.5,
    "verbose": 0,
    "policy_kwargs": {
        "features_extractor_class": CustomFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [256, 128],
    }
}

# ===================== 数据加载 =====================
def load_data():
    feat = pd.read_csv(DATA_PATH, parse_dates=["date"], encoding="utf-8-sig")
    price = pd.read_csv(PRICE_PATH, parse_dates=["date"], encoding="gbk")
    feat = feat.ffill().fillna(0)
    price = price.ffill().fillna(0)
    common_dates = set(feat["date"]) & set(price["date"])
    feat = feat[feat["date"].isin(common_dates)].reset_index(drop=True)
    price = price[price["date"].isin(feat["date"])].reset_index(drop=True)
    feat = feat.shift(1).dropna().reset_index(drop=True)
    price = price.iloc[1:].reset_index(drop=True)
    date_col = feat["date"].copy()
    feat_only = feat.drop(columns=["date"])
    return feat_only, date_col, price

# ===================== 单轮训练 =====================
def train_single_round(model_cls, model_kwargs, feat_only, price_df,
                       train_start, train_end, val_start, val_end, round_id, model_name):
    raw_train_env = PortfolioEnv(
        feature_df=feat_only,
        price_df=price_df,
        start_idx=train_start,
        end_idx=train_end,
        window=RL_WINDOW,
        reward_coef=REWARD_COEF
    )
    raw_val_env = PortfolioEnv(
        feature_df=feat_only,
        price_df=price_df,
        start_idx=val_start,
        end_idx=val_end,
        window=RL_WINDOW,
        reward_coef=REWARD_COEF
    )
    train_env = Monitor(raw_train_env)
    val_env = Monitor(raw_val_env)

    model = model_cls("MlpPolicy", train_env, **model_kwargs)
    best_sharpe = -np.inf
    patience = 0
    best_model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_round{round_id}_best.zip")
    log_path = os.path.join(LOG_SAVE_DIR, f"{model_name}_round{round_id}_train_log.csv")
    log_records = []

    for train_iter in range(30):
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
            "train_iter": train_iter,
            "sharpe": current_sharpe,
            "cum_ret": metrics["累计收益"],
            "vol": metrics["年化波动"],
            "mdd": metrics["最大回撤"]
        })
        print(f"轮次{round_id} 迭代{train_iter} | 夏普:{current_sharpe:.2f} | 累计收益:{metrics['累计收益']:.4f} | 年化波动:{metrics['年化波动']:.4f} | 最大回撤:{metrics['最大回撤']:.4f}")
        if current_sharpe > best_sharpe:
            best_sharpe = current_sharpe
            model.save(best_model_path)
            patience = 0
        else:
            patience += 1
        if patience >= EARLY_STOP_PATIENCE and train_iter >= MIN_ITER:
            print(f"🛑 早停触发，连续{EARLY_STOP_PATIENCE}轮无提升，已完成最少{MIN_ITER}轮迭代")
            break
    pd.DataFrame(log_records).to_csv(log_path, index=False, encoding="utf-8-sig")
    best_model = model_cls.load(best_model_path)
    return best_model

# ===================== 预测 =====================
def predict_round(model, feat_only, date_col, price_df, pred_start, pred_end):
    pred_env = PortfolioEnv(
        feature_df=feat_only,
        price_df=price_df,
        start_idx=pred_start,
        end_idx=pred_end,
        window=RL_WINDOW,
        reward_coef=REWARD_COEF
    )
    obs, _ = pred_env.reset()
    done = False
    net_list = [1.0]
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = pred_env.step(action)
        done = terminated or truncated
        net_list.append(pred_env.unwrapped.net_value)
    date_range = date_col.iloc[pred_start:pred_end + 1].tolist()
    return pd.DataFrame({"date": date_range, "net": net_list})

# ===================== 滚动主流程 =====================
def run_rolling_for_model(model_name, model_cls, model_kwargs):
    feat_only, date_col, price_df = load_data()
    total_len = len(feat_only)
    cur_start = 0
    all_res = []
    round_idx = 0
    print(f"\n========== 开始 {model_name} 滚动训练 ==========")
    while cur_start + TRAIN_WINDOW + VAL_WINDOW + PREDICT_WINDOW <= total_len:
        round_idx += 1
        train_end = cur_start + TRAIN_WINDOW
        val_start = train_end
        val_end = val_start + VAL_WINDOW
        pred_start = val_end
        pred_end = pred_start + PREDICT_WINDOW
        print(f"\n{model_name} 第{round_idx}轮")
        print(f"训练区间: {cur_start} ~ {train_end}")
        print(f"验证区间: {val_start} ~ {val_end}")
        print(f"预测区间: {pred_start} ~ {pred_end}")
        model = train_single_round(
            model_cls=model_cls,
            model_kwargs=model_kwargs,
            feat_only=feat_only,
            price_df=price_df,
            train_start=cur_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            round_id=round_idx,
            model_name=model_name
        )
        pred_df = predict_round(model, feat_only, date_col, price_df, pred_start, pred_end)
        all_res.append(pred_df)
        cur_start += PREDICT_WINDOW
    if len(all_res) == 0:
        print(f"⚠️ {model_name} 无完整滚动窗口，跳过保存")
        return pd.DataFrame(), {}
    full_df = pd.concat(all_res, ignore_index=True).drop_duplicates("date", keep="last")
    save_path = os.path.join(RESULT_SAVE_DIR, f"{model_name}_rolling_net.csv")
    full_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    metrics = calc_portfolio_metrics(full_df["net"])
    print(f"\n【{model_name} 滚动外推最终指标（预测集，真实泛化能力）】")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    return full_df, metrics

def main():
    run_rolling_for_model("PPO", PPO, PPO_KWARGS)
    print("\n✅ PPO滚动训练全部完成！")

if __name__ == "__main__":
    main()
