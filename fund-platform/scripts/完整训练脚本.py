import os
import random
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from portfolio_env import PortfolioEnv
from portfolio_metrics import calc_portfolio_metrics

# ===================== 全局固定随机种子（复现性优化） =====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
# SB3全局种子
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["SB3_ALLOW_GYM_V0"] = "1"

# 自动GPU加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"训练设备：{device}")

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

# ===================== 路径与全局超参（上调EPOCH_STEP） =====================
# 推荐改为与筛选脚本一致
DATA_PATH = "data/clean/train_feature_filtered.csv"
PRICE_PATH = "data/clean/etf_price_clean.csv"
MODEL_SAVE_DIR = "models/multi_model/"
RESULT_SAVE_DIR = "results/multi_model/"
LOG_SAVE_DIR = "logs/train_log/"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULT_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_SAVE_DIR, exist_ok=True)

TRAIN_WINDOW = 800
VAL_WINDOW = 200    # 拉长验证窗口降低指标噪声
PREDICT_WINDOW = 100
EARLY_STOP_PATIENCE = 50
MIN_ITER = 20       # 保底最少迭代轮数
EPOCH_STEP = 6000   # 原3000 → 6000，单轮总步数翻倍，收敛更充分
RL_WINDOW = 40
REWARD_COEF = (6.0, 0.4, 2.5, 0.005)

PPO_KWARGS = {
    "learning_rate": 7e-5,
    "n_steps": 1024,
    "batch_size": 64,
    "gamma": 0.99,
    "clip_range": 0.2,
    "ent_coef": 0.05,
    "max_grad_norm": 0.5,
    "verbose": 0,
    "device": device,
    "seed": SEED,  # PPO内部固定种子
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

    # 关键注释：全局特征shift(1)，t时刻观测 = t-1特征，避免未来泄露
    # 校验PortfolioEnv：step仅用price[i+1]/price[i]计算收益，内部禁止再次shift特征/价格
    feat = feat.shift(1).dropna().reset_index(drop=True)
    price = price.iloc[1:].reset_index(drop=True)
    date_col = feat["date"].copy()
    feat_only = feat.drop(columns=["date"])
    return feat_only, date_col, price

# ===================== 滚动窗口Z-Score标准化（新增，解决特征尺度悬殊） =====================
def get_train_scaler(train_feat_df):
    """仅用训练窗口计算均值、标准差，滚动实时归一化，无未来泄露"""
    mean = train_feat_df.mean()
    std = train_feat_df.std()
    std = std.where(std > 1e-8, 1.0)  # 常量特征标准差置1，避免除0
    return mean, std

def apply_zscore(feat_df, mean, std):
    return (feat_df - mean) / std

# ===================== 单轮训练 =====================
def train_single_round(model_cls, model_kwargs, feat_only, price_df,
                       train_start, train_end, val_start, val_end,
                       round_id, model_name, prev_model_path=None):
    """
    若 prev_model_path 存在，则加载该模型作为初始模型（增量学习）
    新增：滚动窗口Z-Score标准化，仅使用训练区间统计量
    """
    # 1. 提取当前轮训练集，计算归一化参数（仅训练集，无未来数据）
    train_feat_raw = feat_only.iloc[train_start:train_end].copy()
    train_mean, train_std = get_train_scaler(train_feat_raw)
    # 全局统一归一化（训练+验证集全部使用训练窗口均值方差）
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

    # ----- 增量学习：加载上一轮模型（如果存在） -----
    if prev_model_path is not None and os.path.exists(prev_model_path):
        print(f"  加载上一轮模型：{prev_model_path}")
        model = model_cls.load(prev_model_path, env=train_env, device=device)
    else:
        model = model_cls("MlpPolicy", train_env, **model_kwargs)

    best_sharpe = -np.inf
    patience = 0
    best_model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_round{round_id}_best.zip")
    log_path = os.path.join(LOG_SAVE_DIR, f"{model_name}_round{round_id}_train_log.csv")
    log_records = []

    # 迭代50轮，单轮6000步，总30万步，收敛更充分
    for train_iter in range(50):
        model.learn(total_timesteps=EPOCH_STEP, reset_num_timesteps=False)
        # 评估验证集
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
    best_model = model_cls.load(best_model_path, device=device)
    return best_model, train_mean, train_std

# ===================== 预测（复用训练窗口归一化参数，无泄露） =====================
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
    records = []  # 用于存储每一期的详细数据

    step_idx = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = pred_env.step(action)  # 【关键】捕获 info
        done = terminated or truncated

        # 记录当前时间点的详细数据
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
    """
    分析调仓频率与成本
    turnover_series: 每日换手率（0~2.0，例如 0.3 表示当日调换了30%的仓位）
    """
    avg_turnover = turnover_series.mean()
    median_turnover = turnover_series.median()
    max_turnover = turnover_series.max()
    zero_turnover_ratio = (turnover_series == 0).mean()  # 持仓不动天数占比

    # 日均交易成本（按万分之五计算）
    daily_cost = avg_turnover * trade_cost
    # 假设年化交易日 240 天，估算年化摩擦成本
    annual_cost = daily_cost * 240

    print("========== 调仓频率深度分析 ==========")
    print(f"1. 平均每日换手率    : {avg_turnover:.4f} ({avg_turnover * 100:.2f}%)")
    print(f"2. 换手率中位数      : {median_turnover:.4f}")
    print(f"3. 最大单日换手率    : {max_turnover:.4f}")
    print(f"4. 零调仓（躺平）占比: {zero_turnover_ratio * 100:.2f}%")
    print(f"5. 预估日均交易成本  : {daily_cost:.6f} (万分之{daily_cost * 10000:.2f})")
    print(f"6. 预估年化摩擦成本  : {annual_cost:.4f} ({annual_cost * 100:.2f}%)")

    # 调仓风格判断
    if avg_turnover > 0.5:
        print("风格判断：高频轮动（几乎每日大幅调仓，适合高波动市场但成本高昂）")
    elif avg_turnover > 0.15:
        print("风格判断：中频择时（每周调仓1~2次，成本可控）")
    else:
        print("风格判断：低频持有（买入并持有为主，交易成本极低）")

    return {
        "avg": avg_turnover,
        "median": median_turnover,
        "zero_ratio": zero_turnover_ratio,
        "annual_cost": annual_cost
    }

# ===================== 滚动主流程 =====================
def run_rolling_for_model(model_name, model_cls, model_kwargs):
    feat_only, date_col, price_df = load_data()
    total_len = len(feat_only)
    cur_start = 0
    all_res = []
    round_idx = 0
    print(f"\n========== 开始 {model_name} 滚动训练（随机种子{SEED}，复现锁定）==========")
    prev_model_path = None
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

        model, train_mean, train_std = train_single_round(
            model_cls=model_cls,
            model_kwargs=model_kwargs,
            feat_only=feat_only,
            price_df=price_df,
            train_start=cur_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            round_id=round_idx,
            model_name=model_name,
            prev_model_path=prev_model_path
        )
        prev_model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_round{round_idx}_best.zip")
        pred_df = predict_round(model, feat_only, train_mean, train_std, date_col, pred_start, pred_end)
        all_res.append(pred_df)
        cur_start += PREDICT_WINDOW
    if len(all_res) == 0:
        print(f"⚠️ {model_name} 无完整滚动窗口，跳过保存")
        return pd.DataFrame(), {}
    full_df = pd.concat(all_res, ignore_index=True).drop_duplicates("date", keep="last")
    # 在 full_df 生成并保存后添加
    if not full_df.empty and "turnover" in full_df.columns:
        print("\n" + "=" * 40)
        analyze_turnover(full_df["turnover"])
        # 将得分和换手率一起保存给队友
        save_detailed_path = os.path.join(RESULT_SAVE_DIR, f"{model_name}_rolling_detailed.csv")
        full_df.to_csv(save_detailed_path, index=False, encoding="utf-8-sig")
        print(f"详细预测数据（含得分/换手率）已保存至：{save_detailed_path}")
    metrics = calc_portfolio_metrics(full_df["net_value"])
    print(f"\n【{model_name} 滚动外推最终指标（预测集，真实泛化能力）】")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    return full_df, metrics

def main():
    run_rolling_for_model("PPO", PPO, PPO_KWARGS)
    print("\n✅ PPO滚动训练全部完成！种子固定：", SEED)

if __name__ == "__main__":
    import pandas as pd  # 延迟导入避免全局顺序冲突
    main()
