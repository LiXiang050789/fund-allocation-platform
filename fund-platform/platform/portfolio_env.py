import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

ASSET_NAMES = ["hs300", "zz500", "kc50", "consume", "chip", "gold", "bond10"]
N_ASSET = len(ASSET_NAMES)
MIN_W = 0.02
MAX_W = 0.4
CONSTRAINT_COEF = 10.0
DEFAULT_SOFTMAX_TEMP = 0.6

# ===================== 四维打分工具函数 =====================
def calc_val_score(pe_pct):
    if pe_pct <= 0.30:
        return 25.0
    elif pe_pct >= 0.70:
        return 0.0
    else:
        return 25.0 * (1 - (pe_pct - 0.30) / 0.40)

def calc_macro_score(pmi, spread_10y2y):
    if pmi >= 52:
        pmi_score = 15.0
    elif pmi <= 48:
        pmi_score = 0.0
    else:
        pmi_score = 15.0 * (pmi - 48) / 4.0
    if spread_10y2y >= 1.0:
        spread_score = 10.0
    elif spread_10y2y <= -0.5:
        spread_score = 0.0
    else:
        spread_score = 10.0 * (spread_10y2y + 0.5) / 1.5
    return pmi_score + spread_score

def calc_sent_score(margin_change, north_flow):
    if margin_change >= 5.0:
        margin_score = 10.0
    elif margin_change <= -5.0:
        margin_score = 0.0
    else:
        margin_score = 10.0 * (margin_change + 5.0) / 10.0
    if north_flow >= 500:
        north_score = 10.0
    elif north_flow <= -500:
        north_score = 0.0
    else:
        north_score = 10.0 * (north_flow + 500) / 1000.0
    return margin_score + north_score

def calc_trend_score(price, ma_line):
    deviation = (price / ma_line - 1) * 100
    if deviation >= 5.0:
        return 30.0
    elif deviation <= -5.0:
        return 0.0
    else:
        return 30.0 * (deviation + 5.0) / 10.0

# ==========================================================================
class PortfolioEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, feature_df, price_df, start_idx, end_idx,
                 window=40, reward_coef=(6.0, 0.4, 2.5, 0.005), temp=None):
        super().__init__()
        self.df_raw = feature_df.reset_index(drop=True)
        self.feat_np = self.df_raw.to_numpy(dtype=np.float32)
        self.col_map = {col: i for i, col in enumerate(self.df_raw.columns)}

        self.price_array = price_df.reset_index(drop=True)[ASSET_NAMES].values.astype(np.float32)
        self.feature_raw = self.feat_np
        self.temp = temp if temp is not None else DEFAULT_SOFTMAX_TEMP

        train_sample_len = window * 2
        train_start = max(0, start_idx - train_sample_len)
        train_feat_slice = self.feature_raw[train_start: start_idx]
        if train_feat_slice.shape[0] == 0:
            train_feat_slice = self.feature_raw[:200]
        self.feat_mean = train_feat_slice.mean(axis=0)
        self.feat_std = train_feat_slice.std(axis=0)
        self.feat_std[self.feat_std < 1e-6] = 1.0
        self.feature_array = (self.feature_raw - self.feat_mean) / (self.feat_std + 1e-8)

        max_valid_idx = len(self.price_array) - 1
        self.end_idx = min(end_idx, max_valid_idx)
        self.start_idx = start_idx
        self.max_feature_idx = self.feat_np.shape[0] - 1
        if self.start_idx >= self.end_idx:
            raise ValueError(f"非法区间：start_idx({self.start_idx}) >= end_idx({self.end_idx})")

        self.current_idx = self.start_idx
        self.window = window
        self.reward_coef = reward_coef

        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(N_ASSET,), dtype=np.float32)
        self.market_score_dim = 4
        self.obs_dim = self.feature_array.shape[1] + N_ASSET + self.market_score_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self.last_weight = np.zeros(N_ASSET, dtype=np.float32)
        self.last_weight[0] = 1.0
        self.net_value = 1.0
        self.ret_queue = deque(maxlen=window)

        self.val_score = 0.0
        self.macro_score = 0.0
        self.sent_score = 0.0
        self.trend_score = 0.0
        self.total_market_score = 0.0

    def softmax(self, x):
        x = x / self.temp
        max_x = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def calc_daily_market_score(self, idx):
        safe_idx = np.clip(idx, 0, self.max_feature_idx)
        row_np = self.feat_np[safe_idx]
        hs300_pe = row_np[self.col_map["hs300_pe"]]
        pmi = row_np[self.col_map["pmi_制造业采购经理指数"]]
        spread = row_np[self.col_map["bond_10y_2y_spread"]]
        margin_chg = row_np[self.col_map["margin_5d_chg"]]
        north_net = row_np[self.col_map["north_net"]]
        price_series = row_np[self.col_map["hs300_ma5"]]
        ma_series = row_np[self.col_map["zz500_ma5"]]

        self.val_score = calc_val_score(hs300_pe)
        self.macro_score = calc_macro_score(pmi, spread)
        self.sent_score = calc_sent_score(margin_chg, north_net)
        self.trend_score = calc_trend_score(price_series, ma_series)
        self.total_market_score = self.val_score + self.macro_score + self.trend_score + self.sent_score

    def step(self, action):
        target_weight = self.softmax(action)
        target_weight = target_weight / (np.sum(target_weight) + 1e-8)

        self.calc_daily_market_score(self.current_idx)

        # ---- 修正开始 ----
        price_t = self.price_array[self.current_idx]
        price_t1 = self.price_array[self.current_idx + 1]
        # 防止除零
        returns = (price_t1 - price_t) / (price_t + 1e-8)
        # 限制单日收益率在合理范围（可选）
        returns = np.clip(returns, -0.20, 0.20)
        port_ret = np.dot(target_weight, returns)
        # ---- 修正结束 ----

        # 限制极端收益率（保留）
        port_ret = np.clip(port_ret, -0.10, 0.10)
        self.net_value *= (1 + port_ret)
        self.ret_queue.append(port_ret)

        hist_rets = list(self.ret_queue)
        if len(hist_rets) < self.window:
            rand_fill = np.random.uniform(-0.005, 0.005, size=self.window - len(hist_rets))
            hist_rets = hist_rets + rand_fill.tolist()
        hist_arr = np.array(hist_rets)

        r_profit = self.reward_coef[0] * port_ret
        vol = np.std(hist_arr) + 1e-6
        vol_penalty = self.reward_coef[1] * vol

        cum_ret = np.cumprod(hist_arr + 1)
        running_max = np.maximum.accumulate(cum_ret)
        drawdown_series = cum_ret / running_max - 1
        worst_dd = np.min(drawdown_series)
        dd_penalty = self.reward_coef[2] * worst_dd

        turnover = np.sum(np.abs(target_weight - self.last_weight))
        turn_penalty = self.reward_coef[3] * turnover

        upper_violation = np.sum(np.maximum(0.0, target_weight - MAX_W))
        lower_violation = np.sum(np.maximum(0.0, MIN_W - target_weight))
        constraint_penalty = CONSTRAINT_COEF * (upper_violation + lower_violation)

        reward = r_profit - vol_penalty + dd_penalty - turn_penalty - constraint_penalty
        reward = np.tanh(reward)

        self.last_weight = target_weight.copy()
        self.current_idx += 1
        terminated = self.current_idx >= self.end_idx
        truncated = False

        info = {
            "turnover": turnover,
            "net_value": self.net_value,
            "total_market_score": self.total_market_score,
            "val_score": self.val_score,
            "macro_score": self.macro_score,
            "sent_score": self.sent_score,
            "trend_score": self.trend_score,
        }

        obs_idx = self.current_idx if not terminated else self.current_idx - 1
        obs_idx = np.clip(obs_idx, 0, self.max_feature_idx)
        obs_feat = self.feature_array[obs_idx]

        obs = np.empty(self.obs_dim, dtype=np.float32)
        obs[:len(obs_feat)] = obs_feat
        obs[len(obs_feat):len(obs_feat) + N_ASSET] = self.last_weight
        obs[-4:] = np.array([self.val_score, self.macro_score, self.sent_score, self.trend_score])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3)
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = self.start_idx
        self.net_value = 1.0
        self.last_weight = np.zeros(N_ASSET, dtype=np.float32)
        self.last_weight[0] = 1.0
        self.ret_queue.clear()
        self.calc_daily_market_score(self.start_idx)
        obs_feat = self.feature_array[self.start_idx]

        obs = np.empty(self.obs_dim, dtype=np.float32)
        obs[:len(obs_feat)] = obs_feat
        obs[len(obs_feat):len(obs_feat) + N_ASSET] = self.last_weight
        obs[-4:] = np.array([self.val_score, self.macro_score, self.sent_score, self.trend_score])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3)
        info = {}
        return obs, info

    def render(self, mode="human"):
        turnover = np.sum(np.abs(self.last_weight))
        print(f"Step:{self.current_idx}, Net:{self.net_value:.4f}, TrendScore:{self.trend_score:.1f}, MarketScore:{self.total_market_score:.1f}, Turn:{turnover:.3f}")
