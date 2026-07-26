import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque

ASSET_NAMES = ["hs300", "zz500", "kc50", "consume", "chip", "gold", "bond10"]
N_ASSET = len(ASSET_NAMES)
DEFAULT_SOFTMAX_TEMP = 0.6
ANNUAL_DAYS = 252  # 年化交易日，用于夏普计算

# ===================== 四维打分工具函数（完全对标calc_market_score.py规则） =====================
def calc_val_score(pe_pct):
    if pe_pct <= 0.30:
        res = 25.0
    elif pe_pct >= 0.70:
        res = 0.0
    else:
        res = 25.0 * (1 - (pe_pct - 0.30) / 0.40)
    return np.clip(res, 0.0, 25.0)

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
    return np.clip(pmi_score + spread_score, 0.0, 25.0)

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
        north_score = 10.0 * (north_flow + 500.0) / 1000.0
    return np.clip(margin_score + north_score, 0.0, 20.0)

def calc_trend_score(price, ma_line):
    deviation = (price / ma_line - 1) * 100
    if deviation >= 5.0:
        res = 30.0
    elif deviation <= -5.0:
        res = 0.0
    else:
        res = 30.0 * (deviation + 5.0) / 10.0
    return np.clip(res, 0.0, 30.0)

# ==========================================================================
class PortfolioEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, feature_df, price_df, start_idx, end_idx,
                 window=40, reward_coef=(6.0, 0.5, 1, 0.005), temp=None,
                 min_w=0.02, max_w=0.4, trade_cost=0.0005, constraint_coef=10.0):
        super().__init__()
        self.df_raw = feature_df.reset_index(drop=True)
        num_cols = self.df_raw.select_dtypes(include=[np.number]).columns
        self.df_raw = self.df_raw[num_cols]
        self.feat_np = self.df_raw.to_numpy(dtype=np.float32)
        self.col_map = {col: i for i, col in enumerate(self.df_raw.columns)}

        # 轻量化：仅保留最近2期资产收益
        self.hist_returns = deque(maxlen=2)
        self.hist_ret_dim = N_ASSET * 2  # 定义历史收益观测维度
        self.vol_ewma = 0.0
        self.peak_value = 1.0
        self.sharpe_rolling = 0.0  # 初始化滚动夏普缓存

        self.price_df_raw = price_df.reset_index(drop=True)
        self.price_array = self.price_df_raw[ASSET_NAMES].values.astype(np.float32)
        self.feature_array = self.feat_np

        # 外部可调超参
        self.MIN_W = min_w
        self.MAX_W = max_w
        self.TRADING_COST_RATE = trade_cost
        self.CONSTRAINT_COEF = constraint_coef
        self.temp = temp if temp is not None else DEFAULT_SOFTMAX_TEMP

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
        # 修正观测维度，匹配2期历史收益
        self.obs_dim = self.feature_array.shape[1] + N_ASSET + self.market_score_dim + self.hist_ret_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self.last_weight = np.zeros(N_ASSET, dtype=np.float32)
        self.last_weight[0] = 1.0
        self.net_value = 1.0
        # 原有代码不变
        self.ret_queue = deque(maxlen=window)
        # 新增：存储窗口内净值序列，用于中长期收益计算
        self.window_net_cache = deque(maxlen=window)

        self.val_score = 0.0
        self.macro_score = 0.0
        self.sent_score = 0.0
        self.trend_score = 0.0
        self.total_market_score = 0.0

        # 北向资金20日滚动累计
        north_raw = self.feat_np[:, self.col_map["north_net"]]
        self.north_20roll = pd.Series(np.nan_to_num(north_raw, nan=0.0)).rolling(20, min_periods=1).sum().values

        # ========== 修复时序泄露：正确滞后1期，弃用np.roll ==========
        spread_raw = self.feat_np[:, self.col_map["bond_10y_2y_spread"]]
        margin_raw = self.feat_np[:, self.col_map["margin_5d_chg"]]
        self.spread_shift = np.zeros_like(spread_raw)
        self.margin_shift = np.zeros_like(margin_raw)
        if len(spread_raw) > 1:
            self.spread_shift[1:] = spread_raw[:-1]
            self.margin_shift[1:] = margin_raw[:-1]

        # 预计算沪深300 MA20
        hs300_series = self.price_df_raw["hs300"]
        self.hs300_ma20_series = hs300_series.rolling(20, min_periods=1).mean().values

    def softmax(self, x):
        x = x / self.temp
        max_x = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    # 新增缺失的夏普计算方法
    def calc_rolling_sharpe(self, ret_list):
        rets = np.array(ret_list)
        mean_r = np.mean(rets)
        std_r = np.std(rets) + 1e-8
        sharpe = mean_r / std_r * np.sqrt(ANNUAL_DAYS)
        return sharpe

    def calc_daily_market_score(self, idx):
        safe_idx = np.clip(idx, 0, self.max_feature_idx)
        row_np = self.feat_np[safe_idx]
        hs300_pe = row_np[self.col_map["hs300_pe_quantile"]]
        pmi = row_np[self.col_map["pmi_制造业采购经理指数"]]
        spread = self.spread_shift[safe_idx]
        margin_chg = self.margin_shift[safe_idx]
        north_flow = self.north_20roll[safe_idx]

        if np.isnan(hs300_pe):
            hs300_pe = 0.5

        hs300_close = self.price_array[idx][0]
        hs300_ma20 = self.hs300_ma20_series[idx]

        val_raw = calc_val_score(hs300_pe)
        macro_raw = calc_macro_score(pmi, spread)
        sent_raw = calc_sent_score(margin_chg, north_flow)
        trend_raw = calc_trend_score(hs300_close, hs300_ma20)

        self.val_score = val_raw / 25.0
        self.macro_score = macro_raw / 25.0
        self.sent_score = sent_raw / 20.0
        self.trend_score = trend_raw / 30.0
        self.total_market_score = self.val_score + self.macro_score + self.trend_score + self.sent_score

    def step(self, action):
        target_weight = self.softmax(action)
        target_weight = np.clip(target_weight, self.MIN_W, self.MAX_W)
        target_weight = target_weight / np.sum(target_weight)

        self.calc_daily_market_score(self.current_idx)

        # 资产收益计算
        price_t = self.price_array[self.current_idx]
        price_t1 = self.price_array[self.current_idx + 1]
        returns = (price_t1 - price_t) / (price_t + 1e-8)
        returns = np.clip(returns, -0.20, 0.20)
        port_ret = np.dot(target_weight, returns)

        # 交易成本扣减
        turnover = np.sum(np.abs(target_weight - self.last_weight))
        port_ret -= turnover * self.TRADING_COST_RATE
        port_ret = np.clip(port_ret, -0.10, 0.10)

        self.net_value *= (1 + port_ret)
        self.ret_queue.append(port_ret)
        # 新增：记录每日净值，用于周期收益
        self.window_net_cache.append(self.net_value)

        self.hist_returns.append(returns)

        # ============ 滚动夏普计算（保留你8:2混合夏普逻辑不变） ============
        hist_rets = list(self.ret_queue)
        if len(hist_rets) < self.window:
            pad = [0.0] * (self.window - len(hist_rets))
            hist_rets = hist_rets + pad
        current_sharpe = self.calc_rolling_sharpe(hist_rets)
        sharpe_delta = current_sharpe - self.sharpe_rolling
        self.sharpe_rolling = current_sharpe

        # 混合夏普主奖励 0.8绝对夏普 + 0.2增量
        coef_profit, coef_vol, coef_dd, coef_turn = self.reward_coef
        weight_abs = 0.8
        weight_delta = 0.2
        r_abs = current_sharpe / 100
        r_sharpe = coef_profit * (weight_abs * r_abs + weight_delta * sharpe_delta)

        # ============ 新增：中长期窗口累计收益奖励（核心改进，解决短期噪声过拟合） ============
        long_term_reward = 0.0
        # 缓存满窗口再计算周期收益
        if len(self.window_net_cache) >= self.window:
            start_net = list(self.window_net_cache)[0]
            end_net = self.net_value
            window_cum_return = (end_net / (start_net + 1e-8)) - 1.0
            # 中长期收益权重0.3，平滑短期单日波动
            long_term_reward = 0.3 * window_cum_return

        # ============ 原有风险惩罚不变 ============
        self.vol_ewma = 0.9 * self.vol_ewma + 0.1 * abs(port_ret)
        vol_penalty = coef_vol * self.vol_ewma

        if self.net_value > self.peak_value:
            self.peak_value = self.net_value
        drawdown = (self.net_value / self.peak_value) - 1.0
        dd_penalty = coef_dd * max(0.0, -drawdown - 0.05)
        turn_penalty = coef_turn * turnover

        upper_violation = np.sum(np.maximum(0.0, target_weight - self.MAX_W))
        lower_violation = np.sum(np.maximum(0.0, self.MIN_W - target_weight))
        constraint_penalty = self.CONSTRAINT_COEF * (upper_violation + lower_violation)

        # ============ 总奖励 = 夏普奖励 + 中长期收益奖励 - 各类惩罚 ============
        reward = r_sharpe + long_term_reward - vol_penalty - dd_penalty - turn_penalty - constraint_penalty
        reward = np.clip(reward, -5.0, 5.0)

        self.last_weight = target_weight.copy()
        self.current_idx += 1
        terminated = self.current_idx >= self.end_idx
        truncated = False

        info = {
            "turnover": turnover,
            "net_value": self.net_value,
            "sharpe": current_sharpe,
            "total_market_score": self.total_market_score,
            "val_score": self.val_score,
            "macro_score": self.macro_score,
            "sent_score": self.sent_score,
            "trend_score": self.trend_score,
        }
        # step中构造obs前

        # 构造观测向量（适配2期历史收益）
        obs_idx = self.current_idx if not terminated else self.current_idx - 1
        obs_idx = np.clip(obs_idx, 0, self.feature_array.shape[0] - 1)
        obs_feat = self.feature_array[obs_idx]
        # ========== 重构hist_flat生成逻辑，固定输出14维 ==========
        # 先创建全0数组，固定14维（2期×7资产）
        hist_flat = np.zeros(self.hist_ret_dim, dtype=np.float32)
        # 取出当前缓存的历史收益列表
        hist_list = list(self.hist_returns)
        # 最多取最近2期，按顺序填充
        fill_len = min(len(hist_list), 2)
        for i in range(fill_len):
            # 第i期资产收益，写入对应位置
            hist_flat[i * N_ASSET: (i + 1) * N_ASSET] = hist_list[i]
        obs = np.concatenate([
            obs_feat,
            self.last_weight,
            np.array([self.val_score, self.macro_score, self.sent_score, self.trend_score]),
            hist_flat
        ])
        total = len(obs_feat) + 7 + 4 + len(hist_flat)
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = self.start_idx
        self.net_value = 1.0
        self.last_weight = np.zeros(N_ASSET, dtype=np.float32)
        self.last_weight[0] = 1.0
        self.ret_queue.clear()
        self.hist_returns.clear()
        # 新增清空中长期净值缓存
        self.window_net_cache.clear()

        self.vol_ewma = 0.0
        self.peak_value = 1.0
        self.sharpe_rolling = 0.0  # 重置夏普缓存

        self.calc_daily_market_score(self.start_idx)
        obs_feat = self.feature_array[self.start_idx]
        # reset中替换原有hist_flat代码
        hist_flat = np.zeros(self.hist_ret_dim, dtype=np.float32)

        obs = np.concatenate([
            obs_feat,
            self.last_weight,
            np.array([self.val_score, self.macro_score, self.sent_score, self.trend_score]),
            hist_flat
        ])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3)
        info = {}
        return obs, info

    def render(self, mode="human"):
        turnover = np.sum(np.abs(self.last_weight))
        print(f"Step:{self.current_idx}, Net:{self.net_value:.4f}, Sharpe:{self.sharpe_rolling:.2f}, TrendScore:{self.trend_score:.1f}, Turn:{turnover:.3f}")
