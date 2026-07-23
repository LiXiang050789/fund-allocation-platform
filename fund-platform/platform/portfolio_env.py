import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

ASSET_NAMES = ["hs300", "zz500", "kc50", "consume", "chip", "gold", "bond10"]
N_ASSET = len(ASSET_NAMES)
DEFAULT_SOFTMAX_TEMP = 0.6

# ===================== 四维打分工具函数（适配真实数据尺度+区间截断） =====================
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
    # 适配数据集north_net波动±100
    if north_flow >= 100:
        north_score = 10.0
    elif north_flow <= -100:
        north_score = 0.0
    else:
        north_score = 10.0 * (north_flow + 100) / 200.0
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
                 window=40, reward_coef=(6.0, 0.4, 2.5, 0.005), temp=None,
                 min_w=0.02, max_w=0.4, trade_cost=0.0005, constraint_coef=10.0):
        super().__init__()
        self.df_raw = feature_df.reset_index(drop=True)
        self.feat_np = self.df_raw.to_numpy(dtype=np.float32)
        self.col_map = {col: i for i, col in enumerate(self.df_raw.columns)}
        self.hist_returns = deque(maxlen=5)  # 保存最近5期资产收益
        self.vol_ewma = 0.0
        self.peak_value = 1.0
        self.price_array = price_df.reset_index(drop=True)[ASSET_NAMES].values.astype(np.float32)
        # 外部训练脚本已完成滚动窗口Z-Score标准化，Env不再二次归一化
        self.feature_array = self.feat_np

        # 外部可调超参，取消全局硬编码
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
        self.obs_dim = self.feature_array.shape[1] + N_ASSET + self.market_score_dim + N_ASSET * 5
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
        # 基础打分指标
        hs300_pe = row_np[self.col_map["hs300_pe"]]
        pmi = row_np[self.col_map["pmi_生产经营活动预期指数"]]
        spread = row_np[self.col_map["bond_10y_2y_spread"]]
        margin_chg = row_np[self.col_map["margin_5d_chg"]]
        north_net = row_np[self.col_map["north_net"]]

        # 【修复2：趋势分改为同资产沪深300收盘价 vs 沪深300 MA5，不再跨指数对比】
        hs300_close = self.price_array[idx][0]  # 当前t日沪深300真实收盘价
        hs300_ma5 = row_np[self.col_map["hs300_ma5"]]

        # 计算原始分数并截断区间
        val_raw = calc_val_score(hs300_pe)
        macro_raw = calc_macro_score(pmi, spread)
        sent_raw = calc_sent_score(margin_chg, north_net)
        trend_raw = calc_trend_score(hs300_close, hs300_ma5)

        # 归一化到 [0,1]
        self.val_score = val_raw / 25.0
        self.macro_score = macro_raw / 25.0
        self.sent_score = sent_raw / 20.0
        self.trend_score = trend_raw / 30.0
        self.total_market_score = self.val_score + self.macro_score + self.trend_score + self.sent_score

    def step(self, action):
        # ----- 1. 动作->权重 -----
        target_weight = self.softmax(action)
        target_weight = target_weight / (np.sum(target_weight) + 1e-8)
        target_weight = np.clip(target_weight, self.MIN_W, self.MAX_W)
        target_weight = target_weight / np.sum(target_weight)

        self.calc_daily_market_score(self.current_idx)

        # ----- 2. 资产收益计算 -----
        price_t = self.price_array[self.current_idx]
        price_t1 = self.price_array[self.current_idx + 1]
        returns = (price_t1 - price_t) / (price_t + 1e-8)
        returns = np.clip(returns, -0.20, 0.20)
        port_ret = np.dot(target_weight, returns)

        # ----- 3. 换手率与交易成本（外部传入参数） -----
        turnover = np.sum(np.abs(target_weight - self.last_weight))
        port_ret -= turnover * self.TRADING_COST_RATE
        port_ret = np.clip(port_ret, -0.10, 0.10)

        # ----- 4. 更新账户净值与收益队列 -----
        self.net_value *= (1 + port_ret)
        self.ret_queue.append(port_ret)

        # ----- 5. 历史收益序列，删除随机噪声填充，缺失补0 -----
        hist_rets = list(self.ret_queue)
        if len(hist_rets) < self.window:
            pad = [0.0] * (self.window - len(hist_rets))
            hist_rets = hist_rets + pad
        hist_arr = np.array(hist_rets)

        # ----- 6. 更新历史资产收益（观测用）
        self.hist_returns.append(returns)

        # ----- 7. 奖励函数（外部系数reward_coef统一调参）-----
        coef_profit, coef_vol, coef_dd, coef_turn = self.reward_coef
        r_profit = coef_profit * port_ret

        # 波动惩罚（指数加权）
        self.vol_ewma = 0.9 * self.vol_ewma + 0.1 * abs(port_ret)
        vol_penalty = coef_vol * self.vol_ewma

        # 回撤惩罚
        if self.net_value > self.peak_value:
            self.peak_value = self.net_value
        drawdown = (self.net_value / self.peak_value) - 1.0
        dd_penalty = coef_dd * max(0.0, -drawdown - 0.05)

        # 换手惩罚
        turn_penalty = coef_turn * turnover

        # 权重约束惩罚
        upper_violation = np.sum(np.maximum(0.0, target_weight - self.MAX_W))
        lower_violation = np.sum(np.maximum(0.0, self.MIN_W - target_weight))
        constraint_penalty = self.CONSTRAINT_COEF * (upper_violation + lower_violation)

        reward = r_profit - vol_penalty - dd_penalty - turn_penalty - constraint_penalty
        reward = np.clip(reward, -5.0, 5.0)

        # ----- 8. 更新状态 -----
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

        # ----- 9. 构造观测 -----
        obs_idx = self.current_idx if not terminated else self.current_idx - 1
        obs_idx = np.clip(obs_idx, 0, self.feature_array.shape[0] - 1)
        obs_feat = self.feature_array[obs_idx]

        if len(self.hist_returns) == 5:
            hist_flat = np.concatenate(list(self.hist_returns))
        else:
            hist_flat = np.zeros(N_ASSET * 5, dtype=np.float32)

        # 【修复1：修正观测拼接，替换重复macro_score为sent_score】
        obs = np.concatenate([
            obs_feat,
            self.last_weight,
            np.array([self.val_score, self.macro_score, self.sent_score, self.trend_score]),
            hist_flat
        ])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3)
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = self.start_idx
        self.net_value = 1.0
        self.last_weight = np.zeros(N_ASSET, dtype=np.float32)
        self.last_weight[0] = 1.0
        self.ret_queue.clear()
        self.hist_returns.clear()
        self.vol_ewma = 0.0
        self.peak_value = 1.0

        self.calc_daily_market_score(self.start_idx)
        obs_feat = self.feature_array[self.start_idx]

        hist_flat = np.zeros(N_ASSET * 5, dtype=np.float32)
        # 【修复1：reset同步修正观测拼接】
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
        print(f"Step:{self.current_idx}, Net:{self.net_value:.4f}, TrendScore:{self.trend_score:.1f}, MarketScore:{self.total_market_score:.1f}, Turn:{turnover:.3f}")
