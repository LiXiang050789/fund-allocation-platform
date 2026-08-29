import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque

ASSET_NAMES = ["hs300", "zz500", "kc50", "consume", "chip", "gold", "bond10"]
N_ASSET = len(ASSET_NAMES)
N_EQUITY = N_ASSET - 1  # 前6只为权益，最后1只为国债
BOND_IDX = -1
DEFAULT_SOFTMAX_TEMP = 0.6
ANNUAL_DAYS = 252
MIN_BOND_WEIGHT = 0.18  # 国债最低硬底仓

# 四维打分工具函数（原版保持不变）
def calc_val_score(pe_pct):
    if pe_pct <= 0.30:
        res = 25.0
    elif pe_pct >= 0.70:
        res = 0.0
    else:
        res = 25.0 * (1 - (pe_pct - 0.30) / 0.40)
    return np.clip(res, 0.0, 25.0)

def regime_one_hot(regime):
    vec = np.zeros(4, dtype=np.float32)
    r = min(max(int(regime), 0), 3)
    vec[r] = 1.0
    return vec

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
    if np.isnan(north_flow):
        north_flow = -500
    if north_flow >= 500:
        north_score = 10.0
    elif north_flow <= -500:
        north_score = 0.0
    else:
        north_score = 10.0 * (north_flow + 500.0) / 1000.0
    return np.clip(margin_score + north_score, 0.0, 20.0)

# ========== 修复除零警告：分母增加1e-8 ==========
def calc_trend_score(price, ma_line):
    deviation = (price / (ma_line + 1e-8) - 1) * 100
    if deviation >= 5.0:
        res = 30.0
    elif deviation <= -5.0:
        res = 0.0
    else:
        res = 30.0 * (deviation + 5.0) / 10.0
    return np.clip(res, 0.0, 30.0)


class PortfolioEnvGlobal(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, feature_df, price_df, start_idx, end_idx,
                 window=40, reward_coef=(8.0, 0.01, 0.15, 0.005), temp=None,
                 min_w=0.01, max_w=0.55, trade_cost=0.0003, constraint_coef=10.0,
                 # 消融实验分散入参，无cfg字典
                 rebalance_monthly=False,         # 月度调仓开关
                 enable_bond_regime_reward=True, # 行情动态国债奖励
                 enable_bond_loss_couple=True,   # 亏损耦合国债奖励
                 enable_ladder_dd=True,          # 阶梯回撤惩罚
                 enable_min_bond_hard_constraint=True, # 国债硬底仓约束
                 enable_reward_norm=True,       # 滚动奖励标准化
                 enable_lgb_pred_obs=False):     # LGB预测观测输入开关
        super().__init__()

        # ========== 修复1：彻底删除所有 self.cfg 相关代码（无cfg入参，杜绝报错） ==========
        # 消融开关直接绑定实例属性，全程用self.xxx读取
        self.rebalance_monthly = rebalance_monthly
        self.enable_bond_regime_reward = enable_bond_regime_reward
        self.enable_bond_loss_couple = enable_bond_loss_couple
        self.enable_ladder_dd = enable_ladder_dd
        self.enable_min_bond_hard_constraint = enable_min_bond_hard_constraint
        self.enable_reward_norm = enable_reward_norm
        self.enable_lgb_pred_obs = enable_lgb_pred_obs

        # 基础数据加载
        self.df_raw = feature_df.reset_index(drop=True)
        num_cols = self.df_raw.select_dtypes(include=[np.number]).columns
        self.df_raw = self.df_raw[num_cols]
        self.feat_np = self.df_raw.to_numpy(dtype=np.float32)
        self.col_map = {col: i for i, col in enumerate(self.df_raw.columns)}
        self.feature_df = feature_df
        self.price_df = price_df

        self.price_df_raw = price_df.reset_index(drop=True)
        self.price_array = self.price_df_raw[ASSET_NAMES].values.astype(np.float32)
        self.feature_array = self.feat_np
        self.base_feat_dim = self.feature_array.shape[1]  # 先定义基础特征维度，避免顺序错误

        # 超参绑定
        self.MIN_W = min_w
        self.MAX_W = max_w
        self.TRADING_COST_RATE = trade_cost
        self.CONSTRAINT_COEF = constraint_coef
        self.temp = temp if temp is not None else DEFAULT_SOFTMAX_TEMP
        self.window = window
        self.reward_coef = reward_coef

        # 区间合法性校验
        max_valid_idx = len(self.price_array) - 1
        self.end_idx = min(end_idx, max_valid_idx)
        self.start_idx = start_idx
        self.max_feature_idx = self.feat_np.shape[0] - 1
        if self.start_idx >= self.end_idx:
            raise ValueError(f"非法区间：start_idx({self.start_idx}) >= end_idx({self.end_idx})")
        self.current_idx = self.start_idx

        # ========== 修复2：统一所有序列维度初始化，消除维度漂移 ==========
        # 固定历史收益维度：固定2期历史收益，覆盖原冲突赋值
        self.hist_ret_dim = N_ASSET * 2
        # 行情onehot固定4维
        self.regime_dim = 4
        # LGB维度根据开关切换
        self.lgb_pred_dim = N_ASSET if self.enable_lgb_pred_obs else 0
        # 额外序列维度固定
        self.market_score_dim = 4
        self.bond_hist_dim = 1
        self.extra_seq_dim = 2 + N_ASSET + self.bond_hist_dim

        # ========== 修复3：全局唯一观测总维度，只计算一次，观测空间严格对齐 ==========
        self.obs_dim = (
            self.base_feat_dim
            + N_ASSET                     # 当前权重
            + self.market_score_dim       # 4个市场打分
            + self.hist_ret_dim           # 历史收益序列
            + self.extra_seq_dim          # 均值序列特征
            + self.regime_dim             # 行情独热编码
            + self.lgb_pred_dim           # LGB预测特征
        )

        # 动作、观测空间定义
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(N_ASSET,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # 权重、净值缓存初始化
        self.last_weight = np.zeros(N_ASSET, dtype=np.float32)
        self.last_weight[0] = 1.0
        self.last_target_weight = self.last_weight.copy()
        self.net_value = 1.0
        self.ret_queue = deque(maxlen=window)
        self.window_net_cache = deque(maxlen=window)

        # 打分缓存
        self.score_cache = deque(maxlen=self.window)
        self.asset_ret_cache = deque(maxlen=self.window)
        self.bond_weight_cache = deque(maxlen=self.window)
        self.val_score = 0.0
        self.macro_score = 0.0
        self.sent_score = 0.0
        self.trend_score = 0.0
        self.total_market_score = 0.0

        # 滞后防泄露指标：宏观、情绪
        north_raw = self.feat_np[:, self.col_map["north_net"]]
        north_ser = pd.Series(north_raw)
        self.north_20roll = north_ser.rolling(20, min_periods=20).sum().values
        spread_raw = self.feat_np[:, self.col_map["bond_10y_2y_spread"]]
        margin_raw = self.feat_np[:, self.col_map["margin_5d_chg"]]
        self.spread_shift = np.zeros_like(spread_raw)
        self.margin_shift = np.zeros_like(margin_raw)
        if len(spread_raw) > 1:
            self.spread_shift[1:] = spread_raw[:-1]
            self.margin_shift[1:] = margin_raw[:-1]

        # HS300 MA20 原始序列（滞后移位消除数据泄露）
        hs300_series = self.price_df_raw["hs300"]
        self.hs300_ma20_series = hs300_series.rolling(20, min_periods=1).mean().values
        self.hs300_ma20_shift = np.zeros_like(self.hs300_ma20_series)
        if len(self.hs300_ma20_series) > 1:
            self.hs300_ma20_shift[1:] = self.hs300_ma20_series[:-1]

        # PE分位数滞后
        pe_raw = self.feat_np[:, self.col_map["hs300_pe_quantile"]]
        self.pe_shift = np.zeros_like(pe_raw)
        if len(pe_raw) > 1:
            self.pe_shift[1:] = pe_raw[:-1]

        # HS300收盘价滞后
        hs300_price_raw = self.price_array[:, 0]
        self.hs300_price_shift = np.zeros_like(hs300_price_raw)
        if len(hs300_price_raw) > 1:
            self.hs300_price_shift[1:] = hs300_price_raw[:-1]

        # 回撤、净值峰值缓存
        self.drawdown_cache = deque(maxlen=self.window)
        self.full_history_dd = []    # 全周期回撤无窗口缓存
        self.global_peak = 1.0       # 全周期历史最高净值
        self.peak_value = 1.0
        self.vol_ewma = 0.0
        self.sharpe_rolling = 0.0
        self.reward_norm_queue = deque(maxlen=60)

        # 时序收益缓存
        self.hist_returns = deque(maxlen=2)

        # 行情状态变量
        self.market_regime = 2
        self.market_onehot = regime_one_hot(self.market_regime)

    def softmax(self, x):
        x = x / self.temp
        max_x = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def calc_rolling_sharpe(self, ret_list):
        rets = np.array(ret_list)
        mean_r = np.mean(rets)
        std_r = np.std(rets) + 1e-8
        sharpe = mean_r / std_r * np.sqrt(ANNUAL_DAYS)
        return sharpe

    # 【移入类内部 + 全部使用滞后数据，彻底无泄露】
    def calc_daily_market_score(self, idx):
        safe_idx = np.clip(idx, 0, self.max_feature_idx)
        row_np = self.feat_np[safe_idx]
        # 全部读取滞后一期数据，不用当日实时行情
        hs300_pe = self.pe_shift[safe_idx]
        pmi = row_np[self.col_map["pmi_制造业采购经理指数"]]
        spread = self.spread_shift[safe_idx]
        margin_chg = self.margin_shift[safe_idx]
        north_flow = self.north_20roll[safe_idx]
        if np.isnan(hs300_pe):
            hs300_pe = 0.5
        # 滞后一天收盘价、均线
        hs300_close = self.hs300_price_shift[safe_idx]
        hs300_ma20 = self.hs300_ma20_shift[safe_idx]

        val_raw = calc_val_score(hs300_pe)
        macro_raw = calc_macro_score(pmi, spread)
        sent_raw = calc_sent_score(margin_chg, north_flow)
        trend_raw = calc_trend_score(hs300_close, hs300_ma20)

        self.val_score = val_raw / 25.0
        self.macro_score = macro_raw / 25.0
        self.sent_score = sent_raw / 20.0
        self.trend_score = trend_raw / 30.0
        self.total_market_score = self.val_score + self.macro_score + self.trend_score + self.sent_score

        # 自动划分4类行情（基于滞后昨日数据，无未来泄露）
        deviation = (hs300_close / (hs300_ma20 + 1e-8) - 1) * 100
        pmi_level = pmi
        if deviation > 5 and pmi_level > 50:
            self.market_regime = 0  # 上行牛市
        elif deviation < -5 and pmi_level < 50:
            self.market_regime = 1  # 下行熊市
        elif deviation < -5:
            self.market_regime = 3  # 震荡偏弱
        else:
            self.market_regime = 2  # 震荡中性
        self.market_onehot = regime_one_hot(self.market_regime)

    def _is_rebalance_day(self, idx):
        current_date = self.price_df_raw.loc[idx, "date"]
        if idx <= 0:
            return True
        prev_date = self.price_df_raw.loc[idx - 1, "date"]
        return current_date.month != prev_date.month

    def _apply_min_bond_constraint(self, weight):
        if not self.enable_min_bond_hard_constraint:
            return weight
        target_weight = weight.copy()
        bond_w = target_weight[BOND_IDX]
        if bond_w < MIN_BOND_WEIGHT:
            gap = MIN_BOND_WEIGHT - bond_w
            target_weight[BOND_IDX] = MIN_BOND_WEIGHT
            equity_sum = np.sum(target_weight[:BOND_IDX])
            target_weight[:BOND_IDX] = target_weight[:BOND_IDX] * (equity_sum - gap) / (equity_sum + 1e-8)
        target_weight = target_weight / np.sum(target_weight)
        return target_weight

    def step(self, action):
        # 月度调仓逻辑
        if self.rebalance_monthly and not self._is_rebalance_day(self.current_idx):
            target_weight = self.last_target_weight.copy()
            update_action = False
        else:
            target_weight = self.softmax(action)
            target_weight = np.clip(target_weight, self.MIN_W, self.MAX_W)
            target_weight = target_weight / np.sum(target_weight)
            target_weight = self._apply_min_bond_constraint(target_weight)
            self.last_target_weight = target_weight.copy()
            update_action = True

        # 打分全部基于昨日滞后数据，无当日泄露
        self.calc_daily_market_score(self.current_idx)
        self.bond_weight_cache.append(target_weight[BOND_IDX])

        # price_t 仅用于计算t→t+1真实收益，不参与任何特征/奖励判定
        price_t = self.price_array[self.current_idx]
        price_t1 = self.price_array[self.current_idx + 1]
        returns = (price_t1 - price_t) / (price_t + 1e-8)
        returns = np.clip(returns, -0.20, 0.20)
        port_ret = np.dot(target_weight, returns)

        turnover = np.sum(np.abs(target_weight - self.last_weight)) if update_action else 0.0
        port_ret -= turnover * self.TRADING_COST_RATE
        port_ret = np.clip(port_ret, -0.10, 0.10)

        self.net_value *= (1 + port_ret)
        self.ret_queue.append(port_ret)
        self.window_net_cache.append(self.net_value)
        self.hist_returns.append(returns)

        # 滚动夏普
        hist_rets = list(self.ret_queue)
        if len(hist_rets) < self.window:
            pad = [0.0] * (self.window - len(hist_rets))
            hist_rets = hist_rets + pad
        current_sharpe = self.calc_rolling_sharpe(hist_rets)
        sharpe_delta = current_sharpe - self.sharpe_rolling
        self.sharpe_rolling = current_sharpe

        coef_profit, coef_vol, coef_dd, coef_turn = self.reward_coef
        weight_abs = 0.8
        weight_delta = 0.2
        r_abs = current_sharpe / 100
        r_sharpe = coef_profit * (weight_abs * r_abs + weight_delta * sharpe_delta)

        # 长期收益奖励
        long_term_reward = 0.0
        if len(self.window_net_cache) >= self.window:
            start_net = list(self.window_net_cache)[0]
            end_net = self.net_value
            window_cum_return = (end_net / (start_net + 1e-8)) - 1.0
            long_term_reward = 1.6 * window_cum_return

        # 波动率惩罚
        self.vol_ewma = 0.9 * self.vol_ewma + 0.1 * abs(port_ret)
        vol_penalty = coef_vol * self.vol_ewma
        ret_list = list(self.ret_queue)
        ret_std = np.std(ret_list)
        cycle_vol_penalty = 0.02 * ret_std

        # 窗口局部峰值回撤
        if self.net_value > self.peak_value:
            self.peak_value = self.net_value
        drawdown = (self.net_value / self.peak_value) - 1.0
        self.drawdown_cache.append(drawdown)
        abs_dd = -drawdown

        # 全周期全局峰值回撤（仅用于惩罚，不进观测，无泄露）
        if self.net_value > self.global_peak:
            self.global_peak = self.net_value
        full_dd = (self.net_value / self.global_peak) - 1.0
        self.full_history_dd.append(full_dd)
        abs_full_dd = -full_dd

        # 短期阶梯回撤惩罚
        dd_penalty = 0.0
        if self.enable_ladder_dd:
            base = coef_dd * abs_dd
            dd_penalty += base * 2.2
            if abs_dd > 0.05:
                dd_penalty += base * 3.0
            if abs_dd > 0.15:
                dd_penalty += base * 7.0
            if abs_dd > 0.20:
                dd_penalty += base * 5.0
        else:
            dd_penalty = coef_dd * abs_dd * 4.0

        # 全周期历史最大回撤分层强惩罚（根治全局33%回撤）
        full_max_dd = min(self.full_history_dd)
        abs_full_max_dd = -full_max_dd
        global_long_penalty = 0.0
        if abs_full_max_dd > 0.15:
            global_long_penalty += coef_dd * abs_full_max_dd * 4.0
        if abs_full_max_dd > 0.22:
            global_long_penalty += coef_dd * abs_full_max_dd * 7.0
        if abs_full_max_dd > 0.28:
            global_long_penalty += coef_dd * abs_full_max_dd * 12.0
        dd_penalty = dd_penalty + global_long_penalty

        # 下半方差惩罚
        down_side_penalty = 0.0
        if len(ret_list) > 20:
            ret_arr = np.array(ret_list)
            down_ret = ret_arr[ret_arr < 0]
            down_std = np.std(down_ret) if len(down_ret) > 0 else 0
            down_side_penalty = 1.2 * down_std

        # 分散奖励
        weight_std = np.std(target_weight)
        diversify_reward = 0.3 * (0.5 - weight_std)

        # 连续下跌惩罚
        consecutive_down = 0
        for r in reversed(ret_list):
            if r < 0:
                consecutive_down += 1
            else:
                break
        down_penalty = 0.02 * consecutive_down

        # 约束惩罚
        turn_penalty = coef_turn * turnover
        upper_violation = np.sum(np.maximum(0.0, target_weight - self.MAX_W))
        lower_violation = np.sum(np.maximum(0.0, self.MIN_W - target_weight))
        constraint_penalty = self.CONSTRAINT_COEF * (upper_violation + lower_violation)

        score_reward = 0.5 * (self.total_market_score / 4.0)

        # 债券奖励：熊市大幅提高持债激励
        bond_weight = target_weight[BOND_IDX]
        bond_reward = 0.0
        if self.enable_bond_regime_reward:
            regime = self.market_regime
            if regime == 1:
                bond_coef = 3   # 下行熊市高奖励
            elif regime == 3:
                bond_coef = 2  # 震荡偏弱
            elif regime == 2:
                bond_coef = 0.35
            else:
                bond_coef = 0.15
            base_bond = bond_coef * bond_weight
            if self.enable_bond_loss_couple:
                bond_contribution = max(-port_ret, 0.0)
                base_bond *= bond_contribution
            if bond_weight >= MIN_BOND_WEIGHT:
                base_bond += 1.0
            if bond_weight < 0.2:
                base_bond -= 1.2
            bond_reward = base_bond

        turnover_reward = 0.15 * np.clip(turnover, 0, 0.2)
        reward = (r_sharpe + long_term_reward + score_reward + diversify_reward + bond_reward + turnover_reward
                  - vol_penalty - cycle_vol_penalty - dd_penalty
                  - turn_penalty - down_penalty - down_side_penalty)

        # 奖励标准化
        if self.enable_reward_norm:
            self.reward_norm_queue.append(reward)
            if len(self.reward_norm_queue) >= 20:
                r_mean = np.mean(self.reward_norm_queue)
                r_std = np.std(self.reward_norm_queue) + 1e-6
                reward = (reward - r_mean) / r_std
            reward = np.clip(reward, -3.0, 3.0)
        else:
            reward = np.clip(reward, -5.0, 5.0)

        self.last_weight = target_weight.copy()
        self.current_idx += 1
        terminated = self.current_idx >= self.end_idx
        truncated = False

        info = {
            "turnover": turnover,
            "net_value": self.net_value,
            "sharpe": current_sharpe,
            "drawdown": drawdown,
            "market_regime": self.market_regime,
            "bond_weight": bond_weight,
            "bond_reward": bond_reward,
            "dd_penalty": dd_penalty,
            "down_side_penalty": down_side_penalty,
            "total_market_score": self.total_market_score,
            "val_score": self.val_score,
            "macro_score": self.macro_score,
            "sent_score": self.sent_score,
            "trend_score": self.trend_score,
            "is_rebalance_day": update_action
        }

        # 观测拼接
        obs_idx = self.current_idx if not terminated else self.current_idx - 1
        obs_idx = np.clip(obs_idx, 0, self.feature_array.shape[0] - 1)
        obs_feat = self.feature_array[obs_idx]

        hist_flat = np.zeros(self.hist_ret_dim, dtype=np.float32)
        hist_list = list(self.hist_returns)
        fill_len = min(len(hist_list), 2)
        for i in range(fill_len):
            hist_flat[i * N_ASSET: (i + 1) * N_ASSET] = hist_list[i]

        base_obs = np.concatenate([
            obs_feat,
            self.last_weight,
            np.array([self.val_score, self.macro_score, self.sent_score, self.trend_score]),
            hist_flat
        ])

        score_mean = np.mean(list(self.score_cache)) if len(self.score_cache) > 0 else 0.0
        drawdown_mean = np.mean(list(self.drawdown_cache)) if len(self.drawdown_cache) > 0 else 0.0
        asset_mean_ret = np.array(list(self.asset_ret_cache)).mean(axis=0) if len(self.asset_ret_cache) > 0 else np.zeros(N_ASSET)
        bond_hist_mean = np.mean(list(self.bond_weight_cache)) if len(self.bond_weight_cache) > 0 else 0.0
        extra_seq_feat = np.concatenate([[score_mean, drawdown_mean], asset_mean_ret, [bond_hist_mean]])
        regime_oh = self.market_onehot

        lgb_pred_feat = np.array([])
        if self.enable_lgb_pred_obs:
            lgb_pred_cols = [f"lgb_pred_{name}" for name in ASSET_NAMES]
            lgb_pred_feat = np.array([obs_feat[self.col_map[c]] for c in lgb_pred_cols], dtype=np.float32)

        obs = np.concatenate([base_obs, extra_seq_feat, regime_oh, lgb_pred_feat])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3)

        # ========== 修复4：强制对齐观测输出维度，彻底解决shape不匹配报错 ==========
        if obs.shape[0] != self.obs_dim:
            fixed_obs = np.zeros(self.obs_dim, dtype=np.float32)
            copy_len = min(obs.shape[0], self.obs_dim)
            fixed_obs[:copy_len] = obs[:copy_len]
            obs = fixed_obs

        self.score_cache.append(self.total_market_score)
        self.asset_ret_cache.append(returns.copy())
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = self.start_idx
        self.net_value = 1.0
        self.last_weight = np.zeros(N_ASSET, dtype=np.float32)
        self.last_weight[0] = 1.0
        self.last_target_weight = self.last_weight.copy()

        # 清空全部缓存
        self.ret_queue.clear()
        self.hist_returns.clear()
        self.window_net_cache.clear()
        self.score_cache.clear()
        self.asset_ret_cache.clear()
        self.drawdown_cache.clear()
        self.full_history_dd.clear()
        self.bond_weight_cache.clear()
        self.reward_norm_queue.clear()
        self.vol_ewma = 0.0
        self.peak_value = 1.0
        self.global_peak = 1.0
        self.sharpe_rolling = 0.0

        self.calc_daily_market_score(self.start_idx)
        obs_feat = self.feature_array[self.start_idx]
        hist_flat = np.zeros(self.hist_ret_dim, dtype=np.float32)
        base_obs = np.concatenate([
            obs_feat,
            self.last_weight,
            np.array([self.val_score, self.macro_score, self.sent_score, self.trend_score]),
            hist_flat
        ])
        extra_seq_feat = np.zeros(self.extra_seq_dim)
        regime_oh = self.market_onehot
        lgb_pred_feat = np.array([]) if not self.enable_lgb_pred_obs else np.zeros(N_ASSET)
        obs = np.concatenate([base_obs, extra_seq_feat, regime_oh, lgb_pred_feat])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3)

        # 强制对齐维度
        if obs.shape[0] != self.obs_dim:
            fixed_obs = np.zeros(self.obs_dim, dtype=np.float32)
            copy_len = min(obs.shape[0], self.obs_dim)
            fixed_obs[:copy_len] = obs[:copy_len]
            obs = fixed_obs

        info = {}
        return obs, info

    def render(self, mode="human"):
        turnover = np.sum(np.abs(self.last_weight))
        bond_w = self.last_weight[BOND_IDX]
        print(f"Step:{self.current_idx}, Net:{self.net_value:.4f}, Sharpe:{self.sharpe_rolling:.2f}, BondW:{bond_w:.2f}, Regime:{self.market_regime}, Turn:{turnover:.3f}")
