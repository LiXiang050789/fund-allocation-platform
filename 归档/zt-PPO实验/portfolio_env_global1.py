import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque

ASSET_NAMES = ["hs300", "zz500", "kc50", "consume", "chip", "gold", "bond10"]
N_ASSET = len(ASSET_NAMES)
N_EQUITY = N_ASSET - 1
BOND_IDX = -1
DEFAULT_SOFTMAX_TEMP = 1.0
ANNUAL_DAYS = 252
MIN_BOND_WEIGHT = 0.18


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
                 window=40, reward_coef=(3.0, 0.18, 1.3, 0.65), temp=None,
                 min_w=0.01, max_w=0.55, trade_cost=0.0003, constraint_coef=10.0,
                 rebalance_monthly=False,
                 enable_bond_regime_reward=True,
                 enable_bond_loss_couple=True,
                 enable_ladder_dd=True,
                 enable_min_bond_hard_constraint=True,
                 enable_lgb_pred_obs=False,
                 enable_reward_norm=True,
                 reward_mode="raw",
                 min_bond_weight=None,
                 action_bound=5.0,
                 enable_residual=False,
                 residual_scale=0.30,
                 enable_equity_only=False):
        super().__init__()
        self.rebalance_monthly = rebalance_monthly
        self.enable_bond_regime_reward = enable_bond_regime_reward
        self.enable_bond_loss_couple = enable_bond_loss_couple
        self.enable_ladder_dd = enable_ladder_dd
        self.enable_min_bond_hard_constraint = enable_min_bond_hard_constraint
        self.enable_reward_norm = enable_reward_norm
        self.enable_lgb_pred_obs = enable_lgb_pred_obs
        self.enable_lgb_pred_obs = enable_lgb_pred_obs
        self.MIN_BOND_WEIGHT = float(min_bond_weight) if min_bond_weight is not None else MIN_BOND_WEIGHT
        self.ACTION_BOUND = float(action_bound)
        self.enable_residual = bool(enable_residual)
        self.residual_scale = float(residual_scale)
        self.enable_equity_only = bool(enable_equity_only)
        self.LADDER_MAX_BOND = 0.50
        self.LADDER_TABLE = [
            (0.05, 0.15),  # 回撤 ≥ 5%  → 债券底仓至少 15%
            (0.10, 0.25),  # 回撤 ≥ 10% → 债券底仓至少 25%
            (0.15, 0.35),
            (0.20, 0.45),
            (0.25, 0.50),
        ]

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
        self.base_feat_dim = self.feature_array.shape[1]

        # —— 修复 4：滞后一天的特征在 __init__ 里一次性构造 ——
        self.feat_np_shifted = np.vstack([
            np.full((1, self.feat_np.shape[1]), np.nan, dtype=np.float32),
            self.feat_np[:-1]
        ])

        self.MIN_W = min_w
        self.MAX_W = max_w
        self.TRADING_COST_RATE = trade_cost
        self.CONSTRAINT_COEF = constraint_coef
        self.temp = temp if temp is not None else DEFAULT_SOFTMAX_TEMP
        self.window = window
        self.reward_coef = reward_coef

        max_valid_idx = len(self.price_array) - 1
        self.end_idx = min(end_idx, max_valid_idx)
        self.start_idx = start_idx
        self.max_feature_idx = self.feat_np.shape[0] - 1
        if self.start_idx >= self.end_idx:
            raise ValueError(f"非法区间：start_idx({self.start_idx}) >= end_idx({self.end_idx})")
        self.current_idx = self.start_idx

        self.hist_ret_dim = N_ASSET * 2
        self.regime_dim = 4
        self.lgb_pred_dim = N_ASSET if self.enable_lgb_pred_obs else 0
        self.market_score_dim = 4
        self.bond_hist_dim = 1
        self.extra_seq_dim = 2 + N_ASSET + self.bond_hist_dim

        # —— 修复 5：obs_dim 使用 actual_weight + last_target_weight 两个 N_ASSET ——
        self.obs_dim = (
            self.base_feat_dim
            + 2 * N_ASSET
            + self.market_score_dim
            + self.hist_ret_dim
            + self.extra_seq_dim
            + self.regime_dim
            + self.lgb_pred_dim
        )

        self.action_space = spaces.Box(low=-self.ACTION_BOUND, high=self.ACTION_BOUND,
                                       shape=(N_ASSET,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.obs_dim,), dtype=np.float32)

        self.last_weight = np.zeros(N_ASSET, dtype=np.float32)
        self.last_weight[0] = 1.0
        self.last_target_weight = self.last_weight.copy()
        self.actual_weight = self.last_weight.copy()
        self.net_value = 1.0
        self.ret_queue = deque(maxlen=window)
        self.window_net_cache = deque(maxlen=window)
        self.score_cache = deque(maxlen=self.window)
        self.asset_ret_cache = deque(maxlen=self.window)
        self.bond_weight_cache = deque(maxlen=self.window)
        self.val_score = 0.0
        self.macro_score = 0.0
        self.sent_score = 0.0
        self.trend_score = 0.0
        self.total_market_score = 0.0

        north_raw = self.feat_np[:, self.col_map["north_net"]]
        north_ser = pd.Series(north_raw)
        self.north_20roll = north_ser.rolling(20, min_periods=20).sum().shift(1).values
        spread_raw = self.feat_np[:, self.col_map["bond_10y_2y_spread"]]
        margin_raw = self.feat_np[:, self.col_map["margin_5d_chg"]]
        self.spread_shift = np.zeros_like(spread_raw)
        self.margin_shift = np.zeros_like(margin_raw)
        if len(spread_raw) > 1:
            self.spread_shift[1:] = spread_raw[:-1]
            self.margin_shift[1:] = margin_raw[:-1]

        hs300_series = self.price_df_raw["hs300"]
        self.hs300_ma20_series = hs300_series.rolling(20, min_periods=1).mean().values
        self.hs300_ma20_shift = np.zeros_like(self.hs300_ma20_series)
        if len(self.hs300_ma20_series) > 1:
            self.hs300_ma20_shift[1:] = self.hs300_ma20_series[:-1]

        pe_raw = self.feat_np[:, self.col_map["hs300_pe_quantile"]]
        self.pe_shift = np.zeros_like(pe_raw)
        if len(pe_raw) > 1:
            self.pe_shift[1:] = pe_raw[:-1]

        hs300_price_raw = self.price_array[:, 0]
        self.hs300_price_shift = np.zeros_like(hs300_price_raw)
        if len(hs300_price_raw) > 1:
            self.hs300_price_shift[1:] = hs300_price_raw[:-1]

        self.drawdown_cache = deque(maxlen=self.window)
        self.full_history_dd = []
        self.global_peak = 1.0
        self.peak_value = 1.0
        self.vol_ewma = 0.0
        self.sharpe_rolling = 0.0
        self.reward_norm_queue = deque(maxlen=60)
        self.hist_returns = deque(maxlen=2)
        self.market_regime = 2
        self.market_onehot = regime_one_hot(self.market_regime)
        dates = pd.to_datetime(self.price_df_raw["date"])
        self.rebalance_flags = dates.dt.to_period("M").ne(
            dates.dt.to_period("M").shift()
        ).values

        # —— 修复 3 / 小修：新增状态 ——
        self.prev_abs_dd = 0.0
        self.last_turnover = 0.0

    # —— 修复 1：_project_weights 移入类内 ——
    def _project_weights(self, raw_weight, listed_mask=None):
        """
        将 raw_weight 投影到可行权重集合：
          - sum(w) = 1
          - 上市资产: MIN_W <= w_i <= MAX_W
          - 未上市资产: w_i = 0
          - 债券: w_bond >= MIN_BOND_WEIGHT (若开启硬约束)
        带盒约束的单纯形投影，二分求 lambda。
        """
        n = N_ASSET
        if listed_mask is None:
            listed_mask = np.ones(n, dtype=bool)
        listed_mask = np.asarray(listed_mask, dtype=bool)

        lo = np.zeros(n, dtype=np.float64)
        hi = np.zeros(n, dtype=np.float64)
        lo[listed_mask] = self.MIN_W
        hi[listed_mask] = self.MAX_W

        if listed_mask[BOND_IDX] and self.enable_min_bond_hard_constraint:
            cur_min_bond = self._get_current_min_bond()
            lo[BOND_IDX] = max(self.MIN_W, cur_min_bond)

        if lo.sum() > 1.0 + 1e-9 or hi.sum() < 1.0 - 1e-9:
            w = np.zeros(n, dtype=np.float64)
            n_listed = int(listed_mask.sum())
            if n_listed > 0:
                w[listed_mask] = 1.0 / n_listed
            return w.astype(np.float32)

        v = np.asarray(raw_weight, dtype=np.float64).copy()
        v[~listed_mask] = 0.0

        a = float((v - hi).min())
        b = float((v - lo).max())
        for _ in range(80):
            c = 0.5 * (a + b)
            w = np.clip(v - c, lo, hi)
            s = float(w.sum())
            if abs(s - 1.0) < 1e-10:
                break
            if s > 1.0:
                a = c
            else:
                b = c

        w = np.clip(v - 0.5 * (a + b), lo, hi)
        w[~listed_mask] = 0.0
        s = float(w.sum())
        if s > 1e-9 and abs(s - 1.0) > 1e-8:
            w = np.clip(w / s, lo, hi)
        return w.astype(np.float32)

    def softmax(self, x):
        x = x / self.temp
        max_x = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    def _rule_baseline_weight(self):
        """
        基于 hs300_pe_quantile 的简单规则权重。
        PE 高（贵）→ 债券多；PE 低（便宜）→ 债券少。
        用 shift 后的 PE 避免未来函数。
        注意：feat 已经做过标准化，PE 是 z-score 尺度。
        """
        idx = int(np.clip(self.current_idx, 0, self.max_feature_idx))
        pe_z = self.pe_shift[idx]
        if np.isnan(pe_z):
            pe_z = 0.0
        # z-score 裁到 [-2, 2]，映射到 [-1, 1]
        norm_pe = np.clip(pe_z / 2.0, -1.0, 1.0)
        # 低估 (norm_pe=-1) → bond=0.05；高估 (norm_pe=+1) → bond=0.55
        bw = 0.30 + 0.25 * norm_pe
        bw = float(np.clip(bw, 0.05, 0.70))

        w = np.zeros(N_ASSET, dtype=np.float64)
        w[:BOND_IDX] = (1.0 - bw) / N_EQUITY
        w[BOND_IDX] = bw
        return w
    def _get_current_min_bond(self):
        """回撤越深，债券硬底仓越高（阶梯式）"""
        if not self.enable_ladder_dd:
            return self.MIN_BOND_WEIGHT
        # 当前回撤（相对全局峰值）
        if self.global_peak > 1e-8:
            abs_dd = max(0.0, 1.0 - self.net_value / self.global_peak)
        else:
            abs_dd = 0.0
        target = self.MIN_BOND_WEIGHT
        for dd_thresh, bond_floor in self.LADDER_TABLE:
            if abs_dd >= dd_thresh:
                target = bond_floor
        return min(target, self.LADDER_MAX_BOND)

    def calc_rolling_sharpe(self, ret_list):
        rets = np.array(ret_list)
        mean_r = np.mean(rets)
        std_r = np.std(rets) + 1e-8
        sharpe = mean_r / std_r * np.sqrt(ANNUAL_DAYS)
        return sharpe

    def calc_daily_market_score(self, idx):
        safe_idx = np.clip(idx, 0, self.max_feature_idx)
        # —— 修复 4：改用滞后特征 ——
        row_np = self.feat_np_shifted[safe_idx]
        hs300_pe = self.pe_shift[safe_idx]
        pmi = row_np[self.col_map["pmi_制造业采购经理指数"]]
        spread = self.spread_shift[safe_idx]
        margin_chg = self.margin_shift[safe_idx]
        north_flow = self.north_20roll[safe_idx]

        if np.isnan(hs300_pe):
            hs300_pe = 0.5
        if np.isnan(pmi):
            pmi = 50.0
        if np.isnan(spread):
            spread = 0.25
        if np.isnan(margin_chg):
            margin_chg = 0.0
        hs300_close = self.hs300_price_shift[safe_idx]
        hs300_ma20 = self.hs300_ma20_shift[safe_idx]
        if np.isnan(hs300_close) or np.isnan(hs300_ma20):
            hs300_close = 1.0
            hs300_ma20 = 1.0

        val_raw = calc_val_score(hs300_pe)
        macro_raw = calc_macro_score(pmi, spread)
        sent_raw = calc_sent_score(margin_chg, north_flow)
        trend_raw = calc_trend_score(hs300_close, hs300_ma20)
        self.val_score = val_raw / 25.0
        self.macro_score = macro_raw / 25.0
        self.sent_score = sent_raw / 20.0
        self.trend_score = trend_raw / 30.0
        self.total_market_score = self.val_score + self.macro_score + self.trend_score + self.sent_score

        deviation = (hs300_close / (hs300_ma20 + 1e-8) - 1) * 100
        pmi_level = pmi
        if deviation > 5 and pmi_level > 50:
            self.market_regime = 0
        elif deviation < -5 and pmi_level < 50:
            self.market_regime = 1
        elif deviation < -5:
            self.market_regime = 3
        else:
            self.market_regime = 2
        self.market_onehot = regime_one_hot(self.market_regime)

    def _is_rebalance_day(self, idx):
        return bool(self.rebalance_flags[idx])

    def _apply_min_bond_constraint(self, weight):
        # 保留作为 fallback（当前 step 主路径已由 _project_weights 保证约束）
        if not self.enable_min_bond_hard_constraint:
            return weight
        target_weight = weight.copy()
        bond_price = self.price_array[self.current_idx, BOND_IDX]
        if bond_price < 1e-6:
            return target_weight / (np.sum(target_weight) + 1e-8)
        bond_w = target_weight[BOND_IDX]
        if bond_w < MIN_BOND_WEIGHT:
            gap = MIN_BOND_WEIGHT - bond_w
            equity_sum = np.sum(target_weight[:BOND_IDX])
            target_weight[BOND_IDX] = MIN_BOND_WEIGHT
            if equity_sum > 1e-6:
                target_weight[:BOND_IDX] = target_weight[:BOND_IDX] * (equity_sum - gap) / (equity_sum + 1e-8)
            else:
                target_weight[:BOND_IDX] = 0.0
        target_weight = target_weight / (np.sum(target_weight) + 1e-8)
        return target_weight

    def step(self, action):
        # ---------- 1) 目标权重 ----------
        if self.rebalance_monthly and not self._is_rebalance_day(self.current_idx):
            target_weight = self.last_target_weight.copy()
            update_action = False
        else:
            price_t = self.price_array[self.current_idx]
            listed_mask = np.isfinite(price_t) & (price_t >= 1e-6)

            if self.enable_equity_only:
                # 债券由规则决定；PPO 学"相对权益等权的偏离"
                rule_w = self._rule_baseline_weight()
                bond_w = float(rule_w[BOND_IDX])

                eq_action = action[:BOND_IDX]  # 前 6 维
                eq_delta = self.softmax(eq_action) - 1.0 / N_EQUITY  # sum=0
                eq_w = 1.0 / N_EQUITY + self.residual_scale * eq_delta
                eq_w = np.clip(eq_w, 0.0, None)
                s_eq = eq_w.sum()
                if s_eq > 1e-8:
                    eq_w = eq_w / s_eq
                else:
                    eq_w = np.ones(N_EQUITY) / N_EQUITY

                raw_w = np.zeros(N_ASSET)
                raw_w[:BOND_IDX] = (1.0 - bond_w) * eq_w
                raw_w[BOND_IDX] = bond_w
            elif self.enable_residual:
                rule_w = self._rule_baseline_weight()
                delta_raw = self.softmax(action)
                delta = delta_raw - 1.0 / N_ASSET
                raw_w = rule_w + self.residual_scale * delta
                raw_w = np.clip(raw_w, 0.0, None)
                s = raw_w.sum()
                if s > 1e-8:
                    raw_w = raw_w / s
                else:
                    raw_w = rule_w
            else:
                raw_w = self.softmax(action)

            target_weight = self._project_weights(raw_w, listed_mask)
            update_action = True

        # ---------- 2) 当日市场评分 ----------
        self.calc_daily_market_score(self.current_idx)

        # ---------- 3) 修复 2：补回 returns 计算 + NaN 处理 ----------
        price_t = self.price_array[self.current_idx]
        price_t1 = self.price_array[self.current_idx + 1]
        unlisted_mask = (
            np.isnan(price_t) | np.isnan(price_t1)
            | (price_t < 1e-6) | (price_t1 < 1e-6)
        )
        returns = np.zeros_like(price_t, dtype=np.float32)
        listed_idx = ~unlisted_mask
        returns[listed_idx] = (
            (price_t1[listed_idx] - price_t[listed_idx])
            / (price_t[listed_idx] + 1e-8)
        )
        returns = np.clip(returns, -0.20, 0.20)

        self.bond_weight_cache.append(target_weight[BOND_IDX])
        self.last_target_weight = target_weight.copy()

        # ---------- 4) 组合收益 ----------
        if update_action:
            turnover = float(np.sum(np.abs(target_weight - self.actual_weight)))
            port_ret = float(np.dot(target_weight, returns)) - turnover * self.TRADING_COST_RATE
        else:
            turnover = 0.0
            port_ret = float(np.dot(self.actual_weight, returns))
        port_ret = float(np.clip(port_ret, -0.10, 0.10))

        self.net_value *= (1 + port_ret)
        self.ret_queue.append(port_ret)
        self.window_net_cache.append(self.net_value)
        self.hist_returns.append(returns)

        # ---------- 5) 权重漂移 ----------
        drift = self.actual_weight * (1.0 + returns)
        drift_sum = np.sum(drift)
        if drift_sum > 1e-8:
            self.actual_weight = drift / (drift_sum + 1e-8)
        else:
            self.actual_weight = np.zeros(N_ASSET, dtype=np.float32)
        if update_action:
            self.actual_weight = target_weight.copy()

        # ---------- 6) 滚动指标 ----------
        hist_rets = list(self.ret_queue)
        if len(hist_rets) < self.window:
            pad = [0.0] * (self.window - len(hist_rets))
            hist_rets = hist_rets + pad
        current_sharpe = self.calc_rolling_sharpe(hist_rets)
        sharpe_delta = current_sharpe - self.sharpe_rolling
        self.sharpe_rolling = current_sharpe

        if self.net_value > self.peak_value:
            self.peak_value = self.net_value
        drawdown = (self.net_value / self.peak_value) - 1.0
        self.drawdown_cache.append(drawdown)
        abs_dd = -drawdown
        if self.net_value > self.global_peak:
            self.global_peak = self.net_value
        full_dd = (self.net_value / self.global_peak) - 1.0
        self.full_history_dd.append(full_dd)

        self.vol_ewma = 0.9 * self.vol_ewma + 0.1 * abs(port_ret)

        # ---------- 7) 奖励 ----------
        base_coef, vol_coef, turn_coef, dd_coef = self.reward_coef

        reward_mode = getattr(self, "reward_mode", "raw")
        if reward_mode == "excess":
            baseline_ret = 0.6 * float(returns[0]) + 0.4 * float(returns[BOND_IDX])
            learn_ret = port_ret - baseline_ret
        elif reward_mode == "excess_rule":
            rule_w = self._rule_baseline_weight()
            rule_ret = float(np.dot(rule_w, returns))
            learn_ret = port_ret - rule_ret
        else:
            learn_ret = port_ret

        reward = (
                base_coef * learn_ret
                - vol_coef * self.vol_ewma
                - turn_coef * turnover
                - dd_coef * max(0.0, abs_dd - self.prev_abs_dd)
        )
        reward += 0.02 * sharpe_delta
        self.prev_abs_dd = abs_dd

        if self.enable_reward_norm:
            self.reward_norm_queue.append(reward)
            if len(self.reward_norm_queue) >= 20:
                r_mean = np.mean(self.reward_norm_queue)
                r_std = np.std(self.reward_norm_queue) + 1e-6
                reward = (reward - r_mean) / r_std
            reward = float(np.clip(reward, -3.0, 3.0))
        else:
            reward = float(np.clip(reward, -5.0, 5.0))

        self.last_weight = target_weight.copy()
        self.last_turnover = turnover
        self.current_idx += 1
        terminated = self.current_idx >= self.end_idx
        truncated = False
        info = {
            "turnover": turnover,
            "net_value": self.net_value,
            "sharpe": current_sharpe,
            "drawdown": drawdown,
            "market_regime": self.market_regime,
            "bond_weight": float(target_weight[BOND_IDX]),
        }

        # ---------- 8) 修复 5：obs 与 reset 统一 ----------
        obs_idx = self.current_idx if not terminated else self.current_idx - 1
        obs_idx = int(np.clip(obs_idx, 0, self.feature_array.shape[0] - 1))
        obs_feat = self.feature_array[obs_idx]
        hist_flat = np.zeros(self.hist_ret_dim, dtype=np.float32)
        hist_list = list(self.hist_returns)
        fill_len = min(len(hist_list), 2)
        for i in range(fill_len):
            hist_flat[i * N_ASSET: (i + 1) * N_ASSET] = hist_list[i]

        base_obs = np.concatenate([
            obs_feat,
            self.actual_weight,
            self.last_target_weight,
            np.array([self.val_score, self.macro_score, self.sent_score, self.trend_score]),
            hist_flat
        ])

        score_mean = np.mean(list(self.score_cache)) if len(self.score_cache) > 0 else 0.0
        drawdown_mean = np.mean(list(self.drawdown_cache)) if len(self.drawdown_cache) > 0 else 0.0
        asset_mean_ret = (
            np.array(list(self.asset_ret_cache)).mean(axis=0)
            if len(self.asset_ret_cache) > 0 else np.zeros(N_ASSET)
        )
        bond_hist_mean = (
            np.mean(list(self.bond_weight_cache))
            if len(self.bond_weight_cache) > 0 else 0.0
        )
        extra_seq_feat = np.concatenate(
            [[score_mean, drawdown_mean], asset_mean_ret, [bond_hist_mean]]
        )
        regime_oh = self.market_onehot
        lgb_pred_feat = np.array([])
        if self.enable_lgb_pred_obs:
            lgb_pred_cols = [f"lgb_pred_{name}" for name in ASSET_NAMES]
            lgb_pred_feat = np.array([
                obs_feat[self.col_map[c]] if c in self.col_map else 0.0
                for c in lgb_pred_cols
            ], dtype=np.float32)

        obs = np.concatenate([base_obs, extra_seq_feat, regime_oh, lgb_pred_feat])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3)
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
        # —— 支持随机起始 ——
        if options is not None and "start_idx" in options:
            self.current_idx = int(options["start_idx"])
        else:
            self.current_idx = self.start_idx

        self.net_value = 1.0
        # 初始权重：权益等权 + 债券满足硬底仓
        self.last_weight = np.ones(N_ASSET, dtype=np.float32) / N_ASSET
        self.last_weight[BOND_IDX] = max(self.MIN_BOND_WEIGHT, 1.0 / N_ASSET)
        self.last_weight[:BOND_IDX] = (1 - self.last_weight[BOND_IDX]) / N_EQUITY
        self.actual_weight = self.last_weight.copy()
        # —— 修复 6：同步 last_target_weight ——
        self.last_target_weight = self.last_weight.copy()

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
        # —— 修复 3 / 小修：重置新状态 ——
        self.prev_abs_dd = 0.0
        self.last_turnover = 0.0

        self.calc_daily_market_score(self.current_idx)
        obs_feat = self.feature_array[self.current_idx]
        hist_flat = np.zeros(self.hist_ret_dim, dtype=np.float32)

        # —— 修复 5：obs 结构与 step 完全一致 ——
        base_obs = np.concatenate([
            obs_feat,
            self.actual_weight,
            self.last_target_weight,
            np.array([self.val_score, self.macro_score, self.sent_score, self.trend_score]),
            hist_flat
        ])
        extra_seq_feat = np.zeros(self.extra_seq_dim)
        regime_oh = self.market_onehot
        lgb_pred_feat = np.array([]) if not self.enable_lgb_pred_obs else np.zeros(N_ASSET)
        obs = np.concatenate([base_obs, extra_seq_feat, regime_oh, lgb_pred_feat])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3)
        if obs.shape[0] != self.obs_dim:
            fixed_obs = np.zeros(self.obs_dim, dtype=np.float32)
            copy_len = min(obs.shape[0], self.obs_dim)
            fixed_obs[:copy_len] = obs[:copy_len]
            obs = fixed_obs
        info = {}
        return obs, info

    def render(self, mode="human"):
        # —— 小修：使用 last_turnover ——
        bond_w = self.last_weight[BOND_IDX]
        print(
            f"Step:{self.current_idx}, Net:{self.net_value:.4f}, "
            f"Sharpe:{self.sharpe_rolling:.2f}, BondW:{bond_w:.2f}, "
            f"Regime:{self.market_regime}, Turn:{self.last_turnover:.3f}"
        )
