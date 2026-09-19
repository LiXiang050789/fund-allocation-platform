"""
三档风险组合回测（C2/C3/C4）
基于现有策略，按目标权益中枢缩放权益仓位
"""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============ 路径 ============
BASE = r"E:/农行杯/2026/1/Streamlit平台"
DATA = r"E:\农行杯\2026\1\clean"
OUT  = r"E:\农行杯\2026\1\回测"
LGB_PRED_PATH = r"E:\农行杯\2026\1\Streamlit平台\data\results\pred_return_lgb_daily.csv"
os.makedirs(OUT, exist_ok=True)

# ============ 全局参数（与回测对比.py 一致）============
INIT_CASH = 1_000_000
SLIPPAGE = 0.0005
FEE = 0.0005
MAX_WEIGHT = 0.30
MIN_WEIGHT = 0.0
RISK_FREE = 0.025
MVO_WINDOW = 60

ETF_CODES = ['hs300','zz500','kc50','consume','chip','gold','bond10']
EQUITY_ETFS = ['hs300','zz500','kc50','consume','chip']
HEDGE_ETFS = ['gold','bond10']
EQUITY_INDEXES = [ETF_CODES.index(e) for e in EQUITY_ETFS]
HEDGE_INDEXES = [ETF_CODES.index(e) for e in HEDGE_ETFS]

# 三档权益中枢
RISK_LEVELS = {
    '保守型C2': 0.20,
    '平衡型C3': 0.50,
    '进取型C4': 0.70,
}
# 原策略的"中性"权益仓位（动态评分的震荡中枢）
BASE_EQUITY_CENTER = 0.35

# ============ 加载数据 ============
price_df = pd.read_csv(f'{DATA}/etf_price_clean.csv', encoding='utf-8-sig', parse_dates=['date'])
listed_df = pd.read_csv(f'{DATA}/etf_price_clean.csv', encoding='utf-8-sig', parse_dates=['date'])
score_df  = pd.read_csv(f'{DATA}/market_score_daily.csv', encoding='utf-8-sig', parse_dates=['date'])

price_df = price_df.set_index('date')
prices = price_df[ETF_CODES].copy()
listed_cols = [f'{e}_listed' for e in ETF_CODES]
avail = listed_df.set_index('date')[listed_cols].copy()
avail.columns = ETF_CODES
state_map = score_df.set_index('date')['market_state']

common_dates = prices.index.intersection(avail.index).intersection(state_map.index)
prices = prices.loc[common_dates]
avail  = avail.loc[common_dates]
state  = state_map.loc[common_dates]
returns = prices.pct_change().fillna(0)

print(f'[数据] 交易日: {len(common_dates)}, {common_dates[0].date()} ~ {common_dates[-1].date()}')

# ============ 工具函数 ============
def mask_available(date_idx):
    return avail.iloc[date_idx].values.astype(bool)

def solve_risk_parity(cov):
    n = cov.shape[0]
    if n == 0: return np.array([])
    vols = np.sqrt(np.diag(cov))
    vols = np.where(vols < 1e-10, 1e-10, vols)
    w = 1.0 / vols
    return w / w.sum()

def solve_mvo(mu, cov):
    n = len(mu)
    if n == 0: return np.array([])
    mu_shrink = 0.6 * mu + 0.4 * np.full_like(mu, mu.mean())
    excess = mu_shrink - RISK_FREE / 252
    try:
        cov_inv = np.linalg.inv(cov + np.eye(n) * 1e-6)
        w = cov_inv @ excess
        w = w / w.sum()
    except np.linalg.LinAlgError:
        w = np.ones(n) / n
    w = np.clip(w, MIN_WEIGHT, MAX_WEIGHT)
    s = w.sum()
    if s > 1e-12: w = w / s
    return w

def get_monthly_rebalance_dates(dates):
    df = pd.DataFrame({'date': dates})
    df['ym'] = df['date'].dt.to_period('M')
    return df.groupby('ym')['date'].first().values

# ============ 动态评分策略 ============
BASE_EQUITY = {
    '上行': 0.55, '震荡偏强': 0.45, '震荡': 0.35,
    '下行': 0.15, '极寒': 0.05
}

def strategy_dynamic_score(dates):
    monthly = get_monthly_rebalance_dates(dates)
    weights = {}
    for d in monthly:
        try:
            loc = dates.get_loc(d)
        except KeyError:
            continue
        try:
            st = state.loc[d]
        except (KeyError, IndexError):
            st = '震荡'
        if pd.isna(st): st = '震荡'
        target_equity = BASE_EQUITY.get(st, 0.35)
        avail_mask = mask_available(loc)
        eq_active = [i for i in EQUITY_INDEXES if avail_mask[i]]
        hedge_active = [i for i in HEDGE_INDEXES if avail_mask[i]]
        w = np.zeros(7)
        if eq_active and target_equity > 0:
            start = max(0, loc - MVO_WINDOW)
            ret_window = returns.iloc[start:loc+1].iloc[:, eq_active]
            if ret_window.shape[1] >= 2:
                mu = ret_window.mean().values
                cov = ret_window.cov().values
                sub_w = solve_mvo(mu, cov)
            else:
                sub_w = np.ones(len(eq_active)) / len(eq_active)
            for j, idx in enumerate(eq_active):
                w[idx] = sub_w[j] * target_equity
        if hedge_active and (1 - target_equity) > 0:
            start = max(0, loc - MVO_WINDOW)
            ret_window = returns.iloc[start:loc+1].iloc[:, hedge_active]
            if ret_window.shape[1] >= 2:
                cov = ret_window.cov().values
                sub_w = solve_risk_parity(cov)
            else:
                sub_w = np.ones(len(hedge_active)) / len(hedge_active)
            for j, idx in enumerate(hedge_active):
                w[idx] = sub_w[j] * (1 - target_equity)
        s = w.sum()
        if s > 0: w = w / s
        else: w[avail_mask] = 1.0 / avail_mask.sum()
        weights[pd.Timestamp(d)] = w
    return pd.DataFrame(weights, index=ETF_CODES).T

# ============ LGB 融合策略 ============
lgb_pred = None
if os.path.exists(LGB_PRED_PATH):
    lgb_pred = pd.read_csv(LGB_PRED_PATH, encoding='utf-8-sig', parse_dates=['date']).set_index('date')

def strategy_lgb_fusion(dates):
    if lgb_pred is None:
        return strategy_dynamic_score(dates)
    monthly = get_monthly_rebalance_dates(dates)
    weights = {}
    for d in monthly:
        try:
            loc = dates.get_loc(d)
        except KeyError:
            continue
        try:
            st = state.loc[d]
        except (KeyError, IndexError):
            st = '震荡'
        if pd.isna(st): st = '震荡'
        target_equity = BASE_EQUITY.get(st, 0.35)
        avail_mask = mask_available(loc)
        eq_active = [i for i in EQUITY_INDEXES if avail_mask[i]]
        hedge_active = [i for i in HEDGE_INDEXES if avail_mask[i]]
        w = np.zeros(7)
        if eq_active and target_equity > 0:
            eq_etfs = [ETF_CODES[i] for i in eq_active]
            lgb_loc = lgb_pred.index.get_indexer([d], method='pad')[0]
            lgb_window = lgb_pred.index[max(0, lgb_loc-21):lgb_loc+1]
            eq_preds = np.zeros(len(eq_etfs))
            for j, e in enumerate(eq_etfs):
                if e in lgb_pred.columns:
                    vals = lgb_pred[e].loc[lgb_window.intersection(lgb_pred.index)].dropna()
                    eq_preds[j] = vals.mean() if len(vals) > 0 else 0
            temp = 0.5
            x = eq_preds / (np.std(eq_preds) + 1e-8) / temp
            exp_x = np.exp(x - np.max(x))
            sub_w = exp_x / exp_x.sum()
            for j, idx in enumerate(eq_active):
                w[idx] = sub_w[j] * target_equity
        if hedge_active and (1 - target_equity) > 0:
            start = max(0, loc - MVO_WINDOW)
            ret_window = returns.iloc[start:loc+1].iloc[:, hedge_active]
            if ret_window.shape[1] >= 2:
                cov = ret_window.cov().values
                sub_w = solve_risk_parity(cov)
            else:
                sub_w = np.ones(len(hedge_active)) / len(hedge_active)
            for j, idx in enumerate(hedge_active):
                w[idx] = sub_w[j] * (1 - target_equity)
        s = w.sum()
        if s > 0: w = w / s
        else: w[avail_mask] = 1.0 / avail_mask.sum()
        weights[pd.Timestamp(d)] = w
    return pd.DataFrame(weights, index=ETF_CODES).T

# ============ 权益仓位缩放（核心）============
def rescale_equity(weight_df, target_center, base_center=BASE_EQUITY_CENTER):
    """
    把策略的权益总仓位按比例缩放到目标中枢
    保留原有债股相对择时，仅整体平移
    """
    scale = target_center / base_center
    new_df = weight_df.copy()
    for d in new_df.index:
        w = new_df.loc[d].values
        eq_sum = w[:6].sum()
        if eq_sum < 1e-8:
            # 全债，直接按目标中枢处理
            new_eq = target_center
            new_df.loc[d, ETF_CODES[:6]] = new_eq / 6
            new_df.loc[d, 'bond10'] = 1 - new_eq
        else:
            # 权益按比例缩放，债券补足
            new_eq = eq_sum * scale
            new_eq = np.clip(new_eq, 0.0, 1.0)
            if new_eq < 1e-8:
                new_df.loc[d, ETF_CODES[:6]] = 0.0
                new_df.loc[d, 'bond10'] = 1.0
            else:
                w_eq = w[:6] * (new_eq / eq_sum)
                # 单资产上限 30%
                w_eq = np.clip(w_eq, 0, MAX_WEIGHT)
                # 重新归一化权益部分
                s = w_eq.sum()
                if s > 1e-8:
                    w_eq = w_eq * (new_eq / s)
                new_df.loc[d, ETF_CODES[:6]] = w_eq
                new_df.loc[d, 'bond10'] = 1 - new_eq
    return new_df

# ============ 回测引擎 ============
def run_backtest(weight_df, dates, returns_df):
    daily_weights = pd.DataFrame(index=dates, columns=ETF_CODES, data=0.0)
    sorted_rebalance = weight_df.index.sort_values()
    for i, rd in enumerate(sorted_rebalance):
        if rd not in daily_weights.index:
            continue
        end = sorted_rebalance[i+1] if i+1 < len(sorted_rebalance) else dates[-1] + pd.Timedelta(days=1)
        mask = (daily_weights.index >= rd) & (daily_weights.index < end)
        daily_weights.loc[mask] = weight_df.loc[rd].values
    first_rebalance = sorted_rebalance[0]
    if first_rebalance in daily_weights.index:
        pre_mask = daily_weights.index < first_rebalance
        if pre_mask.any():
            daily_weights.loc[pre_mask] = weight_df.loc[first_rebalance].values
    daily_weights = daily_weights.reindex(dates).ffill().fillna(0.0)
    row_sums = daily_weights.sum(axis=1)
    daily_weights = daily_weights.div(row_sums.where(row_sums > 0, 1.0), axis=0)

    daily_ret = (daily_weights * returns_df.loc[dates]).sum(axis=1)
    for rd in sorted_rebalance:
        if rd not in daily_weights.index:
            continue
        loc = daily_weights.index.get_loc(rd)
        if loc > 0:
            prev_w = daily_weights.iloc[loc-1].values
            new_w  = daily_weights.iloc[loc].values
            turnover = np.abs(new_w - prev_w).sum()
            daily_ret.iloc[loc] -= turnover * FEE
    daily_ret = daily_ret - SLIPPAGE / 252
    ANNUAL_FEE = np.array([0.002, 0.002, 0.002, 0.006, 0.006, 0.006, 0.002])
    daily_fee = (daily_weights * ANNUAL_FEE).sum(axis=1) / 252
    daily_ret = daily_ret - daily_fee
    nav = (1 + daily_ret).cumprod() * INIT_CASH
    return nav, daily_ret, daily_weights

def calc_metrics(name, daily_returns, weight_df, nav_series):
    ret = daily_returns.dropna()
    if len(ret) == 0: return {}
    ann_ret = ret.mean() * 252
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 1e-8 else 0
    downside = ret[ret < 0]
    sortino_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-8
    sortino = (ann_ret - RISK_FREE) / sortino_vol
    cummax = nav_series.expanding().max()
    drawdown = (nav_series - cummax) / cummax
    max_dd = drawdown.min()
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-8 else 0
    monthly_turnover = 0
    if weight_df is not None and len(weight_df) > 1:
        turnovers = weight_df.diff().abs().sum(axis=1) / 2
        monthly_turnover = turnovers.iloc[1:].mean()
    return {
        '策略': name,
        '年化收益': round(ann_ret, 4),
        '年化波动': round(ann_vol, 4),
        '夏普比率': round(sharpe, 4),
        '最大回撤': round(max_dd, 4),
        '卡玛比率': round(calmar, 4),
        '月均换手': round(monthly_turnover, 4),
    }

# ============ 主流程 ============
if __name__ == "__main__":
    dates_arr = pd.DatetimeIndex(common_dates)
    returns_mat = returns.loc[common_dates]

    # 原始策略权重
    print("\n[1] 生成基础策略权重...")
    w_score = strategy_dynamic_score(common_dates)
    w_lgb = strategy_lgb_fusion(common_dates)

    results = []
    all_navs = {}

    print("\n[2] 三档风险组合回测...")
    for strat_name, w_base in [('动态评分', w_score), ('LGB融合', w_lgb)]:
        for level_name, target_center in RISK_LEVELS.items():
            # 缩放权益仓位
            w_scaled = rescale_equity(w_base, target_center)
            # 回测
            nav, ret, _ = run_backtest(w_scaled, dates_arr, returns_mat)
            m = calc_metrics(f"{level_name}_{strat_name}", ret, w_scaled, nav)
            m['风险档'] = level_name
            m['策略源'] = strat_name
            m['权益中枢'] = target_center
            results.append(m)
            all_navs[f"{level_name}_{strat_name}"] = nav
            print(f"  {level_name} | {strat_name}: "
                  f"年化={m['年化收益']*100:.1f}%, "
                  f"夏普={m['夏普比率']:.3f}, "
                  f"MDD={m['最大回撤']*100:.1f}%")

    # 输出
    df_result = pd.DataFrame(results)
    df_result = df_result[['风险档','权益中枢','策略源','年化收益','年化波动','夏普比率','最大回撤','卡玛比率','月均换手']]
    print("\n" + "="*100)
    print(df_result.to_string(index=False))
    print("="*100)

    df_result.to_csv(f"{OUT}/risk_levels_metrics.csv", index=False, encoding='utf-8-sig')
    pd.DataFrame(all_navs, index=common_dates).to_csv(f"{OUT}/risk_levels_nav.csv", encoding='utf-8-sig')

    print(f"\n✅ 结果保存至 {OUT}/risk_levels_*.csv")