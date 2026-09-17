"""
用新的 LGB 预测文件（pred_return_lgb_daily_fixed.csv）
在和回测对比.py 完全一致的框架下回测
同时跑两个版本做对比：
  版本A：旧预测文件（Streamlit 平台原始输出）
  版本B：新预测文件（修复版 lgb.py 输出）
"""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============ 路径 ============
DATA = r"E:\农行杯\2026\1\clean"
OUT  = r"E:\农行杯\2026\1\回测"

# 旧预测（Streamlit 平台输出，回测对比.py 用的）
LGB_OLD_PATH = r"E:\农行杯\2026\1\Streamlit平台\data\results\pred_return_lgb_daily.csv"
# 新预测（修复版 lgb.py 输出）
LGB_NEW_PATH = r"E:\农行杯\2026\1\results\pred_return_lgb_daily_fixed.csv"

os.makedirs(OUT, exist_ok=True)

# ============ 全局参数（与回测对比.py 完全一致）============
INIT_CASH    = 1_000_000
SLIPPAGE     = 0.0005
FEE          = 0.0005
MAX_WEIGHT   = 0.30
MIN_WEIGHT   = 0.0
RISK_FREE    = 0.025
MVO_WINDOW   = 60

ETF_CODES    = ['hs300','zz500','kc50','consume','chip','gold','bond10']
EQUITY_ETFS  = ['hs300','zz500','kc50','consume','chip']
HEDGE_ETFS   = ['gold','bond10']
EQUITY_INDEXES = [ETF_CODES.index(e) for e in EQUITY_ETFS]
HEDGE_INDEXES  = [ETF_CODES.index(e) for e in HEDGE_ETFS]

# ============ 数据加载 ============
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

def get_monthly_rebalance_dates(dates):
    df = pd.DataFrame({'date': dates})
    df['ym'] = df['date'].dt.to_period('M')
    return df.groupby('ym')['date'].first().values

# ============ LGB 融合（与回测对比.py 一致）============
BASE_EQUITY = {
    '上行': 0.55, '震荡偏强': 0.45, '震荡': 0.35,
    '下行': 0.15, '极寒': 0.05
}

def strategy_lgb_fusion(dates, lgb_pred):
    """
    LGB 融合策略（与回测对比.py 完全一致）
    权益部分：LGB 预测 softmax
    避险部分：风险平价
    """
    if lgb_pred is None:
        print("    ⚠️ LGB 预测为空，回退为等权")
        return None

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
        if pd.isna(st):
            st = '震荡'

        target_equity = BASE_EQUITY.get(st, 0.35)
        avail_mask = mask_available(loc)
        eq_active = [i for i in EQUITY_INDEXES if avail_mask[i]]
        hedge_active = [i for i in HEDGE_INDEXES if avail_mask[i]]
        w = np.zeros(7)

        # 权益部分：LGB 预测加权 softmax
        if eq_active and target_equity > 0:
            eq_etfs = [ETF_CODES[i] for i in eq_active]
            lgb_loc = lgb_pred.index.get_indexer([d], method='pad')[0]
            if lgb_loc < 0:
                # 预测文件没有这一天 → 等权
                sub_w = np.ones(len(eq_active)) / len(eq_active)
            else:
                lgb_window = lgb_pred.index[max(0, lgb_loc-21):lgb_loc+1]
                eq_preds = np.zeros(len(eq_etfs))
                for j, e in enumerate(eq_etfs):
                    if e in lgb_pred.columns:
                        vals = lgb_pred[e].loc[lgb_window.intersection(lgb_pred.index)].dropna()
                        eq_preds[j] = vals.mean() if len(vals) > 0 else 0
                # softmax
                std = np.std(eq_preds)
                if std < 1e-10:
                    sub_w = np.ones(len(eq_active)) / len(eq_active)
                else:
                    x = eq_preds / std / 0.5
                    exp_x = np.exp(x - np.max(x))
                    sub_w = exp_x / exp_x.sum()
            for j, idx in enumerate(eq_active):
                w[idx] = sub_w[j] * target_equity

        # 避险部分：风险平价
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
        if s > 0:
            w = w / s
        else:
            w[avail_mask] = 1.0 / avail_mask.sum()

        weights[pd.Timestamp(d)] = w

    return pd.DataFrame(weights, index=ETF_CODES).T

# ============ 回测引擎（与回测对比.py 完全一致）============
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
        '索提诺比率': round(sortino, 4),
        '最大回撤': round(max_dd, 4),
        '卡玛比率': round(calmar, 4),
        '月均换手率': round(monthly_turnover, 4),
        '胜率': round((ret > 0).mean(), 4),
    }

# ============ 主流程 ============
if __name__ == "__main__":
    dates_arr = pd.DatetimeIndex(common_dates)
    returns_mat = returns.loc[common_dates]

    results = []
    all_navs = {}

    # ===== 版本 A：旧预测文件 =====
    print(f"\n{'='*70}")
    print(f"[版本A] 旧预测文件（Streamlit 平台输出）")
    print(f"  路径: {LGB_OLD_PATH}")
    print(f"{'='*70}")

    if os.path.exists(LGB_OLD_PATH):
        lgb_old = pd.read_csv(LGB_OLD_PATH, encoding='utf-8-sig', parse_dates=['date']).set_index('date')
        print(f"  已加载: {len(lgb_old)} 天")
        print(f"  日期范围: {lgb_old.index.min().date()} ~ {lgb_old.index.max().date()}")
        # 统计非 NaN 覆盖
        non_nan = lgb_old.notna().sum()
        print(f"  各 ETF 非 NaN 天数: {non_nan.to_dict()}")

        w_old = strategy_lgb_fusion(common_dates, lgb_old)
        if w_old is not None:
            nav_a, ret_a, _ = run_backtest(w_old, dates_arr, returns_mat)
            m_a = calc_metrics("LGB旧版", ret_a, w_old, nav_a)
            results.append(m_a)
            all_navs["LGB旧版"] = nav_a
            print(f"  结果: 夏普={m_a['夏普比率']:.4f}, 年化={m_a['年化收益']*100:.2f}%, MDD={m_a['最大回撤']*100:.2f}%")
    else:
        print(f"  ⚠️ 旧预测文件不存在，跳过")

    # ===== 版本 B：新预测文件 =====
    print(f"\n{'='*70}")
    print(f"[版本B] 新预测文件（修复版 lgb.py 输出）")
    print(f"  路径: {LGB_NEW_PATH}")
    print(f"{'='*70}")

    if os.path.exists(LGB_NEW_PATH):
        lgb_new = pd.read_csv(LGB_NEW_PATH, encoding='utf-8-sig', parse_dates=['date']).set_index('date')
        print(f"  已加载: {len(lgb_new)} 天")
        print(f"  日期范围: {lgb_new.index.min().date()} ~ {lgb_new.index.max().date()}")
        non_nan = lgb_new.notna().sum()
        print(f"  各 ETF 非 NaN 天数: {non_nan.to_dict()}")

        w_new = strategy_lgb_fusion(common_dates, lgb_new)
        if w_new is not None:
            nav_b, ret_b, _ = run_backtest(w_new, dates_arr, returns_mat)
            m_b = calc_metrics("LGB新版", ret_b, w_new, nav_b)
            results.append(m_b)
            all_navs["LGB新版"] = nav_b
            print(f"  结果: 夏普={m_b['夏普比率']:.4f}, 年化={m_b['年化收益']*100:.2f}%, MDD={m_b['最大回撤']*100:.2f}%")
    else:
        print(f"  ⚠️ 新预测文件不存在，跳过")
        print(f"  提示：请先运行修复版 lgb.py 生成预测文件")

    # ===== 输出对比 =====
    print(f"\n{'='*100}")
    print("===== 对比结果 =====")
    print(f"{'='*100}")

    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))

        # 加参照（从前面的 9 策略回测结果读取）
        print(f"\n参照（来自 9 策略全时段回测）:")
        print(f"  风险平价    夏普=0.546, 年化=12.52%, MDD=-21.62%")
        print(f"  动态评分    夏普=0.504, 年化=12.37%, MDD=-16.20%")
        print(f"  PPO永久     夏普=0.473, 年化=14.36%, MDD=-36.32%")
        print(f"  SASF        夏普=0.516, 年化=14.07%, MDD=-25.46%")

        df.to_csv(f"{OUT}/lgb_verify_metrics.csv", index=False, encoding='utf-8-sig')

    if all_navs:
        nav_df = pd.DataFrame(all_navs)
        nav_df.index = common_dates
        nav_df.to_csv(f"{OUT}/lgb_verify_nav.csv", encoding='utf-8-sig')

    print(f"\n✅ 结果保存至 {OUT}/lgb_verify_*.csv")