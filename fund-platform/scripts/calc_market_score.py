"""
市场环境评分系统 — 完全对标 sf《评分系统金融规则.docx》
输入: data/clean/train_total_feature.csv (182列)
输出: data/clean/market_score_daily.csv (date + 4维得分 + 总分 + 5状态标签)
"""
import pandas as pd, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 四维打分函数（逐行 copy sf docx 规则，仅把变量名改为英文以适配代码）
# ============================================================

def calc_val_score(pe_pct):
    """估值维度 25分 — 沪深300 PE 5年分位"""
    if pe_pct <= 0.30:
        return 25.0
    elif pe_pct >= 0.70:
        return 0.0
    else:
        return 25.0 * (1.0 - (pe_pct - 0.30) / 0.40)

def calc_macro_score(pmi, spread_10y2y):
    """宏观维度 25分 — PMI(15分) + 10Y-2Y利差(10分)"""
    if pmi >= 52:
        pmi_score = 15.0
    elif pmi <= 48:
        pmi_score = 0.0
    else:
        pmi_score = 15.0 * (pmi - 48.0) / 4.0

    if spread_10y2y >= 1.0:
        spread_score = 10.0
    elif spread_10y2y <= -0.5:
        spread_score = 0.0
    else:
        spread_score = 10.0 * (spread_10y2y + 0.5) / 1.5

    return pmi_score + spread_score

def calc_sent_score(margin_change, north_flow):
    """情绪维度 20分 — 两融月变化率(10分) + 北向月净流入(10分)"""
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

    return margin_score + north_score

def calc_trend_score(price, ma20):
    """趋势维度 30分 — hs300 收盘 vs MA20 偏离度"""
    deviation = (price / ma20 - 1.0) * 100
    if deviation >= 5.0:
        return 30.0
    elif deviation <= -5.0:
        return 0.0
    else:
        return 30.0 * (deviation + 5.0) / 10.0

def map_state(total_score):
    """总分 → 5状态标签"""
    if total_score >= 75:
        return '上行'
    elif total_score >= 60:
        return '震荡偏强'
    elif total_score >= 45:
        return '震荡'
    elif total_score >= 30:
        return '下行'
    else:
        return '极寒'

# ============================================================
# 主流程
# ============================================================
def main():
    print('[1] 加载数据...')
    df = pd.read_csv(f'{BASE}/data/clean/train_total_feature.csv', encoding='gbk')
    df = df.rename(columns={df.columns[0]: 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    n = len(df)
    print(f'    总表: {n}行 × {df.shape[1]}列, 日期 {df.date.min().date()} ~ {df.date.max().date()}')

    # 计算 hs300 MA20（从 etf_price_clean 取，确保无未来泄露——当日MA20用当日及之前19天）
    print('[2] 构造 hs300 MA20...')
    price = pd.read_csv(f'{BASE}/data/clean/etf_price_clean.csv', encoding='utf-8-sig', parse_dates=['date'])
    price = price.set_index('date').sort_index()
    hs300_close = price['hs300'].reindex(df['date']).ffill()
    hs300_ma20 = hs300_close.rolling(20, min_periods=1).mean()

    print('[3] 数据对齐（按 sf docx 发布规则）...')
    # PE分位: 当日可用
    pe_pct = df['hs300_pe_quantile'].values

    # PMI 总指数: sf docx 指的是"制造业采购经理指数"（headline PMI），非子指标
    # 当月最后一天可用 → ffill (月度→日频，已是ffill过的，直接用)
    pmi = df['pmi_制造业采购经理指数'].values

    # 10Y-2Y利差: 次日可用 → shift(1)
    spread = df['bond_10y_2y_spread'].values
    spread = np.roll(spread, 1)
    spread[0] = spread[1]  # 首日补

    # 两融月变化: 次日可用 → shift(1)
    margin_chg = df['margin_5d_chg'].values
    margin_chg = np.roll(margin_chg, 1)
    margin_chg[0] = margin_chg[1]

    # 北向: sf docx 要求"当月累计净流入(亿元)"，north_net 为日度值 → 20日滚动求和近似月度
    # 次日可用 → shift(1)
    north_daily = np.nan_to_num(df['north_net'].values, nan=0.0)
    north_flow = pd.Series(north_daily).rolling(20, min_periods=1).sum().values
    north_flow = np.roll(north_flow, 1)
    north_flow[0] = north_flow[1]

    # 价格/MA20: 当日可用
    price_arr = hs300_close.values
    ma20_arr  = hs300_ma20.values

    print('[4] 逐日计算四维得分...')
    val_scores   = np.full(n, np.nan)
    macro_scores = np.full(n, np.nan)
    sent_scores  = np.full(n, np.nan)
    trend_scores = np.full(n, np.nan)

    for i in range(n):
        val_scores[i]   = calc_val_score(pe_pct[i]) if not np.isnan(pe_pct[i]) else 12.5
        macro_scores[i] = calc_macro_score(pmi[i], spread[i])
        sent_scores[i]  = calc_sent_score(margin_chg[i], north_flow[i])
        # 趋势：MA20 可用前用默认15分
        if ma20_arr[i] > 0 and not np.isnan(ma20_arr[i]):
            trend_scores[i] = calc_trend_score(price_arr[i], ma20_arr[i])
        else:
            trend_scores[i] = 15.0

    total_scores = val_scores + macro_scores + sent_scores + trend_scores
    states = [map_state(s) for s in total_scores]

    out = pd.DataFrame({
        'date': df['date'],
        'val_score':   val_scores.round(2),
        'macro_score': macro_scores.round(2),
        'sent_score':  sent_scores.round(2),
        'trend_score': trend_scores.round(2),
        'total_score': total_scores.round(2),
        'market_state': states,
    })

    # 分布统计
    print(f'\n[5] 输出统计 (n={len(out)})')
    from collections import Counter
    dist = Counter(states)
    for st in ['上行','震荡偏强','震荡','下行','极寒']:
        print(f'    {st}: {dist.get(st,0)}天 ({dist.get(st,0)/len(out)*100:.1f}%)')

    out.to_csv(f'{BASE}/data/clean/market_score_daily.csv', index=False, encoding='utf-8-sig')
    print(f'\n✅ 评分表已保存 → data/clean/market_score_daily.csv')

    # 交叉验证
    print('\n[6] 与已有 market_score_clean.csv 交叉验证...')
    try:
        old = pd.read_csv(f'{BASE}/data/clean/market_score_clean.csv', encoding='utf-8-sig', parse_dates=['date'])
        old = old.sort_values('date')
        old_for_merge = old[['date','score_total','market_state']].rename(
            columns={'market_state':'state_old'})
        j = out.merge(old_for_merge, on='date', how='inner')
        corr = j['total_score'].corr(j['score_total'])
        state_map_old = {1.0:'上行', 0.5:'震荡偏强', 0.0:'震荡', -0.5:'下行', -1.0:'极寒'}
        old_mapped = j['state_old'].map(state_map_old)
        agree = (j['market_state'] == old_mapped).mean()
        print(f'    总分相关系数: {corr:.4f}')
        print(f'    状态标签一致率: {agree:.1%}')
        if corr > 0.8 and agree > 0.7:
            print('    ✅ 与已有评分高度一致')
        else:
            print('    ⚠️ 与已有评分存在显著差异，请核对打分规则')
    except Exception as e:
        print(f'    交叉验证跳过: {e}')

if __name__ == '__main__':
    main()
