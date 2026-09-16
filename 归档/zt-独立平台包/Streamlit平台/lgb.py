"""
LightGBM Walk-Forward — 每 ETF 独立模型，日频预测 60 日收益，月度 MVO 回测
"""
import pandas as pd, numpy as np, lightgbm as lgb, os, warnings
warnings.filterwarnings('ignore')

BASE = r"E:/农行杯/2026/1/Streamlit平台"
REPO_ROOT = BASE
ETF = ['hs300','zz500','kc50','consume','chip','gold','bond10']
N = len(ETF); MAX_W = 0.30; RF = 0.025; HORIZON = 21
TRAIN_START = '2020-01-01'   # 足够特征历史后才开始训练
LGB_KWARGS = dict(objective='regression', metric='l2', learning_rate=0.02,
    max_depth=4, num_leaves=15, reg_alpha=2.0, reg_lambda=3.0,
    n_estimators=150, verbose=-1, n_jobs=-1, min_child_samples=30)

print('[1] 加载...')
feat = pd.read_csv(f'{BASE}/data/clean/train_feature_filtered.csv', encoding='utf-8-sig', parse_dates=['date'])
price = pd.read_csv(f'{BASE}/data/clean/etf_price_clean.csv', encoding='utf-8-sig', parse_dates=['date'])
fc = [c for c in feat.columns if c != 'date']
feat = feat.set_index('date').sort_index()
price = price.set_index('date').sort_index()
common = feat.index.intersection(price.index)
feat, price = feat.loc[common], price.loc[common]
ret21 = price[ETF].pct_change(HORIZON).shift(-HORIZON)
print(f'    日频: {len(feat)}天, 特征{len(fc)}, 预测窗口{HORIZON}日')

print('[2] 每 ETF 独立 Walk-Forward...')
pred_daily = pd.DataFrame(np.nan, index=feat.index, columns=ETF)
ic_log = []

for ty in range(2021, 2027):
    train_end = pd.Timestamp(f'{ty-1}-12-31')
    test_s = pd.Timestamp(f'{ty}-01-01')
    test_e = pd.Timestamp(f'{ty}-12-31')
    test_idx = feat.index[(feat.index >= test_s) & (feat.index <= test_e)]
    train_idx = feat.index[feat.index <= train_end]

    if len(train_idx) < 500 or len(test_idx) == 0:
        print(f'    {ty}: skip (train={len(train_idx)}, test={len(test_idx)})')
        continue

    ic_year = []
    for etf in ETF:
        # 取该 ETF 的有效样本
        valid = ret21[etf].notna()
        t_tr = train_idx.intersection(valid[valid].index)
        t_te = test_idx.intersection(valid[valid].index)
        if len(t_tr) < 200 or len(t_te) < 10:
            continue

        X_tr = feat.loc[t_tr, fc].values
        y_tr = ret21[etf].loc[t_tr].values

        # 去掉 NaN 行
        ok_tr = ~np.isnan(y_tr) & ~np.isnan(X_tr).any(axis=1)
        if ok_tr.sum() < 200:
            continue

        model = lgb.LGBMRegressor(**LGB_KWARGS)
        model.fit(X_tr[ok_tr], y_tr[ok_tr])

        X_te = feat.loc[t_te, fc].values
        y_te = ret21[etf].loc[t_te].values
        ok_te = ~np.isnan(X_te).any(axis=1)
        if ok_te.sum() == 0:
            continue

        preds = model.predict(X_te[ok_te])
        pred_daily.loc[t_te[ok_te], etf] = preds

        # IC
        ok_ic = ok_te & ~np.isnan(y_te)
        if ok_ic.sum() > 10:
            ic = np.corrcoef(preds[ok_ic[ok_te]], y_te[ok_ic])[0,1]
            ic_year.append(ic)

    ic = np.nanmean(ic_year) if ic_year else np.nan
    ic_log.append((ty, ic))
    valid_preds = int(pred_daily.loc[test_idx].notna().sum().sum())
    print(f'    {ty}: train={len(train_idx)}d → pred={len(test_idx)}d, valid_preds={valid_preds}, IC={ic:.4f}')

print(f'    平均IC: {np.nanmean([x[1] for x in ic_log]):.4f}')

# 验证截面差异
valid_days = pred_daily.dropna(how='all')
if len(valid_days) > 0:
    cross_std = valid_days.std(axis=1).mean()
    print(f'    日均ETF间标准差: {cross_std:.6f} (截面差异确认)')

print('[3] 聚合日预测 → 月度 MVO 回测...')
mp = price[ETF].resample('ME').last()
mr = mp.pct_change()
nav = 1.0; navs = [1.0]; prev_w = None

for mi in range(len(mp)):
    md = mp.index[mi]
    if mi < 12:
        navs.append(1.0); continue

    avail = mp.iloc[mi].notna()
    ae = [e for e in ETF if avail[e]]
    na = len(ae)
    if na < 2:
        navs.append(navs[-1]); continue

    # 预测: 取月末前21天的日预测均值
    loc = feat.index.get_indexer([md], method='pad')[0]
    pw = feat.index[max(0, loc-21):loc+1]
    mu = np.array([
        pred_daily[e].loc[pw].mean() if e in pred_daily.columns and pred_daily[e].loc[pw].notna().sum() > 0
        else 0 for e in ae
    ])

    # 协方差
    cs = max(0, mi - 12)
    rh = mr.iloc[cs:mi][ae].dropna()
    cov = rh.cov().values * 12 if rh.shape[0] >= na else np.eye(na) * 0.04

    # MVO
    try:
        w = np.linalg.solve(cov + np.eye(na)*1e-6, mu - RF/12)
        w = np.clip(w, 0, MAX_W)
        w = w / (w.sum() + 1e-8)
    except:
        w = np.ones(na) / na

    fw = np.zeros(N)
    for j, e in enumerate(ae):
        fw[ETF.index(e)] = w[j]

    if mi + 1 < len(mp):
        nr = mr.iloc[mi+1][ETF].fillna(0)
        pr = (nr * fw).sum()
        if prev_w is not None:
            turn_cost = np.abs(fw - prev_w).sum() / 2 * 0.001
            pr -= turn_cost
        prev_w = fw.copy()
        nav *= (1 + pr)
        navs.append(nav)
    else:
        navs.append(navs[-1])

# 指标
mr_log = np.diff(np.log(navs))
ann_ret = np.mean(mr_log[24:]) * 12
ann_vol = np.std(mr_log[24:]) * np.sqrt(12)
sharpe = (ann_ret - RF) / ann_vol if ann_vol > 0 else 0
peak = np.maximum.accumulate(navs)
mdd = min((np.array(navs[24:]) - peak[24:]) / peak[24:])
calmar = ann_ret / abs(mdd) if mdd != 0 else 0

print(f'\n    LGB-MVO: 年化={ann_ret*100:.1f}%  夏普={sharpe:.3f}  回撤={mdd*100:.1f}%  卡玛={calmar:.3f}  净值={nav:.3f}')

# 对比
print('\n[4] 基线对比:')
try:
    bm = pd.read_csv(f'{BASE}/results/backtest/baseline_metrics.csv', encoding='utf-8-sig')
    for _, r in bm.iterrows():
        print(f'    {r["策略"]:6s}  年化={float(r["年化收益"])*100:5.1f}%  回撤={float(r["最大回撤"])*100:5.1f}%  夏普={float(r["夏普比率"]):.3f}  卡玛={float(r["卡玛比率"]):.3f}')
except: pass
print(f'    {"LGB":6s}  年化={ann_ret*100:5.1f}%  回撤={mdd*100:5.1f}%  夏普={sharpe:.3f}  卡玛={calmar:.3f}  ← 独立模型')

pred_daily.to_csv(f'{BASE}/data/results/pred_return_lgb_daily.csv', encoding='utf-8-sig')
print(f'\n✅ 60日预测 → results/pred_return_lgb_daily.csv')