"""LGB Walk-Forward v2 —— 合并版（在官方 lgb_walkforward.py 基础上，采纳 zt 修复版的有效项）

相对官方版的差异（逐项依据见 项目书修改建议.md）：
  [1] 特征缺失分类填充：价格类列 ffill→0；宏观/估值类列 ffill→中位数（官方版对含缺失行整行 drop）
  [2] 加入 7 个上市标记特征（官方版无）
  [3] Walk-Forward 起点提前：TRAIN_START 2015-06、WF 自 2018（官方版 2020/2021 起）
  [4] 逐 ETF 独立训练，单 ETF 样本不足只跳过该 ETF（官方版同此行为，此处显式化）
  [5] IC 双口径输出：横截面 Spearman（学术标准）+ 时序 Pearson（与官方/项目书一致）
  [6] 内置 LGB-MVO 参考回测：mu(21日) 与 cov(月度) 量纲统一（官方版 cov×12 与月度 mu 不一致）

输出（全部写 /tmp，不覆盖官方文件）：
  /tmp/nhb_lgb_v2/pred_return_lgb_daily_v2.csv
  /tmp/nhb_lgb_v2/lgb_v2_ic_log.csv
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # fund-platform
DATA = f'{BASE}/data/clean'
OUT = '/tmp/nhb_lgb_v2'
os.makedirs(OUT, exist_ok=True)

ETF = ['hs300', 'zz500', 'kc50', 'consume', 'chip', 'gold', 'bond10']
N = len(ETF)
MAX_W = 0.30
RF = 0.025
HORIZON = 21
TRAIN_START = '2015-06-01'
WF_START_YEAR = 2018
LGB_KWARGS = dict(objective='regression', metric='l2', learning_rate=0.02,
                  max_depth=4, num_leaves=15, reg_alpha=2.0, reg_lambda=3.0,
                  n_estimators=150, verbose=-1, n_jobs=-1, min_child_samples=30)

print('[1] 加载数据...')
feat = pd.read_csv(f'{DATA}/train_feature_filtered.csv', encoding='utf-8-sig', parse_dates=['date'])
price = pd.read_csv(f'{DATA}/etf_price_clean.csv', encoding='utf-8-sig', parse_dates=['date'])
feat = feat.set_index('date').sort_index()
price = price.set_index('date').sort_index()
common = feat.index.intersection(price.index)
feat, price = feat.loc[common].copy(), price.loc[common].copy()

print('[2] 特征清洗（分类填充）...')
price_prefixes = ['hs300_', 'zz500_', 'kc50_', 'consume_', 'chip_', 'gold_', 'bond10_']
all_cols = list(feat.columns)
price_cols = [c for c in all_cols if any(c.startswith(p) for p in price_prefixes)]
other_cols = [c for c in all_cols if c not in price_cols]
for c in price_cols:
    feat[c] = feat[c].ffill().fillna(0)
for c in other_cols:
    feat[c] = feat[c].ffill()
    med = feat[c].median()
    feat[c] = feat[c].fillna(med if pd.notna(med) else 0)
for etf in ETF:
    feat[f'{etf}_listed_flag'] = (price[etf] > 1e-6).astype(int)
fc = list(feat.columns)
print(f'    价格类 {len(price_cols)} 列 + 宏观/估值类 {len(other_cols)} 列 + 上市标记 {N} 列 = {len(fc)} 维')

ret21 = price[ETF].pct_change(HORIZON).shift(-HORIZON)

print(f'[3] Walk-Forward（{WF_START_YEAR} 起）...')
pred_daily = pd.DataFrame(np.nan, index=feat.index, columns=ETF)
ic_rows = []
for ty in range(WF_START_YEAR, 2027):
    train_end = pd.Timestamp(f'{ty-1}-12-31')
    test_idx = feat.index[(feat.index >= pd.Timestamp(f'{ty}-01-01')) & (feat.index <= pd.Timestamp(f'{ty}-12-31'))]
    train_idx = feat.index[(feat.index >= TRAIN_START) & (feat.index <= train_end)]
    if len(train_idx) < 250 or len(test_idx) == 0:
        print(f'    {ty}: skip (train={len(train_idx)})')
        ic_rows.append((ty, np.nan, np.nan, 0))
        continue

    for etf in ETF:
        valid_label = ret21[etf].notna()
        t_tr = train_idx.intersection(valid_label[valid_label].index)
        t_te = test_idx.intersection(valid_label[valid_label].index)
        if len(t_te) < 10 or len(t_tr) < 200:
            continue
        X_tr, y_tr = feat.loc[t_tr, fc].values, ret21[etf].loc[t_tr].values
        ok = ~np.isnan(y_tr)
        if ok.sum() < 200:
            continue
        model = lgb.LGBMRegressor(**LGB_KWARGS)
        model.fit(X_tr[ok], y_tr[ok])
        X_te = feat.loc[test_idx, fc].values
        pred_daily.loc[test_idx, etf] = model.predict(X_te)

    # 横截面 Spearman（同日 7 资产排序）
    cs_list = []
    for d in test_idx:
        p, y = pred_daily.loc[d].values, (ret21.loc[d].values if d in ret21.index else np.full(N, np.nan))
        m = ~np.isnan(p) & ~np.isnan(y)
        if m.sum() < 3 or np.std(p[m]) < 1e-10 or np.std(y[m]) < 1e-10:
            continue
        ic, _ = spearmanr(p[m], y[m])
        if np.isfinite(ic):
            cs_list.append(ic)
    # 时序 Pearson（逐 ETF，再对 ETF 平均——与官方口径一致）
    tp_list = []
    for etf in ETF:
        p = pred_daily.loc[test_idx, etf].values
        y = ret21.loc[test_idx, etf].values if etf in ret21.columns else np.full(len(test_idx), np.nan)
        m = ~np.isnan(p) & ~np.isnan(y)
        if m.sum() > 10 and np.std(p[m]) > 1e-12 and np.std(y[m]) > 1e-12:
            tp_list.append(float(np.corrcoef(p[m], y[m])[0, 1]))
    cs = float(np.nanmean(cs_list)) if cs_list else np.nan
    tp = float(np.nanmean(tp_list)) if tp_list else np.nan
    ic_rows.append((ty, cs, tp, len(cs_list)))
    print(f'    {ty}: train={len(train_idx):4d}d test={len(test_idx):3d}d '
          f'横截面IC={cs:.4f} 时序IC={tp:.4f} valid_preds={int(pred_daily.loc[test_idx].notna().sum().sum())}')

ic_df = pd.DataFrame(ic_rows, columns=['year', 'cross_sectional_ic', 'temporal_pearson_ic', 'n_days'])
print('\n    平均：横截面 %.4f | 时序 %.4f' % (ic_df['cross_sectional_ic'].mean(), ic_df['temporal_pearson_ic'].mean()))
ic_df.to_csv(f'{OUT}/lgb_v2_ic_log.csv', index=False, encoding='utf-8-sig')
pred_daily.to_csv(f'{OUT}/pred_return_lgb_daily_v2.csv', encoding='utf-8-sig')
print(f'[4] 已保存预测 → {OUT}/pred_return_lgb_daily_v2.csv')

print('[5] 内置 LGB-MVO 参考回测（量纲统一版）...')
mp = price[ETF].resample('ME').last()
mr = mp.pct_change()
nav, navs, prev_w = 1.0, [1.0], None
for mi in range(len(mp)):
    md = mp.index[mi]
    if mi < 12:
        navs.append(1.0)
        continue
    avail = mp.iloc[mi].notna()
    ae = [e for e in ETF if avail[e]]
    na = len(ae)
    if na < 2:
        navs.append(navs[-1])
        continue
    loc = feat.index.get_indexer([md], method='pad')[0]
    if loc < 0:
        navs.append(navs[-1])
        continue
    pw = feat.index[max(0, loc - 21):loc + 1]
    mu = np.full(na, np.nan)
    for j, e in enumerate(ae):
        vals = pred_daily[e].loc[pw].dropna()
        if len(vals) > 0:
            mu[j] = vals.mean()
    if np.isnan(mu).any():
        cm = np.nanmean(mu)
        mu[np.isnan(mu)] = 0.0 if np.isnan(cm) else cm
    cs0 = max(0, mi - 12)
    rh = mr.iloc[cs0:mi][ae].dropna()
    cov = rh.cov().values if rh.shape[0] >= max(na, 6) else np.eye(na) * 0.04
    try:
        w = np.linalg.solve(cov + np.eye(na) * 1e-6, mu - RF / 12)
        w = np.clip(w, 0, MAX_W)
        w = np.ones(na) / na if w.sum() < 1e-8 else w / w.sum()
    except Exception:
        w = np.ones(na) / na
    fw = np.zeros(N)
    for j, e in enumerate(ae):
        fw[ETF.index(e)] = w[j]
    if mi + 1 < len(mp):
        pr = (mr.iloc[mi + 1][ETF].fillna(0) * fw).sum()
        if prev_w is not None:
            pr -= np.abs(fw - prev_w).sum() / 2 * 0.001
        prev_w = fw.copy()
        nav *= (1 + pr)
        navs.append(nav)
    else:
        navs.append(navs[-1])
mr_log = np.diff(np.log(navs))
ann_ret = float(np.mean(mr_log) * 12)
ann_vol = float(np.std(mr_log) * np.sqrt(12))
sharpe = (ann_ret - RF) / ann_vol if ann_vol > 0 else 0.0
peak = np.maximum.accumulate(navs)
mdd = float(min((np.array(navs) - peak) / peak))
print(f'    LGB-MVO(v2): 年化={ann_ret*100:.2f}% 夏普={sharpe:.4f} 回撤={mdd*100:.2f}%')
