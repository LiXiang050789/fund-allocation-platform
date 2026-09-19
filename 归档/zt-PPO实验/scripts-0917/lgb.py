"""
LightGBM Walk-Forward — 每 ETF 独立模型，日频预测 21 日收益，月度回测
修复版：
  [Fix1] 特征缺失分类填充（价格类填0，宏观估值类填中位数）
  [Fix2] Walk-Forward 起点提前，训练集覆盖 2015-2017 熊市
  [Fix3] 训练/预测时保留每个 ETF 的可用性，不再因为单一 ETF 缺失而整体 skip
  [Fix4] 加入上市指示特征，帮助模型区分"未上市"和"预测为0"
  [Fix5] MVO 量纲统一（mu 用月度收益，cov 用月度协方差）
  [Fix6] IC 改为横截面 IC（同一天 7 个资产排序相关）
  [Fix7] mu 缺失时用截面均值填充，而非填 0
"""
import pandas as pd, numpy as np, lightgbm as lgb, os, warnings
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

# ============================================================
# 0. 配置
# ============================================================
BASE = r"E:/农行杯/2026/1/Streamlit平台"
DATA = r"E:/农行杯/2026/1/clean"
ETF = ['hs300','zz500','kc50','consume','chip','gold','bond10']
N = len(ETF)
MAX_W = 0.30
RF = 0.025
HORIZON = 21

# [Fix2] 训练起始提前到数据起点后不久
TRAIN_START = '2015-06-01'
WF_START_YEAR = 2018   # Walk-Forward 从 2018 年开始（前 3 年做初始训练）

LGB_KWARGS = dict(
    objective='regression', metric='l2', learning_rate=0.02,
    max_depth=4, num_leaves=15, reg_alpha=2.0, reg_lambda=3.0,
    n_estimators=150, verbose=-1, n_jobs=-1, min_child_samples=30,
)

# ============================================================
# 1. 加载数据
# ============================================================
print('[1] 加载数据...')
feat = pd.read_csv(f'{DATA}/train_feature_filtered.csv',
                   encoding='utf-8-sig', parse_dates=['date'])
price = pd.read_csv(f'{DATA}/etf_price_clean.csv',
                    encoding='utf-8-sig', parse_dates=['date'])

feat = feat.set_index('date').sort_index()
price = price.set_index('date').sort_index()
common = feat.index.intersection(price.index)
feat, price = feat.loc[common].copy(), price.loc[common].copy()

# ============================================================
# 2. [Fix1] 特征缺失分类填充
# ============================================================
print('[2] 特征清洗...')

# 识别价格衍生列（这些列填 0 是合理的，因为未上市时价格就是 0）
price_prefixes = ['hs300_', 'zz500_', 'kc50_', 'consume_', 'chip_', 'gold_', 'bond10_']
feat_cols_all = [c for c in feat.columns if c != 'date']
price_cols = [c for c in feat_cols_all
              if any(c.startswith(p) for p in price_prefixes)]
other_cols = [c for c in feat_cols_all if c not in price_cols]

print(f'    价格类特征: {len(price_cols)} 列')
print(f'    宏观/估值类特征: {len(other_cols)} 列')

# 价格类：ffill 后填 0
for c in price_cols:
    feat[c] = feat[c].ffill().fillna(0)

# 宏观/估值类：ffill 后填中位数（保留分布）
for c in other_cols:
    feat[c] = feat[c].ffill()
    med = feat[c].median()
    if pd.notna(med):
        feat[c] = feat[c].fillna(med)
    else:
        feat[c] = feat[c].fillna(0)

# ============================================================
# 3. [Fix4] 加上市指示特征
# ============================================================
for etf in ETF:
    feat[f'{etf}_listed_flag'] = (price[etf] > 1e-6).astype(int)

fc = [c for c in feat.columns if c != 'date']
print(f'    最终特征数: {len(fc)}')

# ============================================================
# 4. 计算未来收益（label）
# ============================================================
ret21 = price[ETF].pct_change(HORIZON).shift(-HORIZON)
print(f'    日频: {len(feat)}天, 预测窗口{HORIZON}日')

# 诊断：每个 ETF 的有效 label 覆盖
print('\n    [诊断] 各 ETF 的 ret21 非 NaN 覆盖:')
for etf in ETF:
    n_valid = ret21[etf].notna().sum()
    first_valid = ret21[etf].first_valid_index()
    print(f'      {etf:8s}: {n_valid:4d} 天, 起于 {first_valid.date() if first_valid is not None else "N/A"}')

# ============================================================
# 5. Walk-Forward 训练与预测
# ============================================================
print(f'\n[3] Walk-Forward 训练 (起点 {WF_START_YEAR})...')
pred_daily = pd.DataFrame(np.nan, index=feat.index, columns=ETF)
ic_log = []

for ty in range(WF_START_YEAR, 2027):
    train_end = pd.Timestamp(f'{ty-1}-12-31')
    test_s = pd.Timestamp(f'{ty}-01-01')
    test_e = pd.Timestamp(f'{ty}-12-31')

    test_idx = feat.index[(feat.index >= test_s) & (feat.index <= test_e)]
    train_idx = feat.index[(feat.index >= TRAIN_START) & (feat.index <= train_end)]

    if len(train_idx) < 250 or len(test_idx) == 0:
        print(f'    {ty}: skip (train={len(train_idx)}, test={len(test_idx)})')
        ic_log.append((ty, np.nan, 0))
        continue

    # [Fix3] 每个 ETF 独立训练，不因为单一 ETF 缺失而整体 skip
    per_etf_info = []
    for etf in ETF:
        valid_label = ret21[etf].notna()
        t_tr = train_idx.intersection(valid_label[valid_label].index)
        t_te = test_idx.intersection(valid_label[valid_label].index)

        # 该 ETF 在 test 期完全没上市 → 跳过（合理）
        if len(t_te) < 10:
            per_etf_info.append(f'{etf}:no_test')
            continue

        # 该 ETF 在 train 期样本不足 → 跳过（合理，但记录）
        if len(t_tr) < 200:
            per_etf_info.append(f'{etf}:train_short({len(t_tr)})')
            continue

        X_tr = feat.loc[t_tr, fc].values
        y_tr = ret21[etf].loc[t_tr].values

        ok_tr = ~np.isnan(y_tr) & ~np.isnan(X_tr).any(axis=1)
        if ok_tr.sum() < 200:
            per_etf_info.append(f'{etf}:nan_filtered')
            continue

        model = lgb.LGBMRegressor(**LGB_KWARGS)
        model.fit(X_tr[ok_tr], y_tr[ok_tr])

        # 预测 test 期（不是只在 test 期的 label 有效日）
        X_te_full = feat.loc[test_idx, fc].values
        ok_te_full = ~np.isnan(X_te_full).any(axis=1)
        if ok_te_full.sum() > 0:
            preds = model.predict(X_te_full[ok_te_full])
            pred_daily.loc[test_idx[ok_te_full], etf] = preds
            per_etf_info.append(f'{etf}:ok({ok_te_full.sum()})')
        else:
            per_etf_info.append(f'{etf}:no_pred')

    # 汇总统计
    valid_preds = int(pred_daily.loc[test_idx].notna().sum().sum())
    n_active_etfs = pred_daily.loc[test_idx].notna().any(axis=0).sum()

    # [Fix6] 横截面 IC（同一天 7 个 ETF 排序相关）
    test_ic_list = []
    for d in test_idx:
        if d not in pred_daily.index:
            continue
        p = pred_daily.loc[d].values
        y = ret21.loc[d].values if d in ret21.index else np.full(N, np.nan)
        mask = ~np.isnan(p) & ~np.isnan(y)
        if mask.sum() < 3:
            continue
        if np.std(p[mask]) < 1e-10 or np.std(y[mask]) < 1e-10:
            continue
        ic, _ = spearmanr(p[mask], y[mask])
        if np.isfinite(ic):
            test_ic_list.append(ic)

    ic = np.nanmean(test_ic_list) if test_ic_list else np.nan
    ic_log.append((ty, ic, len(test_ic_list)))

    print(f'    {ty}: train={len(train_idx):4d}d → test={len(test_idx):3d}d, '
          f'active_etfs={n_active_etfs}/7, IC={ic:.4f} (n={len(test_ic_list)})')

mean_ic = np.nanmean([x[1] for x in ic_log])
print(f'\n    平均横截面 IC: {mean_ic:.4f}')

# ============================================================
# 6. [Fix5+7] 月度回测
# ============================================================
print('\n[4] 月度回测...')

mp = price[ETF].resample('ME').last()
mr = mp.pct_change()
nav = 1.0
navs = [1.0]
prev_w = None

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

    # [Fix7] mu：缺失时用截面均值，而非填 0
    loc = feat.index.get_indexer([md], method='pad')[0]
    if loc < 0:
        navs.append(navs[-1])
        continue
    pw = feat.index[max(0, loc-21):loc+1]

    mu = np.full(na, np.nan)
    for j, e in enumerate(ae):
        vals = pred_daily[e].loc[pw].dropna()
        if len(vals) > 0:
            mu[j] = vals.mean()
    # 缺失填充
    if np.isnan(mu).any():
        cross_mean = np.nanmean(mu)
        if np.isnan(cross_mean):
            cross_mean = 0.0
        mu[np.isnan(mu)] = cross_mean

    # [Fix5] 协方差：月度协方差，不 ×12
    cs = max(0, mi - 12)
    rh = mr.iloc[cs:mi][ae].dropna()
    if rh.shape[0] >= max(na, 6):
        cov = rh.cov().values
    else:
        cov = np.eye(na) * 0.04

    # MVO 求解（量纲已统一：mu 是 21 日收益，cov 是月度协方差）
    # 21 日 ≈ 1 个月，直接对齐
    try:
        excess = mu - RF / 12
        cov_reg = cov + np.eye(na) * 1e-6
        w = np.linalg.solve(cov_reg, excess)
        # 单资产上限
        w = np.clip(w, 0, MAX_W)
        # 若全为 0（excess 全负），退化为等权
        if w.sum() < 1e-8:
            w = np.ones(na) / na
        else:
            w = w / w.sum()
    except Exception:
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

# ============================================================
# 7. 指标
# ============================================================
mr_log = np.diff(np.log(navs))
# 用全部数据（不再丢弃前 24 个月）
ann_ret = np.mean(mr_log) * 12
ann_vol = np.std(mr_log) * np.sqrt(12)
sharpe = (ann_ret - RF) / ann_vol if ann_vol > 0 else 0
peak = np.maximum.accumulate(navs)
mdd = min((np.array(navs) - peak) / peak)
calmar = ann_ret / abs(mdd) if mdd != 0 else 0

print(f'\n    LGB-MVO: 年化={ann_ret*100:.1f}%  夏普={sharpe:.3f}  '
      f'回撤={mdd*100:.1f}%  卡玛={calmar:.3f}  净值={nav:.3f}')

# ============================================================
# 8. 对比
# ============================================================
print('\n[5] 基线对比:')
try:
    bm = pd.read_csv(f'{BASE}/results/backtest/baseline_metrics.csv',
                     encoding='utf-8-sig')
    for _, r in bm.iterrows():
        print(f'    {r["策略"]:8s}  '
              f'年化={float(r["年化收益"])*100:5.1f}%  '
              f'回撤={float(r["最大回撤"])*100:6.1f}%  '
              f'夏普={float(r["夏普比率"]):.3f}  '
              f'卡玛={float(r["卡玛比率"]):.3f}')
except Exception as e:
    print(f'    [读取 baseline 失败] {e}')

print(f'    {"LGB-MVO":8s}  '
      f'年化={ann_ret*100:5.1f}%  '
      f'回撤={mdd*100:6.1f}%  '
      f'夏普={sharpe:.3f}  '
      f'卡玛={calmar:.3f}  ← 修复版')

# ============================================================
# 9. 保存
# ============================================================
out_path = f'{DATA}/../results/pred_return_lgb_daily_fixed.csv'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
pred_daily.to_csv(out_path, encoding='utf-8-sig')
print(f'\n✅ 预测文件已保存: {out_path}')

# 保存 IC 日志
ic_df = pd.DataFrame(ic_log, columns=['year', 'cross_sectional_ic', 'n_days'])
ic_df.to_csv(f'{DATA}/../results/lgb_ic_log.csv', index=False, encoding='utf-8-sig')
print(f'✅ IC 日志已保存')