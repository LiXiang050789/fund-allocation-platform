"""
生成 SASF 硬切换与 LGB 融合的月度权重序列
用于 Streamlit 前端展示（读文件秒开）
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{BASE}/data/clean"
RESULTS = f"{BASE}/results"
os.makedirs(RESULTS, exist_ok=True)

ETF_CODES = ['hs300','zz500','kc50','consume','chip','gold','bond10']
EQUITY_INDEXES = [0,1,2,3,4]
HEDGE_INDEXES = [5,6]
RISK_FREE = 0.025
MVO_WINDOW = 60
MAX_WEIGHT = 0.30

BASE_EQUITY = {'上行': 0.55, '震荡偏强': 0.45, '震荡': 0.35, '下行': 0.15, '极寒': 0.05}

# ========== 加载 ==========
price = pd.read_csv(f"{DATA}/etf_price_clean.csv", encoding='utf-8-sig', parse_dates=['date']).set_index('date')
score = pd.read_csv(f"{DATA}/market_score_daily.csv", encoding='utf-8-sig', parse_dates=['date']).set_index('date')
lgb_pred_path = f"{RESULTS}/pred_return_lgb_daily.csv"
ppo_weight_path = f"{RESULTS}/ppo_weight.csv"

prices = price[ETF_CODES].copy()
returns = prices.pct_change().fillna(0)
state = score['market_state']
common = prices.index.intersection(state.index)
prices, returns, state = prices.loc[common], returns.loc[common], state.loc[common]

def get_monthly_dates(dates):
    df = pd.DataFrame({'date': dates})
    df['ym'] = df['date'].dt.to_period('M')
    return df.groupby('ym')['date'].first().values

def solve_risk_parity(cov):
    vols = np.sqrt(np.diag(cov))
    vols = np.where(vols < 1e-10, 1e-10, vols)
    w = 1.0 / vols
    return w / w.sum()

def solve_mvo(mu, cov):
    n = len(mu)
    mu_shrink = 0.6 * mu + 0.4 * np.full_like(mu, mu.mean())
    excess = mu_shrink - RISK_FREE / 252
    try:
        cov_inv = np.linalg.inv(cov + np.eye(n) * 1e-6)
        w = cov_inv @ excess
        w = w / w.sum()
    except np.linalg.LinAlgError:
        w = np.ones(n) / n
    w = np.clip(w, 0, MAX_WEIGHT)
    s = w.sum()
    if s > 1e-12: w = w / s
    return w

def get_avail(loc):
    p = prices.iloc[loc].values
    return np.where(p > 1e-6)[0]

# ========== 动态评分权重 ==========
def gen_dynamic_score():
    monthly = get_monthly_dates(common)
    out = []
    for d in monthly:
        loc = prices.index.get_loc(d)
        st = state.loc[d] if not pd.isna(state.loc[d]) else '震荡'
        eq_center = BASE_EQUITY.get(st, 0.35)
        avail = get_avail(loc)
        eq_act = [i for i in EQUITY_INDEXES if i in avail]
        hd_act = [i for i in HEDGE_INDEXES if i in avail]
        w = np.zeros(7)
        if eq_act and eq_center > 0:
            s0 = max(0, loc - MVO_WINDOW)
            rw = returns.iloc[s0:loc+1].iloc[:, eq_act]
            if rw.shape[1] >= 2:
                sub_w = solve_mvo(rw.mean().values, rw.cov().values)
            else:
                sub_w = np.ones(len(eq_act)) / len(eq_act)
            for j, i in enumerate(eq_act):
                w[i] = sub_w[j] * eq_center
        if hd_act and (1 - eq_center) > 0:
            s0 = max(0, loc - MVO_WINDOW)
            rw = returns.iloc[s0:loc+1].iloc[:, hd_act]
            if rw.shape[1] >= 2:
                sub_w = solve_risk_parity(rw.cov().values)
            else:
                sub_w = np.ones(len(hd_act)) / len(hd_act)
            for j, i in enumerate(hd_act):
                w[i] = sub_w[j] * (1 - eq_center)
        s = w.sum()
        w = w / s if s > 0 else w
        row = {'date': d, 'market_state': st}
        for j, c in enumerate(ETF_CODES):
            row[c] = float(w[j])
        out.append(row)
    return pd.DataFrame(out)

# ========== LGB 融合权重 ==========
def gen_lgb_fusion():
    if not os.path.exists(lgb_pred_path):
        print(f"⚠️ 缺少 LGB 预测文件，跳过")
        return None
    lgb_pred = pd.read_csv(lgb_pred_path, encoding='utf-8-sig', parse_dates=['date']).set_index('date')
    monthly = get_monthly_dates(common)
    out = []
    for d in monthly:
        loc = prices.index.get_loc(d)
        st = state.loc[d] if not pd.isna(state.loc[d]) else '震荡'
        eq_center = BASE_EQUITY.get(st, 0.35)
        avail = get_avail(loc)
        eq_act = [i for i in EQUITY_INDEXES if i in avail]
        hd_act = [i for i in HEDGE_INDEXES if i in avail]
        w = np.zeros(7)
        if eq_act and eq_center > 0:
            eq_etfs = [ETF_CODES[i] for i in eq_act]
            lgb_loc = lgb_pred.index.get_indexer([d], method='pad')[0]
            if lgb_loc >= 0:
                win = lgb_pred.index[max(0, lgb_loc-21):lgb_loc+1]
                preds = np.array([
                    lgb_pred[e].loc[win.intersection(lgb_pred.index)].dropna().mean()
                    if e in lgb_pred.columns else 0.0
                    for e in eq_etfs
                ])
                std = np.std(preds)
                if std < 1e-10:
                    sub_w = np.ones(len(eq_act)) / len(eq_act)
                else:
                    x = preds / std / 0.5
                    ex = np.exp(x - np.max(x))
                    sub_w = ex / ex.sum()
            else:
                sub_w = np.ones(len(eq_act)) / len(eq_act)
            for j, i in enumerate(eq_act):
                w[i] = sub_w[j] * eq_center
        if hd_act and (1 - eq_center) > 0:
            s0 = max(0, loc - MVO_WINDOW)
            rw = returns.iloc[s0:loc+1].iloc[:, hd_act]
            if rw.shape[1] >= 2:
                sub_w = solve_risk_parity(rw.cov().values)
            else:
                sub_w = np.ones(len(hd_act)) / len(hd_act)
            for j, i in enumerate(hd_act):
                w[i] = sub_w[j] * (1 - eq_center)
        s = w.sum()
        w = w / s if s > 0 else w
        row = {'date': d, 'market_state': st}
        for j, c in enumerate(ETF_CODES):
            row[c] = float(w[j])
        out.append(row)
    return pd.DataFrame(out)

# ========== SASF 硬切换权重 ==========
def gen_sasf(dyn_w, ppo_daily):
    monthly = get_monthly_dates(common)
    out = []
    dyn_idx = dyn_w.set_index('date')
    ppo = ppo_daily.copy()
    ppo.index = pd.to_datetime(ppo.index)
    ppo = ppo.reindex(common).ffill().fillna(0)
    # 归一化
    rs = ppo.sum(axis=1)
    ppo = ppo.div(rs.where(rs > 0, 1.0), axis=0)

    for d in monthly:
        st = state.loc[d] if not pd.isna(state.loc[d]) else '震荡'
        if st in ('下行', '极寒'):
            # 动态评分
            if d in dyn_idx.index:
                w = dyn_idx.loc[d, ETF_CODES].values.astype(float)
            else:
                cand = dyn_idx.index[dyn_idx.index <= d]
                w = dyn_idx.loc[cand[-1], ETF_CODES].values.astype(float) if len(cand) > 0 else np.ones(7)/7
            source = 'dynamic_score'
        else:
            # PPO
            if d in ppo.index:
                w = ppo.loc[d, ETF_CODES].values.astype(float)
            else:
                cand = ppo.index[ppo.index <= d]
                w = ppo.loc[cand[-1], ETF_CODES].values.astype(float) if len(cand) > 0 else np.ones(7)/7
            source = 'ppo'
        s = w.sum()
        w = w / s if s > 0 else w
        row = {'date': d, 'market_state': st, 'source': source}
        for j, c in enumerate(ETF_CODES):
            row[c] = float(w[j])
        out.append(row)
    return pd.DataFrame(out)

# ========== 执行 ==========
if __name__ == "__main__":
    print("[1] 生成动态评分权重...")
    dyn_w = gen_dynamic_score()
    dyn_w.to_csv(f"{RESULTS}/dynamic_score_monthly_weights.csv", index=False, encoding='utf-8-sig')
    print(f"    {len(dyn_w)} 个月度权重 -> {RESULTS}/dynamic_score_monthly_weights.csv")

    print("[2] 生成 LGB 融合权重...")
    lgb_w = gen_lgb_fusion()
    if lgb_w is not None:
        lgb_w.to_csv(f"{RESULTS}/lgb_fusion_monthly_weights.csv", index=False, encoding='utf-8-sig')
        print(f"    {len(lgb_w)} 个月度权重")

    print("[3] 生成 SASF 硬切换权重...")
    if os.path.exists(ppo_weight_path):
        ppo_daily = pd.read_csv(ppo_weight_path, parse_dates=['date']).set_index('date')
        sasf_w = gen_sasf(dyn_w, ppo_daily)
        sasf_w.to_csv(f"{RESULTS}/sasf_monthly_weights.csv", index=False, encoding='utf-8-sig')
        print(f"    {len(sasf_w)} 个月度权重")
        # 统计
        n_dyn = (sasf_w['source'] == 'dynamic_score').sum()
        print(f"    动态评分触发 {n_dyn} 次 / PPO 触发 {len(sasf_w) - n_dyn} 次")
    else:
        print(f"    ⚠️ 缺少 PPO 权重文件 {ppo_weight_path}")

    print("\n✅ 全部完成")