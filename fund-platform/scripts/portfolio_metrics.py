import numpy as np
import pandas as pd

def calc_portfolio_metrics(series: pd.Series, annual=252) -> dict:
    ret = series.pct_change().fillna(0)
    total_ret = series.iloc[-1] / series.iloc[0] - 1
    annual_ret = (1 + total_ret) ** (annual / len(series)) - 1

    vol = ret.std() * np.sqrt(annual)
    downside_ret = ret[ret < 0]
    downside_vol = downside_ret.std() * np.sqrt(annual) if len(downside_ret) > 0 else 1e-6

    sharpe = annual_ret / (vol + 1e-8)
    sortino = annual_ret / (downside_vol + 1e-8)
    sharpe = np.clip(sharpe, -30, 100)
    sortino = np.clip(sortino, -30, 100)

    cum_max = series.cummax()
    dd = (series - cum_max) / cum_max
    mdd = dd.min()

    dd_mask = dd < 0
    dur_series = dd_mask.astype(int).groupby((dd_mask != dd_mask.shift()).cumsum()).cumsum()
    max_dd_dur = dur_series.max()

    return {
        "累计收益": round(total_ret, 4),
        "年化收益": round(annual_ret, 4),
        "年化波动": round(vol, 4),
        "夏普比率": round(sharpe, 4),
        "索提诺比率": round(sortino, 4),
        "最大回撤": round(mdd, 4),
        "最大回撤天数": int(max_dd_dur)
    }

def calc_turnover(weight_df: pd.DataFrame) -> float:
    delta = weight_df.diff().abs().sum(axis=1)
    return delta.mean()

def calc_concentration(weight_df: pd.DataFrame) -> float:
    return weight_df.max(axis=1).mean()
