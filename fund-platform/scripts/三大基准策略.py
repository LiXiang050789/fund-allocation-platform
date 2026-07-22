import pandas as pd
import numpy as np
from portfolio_metrics import calc_portfolio_metrics

ASSET_NAMES = ["hs300","zz500","kc50","consume","chip","gold","bond10"]

def get_asset_return(price_df):
    """获取7资产日收益率"""
    return price_df[ASSET_NAMES].pct_change().fillna(0)

# 基准1：等权策略
def strategy_equal_weight(price_df):
    ret_df = get_asset_return(price_df)
    w = np.ones(len(ASSET_NAMES)) / len(ASSET_NAMES)
    port_ret = (ret_df * w).sum(axis=1)
    net = (1+port_ret).cumprod()
    return pd.DataFrame({"date":price_df["date"], "net":net})

# 基准2：风险平价策略（极简方差倒数加权）
def strategy_risk_parity(price_df, window=60):
    ret_df = get_asset_return(price_df)
    net_list = []
    w_list = []
    for i in range(len(ret_df)):
        if i < window:
            w = np.ones(len(ASSET_NAMES))/len(ASSET_NAMES)
        else:
            sub = ret_df.iloc[i-window:i]
            vol = sub.std()
            inv_vol = 1/(vol+1e-6)
            w = inv_vol / inv_vol.sum()
        w_list.append(w)
        port_ret = (ret_df.iloc[i] * w).sum()
        if i ==0:
            net_list.append(1+port_ret)
        else:
            net_list.append(net_list[-1]*(1+port_ret))
    weight_df = pd.DataFrame(w_list, columns=ASSET_NAMES)
    res = pd.DataFrame({"date":price_df["date"], "net":net_list})
    return res, weight_df

# 基准3：20日动量策略
def strategy_momentum(price_df, window=20):
    ret_df = get_asset_return(price_df)
    net_list = [1.0]
    for i in range(1, len(ret_df)):
        if i < window:
            w = np.ones(len(ASSET_NAMES))/len(ASSET_NAMES)
        else:
            past_ret = ret_df.iloc[i-window:i].sum()
            rank = past_ret.rank()
            mask = rank == rank.max()
            w = np.zeros(len(ASSET_NAMES))
            w[mask] = 1.0 / mask.sum()
        port_ret = (ret_df.iloc[i] * w).sum()
        net_list.append(net_list[-1]*(1+port_ret))
    res = pd.DataFrame({"date":price_df["date"], "net":net_list})
    return res
