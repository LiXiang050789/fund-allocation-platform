import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from scipy.stats import pearsonr

# ===================== 路径与超参数（与原代码完全对齐） =====================
RAW_FEAT_PATH = "clean/train_total_feature.csv"
SAVE_FEAT_PATH = "clean/train_feature_filtered.csv"
FEAT_IMPORTANCE_PATH = "result/feature_importance.csv"
os.makedirs("result", exist_ok=True)

ASSET_NAMES = ["hs300", "zz500", "kc50", "consume", "chip", "gold", "bond10"]
SIGMA_THRESH = 3
IC_THRESH = 0.015
CORR_THRESH = 0.92
PRED_DAY = 1
IMPORTANCE_RATIO = 0.95

# ===================== 工具函数 =====================
def encode_text_market(df: pd.DataFrame, text_cols: list):
    """有序编码行情文本：下行=1，震荡=2，偏强=3"""
    map_dict = {"下行": 1, "震荡": 2, "偏强": 3}
    df_new = df.copy()
    for col in text_cols:
        df_new[col] = df_new[col].map(map_dict)
        # 未知文本填充中性震荡2
        df_new[col] = df_new[col].fillna(2)
    return df_new

def sigma_denoise(df: pd.DataFrame, cols: list, sigma: float = 3) -> pd.DataFrame:
    df_clean = df.copy()
    for col in cols:
        # 强制转数字，异常文本转为NaN，彻底杜绝报错
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        mean = df_clean[col].mean()
        std = df_clean[col].std()
        upper = mean + sigma * std
        lower = mean - sigma * std
        df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
    return df_clean

def calc_ic_series(factor_series: pd.Series, ret_series: pd.Series) -> float:
    valid_data = pd.DataFrame({"factor": factor_series, "future_ret": ret_series}).dropna()
    if len(valid_data) < 30:
        return 0.0
    ic, _ = pearsonr(valid_data["factor"], valid_data["future_ret"])
    return ic

def filter_valid_feature(df: pd.DataFrame, feat_cols: list, future_ret) -> list:
    valid_feats = []
    for feat in feat_cols:
        ic = calc_ic_series(df[feat], future_ret)
        if abs(ic) >= IC_THRESH:
            valid_feats.append(feat)
    print(f"IC有效筛选完成：{len(valid_feats)} 维有效特征")
    return valid_feats

def drop_high_corr_feature(df: pd.DataFrame, feat_cols: list, future_ret) -> list:
    corr_df = df[feat_cols].corr()
    drop_set = set()
    for i in range(len(feat_cols)):
        for j in range(i + 1, len(feat_cols)):
            f1, f2 = feat_cols[i], feat_cols[j]
            if abs(corr_df.loc[f1, f2]) > CORR_THRESH:
                ic1 = abs(calc_ic_series(df[f1], future_ret))
                ic2 = abs(calc_ic_series(df[f2], future_ret))
                drop_set.add(f2 if ic1 > ic2 else f1)
    final_feats = [f for f in feat_cols if f not in drop_set]
    print(f"共线性剔除完成，剩余特征：{len(final_feats)} 维")
    return final_feats

# ===================== LightGBM 模型级特征筛选 =====================
def lightgbm_feature_filter(df: pd.DataFrame, feat_cols: list, future_ret) -> list:
    X = df[feat_cols].copy().iloc[:-PRED_DAY]
    y = future_ret.copy().iloc[:-PRED_DAY]

    train_data = lgb.Dataset(X, label=y, free_raw_data=False)
    params = {
        "objective": "regression",
        "metric": "mse",
        "learning_rate": 0.05,
        "verbose": -1
    }
    model = lgb.train(params, train_data, num_boost_round=100)

    imp_df = pd.DataFrame({
        "feature": feat_cols,
        "importance": model.feature_importance(importance_type="gain")
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    imp_df.to_csv(FEAT_IMPORTANCE_PATH, index=False, encoding="utf-8-sig")

    imp_sum = imp_df["importance"].sum()
    imp_df["cum_ratio"] = imp_df["importance"].cumsum() / imp_sum
    final_feats = imp_df[imp_df["cum_ratio"] <= IMPORTANCE_RATIO]["feature"].tolist()

    print(f"LightGBM模型筛选完成，最终保留高贡献特征：{len(final_feats)} 维")
    return final_feats

# ===================== 主流程（修正完整逻辑） =====================
def main():
    # 1. 读取特征数据
    df = pd.read_csv(RAW_FEAT_PATH, parse_dates=["date"], encoding="gbk")
    # 2. 读取价格表，拼接原始资产价格用于构造未来收益标签
    price_df = pd.read_csv("clean/etf_price_clean.csv", parse_dates=["date"], encoding="gbk")
    merge_cols = ["date", "hs300", "zz500", "kc50", "consume", "chip", "gold", "bond10"]
    df = pd.merge(df, price_df[merge_cols], on="date", how="left")

    all_cols = [c for c in df.columns if c != "date"]
    order_text_cols = []
    numeric_cols = []

    # 仅识别market_state为文本列，其余全部数值
    for col in all_cols:
        sample = df[col].dropna().unique()[:30]
        if col == "market_state" and any(v in ["下行", "震荡", "偏强"] for v in sample):
            order_text_cols.append(col)
        else:
            numeric_cols.append(col)
    print(f"有序行情文本列：{len(order_text_cols)} | 原生数值列：{len(numeric_cols)}")

    # 文本编码（修正函数名匹配）
    df_proc = df.copy()
    if order_text_cols:
        df_proc = encode_text_market(df_proc, order_text_cols)

    # 仅数值列执行3σ去噪，编码后的market_state不做极值截断
    df_proc = sigma_denoise(df_proc, numeric_cols, SIGMA_THRESH)

    # 构造未来一期hs300收益标签
    future_ret = df_proc["hs300"].pct_change().shift(-PRED_DAY).fillna(0)

    # 剔除原始价格列，不放入特征集，仅用于计算标签
    drop_price_cols = ["hs300", "zz500", "kc50", "consume", "chip", "gold", "bond10"]
    all_proc_cols = [c for c in df_proc.columns if c != "date" and c not in drop_price_cols]

    # 三层特征筛选流程（补全feat_cols参数）
    ic_feats = filter_valid_feature(df_proc, all_proc_cols, future_ret)
    corr_feats = drop_high_corr_feature(df_proc, ic_feats, future_ret)
    final_feats = lightgbm_feature_filter(df_proc, corr_feats, future_ret)

    # 输出清洗后的特征文件
    save_df = df_proc[["date"] + final_feats].copy()
    save_df.to_csv(SAVE_FEAT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n✅ 特征处理完成，最终保留特征维度：{len(final_feats)}")


if __name__ == "__main__":
    main()
    import pandas as pd
    df = pd.read_csv("clean/train_feature_filtered.csv", encoding="utf-8-sig")
    print(df.drop("date",axis=1).std())
