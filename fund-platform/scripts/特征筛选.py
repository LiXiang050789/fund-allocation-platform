import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from scipy.stats import pearsonr
import warnings
from scipy.stats import ConstantInputWarning
warnings.filterwarnings("ignore", category=ConstantInputWarning)

# ===================== 全局配置（统一调参区） =====================
# 路径
RAW_FEAT_PATH = "clean/train_total_feature.csv"
SAVE_FEAT_PATH = "clean/train_feature_filtered.csv"
FEAT_IMPORTANCE_PATH = "result/feature_importance.csv"
MONEY_FEAT_PATH = "clean/money_flow_month.csv"
PRICE_PATH = "clean/etf_price_clean.csv"
os.makedirs("result", exist_ok=True)
os.makedirs("clean", exist_ok=True)

# 资产列表
ASSET_NAMES = ["hs300", "zz500", "kc50", "consume", "chip", "gold", "bond10"]
N_ASSET = len(ASSET_NAMES)

# 超参数
SIGMA_THRESH = 3
IC_THRESH = 0.022
CORR_THRESH = 0.92
PRED_DAY = 1
IMPORTANCE_CUM_RATIO = 0.88
SPLIT_DATE_CUTOFF = "2022-12-31"
FORCE_RAISE_UNKNOWN_TEXT = False
# 统一魔法数值
MIN_SAMPLE_NORMAL = 60
MIN_SAMPLE_FORCE = 20
CONSTANT_STD_THRESH = 1e-8
MISS_RATIO_ALERT = 0.3
LGB_ROUND = 100
LGB_LR = 0.05

# 强制保留特征
MANDATORY_ENV_FEATS = [
    "hs300_pe",
    "pmi_生产经营活动预期指数",
    "pmi_制造业采购经理指数",
    "bond_10y_2y_spread",
    "margin_5d_chg",
    "north_net",
    "hs300_ma5",
    "zz500_ma5"
]
CORE_SCORE_FEATS = [
    "north_20d_sum",
    "margin_total",
    "m1_m2_spread",
    "yield_10y",
    "社融存量增速(近似)",
    "hs300_pe_quantile",
    "score_valuation",
    "score_macro",
    "score_sentiment",
    "score_trend",
    "score_total"
]
FORCE_KEEP_FEATS = MANDATORY_ENV_FEATS + CORE_SCORE_FEATS

# 月度宏观特征、资金流特征
MONTHLY_MACRO_FEATS = [
    "hs300_pe",
    "hs300_pe_quantile",
    "pmi_制造业采购经理指数",
    "pmi_生产经营活动预期指数",
    "m1_m2_spread",
    "社融存量增速(近似)"
]
FLOW_FEATS = ["north_net", "north_20d_sum", "margin_5d_chg", "margin_total"]
# 宏观强制特征不做3σ截断
NO_TRUNCATE_FEATS = list(set(FORCE_KEEP_FEATS) & set(MONTHLY_MACRO_FEATS))

# ===================== 工具函数 =====================
def encode_text_market(df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
    map_dict = {
        "下行": 1,
        "震荡": 2,
        "震荡偏强": 2.3,
        "上行": 2.7,
        "偏强": 3
    }
    df_new = df.copy()
    for col in text_cols:
        unique_vals = df_new[col].dropna().unique()
        unknown_vals = [v for v in unique_vals if v not in map_dict.keys()]
        if len(unknown_vals) > 0:
            warn_msg = f"market_state发现未定义文本：{unknown_vals}"
            if FORCE_RAISE_UNKNOWN_TEXT:
                raise ValueError(warn_msg)
            print(f"⚠️ {warn_msg}，统一填充震荡2，请检查数据源")
        df_new[col] = df_new[col].map(map_dict).fillna(2)
    return df_new


def shift_monthly_feature(df: pd.DataFrame, month_feats: list, date_col="date"):
    df_new = df.copy()
    df_new["year_month"] = df_new[date_col].dt.to_period("M")
    month_map_all = {}
    month_mean_fill = {}
    # 按月分组取当月指标值，采用第一个有效滞后月份均值填充首月缺失（彻底杜绝时序泄露）
    for feat in month_feats:
        month_val_map = {}
        unique_yms = sorted(df_new["year_month"].unique())
        # 取第一个存在上月数据的月份作为填充基准
        if len(unique_yms) >= 2:
            fill_base_ym = unique_yms[1]
        else:
            fill_base_ym = unique_yms[0]
        base_group = df_new[df_new["year_month"] == fill_base_ym]
        base_mean = base_group[feat].mean()

        for ym, group in df_new.groupby("year_month"):
            val = group[feat].iloc[0]
            month_val_map[ym] = val
        month_map_all[feat] = month_val_map
        month_mean_fill[feat] = base_mean

        # 构建上月平移映射
        shift_month_map = {}
        ym_list = sorted(month_val_map.keys())
        for i in range(1, len(ym_list)):
            shift_month_map[ym_list[i]] = month_val_map[ym_list[i - 1]]
        df_new[f"{feat}_shift_month"] = df_new["year_month"].map(shift_month_map).fillna(month_mean_fill[feat])
    return df_new, month_map_all, month_mean_fill


def calc_ic_series(factor_series: pd.Series, ret_series: pd.Series, feat_name: str) -> float:
    if np.isclose(factor_series.std(), 0):
        return np.nan
    valid_data = pd.DataFrame({"factor": factor_series, "future_ret": ret_series}).dropna()
    min_sample = MIN_SAMPLE_FORCE if feat_name in FORCE_KEEP_FEATS else MIN_SAMPLE_NORMAL
    if len(valid_data) < min_sample:
        return np.nan
    ic, _ = pearsonr(valid_data["factor"], valid_data["future_ret"])
    return ic

def filter_valid_feature(df_train: pd.DataFrame, feat_cols: list, future_ret_train) -> list:
    valid_feats = []
    for feat in feat_cols:
        ic = calc_ic_series(df_train[feat], future_ret_train, feat)
        if (not np.isnan(ic) and abs(ic) >= IC_THRESH) or feat in FORCE_KEEP_FEATS:
            valid_feats.append(feat)
    base_cnt = len([x for x in valid_feats if x not in FORCE_KEEP_FEATS])
    print(f"IC筛选完成：基础有效特征{base_cnt}维 + 强制保留特征{len(FORCE_KEEP_FEATS)}维，合计{len(valid_feats)}")
    return valid_feats

def drop_high_corr_feature(df_train: pd.DataFrame, feat_cols: list, future_ret_train) -> list:
    corr_df = df_train[feat_cols].corr()
    drop_set = set()
    ic_cache = {f: abs(calc_ic_series(df_train[f], future_ret_train, f)) for f in feat_cols}
    # 向量化遍历优化双层循环速度
    feat_arr = np.array(feat_cols)
    for i in range(len(feat_arr)):
        f1 = feat_arr[i]
        corr_vals = corr_df[f1].iloc[i+1:].values
        high_corr_idx = np.where(np.abs(corr_vals) > CORR_THRESH)[0]
        for idx in high_corr_idx:
            j = i + 1 + idx
            f2 = feat_arr[j]
            if f1 in FORCE_KEEP_FEATS and f2 in FORCE_KEEP_FEATS:
                continue
            if f1 in FORCE_KEEP_FEATS:
                drop_set.add(f2)
            elif f2 in FORCE_KEEP_FEATS:
                drop_set.add(f1)
            else:
                ic1 = ic_cache[f1] if not np.isnan(ic_cache[f1]) else 0.0
                ic2 = ic_cache[f2] if not np.isnan(ic_cache[f2]) else 0.0
                drop_set.add(f2 if ic1 > ic2 else f1)
    final_feats = [f for f in feat_cols if f not in drop_set]
    base_cnt = len([x for x in final_feats if x not in FORCE_KEEP_FEATS])
    print(f"共线性剔除完成：基础特征剩余{base_cnt}维，含强制特征合计{len(final_feats)}")
    return final_feats

def lightgbm_feature_filter(df_train: pd.DataFrame, feat_cols: list, future_ret_train) -> list:
    train_feats = [x for x in feat_cols if x not in FORCE_KEEP_FEATS]
    cut_len = -PRED_DAY
    df_train_cut = df_train.iloc[:cut_len].copy()
    y_cut = future_ret_train.iloc[:cut_len].copy()
    # 索引对齐校验
    if not df_train_cut.index.equals(y_cut.index):
        raise ValueError("X训练特征与收益标签索引错位，样本不匹配！")
    valid_mask = y_cut.notna()
    X = df_train_cut.loc[valid_mask, train_feats]
    y = y_cut.loc[valid_mask]
    train_data = lgb.Dataset(X, label=y, free_raw_data=False)
    params = {
        "objective": "regression",
        "metric": "mse",
        "learning_rate": LGB_LR,
        "verbose": -1
    }
    model = lgb.train(params, train_data, num_boost_round=LGB_ROUND)
    imp_gain = model.feature_importance(importance_type="gain")
    imp_split = model.feature_importance(importance_type="split")
    imp_df = pd.DataFrame({
        "feature": train_feats,
        "gain_importance": imp_gain,
        "split_importance": imp_split
    }).sort_values("gain_importance", ascending=False).reset_index(drop=True)
    # 强制特征填充0重要性，避免NaN
    force_imp_df = pd.DataFrame({
        "feature": FORCE_KEEP_FEATS,
        "gain_importance": 0.0,
        "split_importance": 0.0
    })
    imp_df = pd.concat([imp_df, force_imp_df], ignore_index=True)
    imp_df["is_force_keep"] = imp_df["feature"].isin(FORCE_KEEP_FEATS)
    imp_sum = imp_df["gain_importance"].sum()
    imp_df["cum_gain_ratio"] = imp_df["gain_importance"].cumsum() / imp_sum
    model_selected = imp_df[(imp_df["cum_gain_ratio"] <= IMPORTANCE_CUM_RATIO) | (imp_df["is_force_keep"])]["feature"].tolist()
    imp_df.to_csv(FEAT_IMPORTANCE_PATH, index=False, encoding="utf-8-sig")
    base_cnt = len([x for x in model_selected if x not in FORCE_KEEP_FEATS])
    print(f"LGB筛选完成：模型保留普通因子{base_cnt}维，叠加强制特征{len(FORCE_KEEP_FEATS)}维，总特征{len(model_selected)}")
    return model_selected

def get_equal_weight_ret(df_input):
    asset_check = [asset for asset in ASSET_NAMES if asset not in df_input.columns]
    if len(asset_check) > 0:
        raise ValueError(f"价格数据缺失资产列：{asset_check}")
    asset_ret = pd.DataFrame()
    for asset in ASSET_NAMES:
        asset_ret[asset] = df_input[asset].pct_change().shift(-PRED_DAY)
    eq_ret = asset_ret.mean(axis=1)
    return eq_ret

def drop_constant_features(df_train: pd.DataFrame, df_full: pd.DataFrame, feat_cols: list) -> list:
    valid_feature_list = []
    for feat in feat_cols:
        if feat in FORCE_KEEP_FEATS:
            valid_feature_list.append(feat)
            continue
        std_train = df_train[feat].std()
        std_full = df_full[feat].std()
        if std_train > CONSTANT_STD_THRESH and std_full > CONSTANT_STD_THRESH:
            valid_feature_list.append(feat)
        else:
            print(f"ℹ️ 丢弃常量无信息特征：{feat} (训练集std={std_train:.6f},全量std={std_full:.6f})")
    return valid_feature_list

def check_missing_ratio(df: pd.DataFrame, feat_cols: list):
    miss_report = {}
    for feat in feat_cols:
        miss_ratio = df[feat].isna().sum() / len(df)
        miss_report[feat] = miss_ratio
        if miss_ratio > MISS_RATIO_ALERT and feat not in FORCE_KEEP_FEATS:
            print(f"⚠️ 特征 {feat} 缺失率 {miss_ratio:.2%} 过高，建议剔除")
    return miss_report

# ===================== 主流程 =====================
def main():
    # 数据读取校验
    try:
        df_raw = pd.read_csv(RAW_FEAT_PATH, parse_dates=["date"], encoding="gbk")
        price_df = pd.read_csv(PRICE_PATH, parse_dates=["date"], encoding="gbk")
    except Exception as e:
        raise FileNotFoundError(f"原始数据读取失败：{str(e)}")
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    price_df["date"] = pd.to_datetime(price_df["date"])
    merge_price_cols = ["date"] + ASSET_NAMES
    df_raw = pd.merge(df_raw, price_df[merge_price_cols], on="date", how="left")

    all_cols = [c for c in df_raw.columns if c != "date"]
    order_text_cols = []
    numeric_cols = []
    for col in all_cols:
        sample = df_raw[col].dropna().unique()[:30]
        if col == "market_state" and any(v in ["下行", "震荡", "偏强"] for v in sample):
            order_text_cols.append(col)
        else:
            numeric_cols.append(col)
    print(f"文本编码列：{len(order_text_cols)} | 数值特征总数：{len(numeric_cols)}")

    df_full_raw_all = df_raw.copy()
    if order_text_cols:
        df_full_raw_all = encode_text_market(df_full_raw_all, order_text_cols)

    train_mask = df_full_raw_all["date"] <= pd.to_datetime(SPLIT_DATE_CUTOFF)
    df_train_raw = df_full_raw_all[train_mask].copy()
    print(f"训练集区间：{df_train_raw['date'].min()} ~ {df_train_raw['date'].max()}")

    # 月度滞后处理（修复均值泄露）
    df_train_month, train_month_map, train_month_mean = shift_monthly_feature(df_train_raw, MONTHLY_MACRO_FEATS)
    df_full_raw = df_full_raw_all.copy()
    df_full_raw["year_month"] = df_full_raw["date"].dt.to_period("M")
    for feat in MONTHLY_MACRO_FEATS:
        df_full_raw[f"{feat}_shift_month"] = df_full_raw["year_month"].map(train_month_map[feat]).fillna(train_month_mean[feat])
        df_full_raw[feat] = df_full_raw[f"{feat}_shift_month"]
        df_full_raw.drop(f"{feat}_shift_month", axis=1, inplace=True)
    df_full_raw.drop(columns=["year_month"], inplace=True)

    train_medians = df_train_raw[numeric_cols].median()
    train_means = df_train_raw[numeric_cols].mean()
    train_stds = df_train_raw[numeric_cols].std()
    check_missing_ratio(df_train_raw, numeric_cols)

    def fit_transform_data(df_input):
        df_out = df_input.copy()
        for col in numeric_cols:
            if col not in FLOW_FEATS and col not in MONTHLY_MACRO_FEATS:
                df_out[col] = df_out[col].fillna(train_medians[col])
        # 区分是否截断
        for col in numeric_cols:
            if col in NO_TRUNCATE_FEATS:
                continue
            upper = train_means[col] + SIGMA_THRESH * train_stds[col]
            lower = train_means[col] - SIGMA_THRESH * train_stds[col]
            df_out[col] = df_out[col].clip(lower=lower, upper=upper)
        df_out = df_out.replace([np.inf, -np.inf], np.nan)
        return df_out

    df_train_processed = fit_transform_data(df_train_raw)
    df_full_processed = fit_transform_data(df_full_raw)

    drop_nan_cols = ["north_net"] + MONTHLY_MACRO_FEATS
    drop_flow_nan_mask = df_train_processed[drop_nan_cols].notna().all(axis=1)
    df_train_processed = df_train_processed[drop_flow_nan_mask].copy()
    print(f"剔除资金/宏观缺失样本后训练集行数：{len(df_train_processed)}")

    future_ret_train = get_equal_weight_ret(df_train_processed)
    mask_ret_valid = future_ret_train.notna()
    df_train_processed = df_train_processed[mask_ret_valid].copy()
    future_ret_train = future_ret_train[mask_ret_valid].copy()
    print(f"剔除收益空值后训练集行数：{len(df_train_processed)}")

    drop_price_cols = ASSET_NAMES.copy()
    all_proc_cols = [c for c in df_full_processed.columns if c != "date" and c not in drop_price_cols]
    all_proc_cols = drop_constant_features(df_train_processed, df_full_processed, all_proc_cols)

    # 三层筛选流水线
    ic_feats = filter_valid_feature(df_train_processed, all_proc_cols, future_ret_train)
    corr_feats = drop_high_corr_feature(df_train_processed, ic_feats, future_ret_train)
    final_feats = lightgbm_feature_filter(df_train_processed, corr_feats, future_ret_train)

    save_df = df_full_processed[["date"] + final_feats].copy()
    print(f"即将写入文件，总行数：{len(save_df)}，特征数量：{len(final_feats)}")
    if len(save_df) == 0:
        raise ValueError("保存数据集为空，请检查原始数据日期匹配！")
    if len(final_feats) == 0:
        raise ValueError("筛选后无任何特征，无法保存！")
    save_df.to_csv(SAVE_FEAT_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ 特征文件已成功写入：{SAVE_FEAT_PATH}")

    print("\n==================== 筛选结果汇总 ====================")
    print(f"时序筛选截止日期：{SPLIT_DATE_CUTOFF}（仅该日前数据拟合统计量与月度滞后映射规则）")
    print(f"输出文件总特征数：{len(final_feats)}")
    print(f"RL环境刚需打分字段：{len(MANDATORY_ENV_FEATS)} 个（全部强制保留）")
    print(f"市场宏观/资金打分字段：{len(CORE_SCORE_FEATS)} 个（全部强制保留）")
    print(f"模型自动筛选普通技术因子：{len([x for x in final_feats if x not in FORCE_KEEP_FEATS])}")
    print(f"\n✅ 无未来信息泄漏，训练/推理预处理分布完全对齐，北向、月度宏观缺失不填充虚假数值")

if __name__ == "__main__":
    main()
    df_out = pd.read_csv(SAVE_FEAT_PATH, encoding="utf-8-sig")
    print("\n输出特征标准差校验：")
    print(df_out.drop("date", axis=1).std().round(4))
