import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from scipy.stats import pearsonr

# ===================== 全局配置（统一调参区） =====================
# 路径
RAW_FEAT_PATH = "data/clean/train_total_feature.csv"
SAVE_FEAT_PATH = "data/clean/train_feature_filtered.csv"
FEAT_IMPORTANCE_PATH = "results/feature_importance.csv"
MONEY_FEAT_PATH = "data/clean/money_flow_month.csv"
PRICE_TABLE_PATH = "data/clean/etf_price_clean.csv"
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
# 脏文本是否直接中断程序，True更稳健，False仅打印警告
FORCE_RAISE_UNKNOWN_TEXT = False

# 强制保留特征（完全匹配原始数据集字段）
# 1. RL PortfolioEnv四维打分运行必需
MANDATORY_ENV_FEATS = [
    "hs300_pe",
    "pmi_生产经营活动预期指数",
    "bond_10y_2y_spread",
    "margin_5d_chg",
    "north_net",
    "hs300_ma5",
    "zz500_ma5"
]
# 2. 市场综合打分核心宏观、资金、分项得分
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

# ===================== 工具函数 =====================
def encode_text_market(df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
    # 扩充完整行情映射，区分强弱档位
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


def calc_ic_series(factor_series: pd.Series, ret_series: pd.Series, feat_name: str) -> float:
    # 提前拦截常量序列，直接返回0，避免pearson常量警告
    if np.isclose(factor_series.std(), 0):
        return 0.0
    valid_data = pd.DataFrame({"factor": factor_series, "future_ret": ret_series}).dropna()
    min_sample = 60 if feat_name not in FORCE_KEEP_FEATS else 20
    if len(valid_data) < min_sample:
        return 0.0
    ic, _ = pearsonr(valid_data["factor"], valid_data["future_ret"])
    return ic


def filter_valid_feature(df_train: pd.DataFrame, feat_cols: list, future_ret_train) -> list:
    valid_feats = []
    for feat in feat_cols:
        ic = calc_ic_series(df_train[feat], future_ret_train, feat)
        if abs(ic) >= IC_THRESH or feat in FORCE_KEEP_FEATS:
            valid_feats.append(feat)
    base_cnt = len([x for x in valid_feats if x not in FORCE_KEEP_FEATS])
    print(f"IC筛选完成：基础有效特征{base_cnt}维 + 强制保留特征{len(FORCE_KEEP_FEATS)}维，合计{len(valid_feats)}")
    return valid_feats

def drop_high_corr_feature(df_train: pd.DataFrame, feat_cols: list, future_ret_train) -> list:
    corr_df = df_train[feat_cols].corr()
    drop_set = set()
    # IC缓存，消除双重循环重复计算
    ic_cache = {f: abs(calc_ic_series(df_train[f], future_ret_train, f)) for f in feat_cols}
    for i in range(len(feat_cols)):
        for j in range(i + 1, len(feat_cols)):
            f1, f2 = feat_cols[i], feat_cols[j]
            if abs(corr_df.loc[f1, f2]) > CORR_THRESH:
                # 双强制特征高相关，直接保留两者
                if f1 in FORCE_KEEP_FEATS and f2 in FORCE_KEEP_FEATS:
                    continue
                if f1 in FORCE_KEEP_FEATS:
                    drop_set.add(f2)
                elif f2 in FORCE_KEEP_FEATS:
                    drop_set.add(f1)
                else:
                    ic1 = ic_cache[f1]
                    ic2 = ic_cache[f2]
                    drop_set.add(f2 if ic1 > ic2 else f1)
    final_feats = [f for f in feat_cols if f not in drop_set]
    base_cnt = len([x for x in final_feats if x not in FORCE_KEEP_FEATS])
    print(f"共线性剔除完成：基础特征剩余{base_cnt}维，含强制特征合计{len(final_feats)}")
    return final_feats

def lightgbm_feature_filter(df_train: pd.DataFrame, feat_cols: list, future_ret_train) -> list:
    train_feats = [x for x in feat_cols if x not in FORCE_KEEP_FEATS]
    X = df_train[train_feats].copy().iloc[:-PRED_DAY]
    y = future_ret_train.copy().iloc[:-PRED_DAY]

    train_data = lgb.Dataset(X, label=y, free_raw_data=False)
    params = {
        "objective": "regression",
        "metric": "mse",
        "learning_rate": 0.05,
        "verbose": -1
    }
    model = lgb.train(params, train_data, num_boost_round=100)

    imp_gain = model.feature_importance(importance_type="gain")
    imp_split = model.feature_importance(importance_type="split")
    imp_df = pd.DataFrame({
        "feature": train_feats,
        "gain_importance": imp_gain,
        "split_importance": imp_split
    }).sort_values("gain_importance", ascending=False).reset_index(drop=True)

    force_imp_df = pd.DataFrame({
        "feature": FORCE_KEEP_FEATS,
        "gain_importance": np.nan,
        "split_importance": np.nan
    })
    imp_df = pd.concat([imp_df, force_imp_df], ignore_index=True)
    imp_df["is_force_keep"] = imp_df["feature"].isin(FORCE_KEEP_FEATS)
    imp_df.to_csv(FEAT_IMPORTANCE_PATH, index=False, encoding="utf-8-sig")

    imp_sum = imp_df[imp_df["gain_importance"].notna()]["gain_importance"].sum()
    imp_df["cum_gain_ratio"] = imp_df["gain_importance"].fillna(0).cumsum() / imp_sum
    model_selected = imp_df[(imp_df["cum_gain_ratio"] <= IMPORTANCE_CUM_RATIO) | (imp_df["is_force_keep"])]["feature"].tolist()

    base_cnt = len([x for x in model_selected if x not in FORCE_KEEP_FEATS])
    print(f"LGB筛选完成：模型保留普通因子{base_cnt}维，叠加强制特征{len(FORCE_KEEP_FEATS)}维，总特征{len(model_selected)}")
    return model_selected

def get_equal_weight_ret(df_input):
    """多资产等权未来收益，不填充NaN，交由下游dropna处理"""
    asset_ret = pd.DataFrame()
    for asset in ASSET_NAMES:
        asset_ret[asset] = df_input[asset].pct_change().shift(-PRED_DAY)
    eq_ret = asset_ret.mean(axis=1)
    return eq_ret
def drop_constant_features(df: pd.DataFrame, feat_cols: list) -> list:
    """剔除方差接近0的常量特征，强制保留字段不删除"""
    valid_feature_list = []
    for feat in feat_cols:
        if feat in FORCE_KEEP_FEATS:
            valid_feature_list.append(feat)
            continue
        feat_std = df[feat].std()
        if feat_std > 1e-8:
            valid_feature_list.append(feat)
        else:
            print(f"ℹ️ 丢弃常量无信息特征：{feat}")
    return valid_feature_list


# ===================== 主流程（Fit-Transform时序安全预处理） =====================
def main():
    # 1. 读取原始数据
    df_raw = pd.read_csv(RAW_FEAT_PATH, parse_dates=["date"], encoding="gbk")
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    price_df = pd.read_csv(PRICE_TABLE_PATH, parse_dates=["date"], encoding="gbk")
    price_df["date"] = pd.to_datetime(price_df["date"])
    merge_price_cols = ["date"] + ASSET_NAMES
    df_raw = pd.merge(df_raw, price_df[merge_price_cols], on="date", how="left")


    # 区分文本/数值列
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

    # 文本编码（映射规则固定，无泄漏）
    df_full_raw = df_raw.copy()
    if order_text_cols:
        df_full_raw = encode_text_market(df_full_raw, order_text_cols)

    # 3. 时序切分：仅切分前数据用于拟合统计量
    train_mask = df_full_raw["date"] <= pd.to_datetime(SPLIT_DATE_CUTOFF)
    df_train_raw = df_full_raw[train_mask].copy()
    df_full_raw_all = df_full_raw.copy()

    # 4. Fit阶段：仅从训练集提取统计量（零未来泄漏核心）
    train_medians = df_train_raw[numeric_cols].median()
    train_means = df_train_raw[numeric_cols].mean()
    train_stds = df_train_raw[numeric_cols].std()

    # 统一预处理转换函数：使用训练集统计量，全局复用
    def fit_transform_data(df_input):
        df_out = df_input.copy()
        # 第一步：中位数填充（仅训练集中位数，无未来数据）
        for col in numeric_cols:
            df_out[col] = df_out[col].fillna(train_medians[col])
        # 第二步：3σ极值截断（边界来自训练集）
        for col in numeric_cols:
            upper = train_means[col] + SIGMA_THRESH * train_stds[col]
            lower = train_means[col] - SIGMA_THRESH * train_stds[col]
            df_out[col] = df_out[col].clip(lower=lower, upper=upper)
        # 清除无穷值，再次兜底填充
        df_out = df_out.replace([np.inf, -np.inf], np.nan)
        for col in numeric_cols:
            df_out[col] = df_out[col].fillna(train_medians[col])
        return df_out

    # 分别转换训练集、全量数据集（规则完全一致）
    df_train_processed = fit_transform_data(df_train_raw)
    df_full_processed = fit_transform_data(df_full_raw_all)

    # 构造筛选用目标收益（无fillna(0)，保留原生NaN）
    future_ret_train = get_equal_weight_ret(df_train_processed)

    # 剔除原始资产价格列，不作为特征
    drop_price_cols = ASSET_NAMES.copy()
    all_proc_cols = [c for c in df_full_processed.columns if c != "date" and c not in drop_price_cols]
    # 前置过滤常量特征，消除Pearson常量警告
    all_proc_cols = drop_constant_features(df_train_processed, all_proc_cols)

    # 三层筛选流水线（全部基于切分前训练集）
    ic_feats = filter_valid_feature(df_train_processed, all_proc_cols, future_ret_train)
    corr_feats = drop_high_corr_feature(df_train_processed, ic_feats, future_ret_train)
    final_feats = lightgbm_feature_filter(df_train_processed, corr_feats, future_ret_train)

    # 输出：全时序统一预处理后的干净数据集（训练/未来分布完全对齐）
    save_df = df_full_processed[["date"] + final_feats].copy()
    # 新增校验打印
    print(f"即将写入文件，总行数：{len(save_df)}，特征数量：{len(final_feats)}")
    if len(save_df) == 0:
        raise ValueError("保存数据集为空，请检查原始数据日期匹配！")
    if len(final_feats) == 0:
        raise ValueError("筛选后无任何特征，无法保存！")
    save_df.to_csv(SAVE_FEAT_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ 特征文件已成功写入：{SAVE_FEAT_PATH}")

    # 汇总打印
    print("\n==================== 筛选结果汇总 ====================")
    print(f"时序筛选截止日期：{SPLIT_DATE_CUTOFF}（仅该日前数据拟合统计量与筛选）")
    print(f"输出文件总特征数：{len(final_feats)}")

    print(f"RL环境刚需打分字段：{len(MANDATORY_ENV_FEATS)} 个（全部强制保留）")
    print(f"市场宏观/资金打分字段：{len(CORE_SCORE_FEATS)} 个（全部强制保留）")
    print(f"模型自动筛选普通技术因子：{len([x for x in final_feats if x not in FORCE_KEEP_FEATS])}")
    print(f"\n✅ 无未来信息泄漏，训练/推理预处理分布完全一致，可直接用于PPO滚动训练")

if __name__ == "__main__":
    main()
    # 校验输出数据分布
    df_out = pd.read_csv(SAVE_FEAT_PATH, encoding="utf-8-sig")
    print("\n输出特征标准差校验：")
    print(df_out.drop("date", axis=1).std())
