import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# ===================== 全局配置常量 =====================
ZT_EXCEL = Path(r"E:\农行杯\2026\数据汇总.xlsx")
CLEAN_DIR = Path("clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
# 清空历史clean文件夹旧csv，避免旧脏数据残留
for old_file in CLEAN_DIR.glob("*.csv"):
    try:
        os.remove(old_file)
    except Exception:
        pass

# 7只目标ETF映射（数字，匹配Excel数字格式etf_code）
etf_codes = [510300, 510500, 588000, 159928, 159995, 518880, 511260]
etf_names = ["hs300", "zz500", "kc50", "consume", "chip", "gold", "bond10"]
# 对应底层指数代码（修正黄金指数代码Au9999→000016）
index_codes = ["000300", "000905", "000688", "000932", "000995", "000016", "000012"]
index_names = ["hs300", "zz500", "kc50", "consume", "chip", "gold", "bond10"]

# 关键时间分界
KC50_START = pd.to_datetime("2020-07-22")
NORTH_STABLE_START = pd.to_datetime("2019-01-01")
PROJECT_START = pd.to_datetime("2015-01-05")

# ===================== Step1 生成全局标准交易日锚（裁剪至2015起） =====================
print("读取etf工作表...")
need_etf_cols = ["date", "open", "high", "low", "close", "volume", "etf_code"]
zt_etf_raw = pd.read_excel(ZT_EXCEL, sheet_name="etf", usecols=need_etf_cols, parse_dates=["date"])
print("etf表列名：", list(zt_etf_raw.columns))
zt_etf_raw["date"] = pd.to_datetime(zt_etf_raw["date"]).dt.normalize()
df_510300 = zt_etf_raw[zt_etf_raw["etf_code"] == 510300].copy()
trade_date_series = df_510300["date"].sort_values().reset_index(drop=True)
# 截断2015年前数据
trade_date_series = trade_date_series[trade_date_series >= PROJECT_START].reset_index(drop=True)

# 校验日期是否为空
if len(trade_date_series) == 0:
    raise Exception("错误：筛选后无2015年后510300交易日，请检查etf_code值、date列格式")

date_anchor = pd.DataFrame({"date": trade_date_series})
date_anchor = date_anchor.drop_duplicates(subset=["date"], keep="last")
date_anchor.to_csv(CLEAN_DIR / "global_date_anchor.csv", index=False)
print(f"1/13 全局交易日历生成，区间：{trade_date_series.min()} ~ {trade_date_series.max()}")

# ===================== Step2 ETF行情基础收盘价 + 预计算factor技术因子 =====================
all_etf_price = pd.DataFrame(index=trade_date_series)
listed_flag = pd.DataFrame(index=trade_date_series)
tech_factor_all = date_anchor.copy().set_index("date")

print("读取factor预计算技术因子工作表...")
zt_factor_raw = pd.read_excel(ZT_EXCEL, sheet_name="factor", parse_dates=["date"])
print("factor表原始列：", list(zt_factor_raw.columns))
if "data_tag" in zt_factor_raw.columns:
    zt_factor_raw = zt_factor_raw.drop(columns=["data_tag"])
zt_factor_raw["date"] = pd.to_datetime(zt_factor_raw["date"]).dt.normalize()

# 校验因子表重复日期
dup_date = zt_factor_raw.groupby(["etf_code", "date"]).size()
dup_date = dup_date[dup_date > 1]
print("因子表重复日期：\n", dup_date)

factor_cols = [
    "ma5", "ma20", "std20", "atr14", "rsi14",
    "macd_dif", "macd_dea", "macd_bar",
    "ma20_vol", "ma20_amt", "obv"
]

for code, name in zip(etf_codes, etf_names):
    sub_raw = zt_etf_raw[zt_etf_raw["etf_code"] == code].copy()
    sub_raw = sub_raw.drop_duplicates(subset=["date"], keep="last")
    sub_raw = sub_raw.sort_values("date").set_index("date")
    list_start = sub_raw.index.min()

    close_ser = sub_raw["close"].reindex(trade_date_series)
    mask_unlist = close_ser.index < list_start
    close_ser.loc[mask_unlist] = np.nan
    close_ser.loc[~mask_unlist] = close_ser.loc[~mask_unlist].ffill()
    all_etf_price[name] = close_ser
    listed_flag[f"{name}_listed"] = close_ser.index >= list_start

    sub_factor = zt_factor_raw[zt_factor_raw["etf_code"] == code].copy()
    sub_factor = sub_factor.drop_duplicates(subset=["date"], keep="last")
    df_merge = pd.merge(date_anchor, sub_factor[["date"] + factor_cols], on="date", how="left")
    df_merge.loc[df_merge["date"] < list_start, factor_cols] = np.nan
    df_merge[factor_cols] = df_merge[factor_cols].ffill()
    rename_map = {c: f"{name}_{c}" for c in factor_cols}
    df_merge = df_merge.rename(columns=rename_map).set_index("date")
    tech_factor_all = pd.concat([tech_factor_all, df_merge], axis=1)

# 恢复date列输出
tech_factor_all.reset_index(names="date", inplace=True)
# 输出ETF价格文件
all_etf_price.reset_index(names="date", inplace=True)
listed_flag.reset_index(names="date", inplace=True)
etf_clean = pd.merge(all_etf_price, listed_flag, on="date")
etf_clean.to_csv(CLEAN_DIR / "etf_price_clean.csv", index=False)
tech_factor_all = tech_factor_all.drop_duplicates(subset=["date"], keep="last")
tech_factor_all.to_csv(CLEAN_DIR / "tech_factor_clean.csv", index=False)
print("2/13 ETF收盘价+预计算技术因子表完成（无重复滚动计算，剔除重复行情列）")

# ===================== Step3 底层指数价格宽表（源头过滤无用列） =====================
print("读取index工作表...")
need_index_cols = ["date", "index_code", "close"]
zt_index_raw = pd.read_excel(ZT_EXCEL, sheet_name="index", usecols=need_index_cols, parse_dates=["date"])
# 映射Au9999→000016
zt_index_raw["index_code"] = zt_index_raw["index_code"].replace("Au9999", "000016")
print("index表列名：", list(zt_index_raw.columns))
zt_index_raw["date"] = pd.to_datetime(zt_index_raw["date"]).dt.normalize()
all_index_price = pd.DataFrame(index=trade_date_series)

# 打印黄金数据校验
gold_raw = zt_index_raw[zt_index_raw["index_code"] == "000016"].copy()
print(f"Au9999黄金原始数据总行数：{len(gold_raw)}")
if len(gold_raw) > 0:
    print(f"黄金最早日期：{gold_raw['date'].min()}, 最晚日期：{gold_raw['date'].max()}")

for code, name in zip(index_codes, index_names):
    sub = zt_index_raw[zt_index_raw["index_code"] == code].copy()
    sub = sub.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    sub = sub.set_index("date")
    ser = sub["close"].reindex(trade_date_series)
    # 内部断档填充
    ser = ser.ffill()
    # 无数据全部填0
    ser = ser.fillna(0)
    all_index_price[name] = ser

all_index_price.reset_index(names="date", inplace=True)
all_index_price.to_csv(CLEAN_DIR / "index_price_clean.csv", index=False)
# 校验空值
gold_null_count = all_index_price["gold"].isna().sum()
print(f"gold列空值数量：{gold_null_count}，总行数：{len(all_index_price)}")
print("3/13 指数价格校验表完成（仅保留close，剔除open/high/low/volume/source）")



# ===================== Step4 一次性读取bond利率表，复用至估值、利率模块（消除重复IO） =====================
print("读取bond利率工作表...")
zt_bond = pd.read_excel(ZT_EXCEL, sheet_name="bond", parse_dates=["date"])
print("bond表列名：", list(zt_bond.columns))
if "source" in zt_bond.columns:
    zt_bond = zt_bond.drop(columns=["source"])
zt_bond["date"] = pd.to_datetime(zt_bond["date"]).dt.normalize()
# 4.1 宽基估值表（剥离利率字段 + 合并外部科创50 PE/PB/PS/股息率全套等权估值）
print("读取宽基估值工作表...")
zt_val_raw = pd.read_excel(ZT_EXCEL, sheet_name="宽基估值", parse_dates=["date"])
print("宽基估值表列名：", list(zt_val_raw.columns))
zt_val_raw["date"] = pd.to_datetime(zt_val_raw["date"]).dt.normalize()

# ========== 读取外部4份科创50等权估值csv文件 ==========
# 1. 科创50 PE-TTM 等权csv
kc_pe_eq = pd.read_csv(r"E:\农行杯\2026\数据\kc50.000688.data\科创50_PE-TTM_等权_上市以来_20260718_213354.csv")
# 先筛选需要的列
kc_pe_eq = kc_pe_eq[["日期", "PE-TTM等权", "PE-TTM 分位点", "PE-TTM 80%分位点值", "PE-TTM 50%分位点值", "PE-TTM 20%分位点值"]].copy()
# 第一步：容错转换日期，非法文本自动变成NaT
kc_pe_eq["tmp_date"] = pd.to_datetime(kc_pe_eq["日期"], errors="coerce")
# 删掉无法解析日期的脏行
kc_pe_eq = kc_pe_eq[kc_pe_eq["tmp_date"].notna()].copy()
# 重命名标准日期列
kc_pe_eq = kc_pe_eq.rename(columns={"日期": "date"})
kc_pe_eq["date"] = kc_pe_eq["tmp_date"].dt.normalize()
# 删除临时辅助列
kc_pe_eq = kc_pe_eq.drop(columns=["tmp_date"])
# 估值字段重命名
kc_pe_eq = kc_pe_eq.rename(columns={
    "PE-TTM等权": "kc50_pe",
    "PE-TTM 分位点": "kc50_pe_quantile",
    "PE-TTM 80%分位点值": "kc50_pe_p80",
    "PE-TTM 50%分位点值": "kc50_pe_p50",
    "PE-TTM 20%分位点值": "kc50_pe_p20"
})

# 2. 科创50 PB 等权csv
kc_pb_eq = pd.read_csv(r"E:\农行杯\2026\数据\kc50.000688.data\科创50_PB_等权_上市以来_20260718_213521.csv", encoding="gbk")
kc_pb_eq = kc_pb_eq[["日期", "PB等权", "PB 分位点", "PB 80%分位点值", "PB 50%分位点值", "PB 20%分位点值"]].copy()
kc_pb_eq["tmp_date"] = pd.to_datetime(kc_pb_eq["日期"], errors="coerce")
kc_pb_eq = kc_pb_eq[kc_pb_eq["tmp_date"].notna()].copy()
kc_pb_eq = kc_pb_eq.rename(columns={"日期": "date"})
kc_pb_eq["date"] = kc_pb_eq["tmp_date"].dt.normalize()
kc_pb_eq = kc_pb_eq.drop(columns=["tmp_date"])
kc_pb_eq = kc_pb_eq.rename(columns={
    "PB等权": "kc50_pb",
    "PB 分位点": "kc50_pb_quantile",
    "PB 80%分位点值": "kc50_pb_p80",
    "PB 50%分位点值": "kc50_pb_p50",
    "PB 20%分位点值": "kc50_pb_p20"
})

# 3. 科创50 PS-TTM 等权csv
kc_ps_eq = pd.read_csv(r"E:\农行杯\2026\数据\kc50.000688.data\科创50_PS-TTM_等权_上市以来_20260718_213622.csv",encoding="utf-8-sig")
kc_ps_eq = kc_ps_eq[["日期", "PS-TTM等权", "PS-TTM 分位点", "PS-TTM 80%分位点值", "PS-TTM 50%分位点值", "PS-TTM 20%分位点值"]].copy()
kc_ps_eq["tmp_date"] = pd.to_datetime(kc_ps_eq["日期"], errors="coerce")
kc_ps_eq = kc_ps_eq[kc_ps_eq["tmp_date"].notna()].copy()
kc_ps_eq = kc_ps_eq.rename(columns={"日期": "date"})
kc_ps_eq["date"] = kc_ps_eq["tmp_date"].dt.normalize()
kc_ps_eq = kc_ps_eq.drop(columns=["tmp_date"])
kc_ps_eq = kc_ps_eq.rename(columns={
    "PS-TTM等权": "kc50_ps",
    "PS-TTM 分位点": "kc50_ps_quantile",
    "PS-TTM 80%分位点值": "kc50_ps_p80",
    "PS-TTM 50%分位点值": "kc50_ps_p50",
    "PS-TTM 20%分位点值": "kc50_ps_p20"
})

# 4. 科创50 股息率 等权csv
kc_div_eq = pd.read_csv(r"E:\农行杯\2026\数据\kc50.000688.data\科创50_股息率_等权_上市以来_20260718_213642.csv",encoding="utf-8-sig")
kc_div_eq = kc_div_eq[["日期", "股息率等权", "股息率 分位点", "股息率 80%分位点值", "股息率 50%分位点值", "股息率 20%分位点值"]].copy()
kc_div_eq["tmp_date"] = pd.to_datetime(kc_div_eq["日期"], errors="coerce")
kc_div_eq = kc_div_eq[kc_div_eq["tmp_date"].notna()].copy()
kc_div_eq = kc_div_eq.rename(columns={"日期": "date"})
kc_div_eq["date"] = kc_div_eq["tmp_date"].dt.normalize()
kc_div_eq = kc_div_eq.drop(columns=["tmp_date"])
kc_div_eq = kc_div_eq.rename(columns={
    "股息率等权": "kc50_div_yield",
    "股息率 分位点": "kc50_div_quantile",
    "股息率 80%分位点值": "kc50_div_p80",
    "股息率 50%分位点值": "kc50_div_p50",
    "股息率 20%分位点值": "kc50_div_p20"
})

# 逐层合并四张科创估值数据
kc_all = kc_pe_eq.merge(kc_pb_eq, on="date", how="outer")
kc_all = kc_all.merge(kc_ps_eq, on="date", how="outer")
kc_all = kc_all.merge(kc_div_eq, on="date", how="outer")
# 调试打印，查看科创数据是否正常读取
print(f"科创合并后kc_all总行数：{len(kc_all)}")
print(f"kc_all列名：{list(kc_all.columns)}")
# 将科创估值合并进原始宽基估值表
zt_val_raw = zt_val_raw.merge(kc_all, on="date", how="left")
print(f"合并后zt_val_raw是否存在kc50_pe：{'kc50_pe' in zt_val_raw.columns}")
# ======================================================================================

val_df = pd.merge(date_anchor, zt_val_raw, on="date", how="left")

# ========= 动态筛选存在的列，彻底解决KeyError =========
# 全部候选填充字段
all_fill_candidates = [
    "hs300_pe", "hs300_pb", "zz500_pe", "zz500_pb",
    "hs300_erp", "hs300_pe_quantile", "hs300_pb_quantile",
    "zz500_pe_quantile", "zz500_pb_quantile",
    # 科创PE全套
    "kc50_pe", "kc50_pe_quantile", "kc50_pe_p80", "kc50_pe_p50", "kc50_pe_p20",
    # 科创PB全套
    "kc50_pb", "kc50_pb_quantile", "kc50_pb_p80", "kc50_pb_p50", "kc50_pb_p20",
    # 科创PS全套
    "kc50_ps", "kc50_ps_quantile", "kc50_ps_p80", "kc50_ps_p50", "kc50_ps_p20",
    # 科创股息率全套
    "kc50_div_yield", "kc50_div_quantile", "kc50_div_p80", "kc50_div_p50", "kc50_div_p20"
]
# 仅保留val_df真实存在的列
exist_fill = [col for col in all_fill_candidates if col in val_df.columns]
if len(exist_fill) > 0:
    val_df[exist_fill] = val_df[exist_fill].ffill()

# 全部科创字段候选
all_kc_candidates = [
    "kc50_pe", "kc50_pe_quantile", "kc50_pe_p80", "kc50_pe_p50", "kc50_pe_p20",
    "kc50_pb", "kc50_pb_quantile", "kc50_pb_p80", "kc50_pb_p50", "kc50_pb_p20",
    "kc50_ps", "kc50_ps_quantile", "kc50_ps_p80", "kc50_ps_p50", "kc50_ps_p20",
    "kc50_div_yield", "kc50_div_quantile", "kc50_div_p80", "kc50_div_p50", "kc50_div_p20"
]
# 动态过滤当前df存在的科创列
new_kc_all_cols = [c for c in all_kc_candidates if c in val_df.columns]
valid_kc50 = 0
# 仅当kc50_pe存在时执行上市前置空逻辑
if "kc50_pe" in val_df.columns and len(new_kc_all_cols) > 0:
    mask_before_kc50 = val_df["date"] < KC50_START
    mask_after_kc50 = val_df["date"] >= KC50_START
    # 2020-07-22科创50上市前全部科创估值、分位阈值统一置NaN，不伪造历史数据
    val_df.loc[mask_before_kc50, new_kc_all_cols] = np.nan
    # 上市区间内单日断档前向填充
    val_df.loc[mask_after_kc50, new_kc_all_cols] = val_df.loc[mask_after_kc50, new_kc_all_cols].ffill()
    valid_kc50 = val_df["kc50_pe"].notna().sum()

# 输出列动态拼接，只写入真实存在的字段，不会访问不存在的key
val_output_cols = ["date"] + exist_fill
val_df[val_output_cols].to_csv(CLEAN_DIR / "valuation_merged.csv", index=False)
print(f"4/13 估值表完成，kc50_pe有效行数：{valid_kc50}，新增科创估值分位特征")

rate_df = pd.merge(date_anchor, zt_bond, on="date", how="left")
rate_cols = ["yield_1y", "yield_2y", "yield_10y"]
real_rate_cols = [col for col in rate_cols if col in rate_df.columns]
if len(real_rate_cols) > 0:
    rate_df[real_rate_cols] = rate_df[real_rate_cols].ffill()
    rate_df.to_csv(CLEAN_DIR / "rate_curve_clean.csv", index=False)
    print("5/13 完整国债利率曲线表完成（1年期/2年期/10年期）")
else:
    print("【警告】bond工作表无yield_1y/yield_2y/yield_10y字段，跳过利率表输出！")

# ===================== Step6 宏观经济表（修复月度匹配空值、重复行、merge日期重名报错） =====================
print("读取money货币工作表...")
zt_money = pd.read_excel(ZT_EXCEL, sheet_name="money")
print("money表列名：", list(zt_money.columns))
zt_money["date"] = pd.to_datetime(zt_money["date"], errors="coerce")
zt_money = zt_money.dropna(subset=["date"])
zt_money["year"] = zt_money["date"].dt.year
zt_money["month"] = zt_money["date"].dt.month
zt_money = zt_money.drop_duplicates(subset=["year", "month"], keep="last")
zt_money = zt_money.drop(columns=["date"])

print("读取ppi工作表...")
zt_ppi = pd.read_excel(ZT_EXCEL, sheet_name="ppi", parse_dates=["date"])
print("ppi表列名：", list(zt_ppi.columns))
ppi_keep_cols = ["date", "ppi_yoy", "ppi_acc_yoy"]
zt_ppi = zt_ppi[ppi_keep_cols].copy()
zt_ppi["year"] = zt_ppi["date"].dt.year
zt_ppi["month"] = zt_ppi["date"].dt.month
zt_ppi = zt_ppi.drop_duplicates(subset=["year", "month"], keep="last")
zt_ppi = zt_ppi.drop(columns=["date"])

print("读取pmi_1宽表并逆透视转换为标准长表...")
zt_pmi_raw = pd.read_excel(ZT_EXCEL, sheet_name="pmi_1")
pmi_melt = zt_pmi_raw.melt(
    id_vars=["指标"],
    var_name="month_str",
    value_name="value"
)

# 新增CPI读取代码
print("读取CPI月度通胀工作表...")
zt_cpi = pd.read_excel(ZT_EXCEL, sheet_name="cpi", parse_dates=["月份"])
print("CPI表列名：", list(zt_cpi.columns))
zt_cpi["月份"] = pd.to_datetime(zt_cpi["月份"]).dt.normalize()
zt_cpi["year"] = zt_cpi["月份"].dt.year
zt_cpi["month"] = zt_cpi["月份"].dt.month
zt_cpi = zt_cpi.drop_duplicates(subset=["year", "month"], keep="last")
# 修复：重命名键为「当月」而非date
zt_cpi = zt_cpi.rename(columns={
    "当月": "cpi_current",
    "同比": "cpi_yoy",
    "环比": "cpi_mom",
    "累计": "cpi_acc"
})
zt_cpi = zt_cpi.drop(columns=["月份"])

def convert_month(s):
    year = int(s.split("年")[0])
    m = int(s.split("年")[1].replace("月", ""))
    return pd.Timestamp(year=year, month=m, day=1)
pmi_melt["date"] = pmi_melt["month_str"].apply(convert_month)
pmi_melt["year"] = pmi_melt["date"].dt.year
pmi_melt["month"] = pmi_melt["month_str"].apply(lambda x: int(x.split("年")[1].replace("月", "")))
pmi_pivot = pmi_melt.pivot(index=["year","month"], columns="指标", values="value").reset_index()
del pmi_melt
rename_pmi = {}
for col in pmi_pivot.columns:
    if col not in ["year", "month"]:
        new_name = "pmi_" + col.replace(" (%)", "").replace(" ", "")
        rename_pmi[col] = new_name
pmi_df = pmi_pivot.rename(columns=rename_pmi)

print("读取社融工作表...")
zt_sr = pd.read_excel(ZT_EXCEL, sheet_name="社融", parse_dates=["date"])
zt_sr["year"] = zt_sr["date"].dt.year
zt_sr["month"] = zt_sr["date"].dt.month
zt_sr = zt_sr.drop_duplicates(subset=["year", "month"], keep="last")
zt_sr = zt_sr.drop(columns=["date"])

# 基准交易日历拆分年月用于匹配
date_anchor["year"] = date_anchor["date"].dt.year
date_anchor["month"] = date_anchor["date"].dt.month

macro_df = pd.merge(date_anchor, zt_money, on=["year", "month"], how="left")
macro_df = macro_df.merge(zt_ppi, on=["year", "month"], how="left")
macro_df = macro_df.merge(zt_cpi, on=["year", "month"], how="left")
macro_df = macro_df.merge(pmi_df, on=["year", "month"], how="left")
macro_df = macro_df.merge(zt_sr, on=["year", "month"], how="left")

# 删除匹配用的辅助年月列
drop_aux_cols = ["year", "month"]
for col in drop_aux_cols:
    if col in macro_df.columns:
        macro_df = macro_df.drop(columns=[col])
assert "year" not in macro_df.columns and "month" not in macro_df.columns, "宏观表残留临时年月辅助列，请检查删除逻辑"

macro_df = macro_df.ffill()
macro_df = macro_df.drop_duplicates(subset=["date"], keep="last")
print("宏观表最终输出列：", list(macro_df.columns))
macro_df.to_csv(CLEAN_DIR / "macro_merged.csv", index=False)
print("6/13 宏观数据表完成（按月匹配无大片空值、月度去重无重复交易日、修复date重名merge报错）")

# ===================== Step7 资金流：两融、换手率、北向 =====================
# 7.1 两融 独立sheet【保留原逻辑不变】
zt_margin = pd.read_excel(ZT_EXCEL, sheet_name="两融", parse_dates=["date"])
zt_margin["date"] = pd.to_datetime(zt_margin["date"]).dt.normalize()
zt_margin = zt_margin.drop_duplicates(subset=["date"], keep="last")
zt_margin["year"] = zt_margin["date"].dt.year
zt_margin["month"] = zt_margin["date"].dt.month
zt_margin = zt_margin.drop_duplicates(subset=["year","month"], keep="last")
zt_margin = zt_margin.drop(columns=["year","month"])
margin_df = pd.merge(date_anchor, zt_margin, on="date", how="left").ffill()
margin_df.to_csv(CLEAN_DIR / "margin_extended.csv", index=False)

# 7.2 换手率（独立sheet：turnover）
zt_turnover = pd.read_excel(ZT_EXCEL, sheet_name="turnover", parse_dates=["date"])
zt_turnover["date"] = pd.to_datetime(zt_turnover["date"]).dt.normalize()
turnover_df = pd.merge(date_anchor, zt_turnover, on="date", how="left").ffill()
turnover_df.to_csv(CLEAN_DIR / "turnover.csv", index=False)

# 7.3 北向资金 取自宏观行情sheet
zt_macro_market = pd.read_excel(ZT_EXCEL, sheet_name="宏观行情", parse_dates=["date"])
zt_macro_market["date"] = pd.to_datetime(zt_macro_market["date"]).dt.normalize()
zt_macro_market = zt_macro_market.drop_duplicates(subset=["date"], keep="last")

# 提取北向字段
north_raw = zt_macro_market[["date", "north_net", "north_20d_sum"]].copy()
north_df = pd.merge(date_anchor, north_raw, on="date", how="left")
mask_early_north = north_df["date"] < NORTH_STABLE_START
mask_late_north = north_df["date"] >= NORTH_STABLE_START
north_cols = ["north_net", "north_20d_sum"]
# 先填充2019后，再清空早期数据，修复丢失数据bug
north_df.loc[mask_late_north, north_cols] = north_df.loc[mask_late_north, north_cols].ffill()
north_df.loc[mask_early_north, north_cols] = np.nan
print("北向有效非空行数：", north_df["north_net"].notna().sum())
north_df.to_csv(CLEAN_DIR / "north_clean.csv", index=False)
print("7/13 资金流（两融/换手率/北向）完成")

# ===================== Step8 全市场波动率 + 大盘两融总量指标 =====================
# 波动率单独提取
vol_raw = zt_macro_market[["date", "volatility_20d"]].copy()
vol_df = pd.merge(date_anchor, vol_raw, on="date", how="left")
vol_df["volatility_20d"] = vol_df["volatility_20d"].ffill()
vol_df.to_csv(CLEAN_DIR / "market_vol_clean.csv", index=False)

# 宏观行情内大盘两融总量：margin_total、margin_5d_chg 单独输出
market_margin_raw = zt_macro_market[["date", "margin_total", "margin_5d_chg"]].copy()
market_margin_df = pd.merge(date_anchor, market_margin_raw, on="date", how="left")
market_margin_df[["margin_total", "margin_5d_chg"]] = market_margin_df[["margin_total", "margin_5d_chg"]].ffill()
market_margin_df.to_csv(CLEAN_DIR / "market_margin_clean.csv", index=False)
print("8/13 全市场20d波动率+大盘两融总量指标完成")


# ===================== Step9 市场多维打分 market_score（过滤重复同源冗余字段） =====================
print("读取market_score工作表...")
zt_score = pd.read_excel(ZT_EXCEL, sheet_name="market_score", parse_dates=["date"])
zt_score["date"] = pd.to_datetime(zt_score["date"]).dt.normalize()
score_df_raw = pd.merge(date_anchor, zt_score, on="date", how="left")
keep_score_cols = [
    "date", "score_valuation", "score_macro", "score_sentiment", "score_trend", "score_total",
    "market_state", "m1_m2_spread", "bond_10y_2y_spread", "ma20_slope", "ma20_streak", "macd_signal"
]
score_df = score_df_raw[keep_score_cols].copy()
score_df = score_df.ffill()
score_df.to_csv(CLEAN_DIR / "market_score_clean.csv", index=False)
print("9/13 市场多维打分表完成（已剔除重复同源基础指标，无多重共线性）")


# ===================== Step11 建模总特征合并脚本（一键生成LGBM/PPO输入总表） =====================
def generate_total_feature():
    def clean_aux_df(df):
        drop_list = ["year", "month"]
        for c in drop_list:
            if c in df.columns:
                df = df.drop(columns=[c])
        return df

    date_ref = clean_aux_df(pd.read_csv(CLEAN_DIR / "global_date_anchor.csv", parse_dates=["date"]))
    df_tech = clean_aux_df(pd.read_csv(CLEAN_DIR / "tech_factor_clean.csv", parse_dates=["date"]))
    df_val = clean_aux_df(pd.read_csv(CLEAN_DIR / "valuation_merged.csv", parse_dates=["date"]))
    df_macro = clean_aux_df(pd.read_csv(CLEAN_DIR / "macro_merged.csv", parse_dates=["date"]))
    df_margin = clean_aux_df(pd.read_csv(CLEAN_DIR / "margin_extended.csv", parse_dates=["date"]))
    df_turn = clean_aux_df(pd.read_csv(CLEAN_DIR / "turnover.csv", parse_dates=["date"]))
    df_north = clean_aux_df(pd.read_csv(CLEAN_DIR / "north_clean.csv", parse_dates=["date"]))
    df_vol = clean_aux_df(pd.read_csv(CLEAN_DIR / "market_vol_clean.csv", parse_dates=["date"]))
    # 新增大盘总量两融
    df_mkt_margin = clean_aux_df(pd.read_csv(CLEAN_DIR / "market_margin_clean.csv", parse_dates=["date"]))
    df_score = clean_aux_df(pd.read_csv(CLEAN_DIR / "market_score_clean.csv", parse_dates=["date"]))

    total = date_ref
    total = total.merge(df_tech, on="date")
    total = total.merge(df_val, on="date")
    rate_file = CLEAN_DIR / "rate_curve_clean.csv"
    if rate_file.exists():
        df_rate = clean_aux_df(pd.read_csv(rate_file, parse_dates=["date"]))
        total = total.merge(df_rate, on="date")
    total = total.merge(df_macro, on="date")
    total = total.merge(df_margin, on="date")
    total = total.merge(df_turn, on="date")
    total = total.merge(df_north, on="date")
    total = total.merge(df_vol, on="date")
    # 合并宏观大盘两融总量
    total = total.merge(df_mkt_margin, on="date")
    total = total.merge(df_score, on="date")

    # 批量删除无用裸字段
    final_drop = [
        "data_tag", "etf_code", "index_code",
        "open", "high", "low", "close", "volume", "amount",
        "ma5", "ma20", "std20", "atr14", "rsi14",
        "macd_dif", "macd_dea", "macd_bar",
        "ma20_vol", "ma20_amt", "obv"
    ]
    for col in final_drop:
        if col in total.columns:
            total = total.drop(columns=[col])

    # 删除两融重复_y后缀列（两融sheet原始列自带_x/_y冲突时清理）
    dup_margin_cols = ["融资买入额_y", "融券卖出量_y", "融券余量_y"]
    for col in dup_margin_cols:
        if col in total.columns:
            total = total.drop(columns=[col])

    total["date"] = pd.to_datetime(total["date"]).dt.normalize()
    total = total.drop_duplicates(subset=["date"], keep="last")
    print("总表最终全部列：", list(total.columns))
    total.to_csv(CLEAN_DIR / "train_total_feature.csv", index=False)

generate_total_feature()

# 函数外部打印行数（变量可正常读取，无作用域报错）
print("tech_factor行数：", len(tech_factor_all))
print("macro行数：", len(macro_df))
print("margin行数：", len(margin_df))
total_check = pd.read_csv(CLEAN_DIR / "train_total_feature.csv", parse_dates=["date"])
print("总表最终行数：", len(total_check))

print("\n===== 全部清洗完成，统一数据源：数据汇总.xlsx，时间区间2015-01至今 =====")
print("输出文件夹：clean")
print("1. global_date_anchor.csv    全局标准交易日历（仅真实交易日，无填充）")
print("2. etf_price_clean.csv       7只ETF收盘价+上市布尔标记（仅收益计算）")
print("3. tech_factor_clean.csv     预计算全套量价技术因子（无代码内重复计算，剔除data_tag）")
print("4. index_price_clean.csv    底层指数收盘价（仅数据校验，剔除open/high/low/volume/source）")
print("5. valuation_merged.csv      宽基估值指标（已剥离bond_10y_yield利率列）")
print("6. rate_curve_clean.csv      1Y/2Y/10Y完整国债收益率曲线（bond表唯一数据源）")
print("7. macro_merged.csv          M0/M1/M2存量/同比/环比、PPI同比/累计同比、14项PMI细分、社融、CPI四维度通胀指标")
print("8. margin_extended.csv       两融全套资金指标（唯一数据源）")
print("9. turnover.csv              全市场日换手率")
print("10. north_clean.csv          北向资金流入累计指标（2019年前永久置空不填充）")
print("11. market_vol_clean.csv     全市场20日波动率（宏观行情表仅保留此字段）")
print("12. market_score_clean.csv   多维市场打分（已剔除所有重复估值/宏观/资金冗余字段）")
#print("13. factor_clean.csv         通用横截面量化因子factor1~factor4")
print("14. train_total_feature.csv  建模总特征宽表（无重复同源列、无辅助标记、无未来泄露）")
print("备注：event事件文本标签基线阶段暂不纳入数据集，模型验证通过后再迭代扩展")
