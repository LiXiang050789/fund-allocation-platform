import akshare as ak
import pandas as pd
import numpy as np
import time
import random
import os
from tqdm import tqdm
import warnings
import sys   #新增
warnings.filterwarnings("ignore")

# ===================== 全局配置 =====================
START_DATE = "20080101"
END_DATE = "20260601"

# ============新增：读取命令行传入日期参数============
if len(sys.argv) >=3:
    arg_start = sys.argv[1]
    arg_end = sys.argv[2]
    if len(arg_start)==8 and len(arg_end)==8:
        START_DATE = arg_start
        END_DATE = arg_end
        print(f"【脚本接收外部传入时间】START={START_DATE}, END={END_DATE}")
START_DT = pd.to_datetime(START_DATE)
END_DT = pd.to_datetime(END_DATE)
DIR_MACRO = "raw_data/macro"
os.makedirs(DIR_MACRO, exist_ok=True)

# 目标指数（换手率计算用）
INDEX_LIST = ["000300", "000905", "000688", "000932", "000995", "000012"]
SINA_INDEX_MAP = {
    "000300": "sh000300",
    "000905": "sh000905",
    "000688": "sh000688",
    "000932": "sh000932",
    "000995": "sh000995",
    "000012": "sh000012"
}

# 通用安全重试爬虫
def safe_crawl(func, retry=3, base_sleep=3, **kwargs):
    for i in range(retry):
        try:
            df = func(**kwargs)
            time.sleep(random.uniform(base_sleep, base_sleep + 2))
            return df
        except Exception as e:
            print(f"重试{i+1}, 错误:{str(e)}")
            time.sleep(4)
    print("接口抓取失败，返回空表")
    return pd.DataFrame()

# ===================== 1. PPI 月度 =====================
def crawl_ppi():
    print("\n==== 爬取PPI月度数据 ====")
    df = safe_crawl(ak.macro_china_ppi)
    if df.empty:
        print("PPI无数据")
        return df
    print("PPI原始列名：", list(df.columns))
    # 清洗中文年月：2026年06月份 → 2026-06
    df["ym"] = df["月份"].str.replace("年", "-").str.replace("月份", "")
    df["date"] = pd.to_datetime(df["ym"] + "-01", format="%Y-%m-%d")
    df = df[(df["date"] >= START_DT) & (df["date"] <= END_DT)]
    df_out = df[["date", "当月同比增长", "累计"]].rename(
        columns={"当月同比增长":"ppi_yoy", "累计":"ppi_acc_yoy"}
    )
    df_out.to_csv(f"{DIR_MACRO}/ppi.csv", index=False, encoding="utf-8-sig")
    print(f"PPI保存完成，共{len(df_out)}个月")
    return df_out

# ===================== 2. 北向资金 =====================
def crawl_north_money():
    print("\n==== 爬北向资金 ====")
    df = safe_crawl(ak.stock_hsgt_hist_em, symbol="北向资金")
    if df.empty:
        print("北向资金无数据")
        return df
    print("北向资金原始列名：", list(df.columns))
    # 匹配真实字段
    df["date"] = pd.to_datetime(df["日期"])
    df["north_daily"] = df["当日成交净买额"].astype(float)
    df["north_cum"] = df["历史累计净买额"].astype(float)

    df_out = df[["date", "north_daily", "north_cum"]]
    df_out = df_out[(df_out["date"] >= START_DT) & (df_out["date"] <= END_DT)]
    df_out.to_csv(f"{DIR_MACRO}/north_flow.csv", index=False, encoding="utf-8-sig")
    print(f"北向资金保存，共{len(df_out)}交易日")
    return df_out

# ===================== 7. 各指数换手率 =====================
def crawl_index_turnover():
    print("\n==== 爬指数行情计算换手率 ====")
    all_turn = pd.DataFrame()
    for idx_code in tqdm(INDEX_LIST):
        sym = SINA_INDEX_MAP[idx_code]
        df_idx = safe_crawl(ak.stock_zh_index_daily, symbol=sym)
        if df_idx.empty:
            print(f"{idx_code} 指数无数据，跳过")
            continue
        df_idx["date"] = pd.to_datetime(df_idx["date"])
        df_idx["index_code"] = idx_code
        # 简易换手率近似
        df_idx["turnover"] = df_idx["volume"] / df_idx["close"] / 100000000
        sub = df_idx[["date", "index_code", "turnover"]]
        all_turn = pd.concat([all_turn, sub], ignore_index=True)
    all_turn = all_turn[(all_turn["date"] >= START_DT) & (all_turn["date"] <= END_DT)]
    all_turn.to_csv(f"{DIR_MACRO}/index_turn.csv", index=False, encoding="utf-8-sig")
    print("指数换手率保存完成")
    return all_turn

# ===================== 8. 合并所有宏观指标为一张日频总表 =====================
def merge_all_macro():
    print("\n==== 合并全部宏观指标 ====")
    file_map = {
        "ppi.csv": "df_ppi",
        "north_flow.csv": "df_north",
        "margin.csv": "df_margin",
        "dr007_shibor.csv": "df_rate",
        "market_amt.csv": "df_amt",
        "fund_issue.csv": "df_fund"
    }
    data_dict = {}
    for fname, varname in file_map.items():
        fpath = os.path.join(DIR_MACRO, fname)
        if os.path.exists(fpath):
            data_dict[varname] = pd.read_csv(fpath, parse_dates=["date"])
        else:
            data_dict[varname] = pd.DataFrame(columns=["date"])

    df_amt = data_dict["df_amt"]
    df_north = data_dict["df_north"]
    df_margin = data_dict["df_margin"]
    df_rate = data_dict["df_rate"]
    df_ppi = data_dict["df_ppi"]
    df_fund = data_dict["df_fund"]

    # 以市场成交额交易日为基准
    base = df_amt[["date"]].drop_duplicates().sort_values("date")
    res = base.copy()
    # 日频指标左连接
    for df_merge in [df_north, df_margin, df_rate]:
        res = pd.merge(res, df_merge, on="date", how="left")
    # 月度指标前向填充
    if len(df_ppi) > 0:
        res = pd.merge_asof(res.sort_values("date"), df_ppi.sort_values("date"), on="date", direction="backward")
    if len(df_fund) > 0:
        res = pd.merge_asof(res.sort_values("date"), df_fund.sort_values("date"), on="date", direction="backward")
    res = res.sort_values("date").reset_index(drop=True)
    # 衍生特征：融资成交额占市场总成交比例
    res["margin_ratio"] = res["buy_margin"] / res["total_amt"]
    res.to_csv(f"{DIR_MACRO}/macro_all.csv", index=False, encoding="utf-8-sig")
    print(f"宏观总表生成，总行数：{len(res)}")
    return res

# ===================== 主入口 =====================
if __name__ == "__main__":
    print("===== AKShare宏观资金全指标爬虫启动 =====")
    crawl_ppi()
    crawl_north_money()
    crawl_index_turnover()
    print("\n全部宏观数据爬取+合并完成！")