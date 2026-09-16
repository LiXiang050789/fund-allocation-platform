import akshare as ak
import pandas as pd
import numpy as np
import time
import os
from tqdm import tqdm
import warnings
import sys

warnings.filterwarnings("ignore")

# ===================== 全局配置 =====================
START_DATE = "20080101"
END_DATE = "20260601"
if len(sys.argv) >=3:
    arg_start = sys.argv[1]
    arg_end = sys.argv[2]
    # 简单校验格式YYYYMMDD
    if len(arg_start)==8 and len(arg_end)==8:
        START_DATE = arg_start
        END_DATE = arg_end
        print(f"【脚本接收外部传入时间】START={START_DATE}, END={END_DATE}")
# 指数列表：剔除Au9999（单独现货接口处理）
TARGET_INDEX_LIST = ["000300", "000905", "000688", "000932", "000995", "000012"]
TARGET_ETF_LIST = ["510300", "510500", "588000", "159928", "159995", "518880", "511260"]

# ETF <-> 对应指数映射关系
etf_index_rel = {
    "510300": "000300",    # 沪深300ETF
    "510500": "000905",    # 中证500ETF
    "588000": "000688",    # 科创50ETF
    "159928": "000932",    # 华夏消费ETF
    "159995": "000995",    # 华夏芯片ETF
    "518880": "Au9999",    # 华安黄金ETF
    "511260": "000012"     # 国债ETF
}

DIR_RAW_INDEX = "raw_data/index"
DIR_RAW_ETF = "raw_data/etf"
DIR_RAW_BOND = "raw_data/bond"
os.makedirs(DIR_RAW_INDEX, exist_ok=True)
os.makedirs(DIR_RAW_ETF, exist_ok=True)
os.makedirs(DIR_RAW_BOND, exist_ok=True)


# ===================== 安全爬取封装 =====================
def safe_crawl(func, retry=3, sleep=5, **kwargs):
    for i in range(retry):
        try:
            df = func(**kwargs)
            time.sleep(sleep)
            return df
        except Exception as e:
            print(f"【重试 {i + 1}/{retry}】异常信息：{str(e)}")
            time.sleep(2)
    print("【最终失败】接口返回空DataFrame")
    return pd.DataFrame()


# ===================== 指数日线（新浪源 + 黄金现货补充） =====================
def crawl_index_hist():
    all_df = pd.DataFrame()
    # 正确：指数代码 -> 新浪交易所标识
    sina_map = {
        "000300": "sh000300",
        "000905": "sh000905",
        "000688": "sh000688",
        "000932": "sh000932",
        "000995": "sh000995",
        "000012": "sh000012"
    }
    # 循环爬股票类指数
    for code in tqdm(TARGET_INDEX_LIST, desc="指数日线(新浪)"):
        df = safe_crawl(ak.stock_zh_index_daily, symbol=sina_map[code])
        if df.empty:
            print(f"【跳过】指数 {code} 无数据")
            continue
        df["date"] = pd.to_datetime(df["date"])
        df["index_code"] = code
        df["source"] = "sina_index"
        all_df = pd.concat([all_df, df], ignore_index=True)

    # 单独爬黄金现货Au9999（不用新浪指数接口）
    print("\n正在爬取上金所Au9999现货行情")
    df_gold = safe_crawl(ak.spot_hist_sge, symbol="Au99.99")
    if not df_gold.empty:
        df_gold["date"] = pd.to_datetime(df_gold["date"])
        df_gold["index_code"] = "Au9999"
        df_gold["source"] = "sge_spot"
        all_df = pd.concat([all_df, df_gold], ignore_index=True)

    # 时间过滤
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    all_df = all_df[(all_df["date"] >= start_dt) & (all_df["date"] <= end_dt)]
    all_df.to_csv(os.path.join(DIR_RAW_INDEX, "index_daily.csv"), index=False, encoding="utf-8-sig")
    print(f"指数日线保存完成，有效数据 {len(all_df)} 行\n")
    return all_df


# ===================== ETF日线（腾讯兼容接口，修复前缀+中文列名） =====================
import random
def crawl_etf_hist():
    all_df = pd.DataFrame()
    # 市场前缀映射 51沪 / 159/588深
    prefix_map = {
        "510300": "sh510300",
        "510500": "sh510500",
        "588000": "sh588000",
        "159928": "sz159928",
        "159995": "sz159995",
        "518880": "sh518880",
        "511260": "sh511260"
    }

    def safe_sina_fetch(symbol, retry=3):
        """新浪ETF专用稳定抓取，随机休眠防封禁"""
        for i in range(retry):
            try:
                df = ak.fund_etf_hist_sina(symbol=symbol)
                time.sleep(random.uniform(2.8, 4.5))  # 随机2.8~4.5秒，规避固定频率反爬
                return df
            except Exception as e:
                print(f"重试{i+1} 异常:{str(e)}")
                time.sleep(5)
        print(f"{symbol} 抓取全部失败")
        return pd.DataFrame()

    for code in tqdm(TARGET_ETF_LIST, desc="ETF日线(新浪替代腾讯)"):
        sym = prefix_map[code]
        df = safe_sina_fetch(sym)
        if df.empty:
            print(f"【警告】{code} 无数据，跳过")
            continue
        # 新浪原生字段直接对齐，无需中文转英文
        df.rename(columns={
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        }, inplace=True)
        # 新浪接口无amount成交额，用close*volume近似计算
        df["amount"] = df["close"] * df["volume"]
        df["date"] = pd.to_datetime(df["date"])
        # 时间过滤到2008起
        df = df[df["date"] >= pd.to_datetime(START_DATE)]
        df["etf_code"] = code
        df["source"] = "sina_etf"
        all_df = pd.concat([all_df, df], ignore_index=True)

    all_df.to_csv(os.path.join(DIR_RAW_ETF, "etf_daily.csv"), index=False, encoding="utf-8-sig")
    print(f"ETF日线保存完成，有效数据 {len(all_df)} 行\n")
    return all_df

def crawl_bond_yield():
    print("===== 分段循环爬取国债收益率曲线 =====")
    start_dt_total = pd.to_datetime(START_DATE)
    end_dt_total = pd.to_datetime(END_DATE)
    all_bond_raw = pd.DataFrame()

    # 每次取11个月，无重叠分段，减少请求次数
    current_start = start_dt_total
    while current_start <= end_dt_total:
        current_end = current_start + pd.DateOffset(months=11)
        if current_end > end_dt_total:
            current_end = end_dt_total
        s_str = current_start.strftime("%Y%m%d")
        e_str = current_end.strftime("%Y%m%d")
        print(f"正在爬取区间：{s_str} ~ {e_str}")

        # 带分段日期参数请求
        df_slice = safe_crawl(ak.bond_china_yield, sleep=2, start_date=s_str, end_date=e_str)
        if not df_slice.empty:
            all_bond_raw = pd.concat([all_bond_raw, df_slice], ignore_index=True)
        # 窗口不重叠，直接跳到下一段第一天
        current_start = current_end + pd.DateOffset(days=1)

    if all_bond_raw.empty:
        print("【致命错误】国债接口分段爬取全部失败")
        return pd.DataFrame()

    # 去重（防止接口同一天返回多条）
    all_bond_raw.drop_duplicates(subset=["曲线名称", "日期"], keep="last", inplace=True)
    # 只保留国债曲线
    df_treasury_main = all_bond_raw[all_bond_raw["曲线名称"] == "中债国债收益率曲线"].copy()
    df_treasury_main["date"] = pd.to_datetime(df_treasury_main["日期"])
    df_treasury_main = df_treasury_main[
        (df_treasury_main["date"] >= start_dt_total) & (df_treasury_main["date"] <= end_dt_total)]

    # 提取1年、3年（替代2Y）、10年期
    df_wide = df_treasury_main[["date", "1年", "3年", "10年"]].copy()
    df_wide.rename(
        columns={
            "1年": "yield_1y",
            "3年": "yield_2y",  # 3年期代理2年期
            "10年": "yield_10y"
        },
        inplace=True
    )
    df_wide["source"] = "china_bond_yield"

    # 统计有效数据
    count_1y = df_wide["yield_1y"].notna().sum()
    count_2y = df_wide["yield_2y"].notna().sum()
    count_10y = df_wide["yield_10y"].notna().sum()

    if len(df_wide) > 0:
        # 修复关键点：max必须取.max()，不能直接取整列Series
        min_ts = df_wide["date"].min()
        max_ts = df_wide["date"].max()
        print(f"\n数据时间范围：{min_ts.date()} ~ {max_ts.date()}")
        print(f"有效1年期数据条数：{count_1y}")
        print(f"有效3年期(替代2Y)数据条数：{count_2y}")
        print(f"有效10年期数据条数：{count_10y}")
    print("说明：免费akshare接口无法获取2008–2019完整国债日线，2008–2019区间会大量缺失")

    # 保存宽表
    df_wide.to_csv(os.path.join(DIR_RAW_BOND, "bond_yield_wide.csv"), index=False, encoding="utf-8-sig")

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    print("===== 启动全量数据爬取程序 =====")
    crawl_index_hist()
    crawl_etf_hist()
    crawl_bond_yield()

    print("===== 全部数据爬取完成 =====")
    print("【文件目录说明】")
    print("1. raw_data/index/index_daily.csv      指数+黄金现货日线")
    print("2. raw_data/etf/etf_daily.csv          ETF日线（含volume成交量）")
    print("3. raw_data/bond/bond_yield_wide.csv   国债宽表")
    print("4. raw_data/bond/bond_yield_long.csv   国债长表")
    print("备注：yield_2y使用3年期国债替代标准2年期")