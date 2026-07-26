import pandas as pd
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "platform"))
import numpy as np
from portfolio_env import PortfolioEnv

# 1、加载数据集 保留原始完整df（不要删除date列）
feat_full_df = pd.read_csv("../data/clean/train_total_feature.csv", parse_dates=["date"], encoding="gbk")
price_df = pd.read_csv("../data/clean/etf_price_clean.csv", parse_dates=["date"], encoding="gbk")

# 给Env用的纯数值特征副本，单独删除date/文本列，不破坏原始完整表
feat_env_df = feat_full_df.drop(columns=["date"], errors="ignore")

# 截取统一测试区间
test_start = 0
test_end = 500
env = PortfolioEnv(feat_env_df, price_df, test_start, test_end)

# 2、采集Env打分，日期从完整原始表取，保证对齐
env_records = []
for idx in range(test_start, test_end):
    env.calc_daily_market_score(idx)
    env_records.append({
        "date": feat_full_df.loc[idx, "date"],
        "env_val": env.val_score * 25,
        "env_macro": env.macro_score * 25,
        "env_sent": env.sent_score * 20,
        "env_trend": env.trend_score * 30,
        "env_total": (env.val_score * 25) + (env.macro_score * 25) + (env.sent_score * 20) + (env.trend_score * 30)
    })
env_df = pd.DataFrame(env_records)
env_df["date"] = pd.to_datetime(env_df["date"])

# 3、读取标准打分表
std_score_df = pd.read_csv("../data/clean/market_score_daily.csv", parse_dates=["date"], encoding="gbk")

# 4、合并两张表，inner交集，打印匹配行数排查是否为空
compare_df = pd.merge(std_score_df, env_df, on="date", how="inner")
print(f"合并后有效匹配样本行数：{len(compare_df)}")
if len(compare_df) == 0:
    raise Exception("日期无匹配数据，两张打分表日期不重合！")

# 计算分项差值
compare_df["val_diff"] = compare_df["val_score"] - compare_df["env_val"]
compare_df["macro_diff"] = compare_df["macro_score"] - compare_df["env_macro"]
compare_df["sent_diff"] = compare_df["sent_score"] - compare_df["env_sent"]
compare_df["trend_diff"] = compare_df["trend_score"] - compare_df["env_trend"]
compare_df["total_diff"] = compare_df["total_score"] - compare_df["env_total"]
print("各列空值数量：")
print(compare_df[["val_score","env_val","macro_score","env_macro","sent_score","env_sent","total_score","env_total"]].isna().sum())

# 过滤任意分项空值行
valid_compare = compare_df.dropna(
    subset=["val_score","env_val","macro_score","env_macro","sent_score","env_sent","total_score","env_total"]
)
print(f"剔除空值后有效统计行数：{len(valid_compare)}")

# 5、统计指标
print("========== 打分交叉验证统计 ==========")
corr = valid_compare["total_score"].corr(valid_compare["env_total"])

print(f"总分皮尔逊相关系数：{corr:.4f}")
print(f"估值分平均绝对误差：{valid_compare['val_diff'].abs().mean():.2f}")
print(f"宏观分平均绝对误差：{valid_compare['macro_diff'].abs().mean():.2f}")
print(f"情绪分平均绝对误差：{valid_compare['sent_diff'].abs().mean():.2f}")
print(f"趋势分平均绝对误差：{valid_compare['trend_diff'].abs().mean():.2f}")

# 6、输出到clean文件夹，统一路径
out_path = "../data/clean/score_cross_compare.csv"
compare_df.to_csv(out_path, index=False, encoding="gbk")
print(f"\n差异对比文件已输出：{out_path}")
