import pandas as pd
import numpy as np
INPUT_FILE  = r"E:\农行杯\2026\1\Streamlit平台\data\clean\train_feature_filtered.csv"   # 改成你的实际文件
OUTPUT_FILE = r"E:\农行杯\2026\1\Streamlit平台\data\clean\market_score_daily.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

# 直接映射原文件已经计算好的分数，不重算
df["val_score"] = df["score_valuation"]
df["macro_score"] = df["score_macro"]
df["sent_score"] = df["score_sentiment"]
df["trend_score"] = df["score_trend"]
df["total_score"] = df["score_total"]

def get_market_state(total_score):
    if pd.isna(total_score):
        return np.nan
    if total_score <= 20:
        return "极寒"
    elif total_score <= 40:
        return "下行"
    elif total_score <= 60:
        return "震荡"
    elif total_score <= 80:
        return "震荡偏强"
    else:
        return "上行"

df["market_state"] = df["total_score"].apply(get_market_state)

out_cols = ["date","val_score","macro_score","sent_score","trend_score","total_score","market_state"]
df[out_cols].to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(df[["val_score","macro_score","sent_score","trend_score","total_score"]].isna().sum())
