"""LGB 融合推理 runtime"""
import json
import os
import sys
import pandas as pd
import numpy as np

ETF_CODES = ["hs300", "zz500", "kc50", "consume", "chip", "gold", "bond10"]
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LGB_WEIGHT_PATH = os.path.join(BASE, "results", "lgb_fusion_monthly_weights.csv")


def infer_lgb(as_of_date):
    if not os.path.exists(LGB_WEIGHT_PATH):
        raise FileNotFoundError(f"LGB 权重文件缺失：{LGB_WEIGHT_PATH}")
    df = pd.read_csv(LGB_WEIGHT_PATH, encoding="utf-8-sig", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(as_of_date)]
    if df.empty:
        raise ValueError("无有效日期")
    row = df.iloc[-1]
    return {
        "date": str(row["date"].date()),
        "weights": {code: float(row[code]) for code in ETF_CODES},
        "source": "lgb_fusion",
        "market_state": row.get("market_state", "震荡"),
    }


if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else str(pd.Timestamp.today().date())
    print(json.dumps(infer_lgb(date_arg), ensure_ascii=False))