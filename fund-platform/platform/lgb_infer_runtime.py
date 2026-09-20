"""LGB 融合推理 runtime —— 完全对齐离线三档风险缩放逻辑（路线1）
命令行调用示例：
python lgb_infer_runtime.py 2025-12-31      # 返回原生未缩放LGB权重
python lgb_infer_runtime.py 2025-12-31 C2   # C2保守风险适配
python lgb_infer_runtime.py 2025-12-31 C3   # C3平衡风险适配
python lgb_infer_runtime.py 2025-12-31 C4   # C4进取风险适配
"""
import json
import os
import sys
import pandas as pd
import numpy as np

ETF_CODES = ["hs300", "zz500", "kc50", "consume", "chip", "gold", "bond10"]
EQUITY_IDX = [0, 1, 2, 3, 4]   # 5只权益ETF
GOLD_IDX = 5
BOND_IDX = 6

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LGB_WEIGHT_PATH = os.path.join(BASE, "results", "lgb_fusion_monthly_weights.csv")

# 和离线脚本完全保持一致
RISK_TARGET_EQUITY = {
    "C2": 0.20,
    "C3": 0.50,
    "C4": 0.70,
}
BASE_EQUITY_CENTER = 0.35
MAX_WEIGHT = 0.30

def rescale_equity_online(raw_weight_dict, target_center):
    """
    线上推理版rescale_equity，100%复刻离线脚本修复后逻辑
    raw_weight_dict: dict 原始lgb权重
    target_center: float 风险档位目标权益中枢
    return dict: 缩放之后权重字典
    """
    scale = target_center / BASE_EQUITY_CENTER
    w = np.array([raw_weight_dict[code] for code in ETF_CODES], dtype=np.float64)
    w_eq_raw = w[EQUITY_IDX]
    eq_sum = w_eq_raw.sum()

    if eq_sum < 1e-8:
        new_eq = target_center
        eq_w = np.full(5, new_eq / 5)
        bond_w = 1.0 - new_eq
    else:
        new_eq = eq_sum * scale
        new_eq = np.clip(new_eq, 0.0, 1.0)
        if new_eq < 1e-8:
            eq_w = np.zeros(5)
            bond_w = 1.0
        else:
            eq_w = w_eq_raw * (new_eq / eq_sum)
            eq_w = np.clip(eq_w, 0.0, MAX_WEIGHT)
            s = eq_w.sum()
            if s > 1e-8:
                eq_w = eq_w * (new_eq / s)
            bond_w = 1.0 - new_eq

    out_w = np.zeros(7)
    out_w[EQUITY_IDX] = eq_w
    out_w[GOLD_IDX] = 0.0
    out_w[BOND_IDX] = bond_w

    return {ETF_CODES[i]: float(out_w[i]) for i in range(7)}


def infer_lgb(as_of_date, risk_tier=None):
    if not os.path.exists(LGB_WEIGHT_PATH):
        raise FileNotFoundError(f"LGB 权重文件缺失：{LGB_WEIGHT_PATH}")
    df = pd.read_csv(LGB_WEIGHT_PATH, encoding="utf-8-sig", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(as_of_date)]
    if df.empty:
        raise ValueError("无有效日期")
    row = df.iloc[-1]

    raw_weights = {code: float(row[code]) for code in ETF_CODES}

    # 不传风险档位：返回原生LGB权重（兼容旧逻辑）
    if risk_tier is None or risk_tier not in RISK_TARGET_EQUITY:
        return {
            "date": str(row["date"].date()),
            "weights": raw_weights,
            "source": "lgb_fusion_raw",
            "market_state": row.get("market_state", "震荡"),
            "risk_tier": None,
            "raw_equity_sum": float(sum(raw_weights[c] for c in [ETF_CODES[i] for i in EQUITY_IDX]))
        }

    target_eq = RISK_TARGET_EQUITY[risk_tier]
    scaled_weights = rescale_equity_online(raw_weights, target_eq)
    final_equity_sum = sum(scaled_weights[c] for c in [ETF_CODES[i] for i in EQUITY_IDX])

    return {
        "date": str(row["date"].date()),
        "weights": scaled_weights,
        "source": f"lgb_fusion_risk_{risk_tier}",
        "market_state": row.get("market_state", "震荡"),
        "risk_tier": risk_tier,
        "target_equity": target_eq,
        "final_equity_sum": round(final_equity_sum, 4),
    }


if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else str(pd.Timestamp.today().date())
    risk_arg = sys.argv[2] if len(sys.argv) >2 else None
    res = infer_lgb(date_arg, risk_arg)
    print(json.dumps(res, ensure_ascii=False))
