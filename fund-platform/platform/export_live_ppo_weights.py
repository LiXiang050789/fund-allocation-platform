"""用现场推理路径（ppo_infer_runtime.infer_ppo）导出 PPO 日频权重序列。

目的：让离线权重文件与"执行模型对拍"的现场推理**同源**（同一模型、同一环境、同一次重放），
      对拍偏差 → 0。逻辑直接取自 ppo_infer_runtime.infer_ppo，只把"记录最后一步"改为"记录每一步"。

用法：python export_live_ppo_weights.py [输出路径，默认 results/ppo_weight2.csv]
"""
import importlib.util
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RT_PATH = os.path.join(HERE, "ppo_infer_runtime.py")

spec = importlib.util.spec_from_file_location("ppo_rt", RT_PATH)
rt = importlib.util.module_from_spec(spec)
sys.modules["ppo_rt"] = rt
spec.loader.exec_module(rt)

src = open(RT_PATH, encoding="utf-8").read()
body = src[src.index("def infer_ppo("):src.index("def infer_sasf(")]
body = body.replace("    step_idx = 0\n", "    step_idx = 0\n    rows = []\n", 1)
body = body.replace(
    "        last_weight = env.last_weight.copy()\n",
    "        last_weight = env.last_weight.copy()\n        rows.append((pd.Timestamp(current_date), env.last_weight.copy()))\n",
    1,
)
body = body[:body.index("    return {")] + "    return rows\n"

ns = dict(rt.__dict__)
exec(compile(body, RT_PATH, "exec"), ns)
infer_ppo_rows = ns["infer_ppo"]

end_date = rt.date_all.iloc[-1] if hasattr(rt, "date_all") else None
# 与平台调用一致：以数据最后一天为目标日期
target = "2026-07-10"
rows = infer_ppo_rows(target)
df = pd.DataFrame(
    [{"date": d, **{code: float(w[i]) for i, code in enumerate(rt.ETF_CODES)}} for d, w in rows]
).drop_duplicates(subset="date", keep="last")

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(rt.BASE, "results", "ppo_weight2.csv")
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"已导出 {len(df)} 行 → {out}")
print("末尾两行:")
print(df.tail(2).to_string(index=False))
print("权重和(末行):", round(float(df.iloc[-1][rt.ETF_CODES].sum()), 8))
