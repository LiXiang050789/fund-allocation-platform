# 文档问答评测报告

执行方式：`configured_llm`
时间：2026-11-09T13:47:49+08:00

- model: deepseek-flash
- total: 16
- recall_at_5: 0.875
- citation_valid_rate: 1.0
- expected_source_cited_rate: 1.0
- keyword_coverage: 0.9508196721311475
- verified_rate: 0.9375
- latency_p50: 7.12
- latency_p95: 25.67
- llm_error_count: 0
- backend_counts: {'hybrid': 16, 'bm25': 0}

Recall@5 按固定全部16题统计。引用指标要求至少一个真实引用，且全部引用可映射。
keyword_coverage 按所有期望关键词统计。数字与关键词明细记录在 results_doc.json 的 fails。
引用只校验片段存在，不保证语义完备。background/historical 的数字不作为当前事实依据。
Q14 Recall@5=0：golden 锚定 artifact，指定原文片段未进入前五；保留原 source_quote。

| ID | Recall@5 | 引用合法 | 期望来源 | verified | 关键词 | 失败项 |
|---|---|---|---|---|---|---|
| Q01 | 1 | 1 | 1 | False | 4/4 | - |
| Q02 | 1 | 1 | 1 | True | 4/4 | - |
| Q03 | 1 | 1 | 1 | True | 2/2 | - |
| Q04 | 1 | 1 | 1 | True | 2/2 | - |
| Q05 | 1 | 1 | 1 | True | 2/2 | - |
| Q06 | 1 | 1 | 1 | True | 5/5 | - |
| Q07 | 1 | 1 | 1 | True | 4/4 | - |
| Q08 | 1 | 1 | 1 | True | 7/7 | - |
| Q09 | 1 | 1 | 1 | True | 3/3 | - |
| Q10 | 1 | 1 | 1 | True | 5/5 | - |
| Q11 | 1 | 1 | 1 | True | 3/3 | - |
| Q12 | 0 | 1 | 1 | True | 2/2 | - |
| Q13 | 1 | 1 | 1 | True | 3/3 | - |
| Q14 | 0 | 1 | 1 | True | 0/3 | 缺少关键词 市场; 缺少关键词 持仓; 缺少关键词 净值 |
| Q15 | 1 | 1 | 1 | True | 6/6 | - |
| Q16 | 1 | 1 | 1 | True | 6/6 | - |
