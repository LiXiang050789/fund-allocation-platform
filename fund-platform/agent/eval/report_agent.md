# 25 题评测报告

- 模式：`agent`
- 执行方式：`configured_llm`
- 时间：2026-11-08T17:10:28
- 总题数：25
- 通过：25
- 准确率：100.0%
- 引用支持率：1.0（按全部题目统计成功工具 source 路径的引用）
- 延迟 p50/p95：3.48s / 4.39s

- 工具选择准确率：0.92

## 分类通过率

| 类别 | 通过/总数 | 通过率 |
|---|---|---|
| market_score | 2/2 | 100% |
| market_state | 1/1 | 100% |
| market_history | 2/2 | 100% |
| strategy | 6/6 | 100% |
| segment | 2/2 | 100% |
| drawdown | 3/3 | 100% |
| risk | 3/3 | 100% |
| data | 3/3 | 100% |
| refusal | 3/3 | 100% |

## 逐题结果

| ID | 结果 | 失败项 |
|---|---|---|
| M01 | PASS | - |
| M02 | PASS | - |
| M03 | PASS | - |
| M04 | PASS | - |
| M05 | PASS | - |
| S01 | PASS | - |
| S02 | PASS | - |
| S03 | PASS | - |
| S04 | PASS | - |
| S05 | PASS | - |
| S06 | PASS | - |
| G01 | PASS | - |
| G02 | PASS | - |
| G03 | PASS | - |
| G04 | PASS | - |
| G05 | PASS | - |
| R01 | PASS | - |
| R02 | PASS | - |
| R03 | PASS | - |
| D01 | PASS | - |
| D02 | PASS | - |
| D03 | PASS | - |
| X01 | PASS | - |
| X02 | PASS | - |
| X03 | PASS | - |

基线通过 6/25；Agent 通过 25/25。

工具选择和引用支持均以全部题目为分母。空期望工具按无交集计分。
引用支持率检查答案是否含成功工具的 source 路径；不代表全文语义核验。

## 验证状态

- M01: verified=True
- M02: verified=True
- M03: verified=True
- M04: verified=True
- M05: verified=True
- S01: verified=True
- S02: verified=True
- S03: verified=True
- S04: verified=True
- S05: verified=True
- S06: verified=True
- G01: verified=True
- G02: verified=True
- G03: verified=True
- G04: verified=True
- G05: verified=True
- R01: verified=True
- R02: verified=True
- R03: verified=True
- D01: verified=True
- D02: verified=True
- D03: verified=True
- X01: verified=True
- X02: verified=True
- X03: verified=True