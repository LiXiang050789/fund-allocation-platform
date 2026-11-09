# 投研 AI Agent（LangGraph + RAG）

## Phase 2.1.1（2026-11-09）

已完成 PPO 权威语料、16 题文档评测、命名检索、Unicode 数值处理、真实 healthz、引用 fail-closed 与语料预检。最终真实评测时间：2026-11-09T20:37:03+08:00。
阶段施工书见 [`../../AI-Agent施工书-2026-11-07.md`](../../AI-Agent施工书-2026-11-07.md)。该施工书为本地文件，不纳入本次提交。
Phase 3 简报审核推送、Phase 4 平台集成未开始。

## 环境与配置

使用 Python 3.12 和独立 `.venv/`。在本目录执行：

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt --python .venv/bin/python
uv pip install -r requirements-dev.txt --python .venv/bin/python
UV_TORCH_BACKEND=cpu uv pip install -r requirements-rag.txt --python .venv/bin/python
cp .env.example .env
chmod 600 .env
```

填写配置的模型服务密钥；不得提交 `.env`。真实模型运行会把问题及工具结果（包括检索片段）发送到配置的 DeepSeek 服务。本轮执行有用户明确授权。

## 接口与问答图

```bash
.venv/bin/uvicorn app.main:app --port 8001
curl -s localhost:8001/healthz
curl -s localhost:8001/tools
curl -s -X POST localhost:8001/chat -H 'Content-Type: application/json' \
  -d '{"question":"最新市场评分和状态"}'
```

`POST /chat` 返回 answer、tool_calls、citations、verified、latency_ms、thread_id。空白问题返回 HTTP 422。内部异常返回 HTTP 200 和 `verified=false`。thread_id 标识请求，不持久化对话记忆。

`intake → router → tools → compose → verify → finalize`。工具全部失败时最多再路由一轮。回答校验失败时最多再生成一次。模型异常时，从成功工具证据生成确定性摘要。

9 个工具均只读：get_market_score、get_market_state_summary、get_strategy_metrics、get_segment_performance、get_drawdown_recovery、get_risk_tier、get_feature_importance、get_data_caveats、search_docs。
`search_docs(query, top_k=5)` 的 query 为 1–200 字，top_k 为 1–10；严格类型校验，拒绝额外参数和外部文件路径。
规则、设计、问卷与费率问题检索文档；实际指标读取数据工具；混合问题同时查询。

`/healthz` 每次调用 validate，检查真实语料、清单与索引哈希，不加载 embedding。
成功返回 `status=ok`、`rag_enabled=true`、`rag_backend` 和 `rag_error=null`。
缺失或过期返回 HTTP 200、`status=degraded`、`rag_enabled=false`、`rag_backend=null`；rag_error 提供重建命令，不暴露任意异常内容。
`vector_rag_enabled` 表示向量开关。开关关闭时仍可使用 BM25。

## 语料与权威级别

路径相对仓库根目录：

| 文件 | authority |
|---|---|
| `项目彻底理解手册-答辩备战版.md` | authoritative |
| `fund-platform/README.md` | authoritative |
| `fund-platform/docs/评分系统金融规则.docx` | authoritative |
| `fund-platform/docs/风险等级划分规则.docx` | authoritative |
| `fund-platform/docs/用户画像测评问卷+打分规则.docx` | authoritative |
| `fund-platform/docs/已有信息下ETF实际费率与5bp的对比(1).docx` | authoritative |
| `fund-platform/docs/强化学习.txt` | background |
| `项目书修订说明.md` | historical |
| `fund-platform/docs/PPO奖励口径-最终代码.md` | authoritative |

authoritative 文本可提供数字依据。background/historical 只解释背景，不提供当前事实数字。禁用旧值只有明确标记时才能作为旧值解释。
PPO 最终奖励以新增文档及 env_16.py / H_equity_residual 配置为准；早期《强化学习.txt》保留为 background。
Q15 先回答设计稿的收益奖励与三类惩罚，再披露最终实现的差异；Q16 回答最终公式、系数、交易成本与 clip。Q15 原 source_quote 和 Q14 原锚点未改动。
文档中的逐字引用由 source_quote 定位；golden_answer 为评测期望说明，不声称全文逐字摘录。

`项目彻底理解手册-答辩备战版.md`、`项目书修订说明.md` 被根 `.gitignore` 忽略，是本地专用语料。
fresh clone 必须先把这两个文件放到仓库根目录，否则预检和构建会 fail-closed。
不得 `git add -f` 这两个文件；不得修改根 `.gitignore` 规避检查。
新增 `fund-platform/docs/PPO奖励口径-最终代码.md` 已入库，保证最终奖励口径可复现。

## 构建与缺失前置

```bash
.venv/bin/python -m app.rag.build_index --check-corpus
.venv/bin/python -m app.rag.build_index
.venv/bin/python -m app.rag.build_index --check-golden
```

预检成功输出 `corpus_ok=9`；缺失时输出全部 missing 路径和补齐/移除建议，退出码 1。构建在计算 digest 前检查全部文件，不静默跳过。
validate 区分“索引缺失”“语料缺失”和“索引过期”。每次检索检查完整性，拒绝使用过期内容。

本轮索引：`documents=9 chunks=124 backend=hybrid`。
索引目录 `data/index/` 的 chunks.jsonl、bm25.json、faiss.index、manifest.json 不入库；`.gitkeep` 本地保留、不入库。
使用 CPU BAAI/bge-small-zh-v1.5、L2 归一化与 FAISS 内积；BM25 使用 jieba，RRF 为 sum(1/(60+rank))。
本轮构建使用已有模型缓存并设置 `HF_HUB_OFFLINE=1`，未下载模型。新环境首次构建需要准备模型或允许下载；embedding 不可用时记录异常类型并构建 BM25-only 索引。
运行时只读本地模型；加载失败回退 BM25。`ENABLE_VECTOR_RAG=0` 或构建时 `--bm25-only` 可关闭向量路径。
短段落与表格合并，单块上限 800 字，重叠 80 字；长表按完整行拆分并重复表头。

## 命名检索、引用与数值校验

查询按大小写和空白归一化识别完整文件名、去后缀名称与明确别名。保留 BM25/FAISS/RRF 主排名，普通查询不改写排名。
命名文档无论是否已进入 RRF 前列，都按整份 source 的 BM25 分数排序；同分按 chunks.jsonl 顺序。全源分数为 0 时，按不同 query token 的覆盖数排序，再按 chunk 顺序。
单命名文档保留源内前 2 块（受 top_k 限制），进入前 min(5, top_k)。多命名文档先各保 1 块，剩余 slot 按源内候选分数分配。已在结果中的选中片段保留；补入时只替换未保护 slot，结果不超过 top_k，最后去重。
删除了围绕错误 existing chunk 只看前后各一块的选择逻辑；本轮不使用邻块补充。日志保留 named_sources、named_reserved、named_neighbors（空列表），新增 named_reserved_chunks，逐块记录 source_name、chunk_id、section。
单独“PPO”不是文档别名；“手册”泛称可辅助检索，但不触发强制命名来源校验。
路由不得删掉用户点名的文档。答案先答命名文档；冲突需披露差异并说明最终以代码为准。

数据引用为 `[source: 路径]`；文档引用为 `[来源: 文件名 §章节]`。文档引用必须映射到本轮 chunk；点名文档必须有对应真实引用。
模型未提供引用时，不自动贴 docs[0]。首次校验失败后重试；第二次仍无合法引用则返回“未在检索片段中找到可引用依据，无法回答。不构成投资建议。”，无伪造引用，verified=false。
模型异常时的 evidence_answer 从实际 chunk 摘录并引用，保留此确定性回退。

共享 NFKC 处理全角数字、百分号与 Unicode 负号。数值提取区分负收益、日期、范围、二元减法和中文标点。评分使用分制，收益/回撤可用百分比；检索分数、日期和非权威事实数字不作为数字依据。
回答总长不超过 300 字，含引用与免责声明。verified 仅检查数字、引用、拒答和长度，不保证全文语义完整。

## Phase 2.1.1 验收

Q14 此前未召回相关片段，并回答无法确认三类信息，属于检索/回答失败。此前将其归因于原文锚点的说明错误。
本轮从整份命名文档保留词法前两块，不硬编码题目、章节、关键词或答案。Q14/Q15/Q16 的 golden 与原25题 golden_set.yaml 未改动。

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m app.rag.build_index --check-corpus
.venv/bin/python -m app.rag.build_index
.venv/bin/python -m app.rag.build_index --check-golden
.venv/bin/python eval/run_doc_eval.py --offline
.venv/bin/python eval/run_doc_eval.py
.venv/bin/python eval/run_eval.py --mode agent
# 服务运行时：
.venv/bin/python eval/check_score_units.py
```

- 单测原始摘要：`97 passed, 1 warning in 10.49s`；原91项保留，无 skip/xfail，测试不下载模型。
- 新增临时 mini corpus 回归保持 BM25 与实际 FAISS 查询；构造源内第2块在 RRF 前列之外且远离错误已有块。top_k=3/5/8 三种情况在基线代码下均失败，修复后通过。新增报告条件判定的3个用例。
- 预检：`corpus_ok=9`。
- 索引：`documents=9 chunks=124 backend=hybrid`。
- 原文核验：`golden_quotes=16/16`。
- 离线评测：execution=offline_fallback；Recall@5=1.0；引用合法率=1.0；期望来源=0.8125；关键词=0.5737704918032787；verified=0.875；llm_error_count=34（故意注入 offline 异常）。这不是实际模型表现。

真实文档评测（2026-11-09T20:37:03+08:00）：

| 指标 | 结果 |
|---|---|
| total | 16 |
| recall_at_5 | 1.0 |
| citation_valid_rate | 1.0 |
| expected_source_cited_rate | 1.0 |
| keyword_coverage | 1.0 |
| verified_rate | 1.0 |
| latency_p50 | 7.86 |
| latency_p95 | 21.39 |
| llm_error_count | 0 |
| backend_counts | {'hybrid': 16, 'bm25': 0} |

| 题目 | Recall@5 | expected_source | 关键词 | verified |
|---|---|---|---|---|
| Q14 | 1 | 1 | 3/3 | True |
| Q15 | 1 | 1 | 6/6 | True |
| Q16 | 1 | 1 | 6/6 | True |

Q14 完整回答：

> 精炼市场特征；当前资产持仓权重；组合实时净值。不构成投资建议。
> [来源: 强化学习.txt §二、各模块详细逻辑链条（逐块拆解） / 2.2 环境核心优势总结]

Q14：通过。三类信息与本轮检索到的《强化学习.txt》正确章节对应。报告区分“召回未中但关键词全中”的锚点偏差与“召回未中且关键词缺失”的检索/回答失败。

25题真实回归（2026-11-09T20:35:38）：25/25；工具选择=0.96；引用支持=1.0；verified=1.0；llm_error_count=0；p50/p95=4.19/6.94s。

HTTP：Q14 包含市场、持仓、净值，并引用强化学习.txt 的正确章节；Q15/Q16 保持全部关键词和来源，verified=true；市场题为33.9/下行；工具9个；空问题422；healthz status=ok、rag_enabled=true、backend=hybrid。
连续5+3次评分单位检查：`score_unit_checks=8/8`。验收进程记录在 `/tmp/agent-uvicorn-2.1.1.pid`，只关闭本次PID；`server_stopped=True port_8001_closed=True`。
临时原始日志：`/tmp/phase211-pytest.log`、`/tmp/phase211-red.log`、`/tmp/phase211-green.log`、`/tmp/phase211-doc-offline.log`、`/tmp/phase211-doc-live.log`、`/tmp/phase211-agent-live.log`、`/tmp/phase211-http.log`。

产物：[report_doc.md](eval/report_doc.md)、[results_doc.json](eval/results_doc.json)、[answers_doc.md](eval/answers_doc.md)、[report_agent.md](eval/report_agent.md)、[results_agent.json](eval/results_agent.json)、[answers_agent.md](eval/answers_agent.md)。JSON 来自完整评测运行，未手改结果。

## 已知限制

- 本轮16题没有原文召回、关键词或校验失败项。
- verified 只检查数字、引用、拒答和长度，不保证回答完整；关键词检查也不能替代完整语义审查。
- 命名保留位受 top_k 限制；本轮前两块解决了Q14，未扩大至前三块。
- 多个命名文档会占用更多保留位，可能减少其他未点名来源的片段。普通查询不改写RRF。
- Q10 原文算式矛盾不自行补算；Q07/Q13 原文规则与实际数据不同，分开说明。
- LangGraph 保留一项未来默认值变更警告；本轮未升级依赖。
- 两份根目录本地专用语料仍需 fresh clone 手工补齐，预检保持 fail-closed。

本轮未改弱既有断言，只新增6项参数化回归/报告测试。仅修改 agent/；保留原有四项未跟踪内容，不提交密钥、索引、模型缓存或忽略的手册。未 push。
