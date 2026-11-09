# 投研 AI Agent（LangGraph + RAG）

## Phase 2.1（2026-11-09）

已完成 PPO 权威语料、16 题文档评测、命名检索、Unicode 数值处理、真实 healthz、引用 fail-closed 与语料预检。最终真实评测时间：2026-11-09T13:47:49+08:00。
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

查询按大小写和空白归一化识别完整文件名、去后缀名称与明确别名。保留 BM25/FAISS/RRF；命名文档缺席时，从其片段按 BM25、同分顺序或 token 重叠选取保留位。
保留位进入前 min(5, top_k)；多个文档保护已有保留位，结果总数不超过 top_k，并去重。
已进入排名的命名文档使用同源前后各一块扩展，按 BM25 选择本地候选。日志包含 named_sources、named_reserved、named_neighbors。
单独“PPO”不是文档别名；“手册”泛称可辅助检索，但不触发强制命名来源校验。
路由不得删掉用户点名的文档。答案先答命名文档；冲突需披露差异并说明最终以代码为准。

数据引用为 `[source: 路径]`；文档引用为 `[来源: 文件名 §章节]`。文档引用必须映射到本轮 chunk；点名文档必须有对应真实引用。
模型未提供引用时，不自动贴 docs[0]。首次校验失败后重试；第二次仍无合法引用则返回“未在检索片段中找到可引用依据，无法回答。不构成投资建议。”，无伪造引用，verified=false。
模型异常时的 evidence_answer 从实际 chunk 摘录并引用，保留此确定性回退。

共享 NFKC 处理全角数字、百分号与 Unicode 负号。数值提取区分负收益、日期、范围、二元减法和中文标点。评分使用分制，收益/回撤可用百分比；检索分数、日期和非权威事实数字不作为数字依据。
回答总长不超过 300 字，含引用与免责声明。verified 仅检查数字、引用、拒答和长度，不保证全文语义完整。

## 最终验收

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

- pytest 原始摘要：`91 passed, 1 warning in 11.30s`；无 skip/xfail，全部离线，无模型下载。受限环境的 TestClient 阻塞，最终使用允许本地服务的执行环境。
- 语料预检：`corpus_ok=9`。
- 构建：`documents=9 chunks=124 backend=hybrid`。
- 原文核验：`golden_quotes=16/16`，逐字匹配。
- 离线文档评测：execution=offline_fallback；Recall@5=0.9375；citation_valid_rate=1.0；expected_source_cited_rate=0.8125；keyword_coverage=0.5737704918032787；verified_rate=0.875；p50/p95=0.06/10.34s；llm_error_count=34（故意注入 offline 异常）。离线摘要结果不代表真实模型能力，日志保留在 /tmp/phase21-doc-offline.log。

真实 16 题结果（2026-11-09T13:47:49+08:00，deepseek-flash）：

| 指标 | 结果 |
|---|---|
| total | 16 |
| recall_at_5 | 0.875 |
| citation_valid_rate | 1.0 |
| expected_source_cited_rate | 1.0 |
| keyword_coverage | 0.9508196721311475 |
| verified_rate | 0.9375 |
| latency_p50 | 7.12 |
| latency_p95 | 25.67 |
| llm_error_count | 0 |
| backend_counts | {'hybrid': 16, 'bm25': 0} |

| 题目 | Recall@5 | expected_source | 关键词 | verified |
|---|---|---|---|---|
| Q15 | 1 | 1 | 6/6 | True |
| Q16 | 1 | 1 | 6/6 | True |

25 题真实回归（2026-11-09T13:51:38）：`25/25`；工具选择准确率=0.96；引用支持率=1.0；verified_rate=1.0；p50/p95=4.76/7.35s；llm_error_count=0。
全部题目为指标分母；空期望工具仍按无交集计分。G03 verified=True。baseline 三个文件和原 25 题 golden_set.yaml 未改动。

HTTP：healthz status=ok、rag_enabled=true、backend=hybrid；tools 数量 9；Q15 包含设计四项及规则基准/夏普差异并引用强化学习.txt；Q16 包含全部系数、3bp 与 clip 并引用新文档；两题 verified=true。
数据题返回 33.90/33.9 与“下行”，verified=true；空问题 422。连续 5+3 次原始输出：`score_unit_checks=8/8`。
验收仅记录并关闭本次 uvicorn PID；`server_stopped=True port_8001_closed=True`。HTTP 原始记录为 /tmp/phase21-http.log（临时日志不入库）。

产物：[report_doc.md](eval/report_doc.md)、[results_doc.json](eval/results_doc.json)、[answers_doc.md](eval/answers_doc.md)、[report_agent.md](eval/report_agent.md)、[results_agent.json](eval/results_agent.json)、[answers_agent.md](eval/answers_agent.md)。JSON 均来自评测脚本，没有手工改结果。

## 已知限制与原有断言

- Q14 Recall@5=0：golden 锚定 artifact，指定原文片段未进入前五；保留原 source_quote，不以其他同义片段替换。 本轮未命中原文锚点的题目：Q12, Q14。
- Q10 原文问卷合计与算式矛盾，照录原文，不自行补算新合计。
- Q13 手册原则与北向实际填零不同；Q07 原文 PMI 可用时间与实际滞后不同。规则和实际数据分开说明，实际以数据工具为准。
- 冷启动需加载 CPU 模型；上述延迟来自本轮运行，不是稳定服务 SLA。
- LangGraph 有一项未来默认值变更警告；本次未升级依赖。
- Q01 同时调用市场数据工具；权重百分比触发评分单位校验。保留 verified=false，不改写结果。
- Q01: verified=False；删除证据不支持的数值[]；仅引用tool_results，遵守拒答政策和字数限制。市场评分及分项使用分制，删除错误百分号并保留原始分数，不得改动数值。；golden评分失败项：无。
- Q14: verified=True；无校验错误；golden评分失败项：缺少关键词 市场; 缺少关键词 持仓; 缺少关键词 净值。

本次未改弱任何既有断言。test_api 的 rag_enabled=True 断言保留，新增 validate 成功 fixture，避免依赖用户真实索引；新增独立用例验证 degraded/ok 和错误信息清理。
既有图测试的 offline fixture 新增 embedding 异常模拟，确保不加载/下载真实模型；既有断言保留，混合检索另由 fake encoder 测试。
原有 57 项测试均保留；本轮合计 91 项，无 skip/xfail。

本次指标问题额外约束使用数据工具；风险档、危机和策略主语从成功工具行保留，防止模型省略题目身份。

本次仅修改 agent/ 与新增 PPO 文档。范围外四项原有未跟踪内容保留。未提交密钥、虚拟环境、索引、模型缓存和被忽略手册。未 push。
