# 投研 AI Agent（LangGraph + RAG）

工行杯平台的问答 Agent 与投研简报工作流。设计与阶段计划见
[`../../AI-Agent施工书-2026-11-07.md`](../../AI-Agent施工书-2026-11-07.md)。

## 当前状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| Phase 0 | 已完成 | 独立环境、25 题评测集、基线评测、飞书发卡实测 |
| Phase 1 | 已完成 | 8 个只读工具、LangGraph、`/chat`、30 项单测；真实模型评测 25/25 |
| Phase 2 | 已完成 | docx/md/txt、BM25 + FAISS、search_docs、引用与数字校验；验收见下文 |
| Phase 3 | 未开始 | 简报工作流 + 人工审核 + 飞书推送 |
| Phase 4 | 未开始 | 平台集成 + 本地部署 |

## 环境

- Python 3.12，独立 venv：`.venv/`
- 核心依赖：`requirements.in` → 编译为 `requirements.txt`
- RAG 重型依赖单独放 `requirements-rag.in`（Phase 2 再装）

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt --python .venv/bin/python
uv pip install -r requirements-dev.txt --python .venv/bin/python
```

## 配置

```bash
cp .env.example .env    # 填入 DeepSeek Key、飞书 App ID/Secret
chmod 600 .env          # .env 已在 .gitignore，不提交
```

代码只从环境变量读取密钥，不写死、不进镜像。

## 评测

```bash
.venv/bin/python eval/run_eval.py --mode baseline
```

产物：

- `eval/results.json`：机器可读结果与 system prompt 原文
- `eval/report.md`：汇总报告
- `eval/answers.md`：逐题作答明细

基线结果：25 题通过 6 题（24%）；通过项为当前市场评分/状态与合规拒答。

## 飞书自检

```bash
.venv/bin/python -m app.feishu --list                      # 列出机器人所在群
.venv/bin/python -m app.feishu --send --chat-name "群名"    # 发一张测试卡片
```

通道优先级：自建应用机器人 → 群自定义机器人 Webhook → 抛错。

## 待办

- 平台 `platform/app.py` 的 AI 助手 tab 改为调用 Agent 服务（Phase 4）

## Phase 1（2026-11-08）

启动（在 `fund-platform/agent` 目录）：

```bash
.venv/bin/uvicorn app.main:app --port 8001
curl -s -X POST localhost:8001/chat -H 'Content-Type: application/json' \
  -d '{"question":"最新市场评分和状态","thread_id":"demo"}'
```

接口：`POST /chat`、`GET /healthz`、`GET /tools`。
空白问题返回 422。内部错误返回 200 与 `verified=false`。
`thread_id` 用于请求标识；本阶段不保存对话记忆。
Phase 1 验收时 `rag_enabled=false`；Phase 2 当前行为见下文。

图：`intake → router → tools → compose → verify → finalize`。
工具失败时最多再路由一轮；数字或合规校验失败时最多再生成一次。
模型异常、无工具调用或工具名不在白名单时，路由使用确定性关键词回退。
回答只引用成功工具的证据。`verified` 只表示数字、拒答和长度检查通过，不表示全文语义正确。
模型无法连接时，回答从工具结果生成。

工具清单：

- `get_market_score`：最新或指定日期评分。
- `get_market_state_summary`：五态交易日统计。
- `get_strategy_metrics`：十套策略的全区间指标。
- `get_segment_performance`：六阶段、十套策略表现。
- `get_drawdown_recovery`：危机回撤和修复天数。
- `get_risk_tier`：风险档与策略指标。
- `get_feature_importance`：特征重要性排序。
- `get_data_caveats`：数据覆盖、北向缺口、宏观滞后和维度。

所有工具只读。输入校验禁止额外参数和任意文件路径。
表格缓存按路径、修改时间与大小失效。北向只读取日期和资金列。
`MACRO_LAG` 从源码解析，解析失败返回错误。

验证命令：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python eval/run_eval.py --mode agent --offline
# 真实模型评测会发送题目及工具聚合结果至配置的模型服务：
.venv/bin/python eval/run_eval.py --mode agent
```

2026-11-08 验证结果：

- 单测：`30 passed, 1 warning in 3.12s`。全部 mock LLM，无真实网络。
- 离线 Agent 评测：25/25（100%）；工具选择准确率 92%；引用支持率 96%。
- 真实模型 Agent 评测：25/25（100%）；工具选择准确率 92%；引用支持率 100%。
- 真实模型延迟 p50/p95：3.48s / 4.39s。
- 原基线：6/25（24%）。离线结果单独记录，不代表真实模型表现。
- 真实模型 25 题的 `verified` 均为 true，最长回答 152 字（含引用）。
- 8001 端口后台启动和 curl：评分 33.9、状态「下行」、`verified=true`。
- 空问题 curl：HTTP 422。
- 缺失评分文件：工具 `ok=false`，HTTP 200；原始数据未改动。
- 初次 HTTP 验收禁用模型外呼。授权后的真实模型 curl 同样返回评分 33.90、状态「下行」、`verified=true`；验收后关闭服务。
- `grep -rnE '33\.9|13\.13|0\.646' app/` 无命中。
- LangGraph 依赖有一项未来默认值变更警告；未改动依赖版本。

产物：`eval/results_agent.json`、`eval/report_agent.md`、`eval/answers_agent.md`。
基线产物和 `golden_set.yaml` 保持原样。
工具选择和引用支持均以全部 25 题为分母。
X01 期望的 `search_docs` 属于 Phase 2，不实现。
X02 的期望工具为空，按“与期望有交集”规则记为未命中。
引用支持率仅检查成功工具 source 路径是否出现在回答中，不代表语义判定。

真实模型评测由用户授权后执行。首次联网评测为 24/25，北向题因评分标签误用而缺少主题关键词。提示词改为仅对市场分项使用评分标签；数据口径引用按问题主题选择主来源。修正后完整联网评测 25/25。
限制：`verified` 只检查数字、拒答和长度；引用支持指标只检查路径出现，不覆盖完整语义。
工作区原有 4 项 agent 目录外的未跟踪内容，本次保留，未纳入提交。

## Phase 1.1 与 Phase 2（2026-11-09）

### 当前行为

- 共 9 个只读工具，第 9 个为 `search_docs(query, top_k=5)`。
- `query` 为 1–200 字，`top_k` 为 1–10，严格类型校验，拒绝额外参数。
- 规则/定义/问卷/费率/PPO设计/防前视解释检索文档；具体日期与指标读取数据工具；混合问题同时查询。
- 数据引用：`[source: 路径]`；文档引用：`[来源: 文件名 §章节]`。
- `/healthz` 返回 `rag_enabled=true`，另有 `vector_rag_enabled` 表示向量开关。
- 市场评分和分项使用分制；计数使用原单位；比例字段按百分数保留两位小数。
- verify 拦截评分/分项百分号，保留合法收益和回撤百分比。
- 自然阶段名接受“2021年后震荡期”与“2015年股灾期间”，未知阶段仍返回错误。

### 语料清单与权限

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

- `authoritative`：用于规则解释，文本中的数字可以作为依据。
- `background`：只解释设计背景，事实数字不作为答案依据。
- `historical`：只解释历史背景。只有明确标记的禁用旧值可以作为旧值说明。
- `doc_date` 优先使用文件最近一次 Git 提交日期；无记录时使用 mtime。
- `golden_answer` 与 `source_quote` 来自解析原文摘录，不由模型改写。

### 安装与构建

在 `fund-platform/agent` 目录执行。CPU 版 torch 的编译与安装均指定 `UV_TORCH_BACKEND=cpu`：

```bash
UV_TORCH_BACKEND=cpu uv pip compile requirements-rag.in -o requirements-rag.txt
UV_TORCH_BACKEND=cpu uv pip install -r requirements-rag.txt --python .venv/bin/python
.venv/bin/python -m app.rag.build_index
.venv/bin/python -m app.rag.build_index --check-golden
```

核心 `requirements.txt` 不变；RAG 锁文件约束核心依赖。
Embedding 使用 `BAAI/bge-small-zh-v1.5`，CPU 本地编码、L2 归一化、FAISS 内积索引。
BM25 使用 jieba 分词；结果按 `sum(1/(60+rank))` 做 RRF 融合。
本次 BGE 下载成功，未使用镜像。

直连下载失败时，可以尝试：

```bash
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python -m app.rag.build_index
```

仍失败时，构建器保留 BM25-only 索引，并记录 embedding 异常类型。
运行时不下载模型。服务启动不加载 embedding；缺少缓存或向量后端失败时回退 BM25。

```bash
ENABLE_VECTOR_RAG=0 .venv/bin/uvicorn app.main:app --port 8001
# 只构建 BM25 索引：
ENABLE_VECTOR_RAG=0 .venv/bin/python -m app.rag.build_index
```

关闭向量检索后，`search_docs.data.backend=bm25`，不会加载 embedding。
重新启用后，如果当前只有 BM25 索引，必须重建才有 FAISS 向量。
`--bm25-only` 也可以只构建词法索引。

索引目录：`data/index/`；产物为 `chunks.jsonl`、`bm25.json`、`faiss.index`、`manifest.json`。
索引不入 Git；目录内保留本地 `.gitkeep`。
每次检索检查语料、清单和索引文件哈希；变更时返回“索引过期，请重建”，不会静默使用旧内容。
重建命令仍为 `.venv/bin/python -m app.rag.build_index`。
短章节、表格不足目标块长时保留原文；相邻短段落与表格合并，单块上限为 800 字，递归切分重叠为 80 字。
超长表格按完整行拆分并重复表头。

### 校验与验证边界

- 每条文档引用必须映射到本轮检索片段的文件名与章节前缀。
- 文档解释至少包含一条真实引用。
- 数字依据只含成功数据工具值和 authoritative 文本，排除检索分数、文档日期与 background/historical 事实数字。
- 禁用旧值未标记时校验失败；明确标为旧值时只作为历史说明。
- 校验失败最多重新生成一次；仍失败保留原答并返回 `verified=false`。
- `llm_events` 与评测的 `llm_error_count` 区分真实模型调用与异常回退。
- 引用校验只证明片段存在，不保证语义完备。数字校验不代替全文事实审查。

### 评测与 HTTP

```bash
.venv/bin/python -m pytest -q
.venv/bin/python eval/run_doc_eval.py --offline
.venv/bin/python eval/run_doc_eval.py
.venv/bin/python eval/run_eval.py --mode agent

.venv/bin/uvicorn app.main:app --port 8001
# 另一个终端：
curl -s -X POST localhost:8001/chat -H 'Content-Type: application/json' \
  -d '{"question":"评分系统四个维度的权重分别是多少？"}'
curl -s -X POST localhost:8001/chat -H 'Content-Type: application/json' \
  -d '{"question":"最新市场评分和状态"}'
curl -s localhost:8001/tools
curl -s localhost:8001/healthz
.venv/bin/python eval/check_score_units.py
```

真实模型调用会发送问题和工具结果（包括检索片段）至配置的模型服务。本次执行有用户明确授权。

最终验证输出：

- 单测：`57 passed, 1 warning in 22.98s`。全部离线；原有 30 项无回归。
- 索引：`documents=8 chunks=119 backend=hybrid`。
- 原文核验：`golden_quotes=15/15`，逐字匹配。
- 离线文档评测：`execution=offline_fallback`；Recall@5=13/15；引用合法率100%；关键词覆盖率45.3%。仅为片段摘录能力，不冒充模型结果。
- 真实文档评测：`execution=configured_llm`；Recall@5=86.67%；引用合法率100%；期望来源命中率93.3%；关键词覆盖率98.1%；verified率93.3%。
- 文档延迟 p50/p95：7.22s / 20.32s；LLM调用错误0次；backend={'hybrid': 15, 'bm25': 0}。
- 25题真实回归：25/25；工具选择准确率96%；引用支持率100%；LLM调用错误0次。
- HTTP：文档题有真实章节引用；数据题返回33.9/下行；工具数9；空问题422；RAG和vector开关可查询。
- 评分单位：Task A和Phase 2 HTTP的连续5+3次检查均为`score_unit_checks=8/8`。验收后关闭服务。
- 向量关闭检查：`backend=bm25 chunks=5`，`embedding_loaded=False`。

文档产物：`eval/results_doc.json`、`eval/report_doc.md`、`eval/answers_doc.md`。
数据回归产物仍为 `eval/*_agent.*`，只收录本轮重跑结果。
baseline 的三个文件与原25题 `golden_set.yaml` 不改动。

### 已知限制与文档矛盾

- 指定原文未进入top-5：Q14, Q15。这是严格quote判定；不改题、不删题、不换quote。
- Q10 原文写问卷满分148分，原文算式与该值矛盾；按任务给出的算式相加为152。golden照录148，在caveat说明矛盾。回答不自行补算新合计数。
- Q13 手册原则为置NaN、评分中性、不填0；实际数据存在457行填0。原则回答与实际数据分开，实际数值以`get_data_caveats`为准。
- Q07 原文PMI可用时间与实际宏观滞后实现不同；原文解释保留引用，实际以数据工具和源码为准。
- 离线摘要的关键词覆盖有限；冷启动首次检索加载CPU模型，延迟高于热请求。
- LangGraph依赖有未来默认值变更警告；本任务未升级核心依赖。

### 原有测试断言变更

1. `/tools` 数量从8改9：新增search_docs。
2. `/healthz` 的rag_enabled从false改true，并检查vector_rag_enabled：Phase 2支持BM25，向量开关单独返回。

### 工作区范围

仅修改 `fund-platform/agent/` 与 `fund-platform/.gitignore`。
原有4项范围外未跟踪内容保留，不回滚、不暂存。
不提交密钥、虚拟环境、索引二进制或模型缓存；不push。

### 本轮未通过项

- Q02：verified=False；期望来源引用=True；关键词/数字检查：通过。
- Q15：verified=True；期望来源引用=False；关键词/数字检查：缺少关键词 换手。

固定题集、原文quote与预期值均保留。未通过项不按通过计分。

### 保守校验与回放

- Q02 的原文公式 `pe_pct-0.30` 中，减号被简单数字提取器当作负号，导致该题 `verified=false`。原答和公式保留，不为通过校验改数字。
- 最终校验器对保存的25条真实模型答案重新执行工具并校验：`verification_replay=25/25`。原始产物保留当次运行的verified标记，不改写历史结果。
