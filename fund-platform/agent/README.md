# 投研 AI Agent（LangGraph + RAG）

工行杯平台的问答 Agent 与投研简报工作流。设计与阶段计划见
[`../../AI-Agent施工书-2026-11-07.md`](../../AI-Agent施工书-2026-11-07.md)。

## 当前状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| Phase 0 | 已完成 | 独立环境、25 题评测集、基线评测、飞书发卡实测 |
| Phase 1 | 已完成（离线验收） | 8 个只读工具、LangGraph、`/chat`、29 项单测；真实模型评测待授权 |
| Phase 2 | 未开始 | RAG + 引用 + verifier |
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
`/healthz` 的 `rag_enabled` 固定为 false，本阶段不执行检索。

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

- 单测：`29 passed, 1 warning in 2.92s`。全部 mock LLM，无真实网络。
- 离线 Agent 评测：25/25（100%）；工具选择准确率 92%；引用支持率 96%。
- 原基线：6/25（24%）。本次离线结果不代表真实模型表现。
- 25 题的 `verified` 均为 true，最长回答 132 字（含引用）。
- 8001 端口后台启动和 curl：评分 33.9、状态「下行」、`verified=true`。
- 空问题 curl：HTTP 422。
- 缺失评分文件：工具 `ok=false`，HTTP 200；原始数据未改动。
- HTTP 验收时禁用模型外呼；验收后关闭服务。
- `grep -rnE '33\.9|13\.13|0\.646' app/` 无命中。
- LangGraph 依赖有一项未来默认值变更警告；未改动依赖版本。

产物：`eval/results_agent.json`、`eval/report_agent.md`、`eval/answers_agent.md`。
基线产物和 `golden_set.yaml` 保持原样。
工具选择和引用支持均以全部 25 题为分母。
X01 期望的 `search_docs` 属于 Phase 2，不实现。
X02 的期望工具为空，按“与期望有交集”规则记为未命中。
引用支持率仅检查成功工具 source 路径是否出现在回答中，不代表语义判定。

限制：真实模型评测待外发授权。初次受限网络评测走回退路径，不作真实模型证据。
工作区原有 4 项 agent 目录外的未跟踪内容，本次保留，未纳入提交。
