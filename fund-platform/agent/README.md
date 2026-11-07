# 投研 AI Agent（LangGraph + RAG）

工行杯平台的问答 Agent 与投研简报工作流。设计与阶段计划见
[`../../AI-Agent施工书-2026-11-07.md`](../../AI-Agent施工书-2026-11-07.md)。

## 当前状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| Phase 0 | 已完成 | 独立环境、25 题评测集、基线评测、飞书发卡实测 |
| Phase 1 | 未开始 | 工具层 + LangGraph 路由 + FastAPI `/chat` |
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

- Phase 1：8 个只读工具、LangGraph 问答图、FastAPI `/chat`
- 平台 `platform/app.py` 的 AI 助手 tab 改为调用 Agent 服务（Phase 4）
