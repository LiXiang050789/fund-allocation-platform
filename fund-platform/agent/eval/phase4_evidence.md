# Phase 4 平台集成与本地部署证据（2026-11-10）

## 范围与版本

本轮基于 `master=origin/master=bea8a451832320f3c7511035b77e6563d2c23e11`。
平台/客户端提交 `789b3487`；部署文档提交 `3fecf85f`；验收修复提交 `11063446`（`saved_only` 的 message_id 展示、README 日期层级）。真实飞书单卡验收与最终 25 题复跑在修复后执行。未 push。
日期为 Asia/Shanghai 本地时间。系统为 Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.35。
Agent Python 3.12.6；平台 Python 3.13.11；Streamlit 1.60.0；FastAPI 0.162.1；uvicorn 0.54.0；LangGraph 0.3.34；langgraph-checkpoint-sqlite 2.0.11。
真实模型为 deepseek-flash，沿用既有授权。未修改 Agent 源码、工具、RAG 算法、校验器、golden set、评测脚本与既有测试断言。

## 启动与 HTTP

Agent 实际以五个飞书变量为空的环境启动。load_dotenv 不覆盖已有空值。以下命令没有凭证。

```bash
cd /home/lx/农行杯/fund-platform/agent
FEISHU_APP_ID= FEISHU_APP_SECRET= FEISHU_CHAT_ID= FEISHU_WEBHOOK_URL= FEISHU_WEBHOOK_SECRET= HF_HUB_OFFLINE=1 .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8001 --workers 1
```

启动输出：

```text
INFO:     Started server process [18892]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
```

平台启动命令：

```bash
cd /home/lx/农行杯/fund-platform
AGENT_BASE_URL=http://127.0.0.1:8001 /home/lx/miniconda3/bin/python -m streamlit run platform/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

实际访问地址：`http://127.0.0.1:8501/`。只监听本机回环地址。
HTTP 验证命令为 `/home/lx/miniconda3/bin/python /tmp/phase4_http.py`；使用 requests 查询两个 GET 接口，并提交空问题。
`GET /healthz` 原始 JSON：

```json
{
  "status": "ok",
  "model": "deepseek-flash",
  "rag_enabled": true,
  "vector_rag_enabled": true,
  "rag_backend": "hybrid",
  "rag_error": null
}
```

`GET /tools` 返回 9 个工具：

```json
[
  {
    "name": "get_market_score",
    "description": "获取指定日期或最新交易日的市场总分、五态标签与四项分数。"
  },
  {
    "name": "get_market_state_summary",
    "description": "统计日期区间内五种市场状态天数、交易日总数与末日状态。"
  },
  {
    "name": "get_strategy_metrics",
    "description": "查询十种策略的全区间收益、波动、夏普、回撤和卡玛指标。"
  },
  {
    "name": "get_segment_performance",
    "description": "查询六阶段内各策略的累计收益、年化收益、最大回撤。"
  },
  {
    "name": "get_drawdown_recovery",
    "description": "查询危机阶段各策略的回撤、收益和修复天数；危机名称支持包含匹配。"
  },
  {
    "name": "get_risk_tier",
    "description": "查询保守C2、平衡C3、进取C4下各策略的收益与风险。"
  },
  {
    "name": "get_feature_importance",
    "description": "按重要性降序获取前若干特征，数量限制为1至50。"
  },
  {
    "name": "get_data_caveats",
    "description": "计算数据覆盖、北向缺口、月度宏观滞后、特征维度及历史数据时效。"
  },
  {
    "name": "search_docs",
    "description": "检索文档规则、定义、问卷、费率、PPO设计与防前视解释；不查询当前策略指标。"
  }
]
```

空问题 POST /chat：HTTP `422`。

## pytest 与静态检查

客户端命令（fund-platform 目录）：

```bash
agent/.venv/bin/python -m pytest tests/test_agent_client.py -q
```

原始摘要：

```text
.................................                                        [100%]
33 passed in 0.24s
```

覆盖 URL 配置、请求方法/请求体、连接/DNS/超时、非 2xx、错误脱敏与截断、非 object/坏 JSON、审核动作、本地拒绝、报告文本、无凭证读取/请求头。全部请求均被 monkeypatch；不依赖 Streamlit，不访问网络。
首次新增测试有两项失败：URL 隐藏顺序与环境变量测试的初始化顺序。修复客户端先隐藏 URL；测试先设置哨兵再安装读取守卫。保留原断言，最终 33 项通过。

Agent 命令（agent 目录，飞书变量为空、离线模型缓存）：

```bash
HF_HUB_OFFLINE=1 FEISHU_APP_ID= FEISHU_APP_SECRET= FEISHU_CHAT_ID= FEISHU_WEBHOOK_URL= FEISHU_WEBHOOK_SECRET= .venv/bin/python -m pytest -q
```

原始摘要：

```text
........................................................................ [ 54%]
............................................................             [100%]
132 passed, 1 warning in 27.97s
```

132 项全部保留，无 skip/xfail。警告为 LangChainPendingDeprecationWarning：allowed_objects 默认值未来变化。测试使用现有 LLM/飞书 mock，未调用真实 DeepSeek 或飞书。首次受限沙箱运行在 TestClient 附近停滞；正常本机执行通过后，按精确命令与项目目录终止该停滞进程。

静态命令：

```bash
rg -n 'DEEPSEEK_API_KEY|api\.deepseek\.com|DeepSeek API Key|Authorization' platform/app.py platform/agent_client.py
git diff --check
```

两项均无输出；rg 无匹配时退出码为 1。对比 HEAD 基线：tab4 之前的业务代码、tab5 及之后代码逐字一致；只增加必要导入与替换 tab4。
Windows 脚本检查以下项：脚本目录切换、虚拟环境检查、缺失环境非零退出、UTF-8、回环地址、8001、单 worker、前台运行、失败 pause、保留退出码。输出 `windows_bat_static_ok; Windows execution not performed`。**Windows 仅静态检查，未实机执行。**

## AppTest：服务关闭与真实 HTTP

服务关闭时，执行任务指定 smoke，并增加实时模式与预制模式切换断言：

```bash
/home/lx/miniconda3/bin/python - <<'TEST'
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("platform/app.py", default_timeout=180).run()
assert not at.exception, at.exception
labels = [w.label for w in at.text_input]
assert "DeepSeek API Key" not in labels
print("apptest_ok", len(at.tabs), len(labels))
at.radio[0].set_value("🤖 Agent 实时问答").run()
assert not at.exception, at.exception
assert any("Agent 服务未启动或不可达" in e.value for e in at.error)
at.radio[0].set_value("🎯 预制场景").run()
assert not at.exception, at.exception
assert any(b.label == "📅 定投还有用吗" for b in at.button)
print("offline_fallback_ok history", len(at.session_state["agent_chat_history"]))
TEST
```

原始摘要：

```text
apptest_ok 5 0
offline_fallback_ok history 0
```

服务关闭后，以最终前端版本复跑 smoke 和模式回退，输出相同。最终日志为 `/tmp/phase4_offline_final.log`。

真实 HTTP AppTest 命令：`/home/lx/miniconda3/bin/python /tmp/phase4_live_apptest.py`。
检查页面无异常；发送问题；切换模式后核对历史与标识；生成、编辑、驳回、审核；隐藏终态审核按钮；按需下载；核对 SHA256；后端重复 approve；页面刷新。
AppTest 对显式 st.rerun 的首个树快照保留旧按钮，追加一次无操作 run 后再检查稳定页面，终态隐藏断言通过。没有修改业务断言；真实浏览器另行验证按钮消失。

原始摘要：

```text
apptest_live_ok fee1e7de224e448c969fdc617886807e saved_only revision 2 sha256 36da65902cdaa6257701c2ab268c5e515675b427dd73dd4b50333893efecdcda duplicate True
```

问答原始结果（页面实际调用 /chat）：

```json
{
  "question": "最新市场评分和状态",
  "result": {
    "answer": "截至2026-07-10：市场总分33.9分，状态“下行”。分项：估值0.00分、宏观14.08分、情绪10.65分、趋势9.18分。数据实际以数据工具为准。不构成投资建议。\n[source: data/clean/market_score_daily.csv]",
    "tool_calls": [
      {
        "name": "get_market_score",
        "args": {
          "date": null
        },
        "ok": true
      },
      {
        "name": "get_market_state_summary",
        "args": {
          "start": null,
          "end": null
        },
        "ok": true
      }
    ],
    "citations": [
      "data/clean/market_score_daily.csv"
    ],
    "verified": true,
    "latency_ms": 5924.13,
    "thread_id": "2a3d3ac3b0404fad9e665bd2a9312305"
  }
}
```

页面模式切换后历史和 thread_id 保留。thread_id 只作请求标识，页面历史不代表服务端记忆。

## AppTest 工作流证据

状态迁移（完整线程、日期、修订号、时间）：

```json
[
  {
    "thread_id": "fee1e7de224e448c969fdc617886807e",
    "report_date": "2026-07-10",
    "status": "awaiting_review",
    "revision": 0,
    "updated_at": "2026-11-10T13:44:56+08:00"
  },
  {
    "thread_id": "fee1e7de224e448c969fdc617886807e",
    "report_date": "2026-07-10",
    "status": "awaiting_review",
    "revision": 1,
    "updated_at": "2026-11-10T13:44:57+08:00"
  },
  {
    "thread_id": "fee1e7de224e448c969fdc617886807e",
    "report_date": "2026-07-10",
    "status": "awaiting_review",
    "revision": 2,
    "updated_at": "2026-11-10T13:45:01+08:00"
  },
  {
    "thread_id": "fee1e7de224e448c969fdc617886807e",
    "report_date": "2026-07-10",
    "status": "saved_only",
    "revision": 2,
    "updated_at": "2026-11-10T13:45:15+08:00"
  }
]
```

编辑追加验收文字，修订号 0→1，仍待审。驳回意见非空，修订号 1→2，仍待审。审核人填写后勾选确认框，手动审核通过，状态为 saved_only。
审核历史：

```json
[
  {
    "action": "edit",
    "reviewer": "Phase4-AppTest",
    "comment": "",
    "at": "2026-11-10T13:44:57+08:00",
    "revision": 1
  },
  {
    "action": "reject",
    "reviewer": "Phase4-AppTest",
    "comment": "请核对数据日期与风险提示。",
    "at": "2026-11-10T13:45:01+08:00",
    "revision": 2
  },
  {
    "action": "approve",
    "reviewer": "Phase4-AppTest",
    "comment": "请核对数据日期与风险提示。",
    "at": "2026-11-10T13:45:15+08:00",
    "revision": 2
  }
]
```

报告路径：`agent/reports/2026-07-10_fee1e7de224e448c969fdc617886807e.md`。
SHA256：`36da65902cdaa6257701c2ab268c5e515675b427dd73dd4b50333893efecdcda`。
落盘 bytes、HTTP 报告 UTF-8 bytes、页面待下载 bytes 的 SHA256 相同。

发布对象：

```json
{
  "status": "skipped",
  "message_id": null,
  "error": null,
  "attempts": 0
}
```

重复 approve 原始摘要：

```json
{
  "thread_id": "fee1e7de224e448c969fdc617886807e",
  "status": "saved_only",
  "revision": 2,
  "duplicate": true,
  "publish": {
    "status": "skipped",
    "message_id": null,
    "error": null,
    "attempts": 0
  }
}
```

重复请求的审核历史与报告哈希保持相同。页面终态隐藏审核按钮。
安全模式未发生真实飞书发送。message_id 为 null，逻辑发送尝试数为 0。saved_only 只证明保存报告，不证明飞书发送成功。本轮未复用 Phase 3 单卡授权；授权后的独立单卡发送见下节。

## 真实飞书单卡验收（2026-11-10 授权后）

用户在本轮验收后明确授权再发 1 条卡片。发送前以只读接口核对机器人所在群：共 1 个群，名称「小波❤小蛙」，与 `FEISHU_CHAT_ID` 一致；未向其他群发卡。

启动命令与安全模式相同，但不显式清空飞书变量，由 `agent/.env` 提供配置。`/healthz` 为 `status=ok`、`rag_backend=hybrid`。索引在 README 修复后已重建：发送时为 `documents=9 chunks=128 backend=hybrid`，`corpus_ok=9`；写入本节后最终复跑前重建为 129 块。

工作流线程：`d74b1d9a876a4de0969c22ea801b6d8c`。报告日期：`2026-07-10`。流程为生成 → 待审（revision 0）→ 审核通过。

审批原始结果的关键字段：

```json
{
  "status": "published",
  "revision": 0,
  "publish": {
    "status": "sent",
    "message_id": "om_x100b601830f150a8b4f860c0b35e038",
    "error": null,
    "attempts": 1
  }
}
```

服务日志只有一条发送记录：`brief_send attempt=1 status=sent`。未发生重试或多发。

报告路径：`agent/reports/2026-07-10_d74b1d9a876a4de0969c22ea801b6d8c.md`。报告 SHA256：`5f818f68c789b0d03fc99f41b6ad611643bb980fab97f5e42b0691b03c80bbfd`（3785 bytes）。落盘文件与 HTTP 下载内容哈希相同。

重复 approve 原始摘要：

```json
{
  "status": "published",
  "duplicate": true,
  "publish": {
    "status": "sent",
    "message_id": "om_x100b601830f150a8b4f860c0b35e038",
    "error": null,
    "attempts": 1
  }
}
```

重复请求的 message_id、publish 结果与 updated_at 均未变化；未新增发送日志。

## 真实浏览器 golden path

执行：`/home/lx/miniconda3/bin/python /tmp/phase4_browser.py`。
浏览器：Playwright + 本机 Chromium 1228，1440×1100，无头模式；使用实际 Streamlit 页面与安全 Agent。
逐项验证问答、引用和工具展开、生成、编辑修订号 +1、带意见驳回、勾选审核、saved_only、审核按钮消失、浏览器实际 Markdown 下载与哈希。

浏览器脚本曾因错误标签空格、等待瞬时通知、输入框未失焦、隐藏复选框被可见标签覆盖而停止。改用实际「AI助手」标签、持久修订号、Tab 失焦提交与可见复选框标签点击；保留业务断言。未因此修改 Agent 或放宽验收。
原始摘要：

```text
browser_golden_path_ok 95582bd4f2544914853ce973d9344954 saved_only download_sha256 50dcc9bb0aa2478fa949c516840624ffce8993c28d36af089278a4166fac5034
```

截图（临时文件，不入库）：

- `/tmp/phase4_browser/chat.png`
- `/tmp/phase4_browser/workflow_saved_only.png`

浏览器线程：`95582bd4f2544914853ce973d9344954`。报告日期：`2026-07-10`。
状态为 awaiting_review(revision 0) → awaiting_review(1/edit) → awaiting_review(2/reject) → saved_only(2/approve)。
浏览器实际下载：`/tmp/phase4_browser/brief_2026-07-10_95582bd4.md`。
落盘报告：`agent/reports/2026-07-10_95582bd4f2544914853ce973d9344954.md`。
SHA256：`50dcc9bb0aa2478fa949c516840624ffce8993c28d36af089278a4166fac5034`，等于后端报告 hash。
```json
{
  "status": "skipped",
  "message_id": null,
  "error": null,
  "attempts": 0
}
```

## 最终 25 题

README 属于既有语料。文档追加后，仅重建忽略的本地索引，不改检索算法：

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m app.rag.build_index
.venv/bin/python -m app.rag.build_index --check-corpus
.venv/bin/python -m app.rag.build_index --check-golden
HF_HUB_OFFLINE=1 FEISHU_APP_ID= FEISHU_APP_SECRET= FEISHU_CHAT_ID= FEISHU_WEBHOOK_URL= FEISHU_WEBHOOK_SECRET= .venv/bin/python eval/run_eval.py --mode agent
```

原始摘要：

```text
documents=9 chunks=129 backend=hybrid
corpus_ok=9
Q01 PASS ... Q16 PASS
golden_quotes=16/16
准确率 25/25 = 100.0%
工具选择准确率 0.96; 引用支持率 1.0
产物：results_agent.json / report_agent.md / answers_agent.md
```

上列为修复后的最终复跑（15:59:29）。`chunks=129` 因 README 新增真实飞书段落；修复前为 128 时分块。本轮另有 13:45 与 14:41 两次 25/25 复跑，结论一致。

模型 deepseek-flash；执行方式 configured_llm。评测输出时间是脚本生成报告的本地时间，不是 UTC。摘要：

```json
{
  "mode": "agent",
  "run_at": "2026-11-10T15:59:29",
  "total": 25,
  "pass": 25,
  "accuracy": 1.0,
  "latency_p50": 4.14,
  "latency_p95": 20.13,
  "citation_support_rate": 1.0,
  "tool_selection_accuracy": 0.96,
  "verified_rate": 1.0,
  "llm_error_count": 0,
  "citation_note": "按全部题目统计成功工具 source 路径的引用",
  "execution": "configured_llm"
}
```

p95 高于前两次，原因是 G04 单题耗时 51.58s 与 X01 20.13s；25 题全部通过，LLM 错误数为 0。本次未改模型、提示词或工具。

产物：[最终报告](report_agent.md)、[25 题答案](answers_agent.md)、[机器结果](results_agent.json)。引用支持率按成功工具 source 路径统计，不代表全文语义核验。

## 保留与限制

三个 doc eval 文件在开始与结束时 SHA256 相同，保持用户未提交修改，不暂存、不提交：

```text
3fcc499c062c850220b48664e258c9e925451de3179f354800b38a81581522e7  answers_doc.md
30ba4e05e86df7c5855f9a7e4e0cea3a989640492c91df0ce3a8c21aa673c541  report_doc.md
e268d8c550014081104b42b62da61919e869d5be4d4be5375e4a40bb35e5a83b  results_doc.json
```

- 用户施工书、p.txt、两个归档目录与 .claude/ 保留。无 fetch/rebase/merge/push；无 restore/reset/stash/clean。
- 本轮只提交指定 Phase 4 文件；不提交 .env、索引、SQLite、reports、截图、模型缓存和用户未跟踪目录。
- 前端问答历史只存在页面会话；刷新健康检查不清空 PPO 缓存。
- 审核 API 无认证，仅供受信本机。SQLite 仅单进程。公网、Docker、定时任务和回调未验证，均在范围外。
- 真实飞书单卡已验证（见「真实飞书单卡验收」）；Windows 仅静态检查。截图中的部分 emoji 缺字体，中文与关键指标可读。
- Streamlit 现有 use_container_width 有弃用提示；未扩大修改其他 tab。
- [Phase 4 Quiz](phase4_quiz.md) 含 8 题，仅题目。

服务关闭与端口检查见本文件末尾的最终状态。

## 最终状态

首次检查时间：`2026-11-10T13:54:22.746945+08:00`。Agent PID 18892 与 Streamlit PID 19001 收到 SIGINT 并退出。

真实飞书单卡与最终复跑后再次关闭服务。检查时间：`2026-11-10T16:17:25+08:00`。8001/8501 均无监听，无 uvicorn 进程。

```text
{"8001": "connection_refused_and_reusable_bind_ok", "8501": "connection_refused_and_reusable_bind_ok"}
State Recv-Q Send-Q Local Address:Port Peer Address:PortProcess
```

8001/8501 均无监听，连接拒绝；使用 SO_REUSEADDR 的重绑定通过。普通 bind 曾受 TCP TIME_WAIT 影响返回 EADDRINUSE；未将其误报为仍有服务监听。SQLite 正常关闭，wal/shm 临时文件随连接关闭消失。未 push。
