# Phase 6 实现与验收证据（2026-11-23）

状态：已完成本轮约定范围。用户补充授权后，研究25题、通用12题与真实模型浏览器8场景通过。原文档16题按任务保留既有基线，由验收方隔离复跑。

## 基线与提交

- 基线 master：e40eee89；任务要求只做本地提交，不 push。
- 后端提交：3a4b21b6。四个模块、main.py、四份新测试、离线测试入口与 SQLite 忽略规则。
- 前端提交：19a772cf。agent_client.py、app.py 的 tab4、test_frontend.py。
- 文档/评测提交：包含本文件的提交；用 git log -1 --format=%h -- 此文件查询。
- 三个受保护 doc eval 文件开工时已有用户修改，本轮未还原、未提交、未覆盖。
- p.txt、施工书、.claude/、归档/、fund-platform-handoff-2026-11-19/ 保持原状。

## 验收表

PASS 只表示表内注明的方法和范围。文档16题为既有基线保留，不声称本轮重跑。

| 项 | 结果 | 范围与证据 |
| --- | --- | --- |
| 原 25 题 | PASS | results_agent.json 为本轮 configured_llm：25/25、工具0.96、引用1.0、verified1.0、模型错误0 |
| 文档 16 题 | PASS（基线） | 现有 results_doc.json 为16/16；本轮不运行 run_doc_eval.py；SHA256保持开工时内容；原文引用预检16/16 |
| chat 12 题 | PASS | results_chat.json 为本轮 configured_llm、12/12；逐条模式/关键词/契约检查通过 |
| SSE 契约 | PASS | tests/test_stream.py：meta在先、chat有token、research无token、final字段、error无半轮记忆 |
| 前端 golden path | PASS | AppTest及假LLM原覆盖保留；真实模型+Chromium 8场景见 /tmp/fund-phase6-live-browser/results.json |
| 多会话隔离 | PASS | test_conversation.py、test_stream.py；浏览器新建、切换、刷新、清空 |
| 拒答与合规 | PASS（确定性范围） | test_intent.py、test_chat.py；拒答、句级守卫、提示注入固定安全答复；真实12题及8场景通过，不代表穷尽合规验证 |
| 无密钥/无越权 | PASS（检查范围） | 新暂存内容与本轮日志对实际配置密钥做精确匹配扫描；研究工具白名单未改；服务仅本机 |
| 安全模式 | PASS | 本机验收 AGENT_FEISHU_DISABLED=1；首轮禁止外连，补充授权后仅调用现有DeepSeek评测；未调用飞书 |
| Git 状态 | PASS（纪律范围） | 两个实现提交及本文件提交均为本地；git diff --check、受保护哈希检查；无push |

## 测试命令与结果

```text
.venv/bin/python eval/run_offline_tests.py -q
191 passed, 1 warning in 36.24s
```

原有135项未减少；新增56项。原始完整输出：/tmp/fund-phase6-final-tests.log。
首次用全局 AGENT_FEISHU_DISABLED=1 跑旧测试，8项模拟发布测试因预期 published、实际 saved_only 失败。
未修改这些测试或发布逻辑。离线入口先禁止全部 socket 外连，再只在测试进程内给假发送器测试提供普通配置。
最初隔离基线为135 passed, 1 warning in 15.44s。当前唯一 pytest warning 来自原有 LangGraph 默认值弃用提示。
AppTest 单独结果：4 passed in 22.43s，原始输出 /tmp/fund-phase6-frontend-tests.log。

## 首轮浏览器与 HTTP（假LLM）

- 执行器：已安装的 Playwright + Chromium；无新增依赖。
- 模型：假LLM；研究数据使用真实本地只读工具与确定性回退。
- 服务：127.0.0.1:8001、127.0.0.1:8501；记忆放 /tmp/fund-phase6-browser/conversations.sqlite。
- 8场景：自我介绍、你好、今天几号、金融常识、平台数据、多轮追问、投资建议拒答、空输入。
- 额外3项：刷新/隔离、清空、停止；全部通过，最终模式标签也逐条检查。
- 本机 HTTP：动态评分夏普0.5465；“那LGB呢”0.6462；再追问PPO获得月度/日频指标；均 research/verified=true。
- 显式空 thread_id 返回422；身份回答 chat/verified=null；回撤缺条件先澄清。
- 输入框在列内可用，保留st.chat_input，未切换兜底输入控件。
- 停止按钮使用原生submit_mode=stop；关闭SSE，服务端未存储未完成轮次。未完成标记在下次脚本重跑恢复。
- 截图：/tmp/fund-phase6-browser/case-1.png 至 case-7.png（不入Git）。
- 原始浏览器脚本：/tmp/fund_phase6_browser.py；最终日志：/tmp/fund-phase6-browser/walkthrough-final.log。
- 日志：agent.log、agent-final.log、platform.log；实际密钥精确扫描通过。

healthz 最终 JSON（/tmp/fund-phase6-browser/health-final.json）：

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

本轮记录的Agent和平台PID均用SIGTERM关闭；关闭前核对命令行属于本轮服务。8001/8501结束后无监听。
/tmp/fund-phase6-browser/ports-after.txt为空。

## 语料与保护检查

```text
corpus_ok=9
documents=9 chunks=132 backend=hybrid
golden_quotes=16/16
```

两份README更新使原索引过期，因此使用既有模型缓存离线重建；未提交索引或缓存。
研究图、两套原golden和三个受保护文件均通过开工时SHA256复核。
其余四标签页与e40eee89做AST对比，节点完全相同；只有AI助手tab4和其客户端导入变动。

## 未验证与限制

- 首轮自动审批因数据外传授权未确认而拒绝真实评测；用户随后明确授权现有DeepSeek评测，阻塞解除。
- 首次拒绝时命令未执行；补充授权后正常执行研究与通用真实评测，更新非受保护的agent/chat评测产物。
- 单次样本的首token和研究p50/p95见下文；并发、持续压力与其他网络环境未验证。
- 句级守卫会等句子完成才发送字符，首token可能晚于原生模型流。离线报告耗时不代表真实模型性能。
- SQLite目前串行化整个轮次；并发吞吐未验证。API仍无用户登录/所有权鉴权，只用于受信本机。
- 原生停止后的本地未完成提示在下一次脚本重跑出现；服务端取消无半轮写入已有测试/浏览器证据。
- 没有公网部署，没有开启飞书，没有push。

## 补充授权后的真实模型验收

用户明确回复“授权现有 DeepSeek 评测”。只向现有配置服务发送固定评测问题和检索片段；密钥未输出。

```text
AGENT_FEISHU_DISABLED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python eval/run_eval.py --mode agent
准确率 25/25 = 100.0%
工具选择准确率 0.96; 引用支持率 1.0
verified_rate=1.0; llm_error_count=0
latency_p50=3.69s; latency_p95=6.55s

AGENT_FEISHU_DISABLED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python eval/run_chat_eval.py
execution=configured_llm; total=12; passed=12
```

- 原始研究日志：/tmp/fund-phase6-research-live.log；报告：report_agent.md、results_agent.json、answers_agent.md。
- 原始通用日志：/tmp/fund-phase6-chat-live.log；报告：report_chat.md、results_chat.json、answers_chat.md。
- 三道实际调用通用模型的常识题C05/C06/C07首token为2.379/1.749/1.257秒；模板题不计作模型性能样本。
- 12题全部通过，模式误判0。数据追问保留research；非研究verified=null、无工具/引用。
- 原研究基线工具0.96、引用1.0、verified1.0保持不降；p95从旧基线20.13秒到本轮6.55秒，仅为各次样本统计。
- 真实模型浏览器：/tmp/fund-phase6-live-browser/results.json，execution=real_browser_configured_llm，固定8场景全部通过。
- 浏览器脚本：/tmp/fund_phase6_browser_live.py；日志：/tmp/fund-phase6-live-browser/walkthrough.log；截图：case-1.png至case-7.png，不入Git。
- 真实服务使用/tmp/fund-phase6-live-browser/conversations.sqlite，与用户记忆隔离；飞书始终关闭。
- 关闭前healthz：status=ok、rag_enabled=true、rag_backend=hybrid；原始JSON见/tmp/fund-phase6-live-browser/health-final.json。
- 按记录的PID关闭真实验收Agent和平台，结束后8001/8501无监听；ports-after.txt为空。
- 真实调用只属于评测，没有新增依赖、研究逻辑改动、公网部署或Git推送。

## 下一步

文档16题仍由验收方在隔离目录复跑，不覆盖本工作区三个受保护文件。
继续保留单worker、本机访问和安全模式。并发吞吐、跨网络性能和公网鉴权不在本轮验收范围。
