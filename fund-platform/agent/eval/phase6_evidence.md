# Phase 6 实现与验收证据（2026-11-23）

状态：部分完成。三个阶段的本地实现、离线测试、浏览器交互完成；真实模型评测因外传授权未确认而未执行。

## 基线与提交

- 基线 master：e40eee89；任务要求只做本地提交，不 push。
- 后端提交：3a4b21b6。四个模块、main.py、四份新测试、离线测试入口与 SQLite 忽略规则。
- 前端提交：19a772cf。agent_client.py、app.py 的 tab4、test_frontend.py。
- 文档/评测提交：包含本文件的提交；用 git log -1 --format=%h -- 此文件查询。
- 三个受保护 doc eval 文件开工时已有用户修改，本轮未还原、未提交、未覆盖。
- p.txt、施工书、.claude/、归档/、fund-platform-handoff-2026-11-19/ 保持原状。

## 验收表

PASS 只表示表内注明的方法和范围。FAIL 表示验收条件尚未满足，不表示执行了一个失败的真实模型评测。

| 项 | 结果 | 范围与证据 |
| --- | --- | --- |
| 原 25 题 | FAIL | 现有 results_agent.json 基线 25/25、工具0.96、引用1.0、verified1.0；本轮真实回归未执行 |
| 文档 16 题 | PASS（基线） | 现有 results_doc.json 为16/16；本轮不运行 run_doc_eval.py；SHA256保持开工时内容；原文引用预检16/16 |
| chat 12 题 | FAIL（真实未验证） | results_chat.json 为 offline_fake_llm、12/12；不能当作真实模型质量 |
| SSE 契约 | PASS | tests/test_stream.py：meta在先、chat有token、research无token、final字段、error无半轮记忆 |
| 前端 golden path | PASS（假LLM） | tests/test_frontend.py AppTest；真实 Chromium 8场景，见 /tmp/fund-phase6-browser/results.json |
| 多会话隔离 | PASS | test_conversation.py、test_stream.py；浏览器新建、切换、刷新、清空 |
| 拒答与合规 | PASS（确定性范围） | test_intent.py、test_chat.py；拒答、句级守卫、提示注入固定安全答复；真实模型覆盖未验证 |
| 无密钥/无越权 | PASS（检查范围） | 新暂存内容与本轮日志对实际配置密钥做精确匹配扫描；研究工具白名单未改；服务仅本机 |
| 安全模式 | PASS | 本机验收 AGENT_FEISHU_DISABLED=1、假LLM、禁止Agent外连；未调用飞书 |
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

## 浏览器与 HTTP

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

- 自动审批拒绝真实DeepSeek评测：问题、工具证据和检索文档片段会发送到配置的外部服务，数据外传授权未确认。
- 上述拒绝发生在命令执行前；本轮未因此调用真实模型，未覆盖原25题结果。
- chat的3秒首token目标、真实模型p50/p95、完整真实模型端到端链路未验证。
- 句级守卫会等句子完成才发送字符，首token可能晚于原生模型流。离线报告耗时不代表真实模型性能。
- SQLite目前串行化整个轮次；并发吞吐未验证。API仍无用户登录/所有权鉴权，只用于受信本机。
- 原生停止后的本地未完成提示在下一次脚本重跑出现；服务端取消无半轮写入已有测试/浏览器证据。
- 没有公网部署，没有开启飞书，没有push。

## 后续授权与复验

1. 推荐：明确授权固定评测问题及检索片段发送到现有配置的DeepSeek服务，再运行研究25题与chat12题真实评测。
2. 或仅接受当前离线实现，保留真实质量/性能未验证标记，交由验收方运行。
3. 文档16题仍由验收方在隔离目录复跑，不覆盖本工作区三个受保护文件。

获得授权后使用安全模式和既有密钥配置；不打印.env。默认run_chat_eval.py为真实模型模式，--offline为明确的测试替身模式。
