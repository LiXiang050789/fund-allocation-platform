# Phase 7 P0 实现与验收证据（2026-11-24）

已完成：策略对比与每日评分区间两项 P0。离线与假模型浏览器验收 PASS。未 push。
真实模型回归未运行，不能把本轮结果解释为真实模型质量验收。

## 基线、授权与本地提交

基线：master / origin/master = `5f102972`。本轮不 fetch、不联网评测、不 push。
开工时三份 doc 评测已有用户修改；以开工 SHA256 为保护基线，未还原、未覆盖、未提交。
用户补充授权：“授权修改这一条断言，校验完整的 11 个工具名称。继续完成全部任务”。
仅修改 `tests/test_graph.py` 原工具数量断言：校验数量 11，并校验全部名称集合；其余既有断言不变，无 skip/xfail。

| 提交 | 内容 | 文件（均相对 fund-platform/） |
| --- | --- | --- |
| `8a0d1a0a` | Phase 6 句末标点 | agent/app/text.py、agent/app/graph.py、agent/app/chat.py、agent/tests/test_text.py |
| `97540d26` | Phase 6 width 迁移 | platform/app.py、agent/tests/test_frontend.py |
| `79546872` | Phase 7 P0 工具与查询 | agent/app/tools/registry.py、agent/app/graph.py、agent/app/dates.py、agent/app/intent.py、agent/app/orchestrator.py、agent/tests/test_platform_queries.py、agent/tests/test_graph.py |
| 本文件所在提交 | Phase 7 P0 预设与文档 | platform/app.py、agent/tests/test_frontend.py、agent/README.md、README.md、agent/eval/phase7_p0_evidence.md |

第4个提交 hash：执行 `git log -1 --format=%h -- fund-platform/agent/eval/phase7_p0_evidence.md` 获取；最终用户汇报提供具体值。
四个提交信息均带证据路径。没有提交 SQLite、索引、截图、缓存或 .venv。

## 改动范围与验收结论

| 项目 | 结果 | 方法与范围 |
| --- | --- | --- |
| 策略对比 | PASS | CSV 实读；别名、展开、去重、顺序、缺行、未知与数量边界；假模型故意误选最新评分，路由仍选 compare_strategies |
| 每日评分区间 | PASS | 中文/斜线/ISO、省略年、四种分隔符；1/5/6/10/11交易日；周末、无效、倒置与越界 |
| 短答完整性 | PASS | 两个新查询使用结构化证据排版；默认四项指标与所有每日行保留；引用与免责声明计入300字 |
| 日期澄清 | PASS | 新日期错误经HTTP/SSE统一封装为明确澄清且verified=false；不返回末日评分冒充区间回答 |
| 裸日期区间 | PASS | intent仅新增日期区间research识别；“7月8日到7月10日”实际HTTP与浏览器通过 |
| 标点 | PASS | 研究/通用共享纯函数；12种句末、空正文、双suffix、footer与三条旧正文精确回放；300/400字预算 |
| width | PASS | 实际Streamlit1.60四类签名均支持width；24处True改stretch；原来没有False；AppTest与五标签页浏览器通过 |
| 预设 | PASS | 实际点击“LGB融合与PPO对比”，返回LGB融合、PPO月度、PPO原生日频指标；布局不变 |
| 原校验政策 | PASS（静态） | verify、numeric_values、strip_dates、score_unit_error、refusal与5f102972的函数AST完全相同 |
| 原基线 | PASS（确定性范围） | 原25/16题的data_fallback_calls逐题与5f102972相同；既有评测与golden全部SHA256不变；不声称真实模型重跑 |

intent.py 和 orchestrator.py 的小范围接入属于日期任务：前者识别裸日期区间，后者保留已知日期错误的澄清，避免旧NO_EVIDENCE覆盖具体原因。未知异常仍使用原安全兜底。
新比较/历史工具的确定性格式化不覆盖原拒答政策；新增回归已验证这一点。

## 测试命令与原始摘要

工作目录：`/home/lx/农行杯/fund-platform/agent`。

```text
# 开工基线
.venv/bin/python eval/run_offline_tests.py -q
191 passed, 1 warning in 41.01s

# 最终全量：阻止全部socket外连；旧工作流仅使用原假发送器
.venv/bin/python eval/run_offline_tests.py -q
276 passed, 1 warning in 58.11s

# 任务指定五文件回归（环境中禁用飞书、HF与Transformers联网）
.venv/bin/python -m pytest tests/test_graph.py tests/test_rag.py tests/test_chat.py tests/test_frontend.py tests/test_platform_queries.py -q
166 passed, 1 warning in 42.74s

.venv/bin/python -m app.rag.build_index --check-corpus
corpus_ok=9

.venv/bin/python -m app.rag.build_index --check-golden
golden_quotes=16/16
```

原191项全部保留。新增85项：test_text.py 21项、test_platform_queries.py 62项、test_frontend.py 2项。
最终AppTest 6项均随全量与五文件命令通过。width提交时单独AppTest为5 passed in 24.88s；之后新增预设用例。
唯一pytest warning是原有LangGraph `allowed_objects` 默认值将变更提示，未改依赖。
新增测试开发时曾发现“明天”触发原拒答政策；日期解析失败测试采用非相对日期的非法中文端点，保留原拒答语义。

原始输出：`/tmp/phase7-tests-full.log`（首次274项）；`/tmp/phase7-tests-final.log`（最终276项）；`/tmp/phase7-tests-targeted.log`（166项）。
路由/AST/保护复核脚本：`/tmp/phase7_audit.py`。样例脚本和原始工具响应：`/tmp/phase7_validate.py`、`/tmp/phase7-samples.json`。

两份README属于索引语料。文档更新后使用现有缓存离线重建：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m app.rag.build_index`。
结果：`documents=9 chunks=134 backend=hybrid`。未下载模型，未提交索引。构建日志说明HF离线并使用本地缓存。

## 工具与回答样例

以下内容来自本轮真实CSV，模型为本地故障假实现，研究链路使用确定性证据回退。

### LGB 与动态评分对比

路由：`compare_strategies`；verified=true；总长145字。

```text
LGB融合：年化11.99%、夏普0.6462、回撤-15.02%、卡玛0.7977；动态评分：年化13.13%、夏普0.5465、回撤-14.27%、卡玛0.9204。不构成投资建议。
[source: results/backtest/all_strategy_metrics.csv]
```

### LGB融合与PPO对比

路由：`compare_strategies`；verified=true；总长191字。

```text
LGB融合：年化11.99%、夏普0.6462、回撤-15.02%、卡玛0.7977；PPO月度：年化14.36%、夏普0.4728、回撤-36.32%、卡玛0.3953；PPO原生日频：年化14.59%、夏普0.4803、回撤-36.32%、卡玛0.4016。不构成投资建议。
[source: results/backtest/all_strategy_metrics.csv]
```

### 查询2026年7月8日到7月10日，每天的市场评分

路由：`get_market_score_history`；verified=true；总长186字。

```text
括号依次为估值/宏观/情绪/趋势：2026-07-08 32.63下行(0/14.08/10.65/7.91)；2026-07-09 39.43下行(0/14.08/10.65/14.71)；2026-07-10 33.9下行(0/14.08/10.65/9.18)。不构成投资建议。
[source: data/clean/market_score_daily.csv]
```

实际工具数据（分项未重算）：

```json
{
  "mode": "detail",
  "total_days": 3,
  "rows": [
    {
      "date": "2026-07-08",
      "total_score": 32.63,
      "market_state": "下行",
      "val_score": 0.0,
      "macro_score": 14.08,
      "sent_score": 10.65,
      "trend_score": 7.91
    },
    {
      "date": "2026-07-09",
      "total_score": 39.43,
      "market_state": "下行",
      "val_score": 0.0,
      "macro_score": 14.08,
      "sent_score": 10.65,
      "trend_score": 14.71
    },
    {
      "date": "2026-07-10",
      "total_score": 33.9,
      "market_state": "下行",
      "val_score": 0.0,
      "macro_score": 14.08,
      "sent_score": 10.65,
      "trend_score": 9.18
    }
  ]
}
```

裸输入“7月8日到7月10日”与完整日期问题得到相同三行及186字回答。
工具样例的卡玛数值直接取CSV；未采用任务中的示意数值替换真实指标。

## 浏览器与日志

执行器：已有Playwright + Chromium，未安装新依赖。
执行：`python /tmp/phase7_browser.py`；模型由 `/tmp/fund_phase7_fake_agent.py` 替换为本地假实现。
Agent禁止非本机socket连接，AGENT_FEISHU_DISABLED=1，未发送飞书。Streamlit关闭usage telemetry。
实际服务只监听127.0.0.1:8001和127.0.0.1:8501；会话数据库在/tmp独立目录。
启动前逐端口bind检查空闲；结束finally按自有Popen进程关闭；两端口connect_ex确认无监听。没有kill未知进程。

13个断言场景通过：五个标签页、预设对比、手输对比、完整中文三天查询、裸中文区间、周末、越界、倒置、浏览器pageerror为空。
这包含7个问题场景；成功的两项P0回答均显示“证据回答”与数字/引用校验通过。
日期错误显示明确原因与verified=false；前端沿用原有校验未通过提示。

原始证据：

- `/tmp/fund-phase7-browser/results.json`：逐场景文本及services_stopped=true。
- `/tmp/fund-phase7-browser/health.json`：status=ok、rag_enabled=true、hybrid、rag_error=null。
- `/tmp/fund-phase7-browser/p0-final.png`：最终倒置区间澄清页，未入Git。
- `/tmp/fund-phase7-browser/pids.json`、agent.log、platform.log；`/tmp/phase7-browser.log`。

width警告：platform.log中没有use_container_width。
**已知日志异常**：浏览器期间platform.log出现3次静态资源响应的ASGI异常：
`RuntimeError: Expected ASGI message 'http.response.start', but got 'http.response.body'.`
五页与查询场景仍通过，浏览器pageerror为空。异常根因未确认。本轮未改依赖或服务框架，不声称日志零异常。

全仓库rg搜索仍有归档旧平台快照147处旧参数；它们属于禁止修改目录，未修改。当前运行platform/app.py为零处。
其余3处文本命中是test_frontend.py两个检查及旧phase4_evidence.md记录，不是运行控件调用。
搜索原始输出：`/tmp/phase7-width-search.txt`。

## 保护与安全检查

新内容与本轮日志/JSON对配置中真实凭证做精确匹配扫描：`secret_scan=PASS files=27`。未输出凭证。
截图已目视检查，页面仅含测试问题和公开CSV数据。未打印或提交.env。
所有既有eval文件（包含25/16/12题产物和三套golden）SHA256均与开工时一致。
三个受保护文件的开工SHA256：

- `results_doc.json`：`e268d8c550014081104b42b62da61919e869d5be4d4be5375e4a40bb35e5a83b`。
- `answers_doc.md`：`3fcc499c062c850220b48664e258c9e925451de3179f354800b38a81581522e7`。
- `report_doc.md`：`30ba4e05e86df7c5855f9a7e4e0cea3a989640492c91df0ce3a8c21aa673c541`。

未修改或提交p.txt、施工书、.claude、归档、handoff和其他用户文件。未git add -A、未stash、未还原用户改动。

## 未做与限制

- 未运行run_doc_eval.py、真实run_eval.py、真实run_chat_eval.py。原25/16/12题模型产物未改；真实模型质量、延迟和并发回归未验证。
- 未实现P1净值、PPO权重、流水线IC。
- 区间超过10个交易日仅汇总；缩小范围后查看每日明细。年份省略固定采用当前数据最新年份2026。
- 原服务仍限制单worker；无用户鉴权或会话所有权隔离；仅用于受信本机。
- 静态资源ASGI异常根因未确认，详见日志段。未执行依赖升级或降级。
- 四个提交仅保存在本地。未push。

## 验收方补充（2026-11-24）

- 我独立复跑原 276 项离线测试：`276 passed, 1 warning in 48.37s`。
- 额外发现一项 P0 遗漏：`截至2026年7月10日的市场评分`、`截止到2026-07-10的评分`被 `range_requested` 当成区间，返回“无法解析日期区间”，而不是单日评分。
- 修复：`app/dates.py` 在区间判断和日期解析前剥离“截至/截止/截止到”语义；新增 5 个单日解析用例和 3 个路由用例。全量复跑 `284 passed, 1 warning in 48.24s`。
- 假模型证据回放验证：上述“截至/截止”问题均命中 `get_market_score`，返回 2026-07-10 总分 33.9、状态下行，`verified=true`；原对比与三日区间仍通过。
- 真实模型未运行；P1 未实施；本轮新增修复提交 `0f7d03ec`，仍未 push。
