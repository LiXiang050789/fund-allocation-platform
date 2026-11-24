# Phase 7 P1 平台只读事实覆盖证据（2026-11-24）

已完成：四个只读工具与三项扩展。离线、HTTP假模型与浏览器验收PASS。未push。
本轮未运行真实模型。以下结果不能解释为真实模型质量、延迟或并发验收。

## 基线与范围

master与origin/master开工均为`43fac3b2`。未fetch、未push；本轮只有四个本地提交。
三份doc评测开工已有用户修改；以开工SHA256保护，不还原、不覆盖、不提交。
任务p.txt §5.5已明确要求把/tools断言改为15个完整名称；仅修改这条工具数量/集合断言，其余既有断言保留。
平台app.py、执行脚本、依赖、25/16/12题golden与既有评测产物全部不改。

## PASS/FAIL

| 项目 | 结果 | 验证范围 |
| --- | --- | --- |
| get_strategy_nav | PASS | 实读CSV；最新/精确日期、1/5/6/10日、别名/PPO展开、四策略限制、冲突/倒置/周末/越界、原始市值与基期倍数；HTTP/compose/verify |
| get_ppo_cached_weights | PASS | 最新07-09；请求07-10实际07-09且exact=false；七资产映射、权重和/最小值、非法日期/首日前/坏权重；无推理/子进程 |
| get_pipeline_status | PASS | 真实12项名称/顺序、3项语义、69/20计数、六主表及三特征表形状、Top1、7504预测、五年度IC、AST三常量、缺失/不可读产物与坏AST |
| get_risk_tier_detail | PASS | C2/C3/C4 × 动态/LGB六行、别名/未知值、最终净值及各档静态权重、多档不返回单一配置；原风险工具不变 |
| 状态百分比 | PASS | counts/total/latest/as_of保留；五状态一位小数百分比，合计100.0%；结构化回答同时包含天数和占比 |
| 收益回撤比 | PASS | 指标与比较两工具同字段同值；动态0.9201、LGB0.7983；只问该指标才输出，默认比较四指标保留 |
| 身份与实体 | PASS | 新增净值、缓存权重、就绪/年度IC、最终净值/配置能力；原身份关键词保留；原分类和拒答回归通过 |

## 四个本地提交及文件

路径相对fund-platform/。

| hash | logical unit | 文件 |
| --- | --- | --- |
| `3b6361f0` | 净值与PPO缓存权重 | agent/app/tools/registry.py、agent/tests/test_p1_queries.py |
| `646ad58c` | 流水线与风险档、既有输出扩展 | agent/app/tools/registry.py、agent/tests/test_p1_queries.py |
| `aa92c286` | 路由、证据与错误 | agent/app/graph.py、app/dates.py、app/intent.py、app/orchestrator.py、app/tools/registry.py、tests/test_p1_queries.py、tests/test_graph.py（后六项同属agent/） |
| 本文件所在提交 | 验收与文档 | README.md、agent/README.md、agent/eval/phase7_p1_evidence.md |

第4个hash由`git log -1 --format=%h -- fund-platform/agent/eval/phase7_p1_evidence.md`查询，最终用户汇报列出具体值。所有提交信息都包含本证据路径。
未提交.env、SQLite、索引、缓存、截图或.venv。

## 实现边界与兼容

- 净值是统一回测组合市值，不是单位净值。倍数以该策略2015-01-05实际首值为基数；不硬编码100万元，不与风险档最终净值混用。
- 四策略长区间采用“万”单位，summary增加显式display_nav_10k换算证据；原start_nav/end_nav/max/min完整保留。四策略短区间超过短答预算时明确仅列首末，工具仍返回全部每日行。未指定策略只在短答列期末最高/最低，工具保留十列。
- 中文资产名带300/500/50。工具增加asset_names证据，回答使用分隔符，避免与相邻权重数字拼接。PPO回答明确“历史缓存权重”；当前缓存是历史数据，12.4%等合法权重不冒充旧设计事实。未改OLD_VALUES或verify。
- 流水线global_best.zip显示名与平台保持一致，实际检查models/H_equity_residual_best.zip。仅读取目录/CSV/AST，不导入或运行backtest_unified.py。
- 年度IC与平台同口径：预测/价格按日期合并，未来21交易日收益，七资产各自相关，样本少于2/近零方差为0，再取年度均值。只返回2022–2026。
- 建模表维度问题保留原get_data_caveats，符合覆盖矩阵§17.3第8项，避免原25题D03工具路径改变；其他流水线事实用新工具。
- 权重设计仍查文档；最新PPO权重和规则为docs+data。模型误选评分/文档时，P1数据问题采用确定性expected_data。新工具的回答直接按证据排版，混合问题保留真实文档引用。
- C3“推荐配置每个ETF多少”为平台静态配置。仅对明确C2/C3/C4/中文档位的“推荐配置”短语作静态识别；其余推荐ETF、买卖、保证、全仓仍通过原拒答检查。REFUSAL正则与研究refusal函数不改，回归已覆盖同句恶意请求。
- 新工具失败仅透出固定白名单原因，verified=false。任意异常、后缀和绝对路径不直接回显。

## 测试命令与完整摘要

工作目录：`/home/lx/农行杯/fund-platform/agent`。

```text
# 开工
.venv/bin/python eval/run_offline_tests.py -q
284 passed, 1 warning in 60.17s (0:01:00)

# 最终全量
.venv/bin/python eval/run_offline_tests.py -q
396 passed, 1 warning in 81.67s (0:01:21)

# 新测试数量
.venv/bin/python -m pytest tests/test_p1_queries.py --collect-only -q
112 tests collected in 3.38s

# 路由阶段相关回归
.venv/bin/python eval/run_offline_tests.py tests/test_p1_queries.py tests/test_graph.py tests/test_platform_queries.py tests/test_intent.py -q
248 passed, 1 warning in 16.07s

.venv/bin/python -m app.rag.build_index --check-corpus
corpus_ok=9

.venv/bin/python -m app.rag.build_index --check-golden
Q01至Q16全部PASS
golden_quotes=16/16
```

原284项不减少；新增112项；无skip/xfail。原六项AppTest文件未修改，六项随最终全量全部通过，覆盖五标签页与原预设。
唯一pytest警告为原有LangGraph allowed_objects默认值将变更提示。没有改依赖。
全量入口阻止全部socket外连；旧工作流只使用其原假发送器。新增no_execution fixture拦截subprocess.run/Popen并核对未新增推理模块导入。

首次新增HTTP测试发现：资产名与权重粘连、资产标签数值缺少证据、四策略汇总超长，以及新能力实体缺失。修复实现后通过；未删除或弱化这些新断言。最终又补充C1/C22未知档位与C2不支持PPO策略的HTTP错误回归，防止忽略参数或C22误匹配C2。

原始日志：`/tmp/p1-baseline.log`、`/tmp/p1-tools1-tests.log`、`/tmp/p1-tools2-tests.log`、`/tmp/p1-routing-tests3.log`、`/tmp/p1-full-tests.log`（初轮393项）、`/tmp/p1-final-complete-tests.log`（最终396项）。
审计脚本`/tmp/p1_audit.py`确认原25/16题fallback_calls逐题与43fac3b2一致；verify/numeric_values/strip_dates/score_unit_error/refusal、原get_risk_tier函数AST一致。

README是索引语料，文档更新后使用已有缓存离线重建：
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m app.rag.build_index`。
结果`documents=9 chunks=137 backend=hybrid`，日志`/tmp/p1-index.log`。模型未下载，索引未提交。

## 工具与回答样例

以下为真实本地产物+假模型异常回退，不是真实模型输出。十个样例全部verified=true，均含[source:]，最长281字。

### LGB融合最新净值

工具：`get_strategy_nav`；回答105字。

```text
2026-07-10组合净值（市值）/基期倍数：LGB融合：3399752.14/3.3998×。不构成投资建议。
[source: results/backtest/all_strategy_nav.csv]
```

### 最新PPO权重

工具：`get_ppo_cached_weights`；回答142字。

```text
PPO历史缓存权重日期2026-07-09：沪深300：15.0%、中证500：14.0%、科创50：12.3%、消费：12.4%、芯片：15.2%、黄金：13.2%、十年国债：18.0%；权重和1。不构成投资建议。
[source: results/ppo_weight2.csv]
```

### 2026年7月10日PPO调仓权重

工具：`get_ppo_cached_weights`；回答160字。

```text
请求2026-07-10无当日缓存；PPO历史缓存权重日期2026-07-09：沪深300：15.0%、中证500：14.0%、科创50：12.3%、消费：12.4%、芯片：15.2%、黄金：13.2%、十年国债：18.0%；权重和1。不构成投资建议。
[source: results/ppo_weight2.csv]
```

### 流水线文件就绪几个

工具：`get_pipeline_status`；回答82字。

```text
文件 12/12、就绪校验 3/3、原始 69、清洗 20。不构成投资建议。
[source: data/clean/market_score_daily.csv]
```

### LGB 2022–2026年度IC

工具：`get_pipeline_status`；回答118字。

```text
年度IC：2022年0.2317、2023年0.2707、2024年0.0356、2025年-0.3316、2026年0.0162。不构成投资建议。
[source: results/pred_return_lgb_daily.csv]
```

### C2保守型最终净值

工具：`get_risk_tier_detail`；回答135字。

```text
保守型C2动态评分：年化8.59%、回撤-9.72%、夏普0.5336、最终净值2.44×；保守型C2LGB融合：年化7.90%、回撤-10.26%、夏普0.6263、最终净值2.31×。不构成投资建议。
[source: results/risk_tiers.csv]
```

### C3推荐配置每个ETF多少

工具：`get_risk_tier_detail`；回答193字。

```text
平衡型C3动态评分：年化16.14%、回撤-21.44%、夏普0.5243、最终净值4.48×；平衡型C3LGB融合：年化14.58%、回撤-21.83%、夏普0.6118、最终净值4.18×；静态推荐配置：五个权益ETF（沪深300/中证500/科创50/消费/芯片）各10%，黄金/十年国债各25%。不构成投资建议。
[source: results/risk_tiers.csv]
```

### 全期各状态占比

工具：`get_market_state_summary`；回答142字。

```text
震荡 1317 天（47.1%）；震荡偏强 756 天（27.0%）；下行 580 天（20.7%）；上行 106 天（3.8%）；极寒 39 天（1.4%）；合计 2798 天。不构成投资建议。
[source: data/clean/market_score_daily.csv]
```

### 回测滑点/双边成本/单资产上限

工具：`get_pipeline_status`；回答80字。

```text
回测口径：单边滑点0.05%，双边成本0.10%，单资产上限0.3。不构成投资建议。
[source: scripts/backtest_unified.py]
```

### 十套策略收益回撤比

工具：`get_strategy_metrics`；回答281字。

```text
沪深300：年化收益/最大回撤 0.102；等权：年化收益/最大回撤 0.3596；风险平价：年化收益/最大回撤 0.5791；MVO：年化收益/最大回撤 0.5036；动量：年化收益/最大回撤 0.354；动态评分：年化收益/最大回撤 0.9201；LGB融合：年化收益/最大回撤 0.7983；PPO月度：年化收益/最大回撤 0.3954；SASF硬切换：年化收益/最大回撤 0.622；PPO原生日频：年化收益/最大回撤 0.4017。不构成投资建议。
[source: results/backtest/all_strategy_metrics.csv]
```

原始完整工具数据：`/tmp/p1-samples.json`，生成脚本`/tmp/p1_samples.py`。净值原始值、七权重、形状、语义结果与summary均保留在该JSON中。

## 浏览器 golden path

执行：已有Playwright/Chromium，`python /tmp/p1_browser.py`。
后端：`/tmp/fund_phase7_p1_fake_agent.py`，研究LLM固定抛本地异常，使用真实CSV与确定性回退。无真实模型调用。
浏览器禁止非本机资源请求；后端禁止非本机socket；Streamlit usage telemetry关闭，飞书禁用。

12个断言场景全部通过：

1. 五标签页可渲染：市场仪表盘、自动化数据流水线、配置推荐、AI助手、风险划分。
2. 六题逐项通过：LGB融合最新净值、最新PPO权重、流水线文件就绪几个、LGB 2022–2026年度IC、C2保守型最终净值、全期各状态占比。
3. 浏览器pageerror为空。

六题均检查CSV引用和“数字与引用校验：通过”，即verified=true；回答不含“实时推理”或“今日实时”。未点击执行推理/流水线/报告按钮。

证据目录：`/tmp/fund-phase7-p1-browser/`。

- `results.json`：逐题回答及services_stopped=true。
- `p1-case-1.png`至`p1-case-6.png`：六题截图；已查看年度IC截图，未发现凭证。
- `p0-final.png`：脚本沿用截图文件名，内容为本轮最后一题状态分布，不是P0新验收。
- `health.json`：status=ok、rag_enabled=true、hybrid、rag_error=null。
- `pids.json`、agent.log、platform.log，外层`/tmp/p1-browser.log`。

只监听127.0.0.1:8001与8501。启动前bind确认空闲；finally仅关闭自有Popen进程，随后connect_ex确认两端口无监听；没有kill未知进程。

platform.log无use_container_width警告。仍观察到3次静态资源ASGI异常：
`RuntimeError: Expected ASGI message 'http.response.start', but got 'http.response.body'.`
P0也记录过相同异常。本轮五标签页/六题仍通过，根因未确认。没有为此调整依赖或平台代码，不声称日志零异常。

## 保护、安全与未做

既有eval与三套golden、平台app.py、backtest_unified.py的SHA256均与开工一致。
三份受保护doc文件开工/收工SHA256：

- `results_doc.json`：`e268d8c550014081104b42b62da61919e869d5be4d4be5375e4a40bb35e5a83b`。
- `answers_doc.md`：`3fcc499c062c850220b48664e258c9e925451de3179f354800b38a81581522e7`。
- `report_doc.md`：`30ba4e05e86df7c5855f9a7e4e0cea3a989640492c91df0ce3a8c21aa673c541`。

凭证检查：对本轮变更、日志与JSON精确匹配配置中的真实密钥/secret/token/webhook，结果记录于`/tmp/p1-security.log`；不输出密钥内容。未cat.env、未提交.env。
未修改用户施工书、p.txt、.claude、归档、handoff或其他用户文件；未git add -A、未stash、未还原用户改动。

未做：实时推理、流水线/爬虫/训练/回测执行、报告生成、飞书发送、图表图片生成。仅保留验收截图在/tmp。
未运行run_eval.py、run_chat_eval.py或run_doc_eval.py；原25/16/12题真实模型产物不变。真实模型质量与延迟、并发吞吐未验证。
已知限制：历史数据；无年份日期默认2026；净值超过5日仅摘要，多策略短答按预算给首末；数据就绪按本机产物，不代表训练质量。服务仍单worker、无鉴权，仅供受信本机使用。静态资源ASGI异常根因未确认。

四个提交仅在本地。未push。

## 独立验收补充（2026-11-24，Claude）

结论：P1 通过独立离线验收；发现并修复一处确定性误路由。未 push。

- 独立复跑：`402 passed, 1 warning`（原 396 + 本轮 6 条回归）；`corpus_ok=9`；golden `16/16`。无 skip/xfail。
- 源码对拍：净值末值 3399752.140846137、倍数 3.3998；PPO 权重和 0.999999985；风险档六行数值与 CSV 一致。年度 IC 与 `platform/app.py` 的未来 21 日收益算法一致；预测条数与 `platform/app.py:696` 的 `notna().sum().sum()` 一致；12 项/3 项检查与平台 `_scan_pipeline_assets` 同名同序。
- 缺陷：显式档位 + “最新/当前” + “推荐配置/ETF权重” 被 PPO 分支抢先命中，例如“C2最新推荐配置”返回 PPO 缓存权重。修复见提交 `d915e9f1`：档位静态配置优先于 PPO；同时补“原始文件/清洗CSV/有效预测”同义触发与实体，预测条数只答所问。新增 6 条回归，未弱化既有断言。
- 保护：三份 doc 文件 SHA256 与开工一致；`git diff` 与证据文件对 6 个 `.env` 值零命中；工作树仅本轮两处源码与测试改动（已提交）。
- 未复跑：真实模型、浏览器（沿用本文件上文假模型浏览器证据）。平台日志 3 次静态资源 ASGI 异常仍未定位。
