# Phase 3 真实运行证据（2026-11-09）

## 结论

已完成：工作流、人工审核、原子落盘、SQLite重启恢复与真实飞书发布验收。截图未获取，不作为阻塞门槛。
实现与离线测试使用临时报告目录/SQLite和模拟飞书；本节的message_id仅来自真实API成功返回，不混用模拟值。

## 真实线程与状态迁移

- thread_id：`d5d133fed32d4961b3618c59f6a888db`。
- report_date：`2026-07-10`，取本轮市场工具as_of，不用系统今日冒充。
- 初始/重启前GET/重启后GET：`awaiting_review` / `awaiting_review` / `awaiting_review`。
- 重启前后草稿一致：`True`。
- 草稿SHA256：`1ef2d6ca701f84bf0272aa50cb3581f47665a91b876af3bdd3d513c5515bc841`。
- 审核人：`LiXiang`；意见：`同意发布`；审核时间：`2026-11-09T23:00:18+08:00`。
- approve后：`published`；内部publish_status：`sent`；error：无。
- 实际报告路径：`reports/2026-07-10_d5d133fed32d4961b3618c59f6a888db.md`。
- 实际报告SHA256：`033c82af49f452e0e52840d4df097932799710908d6d5300f2dc63b79269f5e3`，再次读取文件重算一致。
- `GET /workflow/d5d133fed32d4961b3618c59f6a888db/report`：HTTP200，`text/markdown; charset=utf-8`，正文与落盘文件一致。
- 审核记录：一次approve、revision=0；重复approve没有新增审核记录或重写报告。

七节标题：

- `## 一、市场状态与评分`
- `## 二、策略表现与相对排名`
- `## 三、风险提示`
- `## 四、模型观点`
- `## 五、数据与口径备注`
- `## 六、免责声明`
- `## 七、审核状态与审核人`

## 飞书真实发布与幂等

- 用户明确授权的唯一测试群：`小波❤小蛙`；通过机器人群列表核对名称与配置目标一致，不输出完整chat_id。
- 配置：用户明确要求写入FEISHU_CHAT_ID；写入前原值为空，没有覆盖不同目标；`.env`权限600且被gitignore。
- message_id：`om_x100b602a4e9568a0cf21d724d71dae1`。
- 发送完成时间：`2026-11-09T23:00:20+08:00`（来源：发布完成 checkpoint updated_at；消息读取未确认：FeishuError）。
- 逻辑发送尝试次数：`1`；真实卡片数：1。
- 重复approve：`duplicate=true`；message_id、publish结果、updated_at均未变化。
- 服务日志仅有一次 `brief_send attempt=1 status=sent` 和一次本线程的publish节点完成记录；重复approve没有第二次发送。
- 幂等键：`brief-2026-07-10-d5d133fed32d4961b3618c59f6a888db`。
- 卡片仅含日期、评分/状态、三模型聚合摘要、风险、报告链接与免责声明，不含逐行原始数据。
- 截图：未获取。未伪造截图或message_id。

最初运行配置没有群ID，自动审批拒绝了目标未知的发布命令；该命令没有执行。用户随后明确授权测试群并要求写入配置。本次发送在核对新授权后执行，没有绕过先前拒绝，没有发送到其他群。

## 验收命令与原始输出摘要

| 命令 | 实际输出/结果 |
|---|---|
| `.venv/bin/python -c "from langgraph.checkpoint.sqlite import SqliteSaver; print('sqlite_ok')"` | `sqlite_ok` |
| `uv pip check --python .venv/bin/python` | `Checked 100 packages in 4ms`；`All installed packages are compatible` |
| `.venv/bin/python -m pytest -q` | `132 passed, 1 warning in 13.66s`；无skip/xfail；原97项保留 |
| `.venv/bin/python -m app.rag.build_index --check-corpus` | `corpus_ok=9` |
| `.venv/bin/python -m app.rag.build_index --check-golden` | `golden_quotes=16/16` |
| `.venv/bin/python eval/run_eval.py --mode agent` | `准确率 25/25 = 100.0%`；工具选择0.96；引用支持1.0；verified1.0；LLM错误0 |
| HTTP run/GET/重启/GET report | `restart_prepare=PASS`；同一线程awaiting_review，草稿一致 |
| HTTP approve/重复approve/报告下载 | `publish_acceptance=PASS`；published、duplicate=true、报告hash一致 |
| HTTP healthz/tools/chat/空问题 | healthz ok；tools=9；chat=33.90/下行/verified=true；空问题=422 |

## 离线测试覆盖

新增35项，合计132项：七节草稿与日期、十策略排序和数值溯源、报告hash、重复/并发审核、编辑仍待审、三次驳回、编辑重置连续驳回计数、飞书失败保存报告与脱敏、同键重试、无配置saved_only、缺失message_id、五种必需工具失败、保存失败不发送、严格字段校验与API状态码、报告下载、临时SQLite关闭后重开并恢复、可选权重缓存、配置repr脱敏。
测试不调用真实DeepSeek或飞书，不写真实用户reports/与checkpoint。现有97项测试及断言保留。

## 文件范围与服务关闭

- 生产checkpoint：`data/checkpoints.sqlite`；报告位于忽略的reports/，二者未入库。
- 六份原eval产物保留为未提交修改，本轮agent回归覆盖也不纳入Phase3提交。
- 四项原有未跟踪内容保留；根.gitignore、chat路由、RAG、工具schema和golden不改动。
- 服务PID：准备/重启 `[86040, 86105]`，发布 `88710`；只关闭本次进程。
- 原始输出：`server_stopped=True port_8001_closed=True`；最终使用ss确认8001无监听。
- 未push。

## 限制

- 精确飞书create_time未获取（读取返回FeishuError）；记录的是发送成功后的持久化完成时间，没有伪称消息原生时间。
- 截图未获取；message_id为主证据。
- 卡片的localhost报告链接仅在本机服务运行时可访问；手机/外网需配置可达的REPORT_URL_BASE。
- 当前为单进程SQLite、本地受信审核API；不提供公网鉴权、分布式锁或卡片按钮回调。
- Webhook不提供应用消息UUID同等的服务端去重保证；本次使用自建应用机器人。

临时原始日志：`/tmp/phase3-pytest.log`、`/tmp/phase3-agent-live.log`、`/tmp/phase3-http-prepare.log`、`/tmp/phase3-http-publish.log`、`/tmp/phase3-http-evidence.json`、`/tmp/phase3-uvicorn.log`。凭证与完整chat_id不写入本证据。
