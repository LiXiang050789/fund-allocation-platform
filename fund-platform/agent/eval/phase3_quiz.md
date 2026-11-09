# Phase 3 Quiz（2026-11-09）

1. 简报的 `report_date` 为什么取市场工具的 `as_of`，而不是服务器今天的日期？
2. `POST /workflow/run` 返回 `awaiting_review` 时，图停在哪个节点？此时会保存最终报告或发送飞书吗？
3. `edit` 与 `approve` 对草稿、修订号、审核历史和发布动作分别有什么影响？
4. 连续三次 `reject` 后，工作流分别处于什么状态？中间插入一次 `edit` 会怎样？
5. 飞书三次发送失败时，报告文件和 API 状态分别怎样？再次 approve 如何恢复？
6. 重复 approve 如何防止再次发群？本地锁、终态检查和飞书幂等键各解决什么问题？
7. 服务重启后，待审草稿从哪里恢复？为什么不能用进程内字典代替 SQLite checkpoint？
8. 卡片报告链接使用 localhost 时，手机为什么可能打不开？应怎样配置 `REPORT_URL_BASE`？
