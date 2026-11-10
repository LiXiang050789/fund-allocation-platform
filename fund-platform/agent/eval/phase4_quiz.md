# Phase 4 Quiz（2026-11-10）

1. 为什么 Streamlit 只通过 HTTP 请求 Agent？两个进程分别使用哪个端口和 Python 环境？
2. `/chat` 的 `thread_id` 与页面问答历史有什么区别？切换模式与关闭页面分别怎样影响历史？
3. 客户端怎样区分连接失败、HTTP 错误与无效 JSON？哪些诊断信息不得显示在页面？
4. 为什么健康检查使用短 TTL 和 nonce？为什么不得清空全部 Streamlit 缓存？
5. 草稿 widget 的 key 为什么包含线程、修订号和状态？编辑与驳回分别怎样改变审核状态？
6. `saved_only`、`publish_failed`、`published` 分别证明什么？为什么重复审批必须检查幂等？
7. 本轮怎样禁用真实飞书发送？为什么 Phase 3 的授权不能复用于 Phase 4？
8. 如何核对下载报告与落盘报告的 SHA256？为什么 SQLite checkpoint 必须使用单 worker？
