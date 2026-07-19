# Codex CLI + cc-switch + DeepSeek 配置指南

> 适用场景：在国内网络环境下，通过 Codex CLI 使用 DeepSeek API 进行 AI 辅助编程。
> 已知良好版本：Codex v0.142.4 + cc-switch + DeepSeek v4-pro

## 架构

```
codex CLI → http://127.0.0.1:15721/v1 → cc-switch 代理 → https://api.deepseek.com
```

cc-switch 是一个本地代理工具，监听 `127.0.0.1:15721`，将 Codex 的 API 请求转发到 DeepSeek，同时自动管理 Codex 的认证和模型配置。

## 安装

1. 安装 Codex CLI：`npm install -g @openai/codex`
2. 安装 cc-switch：从其官方渠道获取 Linux 版本
3. 获取 DeepSeek API Key：在 platform.deepseek.com 注册并创建 API Key

## 关键配置文件

### ~/.codex/config.toml（Codex 配置）

```toml
wire_api = "responses"
experimental_bearer_token = "PROXY_MANAGED"
base_url = "http://127.0.0.1:15721/v1"
openai_base_url = "http://127.0.0.1:15721/v1"
model = "deepseek-v4-pro"
model_catalog_json = "/home/lx/.codex/cc-switch-model-catalog.json"

[projects]
[projects."/home/lx"]
trust_level = "trusted"

[projects."/home/lx/your-project-dir"]
trust_level = "trusted"

[features]
responses_websockets = false
responses_websockets_v2 = false
```

关键说明：

- `openai_base_url` 和 `base_url` **两个键都要写**。cc-switch 接管时只写 `base_url`，但 Codex 实际读取的是 `openai_base_url`。只写 `base_url` 会导致 Codex 绕过代理直连 OpenAI（被墙，超时）。
- `model_catalog_json` 指向 cc-switch 生成的模型目录文件，缺少它会导致 "Model metadata not found" 警告，影响性能。
- `[features]` 段禁用 WebSocket 的原因是 cc-switch 代理不支持 WebSocket 协议升级，不禁用会导致每次请求先等 WebSocket 超时再回退 HTTPS，浪费约 1 分钟。

### ~/.codex/auth.json（API Key）

```json
{
  "OPENAI_API_KEY": "sk-你的deepseek-api-key",
  "auth_mode": "apikey"
}
```

`OPENAI_API_KEY` 的值就是 DeepSeek 的 API Key。

### ~/.cc-switch/settings.json（cc-switch 关键设置）

```json
{
  "enableLocalProxy": true,
  "proxyConfirmed": true,
  "visibleApps": {
    "codex": true
  }
}
```

## 首次启动流程

1. 启动 cc-switch（它会自动接管 Codex 配置，生成模型目录，启动本地代理）
2. 确认 cc-switch 代理端口正常：

```bash
curl -s -w "\nHTTP %{http_code}\n" --max-time 10 \
  http://127.0.0.1:15721/v1/responses \
  -H "Authorization: Bearer PROXY_MANAGED" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-pro","input":"hi"}'
```

返回 HTTP 200 即为正常。

3. **重要：** 检查 cc-switch 接管后 config.toml 是否完整，被覆写后需手动补全（见上方模板），然后**立即锁定文件**：

```bash
chmod 444 ~/.codex/config.toml
```

这一步是防止日后 cc-switch 重启时再次覆写配置的关键。需要手动编辑时先 `chmod 644 ~/.codex/config.toml` 解锁。

4. 在 VSCode 中 `Ctrl+Shift+P` → `Developer: Reload Window`，或终端直接运行 `codex`。

## 常见问题与修复

### 1. Connection refused (os error 111)：cc-switch 没有运行

**症状：**

```
⚠ Falling back from WebSockets to HTTPS transport. stream disconnected before
  completion: Connection refused (os error 111)
```

**原因：** cc-switch 进程未启动（升级/重启后最常见）。

**修复：**

```bash
# 杀掉残留进程
pkill cc-switch

# 重启（后台运行）
nohup cc-switch > /dev/null 2>&1 &

# 验证端口
ss -tlnp | grep 15721
```

### 2. cc-switch 运行但代理不工作：enabled 字段被重置

**症状：** cc-switch 启动正常但端口 15721 没有监听。

**原因：** 反复强杀 cc-switch 后，数据库 `proxy_config` 表中 `enabled` 字段可能被重置为 0。

**修复：**

```bash
sqlite3 ~/.cc-switch/cc-switch.db \
  "UPDATE proxy_config SET enabled=1 WHERE app_type='codex';"
```

然后重启 cc-switch。

### 3. 多个 cc-switch 进程冲突

**症状：** 端口时而可用时而不可用，行为不稳定。

**原因：** 反复启动导致多个 cc-switch 进程共存，争抢端口和配置。

**修复：**

```bash
pkill cc-switch
sleep 1
cc-switch &
```

保证只有一个实例。

### 4. "Model metadata not found" + WebSocket 超时：config.toml 被覆写

**症状：**

```
⚠ Model metadata for `deepseek-v4-pro` not found. Defaulting to fallback metadata
⚠ Falling back from WebSockets to HTTPS transport. request timed out
```

**原因：** 这是最隐蔽的问题。cc-switch 每次执行"Live 接管"时会**覆写 config.toml**，导致三项关键配置丢失：

- `openai_base_url` 被替换为 `base_url`（Codex 不认这个键名）
- `model_catalog_json` 被删除
- `[features]` 段被删除

**修复：** 手动补全 config.toml（参考上方完整模板）。

验证修复：

```bash
codex doctor 2>&1 | grep -E "websocket|reachability"
```

### 5. cc-switch 窗口不显示（WSL/Linux）

**症状：** 日志显示"主窗口已显示"，但实际上看不到窗口。

**原因：** cc-switch 默认 `showInTray: true` + `minimizeToTrayOnClose: true`，启动后缩到系统托盘。在 WSLg 环境下，托盘图标可能不显示。

**修复：**

在 `~/.cc-switch/settings.json` 中设置：

```json
"showInTray": false,
"minimizeToTrayOnClose": false
```

重启 cc-switch 后窗口会直接显示。

## 排查命令速查

```bash
# 1. 确认 cc-switch 进程
pgrep -a cc-switch

# 2. 确认代理端口
ss -tlnp | grep 15721

# 3. 测试代理连通性（responses API）
curl -s -w "\nHTTP %{http_code}\n" --max-time 10 \
  http://127.0.0.1:15721/v1/responses \
  -H "Authorization: Bearer PROXY_MANAGED" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-pro","input":"hi","max_output_tokens":5}'

# 4. 检查代理配置状态
sqlite3 ~/.cc-switch/cc-switch.db \
  "SELECT app_type, proxy_enabled, enabled FROM proxy_config WHERE app_type='codex';"
# 期望: codex|1|1

# 5. 检查 Codex 配置
codex doctor 2>&1 | head -30

# 6. 查看 cc-switch 日志
tail -50 ~/.cc-switch/logs/cc-switch.log | grep -E "proxy|SRV|接管|15721"
```

### 6. config.toml 被反复覆写（根治方案）

**症状：** 明明刚修好 config.toml，过一阵子（特别是 cc-switch 重启后）问题 4 又复现了。

**原因：** cc-switch 每次启动都会执行"Live 接管"，覆写 config.toml 为自己的最小版本。这不是一次性事件，而是**每次 cc-switch 启动都会发生**。

**根治方法：** 将正确的 config.toml 写入后，锁定为只读：

```bash
chmod 444 ~/.codex/config.toml
```

cc-switch 接管时无法写入只读文件，config.toml 始终保持正确内容。需要手动修改时先解锁：

```bash
chmod 644 ~/.codex/config.toml
# 编辑完成后再次锁定
chmod 444 ~/.codex/config.toml
```

## 已知限制

- cc-switch 代理不支持 WebSocket，必须通过 `[features]` 禁用以避免超时
- Linux/WSL 上 cc-switch 的 `live_takeover_active` 始终为 0（不影响实际使用）
- Codex 版本更新后，config.toml 键名可能变化，需关注 `codex doctor` 输出
- cc-switch 每次接管会覆写 config.toml，**正确的 config.toml 写入后必须 `chmod 444` 锁定防止覆写**；手动编辑时先 `chmod 644` 解锁
