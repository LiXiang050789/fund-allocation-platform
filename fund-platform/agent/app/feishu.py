"""飞书通知：自建应用机器人优先，群自定义机器人 Webhook 兜底。

只发聚合指标和报告链接，不发原始数据。
失败重试 3 次；推送失败不阻塞报告落盘（由调用方保证）。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any

import requests

from .config import settings

log = logging.getLogger(__name__)

FEISHU_BASE = "https://open.feishu.cn/open-apis"
_TOKEN_INVALID_CODES = {99991661, 99991663, 99991668}


class FeishuError(RuntimeError):
    def __init__(self, code: int, msg: str, detail: Any = None):
        super().__init__(f"feishu code={code} msg={msg}")
        self.code = code
        self.msg = msg
        self.detail = detail


_token_cache: dict[str, Any] = {"token": None, "expires_at": 0.0}


def get_tenant_access_token(force: bool = False) -> str:
    """获取并缓存 tenant_access_token（提前 60 秒过期）。"""
    now = time.time()
    if not force and _token_cache["token"] and now < _token_cache["expires_at"] - 60:
        return _token_cache["token"]

    resp = requests.post(
        f"{FEISHU_BASE}/auth/v3/tenant_access_token/internal",
        json={"app_id": settings.feishu_app_id, "app_secret": settings.feishu_app_secret},
        timeout=15,
    )
    data = resp.json()
    if data.get("code") != 0:
        raise FeishuError(data.get("code", -1), data.get("msg", "token failed"), data)
    _token_cache["token"] = data["tenant_access_token"]
    _token_cache["expires_at"] = now + int(data.get("expire", 7200))
    return _token_cache["token"]


def _request(method: str, path: str, *, params: dict | None = None, body: dict | None = None) -> dict:
    """带一次 token 失效重试的 OpenAPI 调用。"""
    for attempt in range(2):
        token = get_tenant_access_token(force=attempt > 0)
        resp = requests.request(
            method,
            f"{FEISHU_BASE}{path}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            params=params,
            json=body,
            timeout=20,
        )
        try:
            data = resp.json()
        except ValueError as exc:
            raise FeishuError(-1, f"non-json response {resp.status_code}", resp.text[:300]) from exc
        if data.get("code") == 0:
            return data
        if data.get("code") in _TOKEN_INVALID_CODES and attempt == 0:
            log.warning("tenant_access_token invalid, refreshing")
            continue
        raise FeishuError(data.get("code", -1), data.get("msg", "unknown"), data)
    raise FeishuError(-1, "token retry exhausted")


def list_chats(page_size: int = 50) -> list[dict]:
    """列出机器人所在的群。需要 im:chat:readonly。"""
    items: list[dict] = []
    page_token = ""
    while True:
        params = {"page_size": page_size}
        if page_token:
            params["page_token"] = page_token
        data = _request("GET", "/im/v1/chats", params=params)
        payload = data.get("data", {})
        items.extend(payload.get("items", []))
        page_token = payload.get("page_token", "")
        if not payload.get("has_more"):
            break
    return items


def find_chat_by_name(name: str) -> dict | None:
    for chat in list_chats():
        if chat.get("name") == name:
            return chat
    return None


def send_card(chat_id: str, card: dict, *, idempotency_key: str = "") -> dict:
    """发送交互卡片到群。需要 im:message:send_as_bot。"""
    body: dict[str, Any] = {
        "receive_id": chat_id,
        "msg_type": "interactive",
        "content": json.dumps(card, ensure_ascii=False),
    }
    if idempotency_key:
        body["uuid"] = idempotency_key[:50]
    return _request("POST", "/im/v1/messages", params={"receive_id_type": "chat_id"}, body=body)


def _webhook_sign(timestamp: str, secret: str) -> str:
    string_to_sign = f"{timestamp}\n{secret}"
    digest = hmac.new(string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def send_webhook_card(card: dict) -> dict:
    """群自定义机器人 Webhook（备用通道）。"""
    body: dict[str, Any] = {"msg_type": "interactive", "card": card}
    if settings.feishu_webhook_secret:
        ts = str(int(time.time()))
        body["timestamp"] = ts
        body["sign"] = _webhook_sign(ts, settings.feishu_webhook_secret)
    resp = requests.post(settings.feishu_webhook_url, json=body, timeout=20)
    data = resp.json()
    if data.get("code") not in (0, None) and data.get("StatusCode") not in (0, None):
        raise FeishuError(data.get("code", -1), data.get("msg", "webhook failed"), data)
    return data


def notify(card: dict, *, chat_id: str = "", idempotency_key: str = "") -> dict:
    """统一出口：应用机器人 → Webhook → 抛错。"""
    if settings.feishu_app_enabled:
        target = chat_id or settings.feishu_chat_id
        if not target:
            raise FeishuError(-1, "FEISHU_CHAT_ID 未配置，且未传入 chat_id")
        return send_card(target, card, idempotency_key=idempotency_key)
    if settings.feishu_webhook_enabled:
        return send_webhook_card(card)
    raise FeishuError(-1, "未配置飞书应用凭证或 Webhook URL")


def build_brief_card(
    *,
    report_date: str,
    score: float,
    state: str,
    strategy_summary: str,
    risk_note: str,
    report_url: str,
    disclaimer: str = "仅供研究演示，不构成投资建议。市场有风险，投资需谨慎。",
) -> dict:
    """简报卡片：日期、评分/状态、策略摘要、风险提示、报告链接、免责声明。"""
    color = {"上行": "green", "震荡偏强": "green", "震荡": "blue", "下行": "orange", "极寒": "red"}.get(
        state, "blue"
    )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": color,
            "title": {"tag": "plain_text", "content": f"投研简报 {report_date}"},
        },
        "elements": [
            {
                "tag": "div",
                "fields": [
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**市场评分**\n{score:.1f}/100"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**市场状态**\n{state}"}},
                ],
            },
            {"tag": "hr"},
            {"tag": "div", "text": {"tag": "lark_md", "content": f"**策略摘要**\n{strategy_summary}"}},
            {"tag": "div", "text": {"tag": "lark_md", "content": f"**风险提示**\n{risk_note}"}},
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "查看完整报告"},
                        "type": "primary",
                        "url": report_url,
                    }
                ],
            },
            {"tag": "hr"},
            {"tag": "note", "elements": [{"tag": "plain_text", "content": disclaimer}]},
        ],
    }


def build_test_card() -> dict:
    return build_brief_card(
        report_date="2026-07-10",
        score=33.9,
        state="下行",
        strategy_summary="动态评分：年化 13.13%，最大回撤 -14.27%；LGB 融合：夏普 0.646（全场最高）。",
        risk_note="当前处于下行状态，权益仓位中枢建议 15%。以上为历史回测口径，不代表未来收益。",
        report_url="http://localhost:8501",
        disclaimer="Phase 0 冒烟测试卡片。仅供研究演示，不构成投资建议。",
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description="飞书通道自检")
    parser.add_argument("--list", action="store_true", help="列出机器人所在群")
    parser.add_argument("--send", action="store_true", help="发送测试卡片")
    parser.add_argument("--chat-name", default="", help="按群名发送")
    parser.add_argument("--chat-id", default="", help="按 chat_id 发送")
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level)
    if args.list:
        for chat in list_chats():
            print(json.dumps({"chat_id": chat.get("chat_id"), "name": chat.get("name")}, ensure_ascii=False))
        return 0

    if args.send:
        chat_id = args.chat_id
        if not chat_id and args.chat_name:
            chat = find_chat_by_name(args.chat_name)
            if not chat:
                print(f"未找到群：{args.chat_name}", flush=True)
                return 1
            chat_id = chat["chat_id"]
        result = notify(build_test_card(), chat_id=chat_id)
        print(json.dumps(result, ensure_ascii=False)[:500])
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
