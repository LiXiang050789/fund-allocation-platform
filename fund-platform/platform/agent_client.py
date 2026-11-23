"""Independent HTTP adapter for the local Agent service."""
import os
import re
from urllib.parse import quote, urlsplit

import requests


class AgentError(Exception):
    """Safe error for display in the platform."""


class AgentConfigError(AgentError):
    pass


class AgentUnavailable(AgentError):
    pass


class AgentHTTPError(AgentError):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = safe_detail(detail)
        super().__init__(f"HTTP {status_code}: {self.detail}")


def safe_detail(value) -> str:
    """Keep validation messages, omit inputs, credentials and diagnostics."""
    if isinstance(value, list):
        value = "; ".join(str(v.get("msg", "请求不合法")) for v in value if isinstance(v, dict))
    if not isinstance(value, str):
        return "Agent 请求失败"
    value = re.sub(r"https?://\S+", "[地址已隐藏]", value, flags=re.I)
    # Diagnostic dumps may contain unlabelled secrets. Do not display them.
    if re.search(r"traceback|\bheaders?\b|\bbearer\b|secret|token|api.?key|password|sk-[\w-]+", value, re.I):
        return "Agent 请求失败（敏感诊断已隐藏）"
    value = " ".join(value.split())
    return value[:300] or "Agent 请求失败"


def _base_url(value: str) -> str:
    try:
        value = value.strip().rstrip("/")
        parsed = urlsplit(value)
        valid = (parsed.scheme in {"http", "https"} and parsed.hostname
                 and parsed.port != 0 and not parsed.username and not parsed.password
                 and not parsed.query and not parsed.fragment and not re.search(r"\s", value))
    except (AttributeError, TypeError, ValueError):
        valid = False
    if not valid:
        raise AgentConfigError("AGENT_BASE_URL 必须是有效的 HTTP 或 HTTPS 服务地址") from None
    return value


def get_base_url() -> str:
    return _base_url(os.environ.get("AGENT_BASE_URL", "http://127.0.0.1:8001"))


def _request(method, path, base_url, timeout, *, body=None, text=False):
    base = get_base_url() if base_url is None else _base_url(base_url)
    kwargs = {"timeout": timeout}
    if body is not None:
        kwargs["json"] = body
    try:
        response = requests.request(method, base + path, **kwargs)
    except requests.RequestException:
        raise AgentUnavailable("Agent 服务未启动或不可达") from None
    if not 200 <= response.status_code < 300:
        try:
            payload = response.json()
            detail = payload.get("detail") if isinstance(payload, dict) else None
        except (ValueError, requests.RequestException):
            detail = None
        raise AgentHTTPError(response.status_code, safe_detail(detail)) from None
    if text:
        return response.text
    try:
        result = response.json()
    except (ValueError, requests.RequestException):
        raise AgentError("Agent 返回了无效 JSON") from None
    if not isinstance(result, dict):
        raise AgentError("Agent 响应必须是 JSON object")
    return result


def _thread_path(thread_id):
    return "/workflow/" + quote(str(thread_id), safe="")


def health(base_url: str | None = None, timeout: float = 3.0) -> dict:
    return _request("GET", "/healthz", base_url, timeout)


def chat(question: str, thread_id: str, base_url: str | None = None, timeout: float = 90.0) -> dict:
    return _request("POST", "/chat", base_url, timeout, body={"question": question, "thread_id": thread_id})


def start_workflow(base_url: str | None = None, timeout: float = 60.0) -> dict:
    return _request("POST", "/workflow/run", base_url, timeout)


def get_workflow(thread_id: str, base_url: str | None = None, timeout: float = 15.0) -> dict:
    return _request("GET", _thread_path(thread_id), base_url, timeout)


def review_workflow(thread_id: str, action: str, reviewer: str, comment: str = "",
                    edited_markdown: str = "", base_url: str | None = None,
                    timeout: float = 60.0) -> dict:
    if action not in {"approve", "edit", "reject"}:
        raise AgentError("审核动作必须为 approve / edit / reject")
    return _request("POST", _thread_path(thread_id) + "/approve", base_url, timeout,
                    body=dict(action=action, reviewer=reviewer, comment=comment, edited_markdown=edited_markdown))


def get_report(thread_id: str, base_url: str | None = None, timeout: float = 15.0) -> str:
    return _request("GET", _thread_path(thread_id) + "/report", base_url, timeout, text=True)


# Phase 6: chat transport remains independent of all Agent imports and secrets.
def chat_history(thread_id, base_url=None, timeout=15.0):
    return _request("GET", "/chat/history/" + quote(str(thread_id), safe=""), base_url, timeout)


def clear_chat(thread_id, base_url=None, timeout=15.0):
    return _request("DELETE", "/chat/" + quote(str(thread_id), safe=""), base_url, timeout)


def stream_chat(question, thread_id, base_url=None):
    """Strict SSE reader. Closing the generator closes its response/socket."""
    import json
    base = get_base_url() if base_url is None else _base_url(base_url)
    response = None
    try:
        response = requests.post(base + "/chat/stream", stream=True, timeout=(5, 120),
                                 json={"question": question, "thread_id": thread_id})
        if not 200 <= response.status_code < 300:
            try:
                detail = response.json().get('detail')
            except (ValueError, AttributeError):
                detail = None
            raise AgentHTTPError(response.status_code, safe_detail(detail))
        response.encoding = 'utf-8'
        event, lines, finished = '', [], False
        for line in response.iter_lines(decode_unicode=True, chunk_size=1):
            if line == '':
                if lines:
                    try:
                        data = json.loads("\n".join(lines))
                    except ValueError:
                        raise AgentError('SSE 数据无效，回答未完成') from None
                    if event not in {'meta', 'status', 'token', 'final', 'error'} or not isinstance(data, dict):
                        raise AgentError('SSE 事件无效，回答未完成')
                    if event == 'error':
                        raise AgentError(safe_detail(data.get('message')))
                    yield event, data
                    if event == 'final':
                        finished = True
                        break
                event, lines = '', []
            elif line.startswith('event:'):
                event = line[6:].strip()
            elif line.startswith('data:'):
                lines.append(line[5:].lstrip())
        if not finished:
            raise AgentError('SSE 连接中断，回答未完成')
    except requests.RequestException:
        raise AgentUnavailable('Agent 连接中断或超时，回答未完成') from None
    finally:
        if response is not None:
            response.close()
