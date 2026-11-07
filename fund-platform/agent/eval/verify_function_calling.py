#!/usr/bin/env python3
"""Phase 0 最小验证：DeepSeek 是否支持 function calling。

用法：.venv/bin/python eval/verify_function_calling.py
退出码 0 = 所有被测模型都正确返回 tool_calls。
"""

from __future__ import annotations

import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_market_score",
            "description": "查询指定日期的市场评分和状态",
            "parameters": {
                "type": "object",
                "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}},
                "required": ["date"],
            },
        },
    }
]

MODELS = ["deepseek-flash", "deepseek-v4-pro"]


def check(model: str) -> bool:
    resp = requests.post(
        f"{os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')}/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": "查一下 2026-07-10 的市场评分"}],
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0,
        },
        timeout=60,
    )
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    ok = bool(calls) and calls[0]["function"]["name"] == "get_market_score"
    print(f"{model}: finish_reason={choice.get('finish_reason')} tool_calls={len(calls)} -> {'OK' if ok else 'FAIL'}")
    if calls:
        print(f"  args={calls[0]['function']['arguments']}")
    return ok


def main() -> int:
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("DEEPSEEK_API_KEY 未配置", file=sys.stderr)
        return 2
    results = {m: check(m) for m in MODELS}
    print(json.dumps(results, ensure_ascii=False))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
