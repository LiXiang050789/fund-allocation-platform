#!/usr/bin/env python3
"""25 题黄金集评测。

mode=baseline：复现平台现有的「直接调用 DeepSeek」AI 助手（无工具、无检索）。
mode=agent：直接调用 LangGraph 问答图。

用法：
    .venv/bin/python eval/run_eval.py --mode baseline
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))
from app.rag.textnorm import normalize_text, numeric_text

PLATFORM_DIR = AGENT_DIR.parent
EVAL_DIR = Path(__file__).resolve().parent
load_dotenv(AGENT_DIR / ".env")

NUM_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def extract_numbers(text: str) -> list[float]:
    out = []
    text = numeric_text(text or "")
    # Date and range separators never denote a unary negative sign.
    text = re.sub(r"(?<=\d)[-–—](?=\d)", " ", text)
    for raw in NUM_RE.findall(text):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            pass
    return out


def load_latest_score() -> dict:
    df = pd.read_csv(
        PLATFORM_DIR / "data/clean/market_score_daily.csv",
        encoding="utf-8-sig",
        parse_dates=["date"],
    )
    row = df.iloc[-1]
    return {
        "date": str(row["date"].date()),
        "total_score": float(row["total_score"]),
        "market_state": str(row["market_state"]),
        "val_score": float(row["val_score"]),
        "macro_score": float(row["macro_score"]),
        "sent_score": float(row["sent_score"]),
        "trend_score": float(row["trend_score"]),
    }


def build_baseline_prompt(latest: dict) -> str:
    """与 platform/app.py:952-958 的 system_prompt 保持一致。"""
    return f"""你是一个智能投顾助手，为工行客户提供资产配置建议。
当前市场环境: 评分 {latest['total_score']:.0f}/100，状态 {latest['market_state']}。
估值{latest['val_score']:.0f}/25，宏观{latest['macro_score']:.0f}/25，情绪{latest['sent_score']:.0f}/20，趋势{latest['trend_score']:.0f}/30。
无 PPO 权重，仅基于市场评分回答。
资产池: 沪深300/中证500/科创50/消费/芯片/黄金/十年国债 (7只ETF)。
策略: 动态评分配置(年化19.8%/-16.5%回撤)，LGB融合(年化12.4%/夏普0.66)。
回答要简洁专业，不生成具体投资建议。"""


def call_deepseek(system_prompt: str, question: str) -> tuple[str, dict]:
    resp = requests.post(
        f"{os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')}/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "max_tokens": 500,
            # 评测置 0 降低随机性；平台线上运行仍为 0.7
            "temperature": 0.0,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"], data.get("usage", {})


def grade(item: dict, answer: str) -> list[str]:
    fails: list[str] = []
    nums = extract_numbers(answer)
    tol = float(item.get("tolerance", 0.01))

    for exp in item.get("expected_numeric", []):
        if not any(abs(abs(n) - abs(exp)) <= tol + 1e-9 for n in nums):
            fails.append(f"缺少数值 {exp}")

    for kw in item.get("expected_keywords", []):
        if kw.lower() not in answer.lower():
            fails.append(f"缺少关键词 {kw}")

    for exp in item.get("forbidden_numeric", []):
        if any(abs(abs(n) - abs(exp)) <= tol + 1e-9 for n in nums):
            fails.append(f"出现禁用数值 {exp}")

    for kw in item.get("forbidden_keywords", []):
        if kw.lower() in answer.lower():
            fails.append(f"出现禁用词 {kw}")

    if item.get("expect_refusal"):
        if not any(k in answer for k in item.get("refusal_keywords", [])):
            fails.append("未按要求拒答")

    return fails


def run_baseline(items: list[dict]) -> list[dict]:
    latest = load_latest_score()
    system_prompt = build_baseline_prompt(latest)
    print(f"[baseline] latest={latest['date']} score={latest['total_score']} state={latest['market_state']}")
    results = []
    for i, item in enumerate(items, 1):
        t0 = time.time()
        try:
            answer, usage = call_deepseek(system_prompt, item["question"])
            fails = grade(item, answer)
        except Exception as e:  # noqa: BLE001
            answer, usage, fails = "", {}, [f"调用失败: {type(e).__name__}: {e}"]
        elapsed = round(time.time() - t0, 2)
        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "answer": answer,
                "usage": usage,
                "latency_s": elapsed,
                "pass": not fails,
                "fails": fails,
                "golden_answer": item.get("golden_answer", ""),
                "source": item.get("source", ""),
            }
        )
        status = "PASS" if not fails else "FAIL " + "; ".join(fails)
        print(f"[{i:02d}/{len(items)}] {item['id']} {elapsed:5.2f}s {status}")
        time.sleep(0.3)
    return results


def run_agent(items: list[dict]) -> list[dict]:
    sys.path.insert(0, str(AGENT_DIR))
    from app.graph import graph

    results = []
    for index, item in enumerate(items, 1):
        started = time.perf_counter()
        state = {}
        try:
            state = graph.invoke({"question": item["question"], "thread_id": f"eval-{item['id']}"})
            answer = state["answer"]
            fails = grade(item, answer)
        except Exception as exc:
            answer, fails = "", [f"调用失败: {type(exc).__name__}"]
        calls = state.get("tool_calls", [])
        expected = set(item.get("tools", []))
        selected = bool(expected.intersection(c["name"] for c in calls))
        supported = bool(calls) and any(
            r.get("ok") and any(source in answer for source in r.get("source", "").split("; ") if source)
            for r in state.get("tool_results", []))
        result = dict(id=item["id"], category=item["category"], question=item["question"],
                      answer=answer, latency_s=round(time.perf_counter()-started, 2),
                      fails=fails, golden_answer=item.get("golden_answer", ""), source=item.get("source", ""),
                      tool_calls=calls, tool_selection_correct=selected, citation_supported=supported,
                      verified=state.get("verified", False), verification_error=state.get("error"), citations=state.get("citations", []),
                      llm_events=state.get("llm_events", []))
        result["pass"] = not fails
        results.append(result)
        print(f"[{index:02d}/{len(items)}] {item['id']} {result['latency_s']:.2f}s " +
              ("PASS" if not fails else "FAIL " + "; ".join(fails)), flush=True)
    return results


def summarize(results: list[dict], mode: str) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r["pass"])
    by_cat: dict[str, dict] = {}
    for r in results:
        c = by_cat.setdefault(r["category"], {"total": 0, "pass": 0})
        c["total"] += 1
        c["pass"] += int(r["pass"])
    for c in by_cat.values():
        c["rate"] = round(c["pass"] / c["total"], 4)
    lat = sorted(r["latency_s"] for r in results)
    return {
        "mode": mode,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "total": total,
        "pass": passed,
        "accuracy": round(passed / total, 4),
        "by_category": by_cat,
        "latency_p50": lat[len(lat) // 2] if lat else None,
        "latency_p95": lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else None,
        "citation_support_rate": 0.0 if mode == "baseline" else round(sum(r.get("citation_supported", False) for r in results) / total, 4),
        "tool_selection_accuracy": round(sum(r.get("tool_selection_correct", False) for r in results) / total, 4) if mode == "agent" else None,
        "verified_rate": sum(r.get("verified", False) for r in results) / total if mode == "agent" else None,
        "llm_error_count": sum(e["status"] == "error" for r in results for e in r.get("llm_events", [])),
        "citation_note": "baseline 无检索、无引用，按定义记 0" if mode == "baseline" else "按全部题目统计成功工具 source 路径的引用",
    }


def write_report(summary: dict, results: list[dict], system_prompt: str) -> None:
    suffix = "_agent" if summary["mode"] == "agent" else ""
    lines = [
        "# 25 题评测报告",
        "",
        f"- 模式：`{summary['mode']}`",
        f"- 执行方式：`{summary.get('execution', 'configured_llm')}`",
        f"- 时间：{summary['run_at']}",
        f"- 总题数：{summary['total']}",
        f"- 通过：{summary['pass']}",
        f"- 准确率：{summary['accuracy']:.1%}",
        f"- 引用支持率：{summary.get('citation_support_rate')}（{summary.get('citation_note', '')}）",
        f"- 延迟 p50/p95：{summary['latency_p50']}s / {summary['latency_p95']}s",
        "",
        f"- verified_rate：{summary.get('verified_rate')}",
        f"- llm_error_count：{summary.get('llm_error_count')}",
        f"- 工具选择准确率：{summary.get('tool_selection_accuracy')}",
        "",
        "## 分类通过率",
        "",
        "| 类别 | 通过/总数 | 通过率 |",
        "|---|---|---|",
    ]
    for cat, c in summary["by_category"].items():
        lines.append(f"| {cat} | {c['pass']}/{c['total']} | {c['rate']:.0%} |")
    lines += ["", "## 逐题结果", "", "| ID | 结果 | 失败项 |", "|---|---|---|"]
    for r in results:
        lines.append(f"| {r['id']} | {'PASS' if r['pass'] else 'FAIL'} | {'; '.join(r['fails']) or '-'} |")
    if summary["mode"] == "baseline":
        lines += ["", "## 基线 system prompt（原文）", "", "```text", system_prompt, "```", ""]
    else:
        baseline_path = EVAL_DIR / "results.json"
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["summary"]
            lines += ["", f"基线通过 {baseline['pass']}/{baseline['total']}；Agent 通过 {summary['pass']}/{summary['total']}。"]
        lines += ["", "工具选择和引用支持均以全部题目为分母。空期望工具按无交集计分。",
                  "引用支持率检查答案是否含成功工具的 source 路径；不代表全文语义核验。",
                  "", "## 验证状态", ""]
        lines += [f"- {r['id']}: verified={r.get('verified', False)}；{r.get('verification_error') or '通过校验'}" for r in results]
    (EVAL_DIR / f"report{suffix}.md").write_text("\n".join(lines), encoding="utf-8")

    detail = ["# 逐题作答明细", ""]
    for r in results:
        detail += [
            f"## {r['id']} {'PASS' if r['pass'] else 'FAIL'}",
            "",
            f"**问题**：{r['question']}",
            "",
            f"**回答**：{r['answer']}",
            "",
            f"**标准答案**：{r['golden_answer']}",
            "",
            f"**来源**：{r['source']}",
            "",
            f"**判定**：{'; '.join(r['fails']) if r['fails'] else '通过'}",
            "",
        ]
    (EVAL_DIR / f"answers{suffix}.md").write_text(
        "\n".join(line.rstrip() for line in "\n".join(detail).splitlines()) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "agent"], default="baseline")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offline", action="store_true", help="agent 模式禁用外部模型，仅验证确定性回退")
    args = parser.parse_args()
    if args.offline and args.mode != "agent":
        parser.error("--offline 仅用于 --mode agent")

    data = yaml.safe_load((EVAL_DIR / "golden_set.yaml").read_text(encoding="utf-8"))
    items = data["questions"]
    if args.limit:
        items = items[: args.limit]

    suffix = "_agent" if args.mode == "agent" else ""
    if args.mode == "agent":
        system_prompt = ""
        if args.offline:
            from unittest.mock import patch
            sys.path.insert(0, str(AGENT_DIR))
            with patch("app.graph.get_llm", side_effect=RuntimeError("offline evaluation")):
                results = run_agent(items)
        else:
            results = run_agent(items)
    else:
        latest = load_latest_score()
        system_prompt = build_baseline_prompt(latest)
        results = run_baseline(items)
    summary = summarize(results, args.mode)
    summary["execution"] = "offline_fallback" if args.offline else "configured_llm"
    (EVAL_DIR / f"results{suffix}.json").write_text(
        json.dumps(
            {"summary": summary, "system_prompt": system_prompt, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(summary, results, system_prompt)
    print(f"\n准确率 {summary['pass']}/{summary['total']} = {summary['accuracy']:.1%}")
    print(f"工具选择准确率 {summary.get('tool_selection_accuracy')}; 引用支持率 {summary['citation_support_rate']}")
    print(f"产物：results{suffix}.json / report{suffix}.md / answers{suffix}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
