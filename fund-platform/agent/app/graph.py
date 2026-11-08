"""有界 LangGraph 问答：白名单路由、证据回答、数字校验。"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.tools import registry

log = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    question: str
    thread_id: str
    route: list[str]
    tool_results: list[dict]
    answer: str
    citations: list[str]
    verified: bool
    attempts: int
    error: str | None
    tool_calls: list[dict]
    pending: list[dict]
    rounds: int
    correction: str


@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(model=settings.deepseek_model, base_url=settings.deepseek_base_url,
                      api_key=settings.deepseek_api_key, temperature=0,
                      timeout=settings.llm_timeout, max_retries=0)


def refusal(question):
    return any(w in question for w in ("推荐股票", "涨停", "保证收益", "明天涨跌", "全仓", "保证", "明天"))


def fallback_calls(question):
    q = question.upper()
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", q)
    strategy = next((s for s in ("动态评分", "LGB", "SASF", "PPO", "等权", "MVO", "风险平价", "动量", "沪深300") if s in q), None)
    if "PPO" in q and "月度" in q:
        strategy = "PPO月度"
    if any(w in q for w in ("数据区间", "覆盖", "北向", "维度", "滞后", "口径", "多少维", "多少列")):
        name, args = "get_data_caveats", {}
    elif any(w in q for w in ("特征", "重要性")):
        match = re.search(r"(?:TOP|前)\s*(\d+)", q)
        name, args = "get_feature_importance", {"top_n": int(match[1])} if match else {}
    elif any(w in q for w in ("风险档", "C2", "C3", "C4", "保守", "平衡", "进取")):
        level = next((s for s in ("C2", "C3", "C4", "保守", "平衡", "进取") if s in q), None)
        name, args = "get_risk_tier", {"level": level, "strategy": strategy}
    elif any(w in q for w in ("修复", "股灾", "危机")):
        year = re.search(r"\d{4}", q)
        name, args = "get_drawdown_recovery", {"crisis": year[0]} if year else {}
    elif any(w in q for w in ("分段", "阶段", "牛市", "熊市", "震荡期")):
        year = re.search(r"\d{4}", q)
        name, args = "get_segment_performance", {"segment": year[0]} if year else {}
    elif "状态" in q and any(w in q for w in ("各", "多少个", "统计", "分布")):
        name, args = "get_market_state_summary", {}
    elif any(w in q for w in ("策略", "夏普", "年化", "回撤", "卡玛", "换手")) and not refusal(question):
        name, args = "get_strategy_metrics", {"strategy": strategy}
    elif any(w in q for w in ("评分", "状态", "点位", "得分", "交易日", "明天")) or dates:
        name, args = "get_market_score", {"date": dates[0]} if dates else {}
    else:
        return []
    return [{"name": name, "args": args}]


def intake(state):
    return dict(question=state["question"].strip(), route=[], tool_results=[], tool_calls=[],
                pending=[], citations=[], attempts=0, rounds=0, error=None, verified=False,
                answer="当前数据无法回答：问题为空。")


def router(state):
    calls = []
    try:
        response = get_llm().bind_tools(registry.TOOL_SCHEMAS).invoke([
            SystemMessage(content="只选择回答问题必需的只读工具。日期、策略名称必须来自问题。数据维度用get_data_caveats；状态计数用get_market_state_summary。禁止调用白名单外工具。"),
            HumanMessage(content=state["question"] + ("\n上轮结果：" + json.dumps(state["tool_results"], ensure_ascii=False) if state["rounds"] else ""))])
        calls = response.tool_calls
        if not calls or any(c.get("name") not in registry.REGISTRY for c in calls):
            calls = fallback_calls(state["question"])
    except Exception as exc:
        log.warning("router LLM failed: %s", type(exc).__name__)
        calls = fallback_calls(state["question"])
    return dict(pending=[{"name": c["name"], "args": c.get("args", {})} for c in calls[:8]],
                route=state["route"] + [c["name"] for c in calls[:8]], rounds=state["rounds"]+1)


def tools(state):
    results, calls = list(state["tool_results"]), list(state["tool_calls"])
    for call in state["pending"]:
        try:
            result = registry.execute(call["name"], call["args"])
        except Exception as exc:
            result = dict(ok=False, tool=call["name"], data={}, as_of=None, source="", error=f"{type(exc).__name__}: {exc}")
        results.append(result)
        calls.append({**call, "ok": result["ok"]})
    return dict(tool_results=results, tool_calls=calls)


def fmt(value, percent=False):
    if isinstance(value, str):
        return value
    return f"{value*100:.2f}%" if percent else f"{value:g}"


def evidence_answer(question, results):
    """模型不可用时，从结构化证据生成短答；不读取评测集。"""
    parts = []
    for result in results:
        if not result["ok"]:
            continue
        name, data = result["tool"], result["data"]
        if name == "get_market_score":
            parts.append(f"{result['as_of']} 总分 {fmt(data['total_score'])}，状态「{data['market_state']}」；" +
                         "，".join(f"{label} {fmt(data[key])}" for label, key in (("估值", "val_score"), ("宏观", "macro_score"), ("情绪", "sent_score"), ("趋势", "trend_score"))))
        elif name == "get_market_state_summary":
            parts.append("；".join(f"{k} {v} 天" for k, v in data["counts"].items()) + f"；合计 {data['total_days']} 天")
        elif name == "get_data_caveats":
            if "北向" in question:
                parts.append(f"北向：{data['nan_range']} 缺失 {data['nan_rows']} 行；{data['zero_start']} 起连续填零 {data['zero_rows']} 行；最后有效 {data['last_valid_date']}")
            elif any(w in question for w in ("维", "特征", "列")):
                dims = data["dimensions"]
                parts.append(f"全表 {dims['all']['columns']} 列，{dims['all']['features']} 个特征；LGB {dims['LGB']['features']} 维；PPO {dims['PPO']['features']} 维")
            else:
                parts.append(f"数据 {data['start']} 至 {data['end']}，{data['trading_days']} 个交易日；宏观滞后 {data['macro_lag']} 个交易日；截止距今 {data['months_since_cutoff']} 个月")
        else:
            rows = data.get("rows", [])
            strategies = [s for s in ("动态评分", "LGB", "SASF", "等权", "PPO", "MVO", "动量", "风险平价") if s in question.upper()]
            if strategies:
                rows = [r for r in rows if any(s in r.get("策略", r.get("策略源", "")) for s in strategies)]
            if "最高" in question and rows:
                metric = "卡玛比率" if "卡玛" in question else "夏普比率"
                rows = [max(rows, key=lambda r: r[metric])]
            for row in rows:
                if name == "get_feature_importance":
                    parts.append(f"{row['feature']}：{fmt(row['importance'])}")
                    continue
                label = " ".join(str(row[k]) for k in ("阶段", "危机阶段", "风险档", "策略", "策略源") if k in row)
                metrics = [k for k in ("累计收益", "年化收益", "年化波动", "夏普比率", "最大回撤", "卡玛比率", "修复天数") if k in row]
                wanted = [k for k in metrics if k.replace("比率", "") in question or (k == "修复天数" and "修复" in question)]
                metrics = wanted or metrics
                parts.append(label + "：" + "，".join(k + " " + fmt(row[k], k in ("累计收益", "年化收益", "年化波动", "最大回撤")) for k in metrics))
    return "；".join(parts) or "当前数据无法回答"


POLICY = "无法预测明天涨跌或涨停，不能推荐个股或全仓操作，不能保证收益。历史回测不代表未来。不构成投资建议。"


def compose(state):
    usable = [r for r in state["tool_results"] if r["ok"]]
    citations = list(dict.fromkeys(s for r in usable for s in r["source"].split("; ") if s))
    answer = evidence_answer(state["question"], usable)
    if refusal(state["question"]):
        answer = POLICY
    elif usable:
        try:
            response = get_llm().invoke([
                SystemMessage(content="只用提供的tool_results回答，数字只能来自证据。中文简洁回答，正文不超过180字，不写引用路径。分项用估值/宏观/情绪/趋势。比例转换成百分数，保留两位小数。不提供个性化投资建议。"),
                HumanMessage(content=json.dumps({"question": state["question"], "tool_results": usable,
                                                 "correction": state.get("correction", "")}, ensure_ascii=False))])
            if isinstance(response.content, str) and response.content.strip():
                answer = response.content.strip()
        except Exception as exc:
            log.warning("compose LLM failed: %s", type(exc).__name__)
    suffix = "不构成投资建议。"
    if suffix not in answer:
        answer += "。" + suffix
    reference = f"\n[source: {citations[0]}]" if citations else ""
    limit = 300 - len(reference)
    if len(answer) > limit:
        # Keep complete clauses so truncation cannot change a numerical value.
        clauses = re.findall(r"[^。；\n]+[。；\n]?", answer.replace(suffix, ""))
        kept = ""
        for clause in clauses:
            if len(kept + clause + suffix) > limit:
                break
            kept += clause
        answer = (kept or "当前数据无法回答：回答超出长度限制。") + suffix
    answer += reference
    return dict(answer=answer, citations=citations, attempts=state["attempts"]+1)


def numeric_values(value):
    if isinstance(value, dict):
        return [n for v in value.values() for n in numeric_values(v)]
    if isinstance(value, list):
        return [n for v in value for n in numeric_values(v)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, str):
        return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", strip_dates(value))]
    return []


def strip_dates(text):
    text = re.sub(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日", "", text)
    return re.sub(r"(?<!\d)(?:19|20)\d{2}年?(?!\d)", "", text)


def verify(state):
    # Citations are metadata, not numerical claims.
    body = re.sub(r"\[source:.*?\]", "", state["answer"], flags=re.S)
    values = numeric_values([r["data"] for r in state["tool_results"] if r["ok"]])
    allowed = {0, 1, 2, 3, 5, 7, 10, 100}
    unsupported = [n for n in numeric_values(body) if n not in allowed and not any(
        abs(n-v) <= .1+1e-9 or abs(n/100-v) <= .001+1e-9 or abs(n-v*100) <= .1+1e-9 for v in values)]
    compliant = not refusal(state["question"]) or any(w in body for w in ("不构成投资建议", "无法预测", "不能保证"))
    valid = not unsupported and compliant and len(state["answer"]) <= 300
    correction = f"删除证据不支持的数值{unsupported}；仅引用tool_results，遵守拒答政策和字数限制。" if not valid else ""
    return dict(verified=valid, correction=correction, error=correction or None)


def finalize(state):
    return state


def build_graph():
    builder = StateGraph(AgentState)
    for name in ("intake", "router", "tools", "compose", "verify", "finalize"):
        builder.add_node(name, globals()[name])
    builder.add_edge(START, "intake")
    builder.add_conditional_edges("intake", lambda s: "router" if s["question"] else "finalize")
    builder.add_edge("router", "tools")
    builder.add_conditional_edges("tools", lambda s: "router" if s["pending"] and not any(r["ok"] for r in s["tool_results"]) and s["rounds"] < 2 else "compose")
    builder.add_edge("compose", "verify")
    builder.add_conditional_edges("verify", lambda s: "compose" if not s["verified"] and s["attempts"] < 2 else "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


graph = build_graph()
