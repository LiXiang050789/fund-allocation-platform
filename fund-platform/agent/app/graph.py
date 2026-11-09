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
from app.rag.citations import (DOC_REF, citation, citation_matches, document_text,
                               marked_numbers, retrieved_chunks)

from app.rag.named import named_sources

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
    llm_events: list[dict]


@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(model=settings.deepseek_model, base_url=settings.deepseek_base_url,
                      api_key=settings.deepseek_api_key, temperature=0,
                      timeout=settings.llm_timeout, max_retries=0)


def refusal(question):
    return any(w in question for w in ("推荐股票", "涨停", "保证收益", "明天涨跌", "全仓", "保证", "明天"))


def data_fallback_calls(question):
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


def document_question(question):
    words = ("规则", "定义", "为什么", "流程", "问卷", "费率", "PPO设计", "PPO 设计", "文档口径", "2019cut", "前视", "切割", "权重", "满分", "打分", "如何计分", "映射", "状态空间", "奖励", "手册", "设计文档")
    return any(word.lower() in question.lower() for word in words) or question.strip().upper() == "PPO"


def fallback_calls(question):
    if any(word in question for word in ("涨停", "推荐股票", "全仓")):
        return [{"name": "search_docs", "args": {"query": "风险匹配 产品适当性 风险等级"}}]
    if "保证" in question:
        return [{"name": "get_strategy_metrics", "args": {}}]
    data = data_fallback_calls(question)
    if document_question(question):
        docs = [{"name": "search_docs", "args": {"query": question[:200]}}]
        if any(w in question for w in ("最新", "当前", "今天", "实际数据")):
            return docs + data
        return docs
    return data


def intake(state):
    return dict(question=state["question"].strip(), route=[], tool_results=[], tool_calls=[],
                pending=[], citations=[], attempts=0, rounds=0, error=None, verified=False, llm_events=[],
                answer="当前数据无法回答：问题为空。")


def router(state):
    calls = []
    events = list(state.get("llm_events", []))
    try:
        response = get_llm().bind_tools(registry.TOOL_SCHEMAS).invoke([
            SystemMessage(content="只选择回答问题必需的只读工具。日期、策略名称必须来自问题。规则/定义/为什么/流程/问卷/费率/PPO设计/文档口径/前视解释用search_docs，query保留原问题及其关键概念，尤其不得删去问题点名的文件名或文档别名。具体日期市场评分和策略回测指标用数据工具；两者都需要时同时调用。评分系统权重与满分是文档规则，不是最新市场评分；C1-C5产品匹配是文档规则，C2/C3/C4策略收益是get_risk_tier。实际北向缺口与特征维度用get_data_caveats；状态计数用get_market_state_summary。拒答问题可以查风险适当性或历史指标，不得推荐个股。禁止调用白名单外工具。"),
            HumanMessage(content=state["question"] + ("\n上轮结果：" + json.dumps(state["tool_results"], ensure_ascii=False) if state["rounds"] else ""))])
        calls = response.tool_calls
        events.append({"stage": "router", "status": "success"})
        if not calls or any(c.get("name") not in registry.REGISTRY for c in calls):
            calls = fallback_calls(state["question"])
    except Exception as exc:
        events.append({"stage": "router", "status": "error", "error": type(exc).__name__})
        log.warning("router LLM failed: %s", type(exc).__name__)
        calls = fallback_calls(state["question"])
    return dict(pending=[{"name": c["name"], "args": c.get("args", {})} for c in calls[:8]],
                route=state["route"] + [c["name"] for c in calls[:8]], rounds=state["rounds"]+1, llm_events=events)


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
        elif name == "search_docs":
            chunks = data.get("chunks", [])
            if chunks:
                chunk = chunks[0]
                text = document_text(chunk)
                # Extract complete source clauses, never synthesize unsupported figures.
                clauses = re.split(r"(?<=[。；])|\n", text)
                excerpt = ""
                for clause in clauses:
                    clause = clause.strip()
                    if clause and not marked_numbers(clause)[1] and len(excerpt+clause) <= 130:
                        excerpt += clause
                parts.append((excerpt or "文档依据不足，当前数据无法回答。") + citation(chunk))
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
    events = list(state.get("llm_events", []))
    usable = [r for r in state["tool_results"] if r["ok"]]
    docs = retrieved_chunks(usable)
    data_results = [r for r in usable if r["tool"] != "search_docs"]
    sources = list(dict.fromkeys(s for r in data_results for s in r["source"].split("; ") if s))
    if any(r["tool"] == "get_data_caveats" for r in data_results):
        question = state["question"]
        primary = ("scripts/calc_market_score.py" if "滞后" in question else
                   "data/clean/train_total_feature.csv" if any(w in question for w in ("北向", "特征", "维", "列")) else
                   "data/clean/market_score_daily.csv")
        sources.sort(key=lambda source: source != primary)
    answer = evidence_answer(state["question"], usable)
    if refusal(state["question"]):
        answer = POLICY
    elif usable:
        try:
            evidence = []
            for result in usable:
                if result["tool"] == "search_docs":
                    evidence.append({**result, "data": {**result["data"], "chunks": [
                        {**chunk, "text": document_text(chunk), "citation": citation(chunk)}
                        for chunk in result["data"]["chunks"]]}})
                else:
                    evidence.append(result)
            response = get_llm().invoke([
                SystemMessage(content="只用提供的tool_results回答，数字只能来自证据。中文简洁回答，正文不超过140字；引用和免责声明也计入总长300字。仅回答市场评分分项时使用估值/宏观/情绪/趋势标签。其他问题保留原主题名称，不强加评分标签。累计收益、年化收益、最大回撤、年化波动等比例字段，原始小数乘100写成百分比，保留两位小数，避免一位小数损失原始精度；夏普/卡玛/索提诺保留比率原值。市场评分和分项使用分制；交易日、维度/列数、天数、排名使用计数单位，均不得加百分号。问题点名某文档时，先回答该文档的内容，不得用另一文档替代；若与权威手册或最终代码不一致，须用一句话说明差异并明确“最终以代码为准”。文档解释至少逐字复制一条chunk的citation，不得编造引用；多个来源优先引用被点名文档，涉及最终代码差异时也引用最终代码依据；选择直接支持答案的章节。文档中的数字仅authoritative可用，background/historical只解释背景，不得引用其事实数字。旧值/禁用清单只可标为旧值，不得作为当前事实。涉及数据口径时说明实际数据以数据工具为准。发现文档算术矛盾时仅说明矛盾并照录原值，不得计算或写出证据没有的合计数。不提供个性化投资建议。"),
                HumanMessage(content=json.dumps({"question": state["question"], "tool_results": evidence,
                                                 "correction": state.get("correction", "")}, ensure_ascii=False))])
            events.append({"stage": "compose", "status": "success"})
            if isinstance(response.content, str) and response.content.strip():
                answer = response.content.strip()
        except Exception as exc:
            events.append({"stage": "compose", "status": "error", "error": type(exc).__name__})
            log.warning("compose LLM failed: %s", type(exc).__name__)
    if "北向" in state["question"] and any(r["tool"] == "get_data_caveats" for r in usable) and "北向" not in answer:
        answer = "北向资金：" + answer
    matched, _ = citation_matches(answer, docs)
    doc_references = list(dict.fromkeys(citation(c) for c in matched))
    if docs and not doc_references:
        doc_references = [citation(docs[0])]
    # Move citations to a reserved footer so length limiting cannot remove half a citation.
    body = DOC_REF.sub("", answer)
    body = re.sub(r"\[source:.*?\]", "", body, flags=re.S).strip()
    references = [f"[source: {sources[0]}]"] if sources else []
    named = named_sources(state["question"], docs, explicit=True)
    doc_references.sort(key=lambda ref: not any(f"来源: {name} §" in ref for name in named))
    references += doc_references[:2]
    # Preserve a forged model reference for the verifier instead of silently repairing it.
    for name, section in DOC_REF.findall(answer):
        reference = f"[来源: {name} §{section}]"
        if not any(c["source_name"] == name.strip() and c["section"].startswith(section.strip()) for c in docs):
            references.append(reference)
    suffix = "不构成投资建议。"
    if docs and any(w in state["question"] for w in ("口径", "对齐", "缺失", "缺口", "数据", "停更", "满分", "权重")):
        suffix = "实际数据以数据工具为准。" + suffix
    body = body.replace("不构成投资建议。", "").replace("实际数据以数据工具为准。", "").strip()
    footer = "\n" + "\n".join(dict.fromkeys(references)) if references else ""
    limit = 300 - len(footer) - len(suffix)
    if len(body) > limit:
        kept = ""
        for clause in re.findall(r"[^。；\n]+[。；\n]?", body):
            if len(kept + clause) > limit:
                break
            kept += clause
        body = kept or "当前数据无法回答：回答超出长度限制。"
    answer = body.rstrip("。 ") + "。" + suffix + footer
    return dict(answer=answer, citations=list(dict.fromkeys(sources+doc_references)), attempts=state["attempts"]+1, llm_events=events)


def numeric_values(value):
    if isinstance(value, dict):
        return [n for v in value.values() for n in numeric_values(v)]
    if isinstance(value, list):
        return [n for v in value for n in numeric_values(v)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, str):
        text = strip_dates(value)
        text = re.sub(r"(?<=\d)[-–—](?=\d)", " 至 ", text)
        return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text)]
    return []


def strip_dates(text):
    text = re.sub(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日", "", text)
    return re.sub(r"(?<!\d)(?:19|20)\d{2}年?(?!\d)", "", text)


def score_unit_error(body, results):
    if not any(r.get("ok") and r.get("tool") == "get_market_score" for r in results):
        return False
    return bool(re.search(
        r"(?:评分|分项|得分|总分|估值|宏观|情绪|趋势)"
        r"(?:为|是|分别|分数|得分|评分|\s|[:：=、]|\*)*"
        r"[-+]?\d+(?:\.\d+)?\s*[%％]", body))


def verify(state):
    # Citations are metadata, not numerical claims.
    body = DOC_REF.sub("", re.sub(r"\[source:.*?\]", "", state["answer"], flags=re.S))
    for result in state["tool_results"]:
        if result.get("ok"):
            for source in result.get("source", "").split("; "):
                if source:
                    body = body.replace(source, "").replace(source.rsplit("/", 1)[-1], "")
    docs = retrieved_chunks(state["tool_results"])
    matched, invalid_refs = citation_matches(state["answer"], docs)
    values = numeric_values([r["data"] for r in state["tool_results"] if r["ok"] and r.get("tool") != "search_docs"])
    values += numeric_values([c["text"] for c in docs if c["authority"] == "authoritative"])
    obsolete_allowed, obsolete_violations = marked_numbers(body)
    values += obsolete_allowed
    needs_reference = bool(docs) and "当前数据无法回答" not in body
    missing_named = [name for name in named_sources(state['question'], docs, explicit=True)
                     if not any(c['source_name'] == name for c in matched)]
    reference_valid = not invalid_refs and not missing_named and (not needs_reference or bool(matched))
    allowed = {0, 1, 2, 3, 5, 7, 10, 100}
    unsupported = [n for n in numeric_values(body) if n not in allowed and not any(
        abs(n-v) <= .1+1e-9 or abs(n/100-v) <= .001+1e-9 or abs(n-v*100) <= .1+1e-9 for v in values)]
    compliant = not refusal(state["question"]) or any(w in body for w in ("不构成投资建议", "无法预测", "不能保证"))
    unit_error = score_unit_error(body, state["tool_results"])
    valid = not unsupported and compliant and not unit_error and reference_valid and not obsolete_violations and len(state["answer"]) <= 300
    correction = f"删除证据不支持的数值{unsupported}；仅引用tool_results，遵守拒答政策和字数限制。" if not valid else ""
    if not reference_valid:
        correction += "引用缺失或伪造：只使用本轮chunk文件名和章节前缀，至少包含一条合法文档引用。"
    for name in missing_named:
        correction += f"问题点名了文档 {name}，必须引用 {name} 的原文 chunk。"
    if obsolete_violations:
        correction += "出现禁用旧值；删除或明确标记旧值，禁止当作当前事实。"
    if unit_error:
        correction += "市场评分及分项使用分制，删除错误百分号并保留原始分数，不得改动数值。"
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
