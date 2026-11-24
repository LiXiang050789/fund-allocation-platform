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
from app.text import sentence_end
from app.dates import parse_date_range, parse_single_date, range_requested, readonly_error
from app.tools import registry
from app.rag.citations import (DOC_REF, citation, citation_matches, document_text,
                               marked_numbers, retrieved_chunks)

from app.rag.named import named_sources, corpus_sources
from app.rag.textnorm import normalize_text, numeric_text

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
    compact = re.sub(r'\s+', '', q)
    compared = registry.strategies_in_question(question)
    nav = '净值' in q
    tier_match = re.search(r'(?<![A-Z0-9])C\d+(?!\d)', q)
    tier = tier_match[0] if tier_match else next((s for s in ('保守','平衡','进取') if s in q), None)
    tier_detail = tier and any(w in compact for w in ('最终净值','推荐配置','ETF权重','配置权重'))
    weight = any(w in q for w in ('权重','调仓','仓位','配置')) and ('PPO' in q or any(w in q for w in ('最新','当前','今天')))
    rule = any(w in q for w in ('设计','规则','怎么算','为什么','文档','怎么计算'))
    current = any(w in q for w in ('最新','当前','今天'))
    # An explicit tier plus a static-config phrase wins over the PPO branch: "C3最新推荐配置"
    # asks for cached tier configuration, not the PPO cache.
    ppo_query = weight and not tier_detail and '评分' not in q and (not rule or (current and 'PPO' in q))
    if re.search(r'对比|比较|VS', q) and len(compared) >= 2 and not any(w in q for w in ('净值','权重','流水线','仓位','调仓','配置')):
        return [{'name':'compare_strategies','args':{'strategies':compared}}]
    date_tool = 'get_strategy_nav' if nav else 'get_ppo_cached_weights' if ppo_query else 'get_market_score_history'
    try:
        interval = parse_date_range(question)
        single = parse_single_date(question)
    except ValueError:
        args = {'date':''} if date_tool != 'get_market_score_history' else {'start':'','end':''}
        return [{'name':date_tool,'args':args}]
    if not interval and single is None and (nav or ppo_query) and re.search(r'\d{4}[-/年]\d', question):
        return [{'name':date_tool,'args':{'date':''}}]
    if interval:
        counts = '状态' in q and any(w in q for w in ('统计','分布','各','多少个','占比'))
        name = 'get_market_state_summary' if counts else 'get_strategy_nav' if nav or '累计' in q else 'get_market_score_history'
        args = dict(zip(('start','end'), interval))
        if name == 'get_strategy_nav' and compared:
            args['strategies'] = compared
        return [{'name':name,'args':args}]
    if ppo_query:
        return [{'name':'get_ppo_cached_weights','args':{'date':single} if single else {}}]
    if nav and not tier_detail:
        args = {'strategies':compared} if compared else {}
        if single:args['date']=single
        return [{'name':'get_strategy_nav','args':args}]
    # The existing dimension questions already use get_data_caveats (coverage matrix §17.3 #8).
    dimensions = '建模表' in q and any(w in q for w in ('多少列','多少维','维度','特征数'))
    if not dimensions and any(w in compact for w in ('流水线','文件就绪','语义校验','建模表','年度IC','IC','预测条数','预测数','预测多少','有效预测','原始文件','清洗CSV','滑点','双边成本','单资产上限','回测口径','主表形状')):
        return [{'name':'get_pipeline_status','args':{}}]
    if tier_detail:
        args = {'level':tier}
        if len(compared) == 1:
            args['strategy']=compared[0]
        elif any(name not in ('动态评分','LGB融合') for name in compared):
            args['strategy']=next(name for name in compared if name not in ('动态评分','LGB融合'))
        return [{'name':'get_risk_tier_detail','args':args}]
    if single and any(word in q for word in ('评分', '状态', '得分')):
        frame = registry.scores()
        if single < frame.date.iloc[0] or single > frame.date.iloc[-1]:
            return [{"name": "get_market_score_history", "args": {"start": single, "end": single}}]
    dates = [single] if single else []
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
    elif "状态" in q and any(w in q for w in ("各", "多少个", "统计", "分布", "占比")):
        name, args = "get_market_state_summary", {}
    elif any(w in q for w in ("策略", "夏普", "年化", "回撤", "卡玛", "换手")) and not refusal(question):
        name, args = "get_strategy_metrics", {"strategy": strategy}
    elif any(w in q for w in ("评分", "状态", "点位", "得分", "交易日", "明天")) or dates:
        name, args = "get_market_score", {"date": dates[0]} if dates else {}
    else:
        return []
    return [{"name": name, "args": args}]


def document_question(question):
    words = ("规则", "定义", "为什么", "流程", "问卷", "费率", "PPO设计", "PPO 设计", "文档口径", "2019cut", "前视", "切割", "权重规则", "权重设计", "权重怎么算", "权重计算", "评分权重", "满分", "打分", "如何计分", "映射", "状态空间", "奖励", "手册", "设计文档")
    return bool(re.search(r"权重.*(?:设计|规则|怎么算|怎么设计|计算)", question)) or any(word.lower() in question.lower() for word in words) or question.strip().upper() == "PPO"


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
            SystemMessage(content="只选择回答问题必需的只读工具。组合净值用get_strategy_nav；PPO缓存权重用get_ppo_cached_weights，不执行推理；流水线产物/年度IC/成本常量用get_pipeline_status；风险档最终净值与静态ETF配置用get_risk_tier_detail。权重设计/评分权重规则用search_docs；最新PPO权重和规则保留文档与缓存两个工具。日期、策略名称必须来自问题。规则/定义/为什么/流程/问卷/费率/PPO设计/文档口径/前视解释用search_docs，query保留原问题及其关键概念，尤其不得删去问题点名的文件名或文档别名。具体日期市场评分和策略回测指标用数据工具；两者都需要时同时调用。评分系统权重与满分是文档规则，不是最新市场评分；C1-C5产品匹配是文档规则，C2/C3/C4策略收益是get_risk_tier。实际北向缺口与特征维度用get_data_caveats；状态天数统计用get_market_state_summary；两两或多个策略对比用compare_strategies，单策略指标用get_strategy_metrics；日期区间每日评分用get_market_score_history；单日或最新评分用get_market_score。拒答问题可以查风险适当性或历史指标，不得推荐个股。禁止调用白名单外工具。"),
            HumanMessage(content=state["question"] + ("\n上轮结果：" + json.dumps(state["tool_results"], ensure_ascii=False) if state["rounds"] else ""))])
        calls = response.tool_calls
        events.append({"stage": "router", "status": "success"})
        if not calls or any(c.get("name") not in registry.REGISTRY for c in calls):
            calls = fallback_calls(state["question"])
    except Exception as exc:
        events.append({"stage": "router", "status": "error", "error": type(exc).__name__})
        log.warning("router LLM failed: %s", type(exc).__name__)
        calls = fallback_calls(state["question"])
    # Numeric-only questions must use their data tool rather than document prose.
    expected_data = data_fallback_calls(state['question'])
    if expected_data and not document_question(state['question']) and not refusal(state['question']):
        if expected_data[0]['name'] in STRUCTURED_TOOLS or range_requested(state['question']):
            # Date/compare arguments are deterministic and override a model's latest-score guess.
            calls = list(expected_data)
        else:
            calls = [c for c in calls if c.get('name') != 'search_docs']
        for expected in expected_data:
            if not any(c.get('name') == expected['name'] for c in calls):
                calls.append(expected)
    if (expected_data and expected_data[0]['name'] in P1_TOOLS and document_question(state['question'])
            and any(w in state['question'] for w in ('最新','当前','今天')) and not refusal(state['question'])):
        calls = fallback_calls(state['question'])
    # Preserve explicitly requested files when the model rewrites a search query.
    catalog = corpus_sources()
    requested = named_sources(state['question'], catalog, explicit=True)
    for call in calls:
        if call.get('name') == 'search_docs':
            args = dict(call.get('args', {}))
            query = args.get('query', '')
            if isinstance(query, str) and any(name not in named_sources(query, catalog, explicit=True) for name in requested):
                args['query'] = (state['question'] + ' ' + query)[:200]
                call['args'] = args
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


def p1_answer(question, result):
    """Compact evidence-only formatters; numerical verification remains unchanged."""
    name, data = result['tool'], result['data']
    if name == 'get_ppo_cached_weights':
        note = f"请求{data['requested_date']}无当日缓存；" if not data['exact'] else ''
        return note + f"PPO历史缓存权重日期{data['date']}：" + '、'.join(f'{asset}：{value*100:.1f}%' for asset,value in data['weights_cn'].items()) + f"；权重和{data['weight_sum']:g}"
    if name == 'get_pipeline_status':
        q = re.sub(r'\s+', '', question).upper()
        if any(word in q for word in ('滑点','成本','上限','回测口径')):
            p=data['protocol']
            return f"回测口径：单边滑点{p['SLIPPAGE']*100:.2f}%，双边成本{p['FEE']*100:.2f}%，单资产上限{p['MAX_WEIGHT']:g}"
        if 'IC' in q:
            return '年度IC：' + ('、'.join(f'{year}年{value:g}' for year,value in data['annual_ic'].items()) or '预测或价格产物缺失，无法计算')
        if any(word in q for word in ('形状','主表','建模表','特征维')):
            shapes=data['feature_shapes'] if any(word in q for word in ('建模表','特征')) else data['shapes']
            labels={'etf_price_clean':'ETF价格','train_total_feature':'总特征','market_score_daily':'市场评分','valuation_merged':'估值','tech_factor_clean':'技术因子','macro_merged':'宏观','train_feature_filtered':'LGB特征','train_feature_filtered_ppo':'PPO特征'}
            return '；'.join(labels[name]+' '+(item['shape']+' '+item['encoding'] if item else '不可读') for name,item in shapes.items())
        # Only answer the requested count; generic pipeline questions still get the summary below.
        if '预测' in q:
            return f"LGB有效预测{data['lgb_pred_count']}条"
        if any(word in q for word in ('文件','就绪','语义','原始','清洗')):
            return f"文件 {data['files_ready']}/{data['files_total']}、就绪校验 {data['semantics_ready']}/{data['semantics_total']}、原始 {data['raw_files']}、清洗 {data['clean_csvs']}"
        top,weight=data['top_feature'],data['ppo_weight']
        return ('Top1 '+top['feature']+f" {top['importance']:.4f}" if top else 'Top1产物不可读') + f"；LGB有效预测{data['lgb_pred_count']}条；" + (f"PPO缓存{weight['date']}，和{weight['sum']:g}，最小{weight['min']:g}" if weight else 'PPO缓存权重不可读')
    if name == 'get_risk_tier_detail':
        rows=data['rows']
        if len(rows)>2:
            return '风险档最终净值（组合倍数）：'+'；'.join(f"{row['风险档']}{row['策略']}{row['最终净值']:g}×" for row in rows)+'；请指定档位查看指标与配置'
        parts=[f"{row['风险档']}{row['策略']}：年化{row['年化收益']*100:.2f}%、回撤{row['最大回撤']*100:.2f}%、夏普{row['夏普比率']:g}、最终净值{row['最终净值']:g}×" for row in rows]
        if any(word in question for word in ('配置','ETF','权重')) and 'recommended_weights' in data:
            weights=data['recommended_weights']
            parts.append(f"静态推荐配置：五个权益ETF（沪深300/中证500/科创50/消费/芯片）各{weights['沪深300']*100:g}%，黄金/十年国债各{weights['黄金']*100:g}%")
        return '；'.join(parts)
    # NAV CSV values are portfolio market values, not normalized unit NAV.
    names=list(data['strategies'])
    note=''
    if len(names)>4:
        ends={name:data['summary'][name]['end_nav'] if data['mode']=='summary' else data['rows'][-1][name] for name in names}
        names=list(dict.fromkeys((max(ends,key=ends.get),min(ends,key=ends.get))))
        note='十策略仅列期末最高/最低；'
    if data['mode']=='summary':
        rows=[]
        for name in names:
            s=data['summary'][name]
            values = ([f"{s['display_nav_10k'][key]:.2f}" for key in ('start','end','max','min')] if len(names)==4
                      else [f'{value:.1f}' for value in (s['start_nav'],s['end_nav'],s['max']['nav'],s['min']['nav'])])
            rows.append(name+':'+'/'.join(values)+f"/{s['range_multiple']:g}×")
        unit = '（万）' if len(names)==4 else ''
        return note+f"{data['start_date']}至{data['end_date']}组合净值{unit}，首/末/高/低/区间倍数："+'；'.join(rows)+'；超过5个交易日，请缩小范围查看每日明细'
    if len(data['rows'])==1:
        row=data['rows'][0]
        return note+row['date']+'组合净值（市值）/基期倍数：'+'；'.join(f"{name}：{row[name]:.2f}/{data['multiples'][name]:g}×" for name in names)
    header='组合净值（市值），顺序'+ '/'.join(names)+'：'
    def line(row):
        return row['date']+' '+ '/'.join(f'{row[name]:.1f}' for name in names)
    text=header+'；'.join(line(row) for row in data['rows'])
    if len(text)>235:
        text=header+line(data['rows'][0])+'；'+line(data['rows'][-1])+'；仅列首末，请缩小范围查看全部每日值'
    return text


def evidence_answer(question, results):
    """模型不可用时，从结构化证据生成短答；不读取评测集。"""
    if any(r.get('tool') in P1_TOOLS for r in results):
        results = sorted(results, key=lambda r: r.get('tool') not in P1_TOOLS)
    parts = []
    for result in results:
        if not result["ok"]:
            continue
        name, data = result["tool"], result["data"]
        if name in P1_TOOLS:
            parts.append(p1_answer(question, result))
        elif name == "get_market_score":
            parts.append(f"{result['as_of']} 总分 {fmt(data['total_score'])}，状态「{data['market_state']}」；" +
                         "，".join(f"{label} {fmt(data[key])}" for label, key in (("估值", "val_score"), ("宏观", "macro_score"), ("情绪", "sent_score"), ("趋势", "trend_score"))))
        elif name == "compare_strategies":
            labels = {"年化收益": "年化", "夏普比率": "夏普", "最大回撤": "回撤", "卡玛比率": "卡玛",
                      "年化波动": "年化波动", "索提诺比率": "索提诺", "月均换手率": "换手", "胜率": "胜率",
                      "权重集中度HHI": "HHI", "95%VaR": "VaR", "年化收益/最大回撤": "收益回撤比"}
            ratio_question = any(word in question for word in ('收益回撤比','收益/回撤比'))
            wanted = ['年化收益/最大回撤'] if ratio_question else [key for key, label in labels.items() if key in question or label in question]
            # 年化波动 does not imply requesting annual return as well.
            if "年化波动" in question and "年化收益" not in question and "年化收益" in wanted:
                wanted.remove("年化收益")
            metrics = wanted or ["年化收益", "夏普比率", "最大回撤", "卡玛比率"]
            ratios = {"年化收益", "年化波动", "最大回撤", "月均换手率", "胜率", "95%VaR"}
            for row in data['rows']:
                parts.append(row['策略'] + '：' + '、'.join(labels[key] + fmt(row[key], key in ratios) for key in metrics))
        elif name == "get_market_score_history":
            if data['mode'] == 'summary':
                summary = data['summary']
                def item(key):
                    row = summary[key]
                    return row['date'] + ' ' + fmt(row['total_score']) + row.get('market_state', '')
                parts.append(f"{summary['start_date']}至{summary['end_date']}，{data['total_days']}个交易日；"
                             + '；'.join(label + item(key) for label, key in (("首", "first"), ("末", "last"), ("最高", "max"), ("最低", "min")))
                             + '；均值' + fmt(summary['mean_total']) + '；'
                             + '、'.join(f"{key}{value}天" for key, value in summary['state_counts'].items())
                             + '；区间超过10个交易日，请缩小范围查看每日明细')
            else:
                lines = []
                for row in data['rows']:
                    line = row['date'] + ' ' + fmt(row['total_score']) + row['market_state']
                    if data['mode'] == 'detail':
                        line += '(' + '/'.join(fmt(row[key]) for key in ('val_score', 'macro_score', 'sent_score', 'trend_score')) + ')'
                    lines.append(line)
                parts.append(('括号依次为估值/宏观/情绪/趋势：' if data['mode'] == 'detail' else '') + '；'.join(lines))
        elif name == "get_market_state_summary":
            parts.append("；".join(f"{k} {v} 天（{data['percentages'][k]:.1f}%）" for k, v in data["counts"].items()) + f"；合计 {data['total_days']} 天")
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
                    clause = clause.strip().lstrip("①②③④⑤⑥⑦⑧⑨⑩").strip()
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
                metrics = ['年化收益/最大回撤'] if name == 'get_strategy_metrics' and any(w in question for w in ('收益回撤比','收益/回撤比')) else wanted or metrics
                parts.append(label + "：" + "，".join(k + " " + fmt(row[k], k in ("累计收益", "年化收益", "年化波动", "最大回撤")) for k in metrics))
    return "；".join(parts) or "当前数据无法回答"


P1_TOOLS = {'get_strategy_nav', 'get_ppo_cached_weights', 'get_pipeline_status', 'get_risk_tier_detail'}
STRUCTURED_TOOLS = P1_TOOLS | {'compare_strategies', 'get_market_score_history'}


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
    if any(r['tool'] == 'get_pipeline_status' for r in data_results):
        primary = ('scripts/backtest_unified.py' if any(w in state['question'] for w in ('滑点','成本','上限','回测口径')) else
                   'results/pred_return_lgb_daily.csv' if 'IC' in state['question'].upper() or '预测' in state['question'] else 'data/clean/market_score_daily.csv')
        sources.sort(key=lambda source: source != primary)
    answer = evidence_answer(state["question"], usable)
    if not usable:
        answer = readonly_error(state["tool_results"], state["question"]) or answer
    if refusal(state["question"]):
        answer = POLICY + (evidence_answer(state["question"], [r for r in usable if r["tool"] == "search_docs"]) if docs else "")
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
                SystemMessage(content="只用提供的tool_results回答，数字只能来自证据。净值是约百万起步的组合市值及倍数，不能混同风险档最终净值。PPO必须写缓存权重及实际缓存日期，不得写实时推理或今日实时。流水线只答所问就绪数/IC/成本/形状；风险档配置为平台静态展示，不提供个性化建议。策略对比只列请求策略；未点名指标默认年化收益、夏普比率、最大回撤、卡玛比率。日期区间按mode输出：detail含全部每日分项，compact含全部每日总分状态，summary含首末、极值、均值、状态分布及缩小范围提示；不得编造未返回的日期或数字。中文简洁回答，正文不超过140字；引用和免责声明也计入总长300字。直接列短语，省略重复题目和文档全名。解释设计稿与最终代码差异时，须压缩保留证据中的全部关键差异（正向基准、惩罚项、附加奖励、裁剪），不得只列前三项。引用单独放末尾，不在正文重复免责声明。仅回答市场评分分项时使用估值/宏观/情绪/趋势标签。其他问题保留原主题名称，不强加评分标签。累计收益、年化收益、最大回撤、年化波动等比例字段，原始小数乘100写成百分比，保留两位小数，避免一位小数损失原始精度；夏普/卡玛/索提诺保留比率原值。市场评分和分项使用分制；交易日、维度/列数、天数、排名使用计数单位，均不得加百分号。问题点名某文档时，先回答该文档的内容，不得用另一文档替代；若与权威手册或最终代码不一致，须用一句话说明差异并明确“最终以代码为准”。文档解释至少逐字复制一条chunk的citation字段，包括文件名、章节、空格和方括号，不得自行改写、拼接或使用text内的旧引用，不得编造引用；混合问题即使已有数据工具的[source:]，也必须引用chunk的citation，不能用数据引用替代文档引用。多个来源优先引用被点名文档，涉及最终代码差异时也引用最终代码依据；选择直接支持答案的章节。文档中的数字仅authoritative可用，background/historical只解释背景，不得引用其事实数字。旧值/禁用清单只可标为旧值，不得作为当前事实。涉及数据口径时说明实际数据以数据工具为准。公式优先逐字照录权威原文，系数保留原文小数位，不能省略零系数或成本说明。发现文档算术矛盾时仅说明矛盾并照录原值，不得计算或写出证据没有的合计数。不提供个性化投资建议。"),
                HumanMessage(content=json.dumps({"question": state["question"], "tool_results": evidence,
                                                 "correction": state.get("correction", "")}, ensure_ascii=False))])
            events.append({"stage": "compose", "status": "success"})
            if isinstance(response.content, str) and response.content.strip():
                answer = response.content.strip()
        except Exception as exc:
            events.append({"stage": "compose", "status": "error", "error": type(exc).__name__})
            log.warning("compose LLM failed: %s", type(exc).__name__)
    if not refusal(state['question']) and any(
            r['tool'] in STRUCTURED_TOOLS for r in data_results) and (not docs or any(r['tool'] in P1_TOOLS for r in data_results)):
        answer = evidence_answer(state['question'], usable)
    if "北向" in state["question"] and any(r["tool"] == "get_data_caveats" for r in usable) and "北向" not in answer:
        answer = "北向资金：" + answer
    # Preserve the requested identity of structured rows, including tiers and crises.
    question_key = re.sub(r"\s+", "", state['question']).casefold()
    labels = list(dict.fromkeys(str(row.get(key) or '')
                  for result in data_results for row in result['data'].get('rows', [])
                  for key in ('风险档', '危机阶段', '阶段', '策略', '策略源')))
    requested_labels = []
    for label in labels:
        key = re.sub(r"\s+", "", label).casefold()
        identifiers = re.findall(r'[a-z]+\d+|(?:19|20)\d{2}', key)
        if key and (key in question_key or any(token in question_key for token in identifiers)):
            requested_labels.append(label)
    answer_key = re.sub(r"\s+", "", answer).casefold()
    missing_labels = [label for label in requested_labels if re.sub(r"\s+", "", label).casefold() not in answer_key]
    if missing_labels:
        answer = '、'.join(missing_labels) + '：' + answer
    matched, _ = citation_matches(answer, docs)
    doc_references = list(dict.fromkeys(citation(c) for c in matched))
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
    limit = 300 - len(footer) - len(suffix) - 1
    if len(body) > limit:
        kept = ""
        for clause in re.findall(r"[^。；\n]+[。；\n]?", body):
            if len(kept + clause) > limit:
                break
            kept += clause
        body = kept or "当前数据无法回答：回答超出长度限制。"
    answer = sentence_end(body) + suffix + footer
    return dict(answer=answer, citations=list(dict.fromkeys(sources+doc_references)), attempts=state["attempts"]+1, llm_events=events)


def numeric_values(value):
    if isinstance(value, dict):
        return [n for v in value.values() for n in numeric_values(v)]
    if isinstance(value, list):
        return [n for v in value for n in numeric_values(v)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, str):
        text = numeric_text(strip_dates(value))
        text = re.sub(r"(?<=\d)[-–—](?=\d)", " 至 ", text)
        return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text)]
    return []


def strip_dates(text):
    text = normalize_text(text)
    text = re.sub(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日", "", text)
    return re.sub(r"(?<!\d)(?:19|20)\d{2}年?(?!\d)", "", text)


def score_unit_error(body, results):
    body = normalize_text(body)
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
    needs_reference = bool(docs)
    missing_named = [name for name in named_sources(state['question'], docs + corpus_sources(), explicit=True)
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
    docs = retrieved_chunks(state.get('tool_results', []))
    matched, invalid = citation_matches(state.get('answer', ''), docs)
    if docs and not state.get('verified') and (not matched or invalid):
        return dict(answer='未在检索片段中找到可引用依据，无法回答。不构成投资建议。',
                    citations=[], verified=False)
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
