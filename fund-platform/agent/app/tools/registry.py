"""只读工具白名单、输入校验、统一错误协议与文件缓存。"""
from __future__ import annotations

import ast
import functools
import inspect
import json
import logging
import re
import time
from datetime import date as Date
from pathlib import Path
from typing import Annotated, get_type_hints

import pandas as pd
from pydantic import ConfigDict, Field, create_model

from app.config import settings

log = logging.getLogger(__name__)
REGISTRY = {}
TOOL_SCHEMAS = []


@functools.lru_cache(maxsize=32)
def _read(path: str, mtime: int, size: int, encoding: str, cols: tuple | None, header: bool):
    return pd.read_csv(path, encoding=encoding, usecols=list(cols) if cols else None,
                       nrows=0 if header else None)


def table(relative, *, encoding="utf-8-sig", cols=None, header=False):
    path = settings.score_path if relative == "data/clean/market_score_daily.csv" else settings.platform_dir / relative
    stat = path.stat()  # Stat before cache lookup: removed files must fail, even after warmup.
    return _read(str(path), stat.st_mtime_ns, stat.st_size, encoding, cols, header).copy()


def tool(source):
    def decorate(fn):
        sig, hints = inspect.signature(fn), get_type_hints(fn, include_extras=True)
        model = create_model(fn.__name__ + "Args", __config__=ConfigDict(extra="forbid", strict=True),
                             **{k: (hints[k], ... if p.default is inspect.Parameter.empty else p.default) for k, p in sig.parameters.items()})

        @functools.wraps(fn)
        def safe(*positional, **args):
            started = time.perf_counter()
            result = dict(ok=False, tool=fn.__name__, as_of=None, data={}, source=source, error=None)
            try:
                args = sig.bind(*positional, **args).arguments
                validated = model(**args).model_dump()
                payload = fn(**validated)
                data, as_of = payload[:2]
                result.update(ok=True, data=data, as_of=as_of)
                if len(payload) == 3:
                    result["source"] = payload[2]
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            log.info(json.dumps(dict(name=fn.__name__, args=args,
                     latency_ms=round((time.perf_counter()-started)*1000, 2),
                     ok=result["ok"], error=result["error"]), ensure_ascii=False, default=str))
            return result

        REGISTRY[fn.__name__] = safe
        TOOL_SCHEMAS.append({"type": "function", "function": {
            "name": fn.__name__, "description": fn.__doc__ or fn.__name__,
            "parameters": model.model_json_schema()}})
        return safe
    return decorate


def select(df, column, value):
    if value is None:
        return df
    key = str(value).replace(" ", "").lower()
    names = df[column].astype(str).str.replace(" ", "").str.lower()
    if column in {"阶段", "危机阶段"}:
        def normalize_phase(text):
            return text.replace("期间", "").replace("阶段", "").replace("年", "").removesuffix("期")
        key = normalize_phase(key)
        names = names.map(normalize_phase)
    mask = names.eq(key)
    if not mask.any():
        mask = names.str.contains(key, regex=False)
    if not mask.any() and column in {"阶段", "危机阶段"} and len(key) >= 2:
        # Permit omitted qualifiers (e.g. a crisis's half-year qualifier) in natural labels.
        mask = names.str.contains(".*?".join(re.escape(char) for char in key), regex=True)
    if not mask.any():
        raise ValueError(f"未知{column}: {value}；可选值: {', '.join(df[column].unique())}")
    return df[mask]


def records(df):
    return json.loads(df.to_json(orient="records", force_ascii=False))


def scores():
    return table("data/clean/market_score_daily.csv").sort_values("date")


def iso(value):
    return Date.fromisoformat(value).isoformat()


@tool("data/clean/market_score_daily.csv")
def get_market_score(date: str | None = None):
    """获取指定日期或最新交易日的市场总分、五态标签与四项分数。"""
    df = scores()
    if date:
        df = df[df.date <= iso(date)]
    if df.empty:
        raise ValueError("日期越界：早于数据起始日")
    row = records(df.tail(1))[0]
    return {k: v for k, v in row.items() if k != "date"}, row["date"]


@tool("data/clean/market_score_daily.csv")
def get_market_state_summary(start: str | None = None, end: str | None = None):
    """统计日期区间内五种市场状态天数、交易日总数与末日状态。"""
    df = scores()
    if start:
        df = df[df.date >= iso(start)]
    if end:
        df = df[df.date <= iso(end)]
    if df.empty:
        raise ValueError("日期越界或区间无交易日")
    counts = df.market_state.value_counts()
    latest = records(df[["date", "total_score", "market_state"]].tail(1))[0]
    return dict(counts={s: int(counts.get(s, 0)) for s in ("震荡", "震荡偏强", "下行", "上行", "极寒")},
                total_days=len(df), latest=latest), latest["date"]


@tool("results/backtest/all_strategy_metrics.csv")
def get_strategy_metrics(strategy: str | None = None):
    """查询十种策略的全区间收益、波动、夏普、回撤和卡玛指标。"""
    df = select(table("results/backtest/all_strategy_metrics.csv"), "策略", strategy)
    return {"rows": records(df)}, scores().date.iloc[-1]


STRATEGY_ALIASES = {
    "LGB": ("LGB融合",), "LGB融合": ("LGB融合",),
    "SASF": ("SASF硬切换",), "SASF硬切换": ("SASF硬切换",),
    "PPO": ("PPO月度", "PPO原生日频"), "PPO月度": ("PPO月度",),
    "PPO原生": ("PPO原生日频",), "PPO原生日频": ("PPO原生日频",),
    **{name: (name,) for name in ("动态评分", "沪深300", "等权", "风险平价", "MVO", "动量")},
}


def expand_strategies(strategies):
    expanded = []
    for strategy in strategies:
        key = re.sub(r"\s+", "", strategy).upper()
        if key not in STRATEGY_ALIASES:
            available = list(dict.fromkeys(name for names in STRATEGY_ALIASES.values() for name in names))
            raise ValueError(f"未知策略：{strategy}；可用策略：{'、'.join(available)}")
        expanded.extend(STRATEGY_ALIASES[key])
    return list(dict.fromkeys(expanded))


def strategies_in_question(question):
    # Longest aliases consume their whole span, so PPO月度 never expands as bare PPO.
    pattern = '|'.join(re.escape(name) for name in sorted(STRATEGY_ALIASES, key=len, reverse=True))
    return expand_strategies(re.findall(pattern, re.sub(r"\s+", "", question).upper()))


@tool("results/backtest/all_strategy_metrics.csv")
def compare_strategies(strategies: list[str]):
    """按请求顺序对比2至4个策略的CSV指标；PPO同时展开月度与原生日频。"""
    names = expand_strategies(strategies)
    if len(names) < 2:
        raise ValueError("对比至少需要两个策略")
    if len(names) > 4:
        raise ValueError("一次最多对比四个策略")
    df = table("results/backtest/all_strategy_metrics.csv")
    rows = []
    for name in names:
        selected = df[df['策略'].eq(name)]
        if len(selected) != 1:
            raise ValueError(f"策略指标行缺失或不唯一：{name}")
        rows.extend(records(selected))
    return {"rows": rows}, scores().date.iloc[-1]


@tool("data/clean/market_score_daily.csv")
def get_market_score_history(start: str, end: str):
    """精确查询交易日评分区间：最多5日分项、10日总分，长区间返回汇总。"""
    if not all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) for value in (start, end)):
        raise ValueError("日期必须为YYYY-MM-DD")
    try:
        start, end = iso(start), iso(end)
    except ValueError as exc:
        raise ValueError("日期无效，请使用有效年月日") from exc
    if start > end:
        raise ValueError("起止日期倒置")
    df = scores()
    first_date, last_date = df.date.iloc[0], df.date.iloc[-1]
    if start < first_date or end > last_date:
        raise ValueError(f"日期越界：可用范围 {first_date} 至 {last_date}")
    df = df[df.date.between(start, end)]
    if df.empty:
        raise ValueError("区间内无交易日")
    columns = ['date', 'total_score', 'market_state']
    count = len(df)
    mode = 'detail' if count <= 5 else 'compact' if count <= 10 else 'summary'
    data = dict(mode=mode, total_days=count, rows=[])
    if mode != 'summary':
        data['rows'] = records(df[columns + (['val_score', 'macro_score', 'sent_score', 'trend_score'] if mode == 'detail' else [])])
    else:
        counts = df.market_state.value_counts()
        data['summary'] = dict(
            start_date=start, end_date=end,
            first=records(df[columns].head(1))[0], last=records(df[columns].tail(1))[0],
            max=records(df.loc[[df.total_score.idxmax()], ['date', 'total_score']])[0],
            min=records(df.loc[[df.total_score.idxmin()], ['date', 'total_score']])[0],
            mean_total=round(float(df.total_score.mean()), 2),
            state_counts={state: int(counts.get(state, 0)) for state in ('震荡', '震荡偏强', '下行', '上行', '极寒')})
    return data, df.date.iloc[-1]


@tool("results/backtest/segment_all_strategy.csv")
def get_segment_performance(segment: str | None = None):
    """查询六阶段内各策略的累计收益、年化收益、最大回撤。"""
    df = select(table("results/backtest/segment_all_strategy.csv"), "阶段", segment)
    rows = []
    for raw in records(df):
        for key in raw:
            if key.endswith("_累计收益"):
                strategy = key.rsplit("_", 1)[0]
                rows.append({"阶段": raw["阶段"], "交易日天数": raw["交易日天数"], "策略": strategy,
                             **{m: raw[f"{strategy}_{m}"] for m in ("累计收益", "年化收益", "最大回撤")}})
    return {"rows": rows}, scores().date.iloc[-1]


@tool("PPT素材/4-压力测试表.csv")
def get_drawdown_recovery(crisis: str | None = None):
    """查询危机阶段各策略的回撤、收益和修复天数；危机名称支持包含匹配。"""
    df = table("PPT素材/4-压力测试表.csv")
    df[["危机阶段", "起始", "结束"]] = df[["危机阶段", "起始", "结束"]].ffill()
    df = select(df, "危机阶段", crisis)
    return {"rows": records(df)}, scores().date.iloc[-1]


@tool("results/risk_levels_metrics.csv")
def get_risk_tier(level: str | None = None, strategy: str | None = None):
    """查询保守C2、平衡C3、进取C4下各策略的收益与风险。"""
    df = select(table("results/risk_levels_metrics.csv"), "风险档", level)
    df = select(df, "策略源", strategy)
    return {"rows": records(df)}, scores().date.iloc[-1]


@tool("results/feature_importance.csv")
def get_feature_importance(top_n: int = 10):
    """按重要性降序获取前若干特征，数量限制为1至50。"""
    if not 1 <= top_n <= 50:
        raise ValueError("top_n 必须在 1-50")
    df = table("results/feature_importance.csv").sort_values("importance", ascending=False).head(top_n)
    return {"rows": records(df)}, scores().date.iloc[-1]


@tool("data/clean/market_score_daily.csv; data/clean/train_total_feature.csv; data/clean/train_feature_filtered.csv; data/clean/train_feature_filtered_ppo.csv; scripts/calc_market_score.py")
def get_data_caveats():
    """计算数据覆盖、北向缺口、月度宏观滞后、特征维度及历史数据时效。"""
    df = scores()
    north = table("data/clean/train_total_feature.csv", encoding="gbk", cols=("date", "north_net")).sort_values("date")
    north["date"] = pd.to_datetime(north["date"]).dt.strftime("%Y-%m-%d")
    missing = north[north.north_net.isna()]
    valid = north[north.north_net.notna() & north.north_net.ne(0)]
    trailing = north.north_net.eq(0).iloc[::-1].cummin().iloc[::-1]
    zero = north[trailing]
    script = settings.platform_dir / "scripts/calc_market_score.py"
    tree = ast.parse(script.read_text(encoding="utf-8-sig"))
    assignments = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "MACRO_LAG" for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError("无法唯一解析 MACRO_LAG")
    lag = ast.literal_eval(assignments[0])
    if type(lag) is not int or lag < 0:
        raise ValueError("MACRO_LAG 不是非负整数常量")
    dimensions = {}
    for label, filename in (("all", "train_total_feature.csv"), ("LGB", "train_feature_filtered.csv"), ("PPO", "train_feature_filtered_ppo.csv")):
        columns = table("data/clean/" + filename, encoding="gbk" if label == "all" else "utf-8-sig", header=True).columns
        dimensions[label] = {"columns": len(columns), "features": len(columns.drop("date"))}
    cutoff = Date.fromisoformat(df.date.iloc[-1])
    today = Date.today()
    months = max(0, (today.year-cutoff.year)*12 + today.month-cutoff.month - (today.day < cutoff.day))
    return dict(start=df.date.iloc[0], end=df.date.iloc[-1], trading_days=len(df),
                nan_range=[missing.date.iloc[0], missing.date.iloc[-1]] if len(missing) else None,
                nan_rows=len(missing), zero_start=zero.date.iloc[0] if len(zero) else None,
                zero_rows=len(zero), last_valid_date=valid.date.iloc[-1] if len(valid) else None,
                macro_lag=lag, dimensions=dimensions, months_since_cutoff=months,
                caveat="仅为历史回测演示数据，不是实时行情；宏观月度指标按交易日滞后以防前视"), df.date.iloc[-1]


@tool("")
def search_docs(query: Annotated[str, Field(min_length=1, max_length=200)],
                top_k: Annotated[int, Field(ge=1, le=10)] = 5):
    """检索文档规则、定义、问卷、费率、PPO设计与防前视解释；不查询当前策略指标。"""
    if not query.strip():
        raise ValueError("query 不得为空")
    from app.rag.retriever import search
    data, as_of = search(query.strip(), top_k)
    sources = list(dict.fromkeys(c["source_name"] for c in data["chunks"]))[:5]
    return data, as_of, "; ".join(sources)


def execute(name, args):
    started = time.perf_counter()
    try:
        if name not in REGISTRY:
            raise ValueError(f"未知工具: {name}")
        return REGISTRY[name](**args)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log.info(json.dumps(dict(name=name, args=args,
                 latency_ms=round((time.perf_counter()-started)*1000, 2), ok=False, error=error),
                 ensure_ascii=False, default=str))
        return dict(ok=False, tool=name, as_of=None, data={}, source="", error=error)
