"""Sentence ending regressions for both answer paths."""
from types import SimpleNamespace
import pytest
from app import graph as g, chat
from app.text import sentence_end
from app.tools import registry


@pytest.mark.parametrize("ending,expected", [("，","。"),("、","。"),("；","。"),("：","。"),(".","。"),("！","！"),("？","？"),("。","。"),("   ","。"),("。。","。"),("！！","！"),("，。。  ","。")])
def test_both_answer_paths(monkeypatch, ending, expected):
    body = "证据说明" + ending
    assert sentence_end(body) == "证据说明" + expected
    monkeypatch.setattr(g, "get_llm", lambda: SimpleNamespace(invoke=lambda _: SimpleNamespace(content=body)))
    state = g.intake({"question":"最新评分"})
    state['tool_results'] = [registry.get_market_score()]
    result = g.compose(state)['answer']
    assert result == "证据说明" + expected + "不构成投资建议。\n[source: data/clean/market_score_daily.csv]"
    assert chat.guard(body) == ("证据说明" + expected + "不构成投资建议。", "chat")


@pytest.mark.parametrize("body", ["", " ，。； ", "。。", "\n"])
def test_empty_body_fallback(body):
    assert sentence_end(body, "请补充问题") == "请补充问题。"
    assert chat.guard(body)[0] == "请补充问题。不构成投资建议。"


def test_two_suffixes_and_footer(monkeypatch):
    from app.rag.citations import citation
    chunk = dict(source_name="手册.md", section="规则", text="规则说明", authority="authoritative", chunk_id="test")
    body = "规则说明，实际数据以数据工具为准。不构成投资建议。" + citation(chunk)
    monkeypatch.setattr(g, "get_llm", lambda: SimpleNamespace(invoke=lambda _: SimpleNamespace(content=body)))
    state = g.intake({"question":"数据口径"})
    state['tool_results'] = [dict(ok=True,tool="search_docs",data={"chunks":[chunk]},source="手册.md")]
    result = g.compose(state)['answer']
    assert "规则说明。实际数据以数据工具为准。不构成投资建议。" in result
    assert result.endswith(citation(chunk))
    assert result.count("不构成投资建议。") == 1
    assert len(result) <= 300


@pytest.mark.parametrize("body", [
    "SASF硬切换：年化收益16.52%，夏普0.6006，最大回撤-26.56%。数据截至2026-07-10，",
    "PPO月度策略最大回撤：-36.32%。数据截至2026-07-10，",
    "进取型C4（权益中枢70%，LGB融合）：年化收益15.96%，最大回撤-27.70%。数据截至2026-07-10，",
])
def test_exact_phase6_bodies(monkeypatch, body):
    monkeypatch.setattr(g, "get_llm", lambda: SimpleNamespace(invoke=lambda _: SimpleNamespace(content=body)))
    state = g.intake({"question":"回放正文"})
    state['tool_results'] = [registry.get_strategy_metrics()]
    result = g.compose(state)['answer']
    assert "，。" not in result
    assert body.rstrip("，") + "。不构成投资建议。" in result
    assert result.endswith("[source: results/backtest/all_strategy_metrics.csv]")
    assert len(result) <= 300


def test_normalization_respects_budgets(monkeypatch):
    assert len(chat.guard("长"*500)[0]) <= 400
    monkeypatch.setattr(g, "get_llm", lambda: SimpleNamespace(invoke=lambda _: SimpleNamespace(content="长"*500)))
    state = g.intake({"question":"评分"})
    state['tool_results'] = [registry.get_market_score()]
    assert len(g.compose(state)['answer']) <= 300
