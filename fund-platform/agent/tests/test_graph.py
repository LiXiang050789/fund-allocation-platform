import dataclasses
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.graph as g
import app.main as api
from app.tools import registry


class OfflineLLM:
    def bind_tools(self, schemas):
        return self

    def invoke(self, messages):
        raise RuntimeError('offline mock')


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(g, 'get_llm', lambda: OfflineLLM())


@pytest.mark.parametrize('question,name', [
    ('最新评分', 'get_market_score'), ('十套策略夏普', 'get_strategy_metrics'),
    ('2018熊市分段收益', 'get_segment_performance'), ('2015股灾修复天数', 'get_drawdown_recovery'),
    ('C2保守风险档', 'get_risk_tier'), ('特征重要性', 'get_feature_importance'),
    ('北向数据口径', 'get_data_caveats'), ('全区间状态各有多少天', 'get_market_state_summary')])
def test_fallback(question, name):
    result = g.graph.invoke({'question': question})
    assert name in result['route']
    assert len(result['answer']) <= 300


def test_numeric_verification():
    state = {'question': '收益', 'answer': '年化13.13%', 'tool_results': [{'ok': True, 'data': {'return': .1313}}]}
    assert g.verify(state)['verified']
    state['answer'] = '62.7%'
    assert not g.verify(state)['verified']
    state['answer'] = '2026-07-10、2026/7/10、2026年7月10日、2021年，年化13.13%'
    assert g.verify(state)['verified']


@pytest.mark.parametrize('question', ['推荐股票涨停全仓', '保证收益', '明天涨跌'])
def test_refusal(question):
    state = g.graph.invoke({'question': question})
    assert any(w in state['answer'] for w in ('不构成投资建议', '无法预测', '不能保证'))
    assert state['verified']


def test_throwing_tool(monkeypatch):
    def fail(*args):
        raise RuntimeError('injected')
    monkeypatch.setattr(registry, 'execute', fail)
    state = g.graph.invoke({'question': '最新评分'})
    assert not state['tool_results'][0]['ok']
    assert state['rounds'] == 2
    assert '当前数据无法回答' in state['answer']


def test_unknown_and_empty_llm_calls(monkeypatch):
    for calls in ([], [{'name': 'unknown', 'args': {}}]):
        llm = OfflineLLM()
        llm.invoke = lambda messages: SimpleNamespace(tool_calls=calls)
        monkeypatch.setattr(g, 'get_llm', lambda: llm)
        state = g.intake({'question': '最新评分'})
        assert g.router(state | {'question': '最新评分'})['pending'][0]['name'] == 'get_market_score'


def test_verification_retry_bounded(monkeypatch):
    class BadComposer(OfflineLLM):
        def invoke(self, messages):
            if '只选择' in messages[0].content:
                return SimpleNamespace(tool_calls=[])
            return SimpleNamespace(content='评分 9999999')
    monkeypatch.setattr(g, 'get_llm', lambda: BadComposer())
    state = g.graph.invoke({'question': '最新评分'})
    assert not state['verified'] and state['attempts'] == 2
    assert '9999999' in state['answer']


def test_api(monkeypatch, tmp_path):
    client = TestClient(api.app)
    result = client.post('/chat', json={'question': '最新市场评分和状态', 'thread_id': 'test'})
    assert result.status_code == 200
    data = result.json()
    assert '33.9' in data['answer'] and '下行' in data['answer'] and data['verified']
    assert data['thread_id'] == 'test' and data['tool_calls'][0]['ok']
    assert client.post('/chat', json={'question': '  '}).status_code == 422
    assert client.get('/healthz').json()['rag_enabled'] is False
    assert len(client.get('/tools').json()) == 8
    monkeypatch.setattr(registry, 'settings', dataclasses.replace(registry.settings, score_path=tmp_path/'missing.csv'))
    data = client.post('/chat', json={'question': '最新评分'}).json()
    assert not data['tool_calls'][0]['ok'] and '当前数据无法回答' in data['answer']


def test_internal_failure(monkeypatch):
    monkeypatch.setattr(api, 'graph', SimpleNamespace(invoke=lambda _: (_ for _ in ()).throw(RuntimeError('injected'))))
    response = TestClient(api.app).post('/chat', json={'question': '评分'})
    assert response.status_code == 200 and response.json()['answer'] == '服务暂时不可用'


def test_llm_success_and_tool_repair(monkeypatch):
    class WorkingLLM(OfflineLLM):
        rounds = 0

        def invoke(self, messages):
            if '只选择' in messages[0].content:
                self.rounds += 1
                args = {'strategy': '不存在'} if self.rounds == 1 else {'strategy': '动态评分'}
                return SimpleNamespace(tool_calls=[{'name': 'get_strategy_metrics', 'args': args}])
            return SimpleNamespace(content='动态评分：年化13.13%，最大回撤-14.27%。')

    llm = WorkingLLM()
    monkeypatch.setattr(g, 'get_llm', lambda: llm)
    state = g.graph.invoke({'question': '动态评分策略年化收益和最大回撤'})
    assert state['verified'] and state['rounds'] == 2
    assert [r['ok'] for r in state['tool_results']] == [False, True]
    assert state['citations'] == ['results/backtest/all_strategy_metrics.csv']


def test_long_answer_keeps_complete_numbers(monkeypatch):
    class LongComposer(OfflineLLM):
        def invoke(self, messages):
            if '只选择' in messages[0].content:
                return SimpleNamespace(tool_calls=[])
            return SimpleNamespace(content=('历史数据说明；' * 60) + '市场评分33.9。')

    monkeypatch.setattr(g, 'get_llm', lambda: LongComposer())
    state = g.graph.invoke({'question': '最新评分'})
    assert len(state['answer']) <= 300
    assert '不构成投资建议' in state['answer']
    assert state['citations'][0] in state['answer']


def test_caveat_citations_follow_evidence_topic():
    north = g.graph.invoke({'question': '北向资金数据缺口'})
    assert north['citations'][0] == 'data/clean/train_total_feature.csv'
    assert '[source: data/clean/train_total_feature.csv]' in north['answer']
    macro = g.graph.invoke({'question': '宏观指标滞后口径'})
    assert macro['citations'][0] == 'scripts/calc_market_score.py'
    assert '[source: scripts/calc_market_score.py]' in macro['answer']


@pytest.mark.parametrize('answer,valid', [
    ('市场总评分为33.90%', False), ('市场总评分为33.90分', True),
    ('分项：估值0.00%、宏观14.08%', False), ('年化13.13%，最大回撤-14.27%', True),
    ('市场评分33.90分；年化13.13%', True), ('市场评分为 **33.90％**', False),
])
def test_score_units(answer, valid):
    state = {'question': '评分和收益', 'answer': answer, 'tool_results': [
        {'ok': True, 'tool': 'get_market_score', 'data': {'score': 33.9, 'val': 0, 'macro': 14.08}},
        {'ok': True, 'tool': 'get_strategy_metrics', 'data': {'return': .1313, 'drawdown': -.1427}}]}
    assert g.verify(state)['verified'] is valid


def test_score_unit_repair_is_bounded(monkeypatch):
    class ScoreLLM(OfflineLLM):
        def invoke(self, messages):
            if '只选择' in messages[0].content:
                return SimpleNamespace(tool_calls=[])
            return SimpleNamespace(content='市场总评分为33.90%')
    monkeypatch.setattr(g, 'get_llm', lambda: ScoreLLM())
    state = g.graph.invoke({'question': '最新评分'})
    assert not state['verified'] and state['attempts'] == 2
    assert '分制' in state['error']
