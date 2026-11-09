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
    from app.rag import retriever
    monkeypatch.setattr(retriever, 'encode', lambda _: (_ for _ in ()).throw(RuntimeError('offline embedding fixture')))


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
    monkeypatch.setattr(api, 'validate', lambda *_: {'backend': 'hybrid'})
    assert client.get('/healthz').json()['rag_enabled'] is True
    assert client.get('/healthz').json()['vector_rag_enabled'] == api.settings.enable_vector_rag
    assert len(client.get('/tools').json()) == 9
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


def test_caveat_answer_keeps_subject_when_model_omits_it(monkeypatch):
    class BriefLLM(OfflineLLM):
        def invoke(self, messages):
            return SimpleNamespace(content='缺失975行，填零457行。')
    monkeypatch.setattr(g, 'get_llm', lambda: BriefLLM())
    state = g.intake({'question': '北向资金数据缺口'})
    state['tool_results'] = [registry.get_data_caveats()]
    answer = g.compose(state)
    assert answer['answer'].startswith('北向资金：')
    assert g.verify(state | answer)['verified']


@pytest.mark.parametrize('answer,valid', [
    ('市场总评分为−33.90%', False), ('市场总评分为－33.90%', False),
    ('市场总评分为-33.90%', False), ('市场总评分为33.90%', False),
    ('市场总评分为33.90分', True), ('年化−13.13%，最大回撤−14.27%', True),
    ('市场总评分为﹣３３．９０％', False),
])
def test_unicode_numeric_units(answer, valid):
    state = {'question': '评分和收益', 'answer': answer, 'tool_results': [
        {'ok': True, 'tool': 'get_market_score', 'data': {'score': 33.9}},
        {'ok': True, 'tool': 'get_strategy_metrics', 'data': {'return': -.1313, 'drawdown': -.1427}}]}
    assert g.verify(state)['verified'] is valid


def test_unicode_numeric_dates_and_eval():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/'eval/run_eval.py'
    spec = importlib.util.spec_from_file_location('eval_numbers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert g.numeric_values('−33.90') == [-33.9]
    assert module.extract_numbers('−33.90%') == [-33.9]
    assert 975 in module.extract_numbers('2018-12-28，975行')
    assert module.extract_numbers('1,317天') == [1317]
    assert module.extract_numbers('clip[-5,5]') == [-5, 5]
    assert module.extract_numbers('(10.0,0.5,0.0,1.0)') == [10.0, .5, 0.0, 1.0]
    for text in ['2020–2021', '2020-2021', '2020-01-02', '21—40']:
        assert all(n >= 0 for n in g.numeric_values(text))
        assert all(n >= 0 for n in module.extract_numbers(text))
    assert g.numeric_values('−14.27%') == [-14.27]


@pytest.mark.parametrize('broken', [True, False])
def test_healthz_checks_index(monkeypatch, broken):
    def validate(*_):
        if broken:
            raise ValueError('索引缺失或语料缺失，请重建：.venv/bin/python -m app.rag.build_index')
        return {'backend': 'hybrid', 'built_at': 'test'}
    monkeypatch.setattr(api, 'validate', validate)
    monkeypatch.setattr(api, 'settings', dataclasses.replace(api.settings, enable_vector_rag=True))
    response = TestClient(api.app).get('/healthz')
    assert response.status_code == 200
    data = response.json()
    assert data['rag_enabled'] is not broken
    if broken:
        assert data['status'] == 'degraded' and data['rag_backend'] is None
        assert 'app.rag.build_index' in data['rag_error']
    else:
        assert data['status'] == 'ok' and data['rag_backend'] == 'hybrid' and data['rag_error'] is None


def test_healthz_sanitizes_arbitrary_error(monkeypatch):
    monkeypatch.setattr(api, 'validate', lambda *_: (_ for _ in ()).throw(ValueError('secret cache path')))
    assert 'secret' not in api.healthz()['rag_error']


def test_router_preserves_explicit_document_name(monkeypatch):
    class Rewriter(OfflineLLM):
        def invoke(self, _):
            return SimpleNamespace(tool_calls=[{'name': 'search_docs', 'args': {'query': 'PPO奖励函数'}}])
    monkeypatch.setattr(g, 'get_llm', lambda: Rewriter())
    state = g.intake({'question': '强化学习设计文档中，奖励函数是什么？'})
    assert '强化学习设计文档' in g.router(state)['pending'][0]['args']['query']


def test_binary_subtraction_does_not_invent_negative_literals():
    assert g.numeric_values('PMI−48)/4') == [48.0, 4.0]
    assert g.numeric_values('pe_pct-0.30') == [.3]
    assert g.numeric_values('(收益−基准)−0.5×波动−1.0×回撤') == [.5, 1.0]
    assert g.numeric_values('年化−13.13%，最大回撤−14.27%') == [-13.13, -14.27]
    state = {'question': '规则', 'answer': '公式(PMI−48)/4', 'tool_results': [
        {'ok': True, 'tool': 'metric', 'data': {'x': 48, 'denom': 4}}]}
    assert g.verify(state)['verified']
    state['answer'] = '最大回撤−48%'
    assert not g.verify(state)['verified']


def test_strategy_answer_retains_requested_tool_row_label(monkeypatch):
    class BareMetrics(OfflineLLM):
        def invoke(self, _):
            return SimpleNamespace(content='年化16.52%，夏普0.6006，最大回撤-26.56%。')
    monkeypatch.setattr(g, 'get_llm', lambda: BareMetrics())
    state = g.intake({'question': 'SASF 硬切换策略的年化收益、夏普和最大回撤分别是多少？'})
    state['tool_results'] = [registry.get_strategy_metrics(strategy='SASF 硬切换')]
    output = g.compose(state)
    assert 'SASF' in output['answer']
    assert g.verify(state | output)['verified']


def test_numeric_question_cannot_use_docs_instead_of_data(monkeypatch):
    class WrongRoute(OfflineLLM):
        def invoke(self, _):
            return SimpleNamespace(tool_calls=[{'name': 'search_docs', 'args': {'query': '动态评分年化'}}])
    monkeypatch.setattr(g, 'get_llm', lambda: WrongRoute())
    state = g.intake({'question': '动态评分策略全区间年化收益和最大回撤是多少？'})
    names = [c['name'] for c in g.router(state)['pending']]
    assert names == ['get_strategy_metrics']


def test_metric_answer_preserves_tier_and_crisis(monkeypatch):
    class BareRows(OfflineLLM):
        def invoke(self, _):
            return SimpleNamespace(content='动态评分年化8.59%，最大回撤-9.72%。')
    monkeypatch.setattr(g, 'get_llm', lambda: BareRows())
    state = g.intake({'question': 'C2保守档动态评分的年化和最大回撤是多少？'})
    state['tool_results'] = [registry.get_risk_tier(level='C2', strategy='动态评分')]
    answer = g.compose(state)
    assert all(word in answer['answer'] for word in ['C2', '保守'])
    class Crisis(BareRows):
        def invoke(self, _):
            return SimpleNamespace(content='等权修复435天。')
    monkeypatch.setattr(g, 'get_llm', lambda: Crisis())
    state = g.intake({'question': '2018熊市下等权策略的修复天数是多少？'})
    state['tool_results'] = [registry.get_drawdown_recovery(crisis='2018')]
    assert '2018' in g.compose(state)['answer']
