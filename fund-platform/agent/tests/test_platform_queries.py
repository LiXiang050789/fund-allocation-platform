"""Real CSV query regressions; all model responses are local fakes."""
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from app import graph as g, main
from app.tools import registry as t
from app.dates import parse_date_range, parse_single_date
from app.intent import classify
from app.conversation import ConversationStore


class FakeLLM:
    def bind_tools(self, schemas):
        return self

    def invoke(self, messages):
        # Deliberately wrong route and invented answer: structured branches must correct both.
        if '只选择' in messages[0].content:
            return SimpleNamespace(tool_calls=[{'name':'get_market_score', 'args':{}}])
        return SimpleNamespace(content='最新总分999999%。')


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    monkeypatch.setattr(g, 'get_llm', lambda: FakeLLM())


@pytest.mark.parametrize('text,expected', [
    ('2026-07-08', '2026-07-08'), ('2026/7/8', '2026-07-08'),
    ('2026年7月8日', '2026-07-08'), ('7月8日', '2026-07-08'),
    ('2025年，请查询7月8日评分', '2025-07-08'),
])
def test_single_dates(text, expected):
    assert parse_single_date(text) == expected


@pytest.mark.parametrize('text', [
    '2026年7月8日到7月10日', '7月8日到7月10日',
    '2026-07-08至2026-07-10', '2026/7/8~2026/7/10',
    '2026年7月8日—2026年7月10日', '2026年7月8日～7月10日',
])
def test_date_ranges(text):
    assert parse_date_range(text) == ('2026-07-08', '2026-07-10')
    assert parse_single_date(text) is None
    assert classify(text) == 'research'


@pytest.mark.parametrize('text', ['7月10日到7月8日', '2026年2月30日到3月1日', '7月8日到七月十日'])
def test_invalid_ranges(text):
    with pytest.raises(ValueError):
        parse_date_range(text)


def test_non_dates_and_first_year():
    assert parse_date_range('全期状态分布') is None
    assert parse_single_date('最新评分') is None
    assert parse_date_range('2025年，7月8日至7月10日') == ('2025-07-08', '2025-07-10')
    with pytest.raises(ValueError):
        parse_single_date('2026年2月30日')


def test_three_trading_days():
    result = t.get_market_score_history('2026-07-08', '2026-07-10')
    assert result['ok'] and result['as_of'] == '2026-07-10'
    assert result['source'] == 'data/clean/market_score_daily.csv'
    data = result['data']
    assert data['mode'] == 'detail' and data['total_days'] == 3
    assert [(row['date'], row['total_score'], row['market_state']) for row in data['rows']] == [
        ('2026-07-08', 32.63, '下行'), ('2026-07-09', 39.43, '下行'), ('2026-07-10', 33.9, '下行')]
    assert all(set(row) == {'date', 'total_score', 'market_state', 'val_score', 'macro_score', 'sent_score', 'trend_score'} for row in data['rows'])


@pytest.mark.parametrize('count,mode', [(1,'detail'),(5,'detail'),(6,'compact'),(10,'compact'),(11,'summary')])
def test_history_boundaries(count, mode):
    frame = t.scores().tail(count)
    result = t.get_market_score_history(frame.date.iloc[0], frame.date.iloc[-1])
    assert result['ok'], result['error']
    data = result['data']
    assert data['mode'] == mode and data['total_days'] == count
    if mode == 'summary':
        assert data['rows'] == []
        summary = data['summary']
        assert set(summary) == {'start_date','end_date','first','last','max','min','mean_total','state_counts'}
        assert summary['mean_total'] == round(float(frame.total_score.mean()), 2)
        assert summary['max']['total_score'] == frame.total_score.max()
        assert summary['min']['total_score'] == frame.total_score.min()
        assert summary['first']['date'] == frame.date.iloc[0]
        assert summary['last']['date'] == result['as_of']
        assert sum(summary['state_counts'].values()) == count
        assert set(summary['state_counts']) == {'震荡','震荡偏强','下行','上行','极寒'}
    elif mode == 'compact':
        assert all(set(row) == {'date','total_score','market_state'} for row in data['rows'])
    text = g.evidence_answer('每日评分', [result])
    assert '%' not in text
    state = g.intake({'question':'每日评分'}) | {'tool_results':[result]}
    answer = g.compose(state)
    assert len(answer['answer']) <= 300
    assert g.verify(state | answer)['verified']
    if mode == 'summary':
        assert '区间超过10个交易日，请缩小范围查看每日明细' in answer['answer']
    else:
        assert all(row['date'] in answer['answer'] and f"{row['total_score']:g}" in answer['answer'] for row in data['rows'])


@pytest.mark.parametrize('start,end,error', [
    ('2026-07-04','2026-07-05','区间内无交易日'),
    ('2026-07-10','2026-07-08','起止日期倒置'),
    ('2015-01-04','2015-01-05','可用范围 2015-01-05 至 2026-07-10'),
    ('2026-07-10','2026-07-11','可用范围 2015-01-05 至 2026-07-10'),
    ('2026/7/8','2026-07-10','日期必须为YYYY-MM-DD'),
    ('2026-7-08','2026-07-10','日期必须为YYYY-MM-DD'),
    ('2026-02-30','2026-03-01','日期无效'),
])
def test_history_errors(start, end, error):
    result = t.get_market_score_history(start, end)
    assert not result['ok'] and error in result['error']
    assert result['data'] == {} and result['as_of'] is None


@pytest.mark.parametrize('names,expected', [
    (['LGB','动态评分'], ['LGB融合','动态评分']),
    (['LGB','PPO'], ['LGB融合','PPO月度','PPO原生日频']),
    (['PPO月度','LGB融合','PPO原生','LGB'], ['PPO月度','LGB融合','PPO原生日频']),
    (['SASF','沪深300','MVO','等权'], ['SASF硬切换','沪深300','MVO','等权']),
    (['PPO'], ['PPO月度','PPO原生日频']),
    (['风险平价','动量'], ['风险平价','动量']),
])
def test_compare_aliases_order_dedup(names, expected):
    result = t.compare_strategies(names)
    assert result['ok'], result['error']
    assert result['as_of'] == '2026-07-10'
    assert result['source'] == 'results/backtest/all_strategy_metrics.csv'
    assert [row['策略'] for row in result['data']['rows']] == expected
    originals = t.get_strategy_metrics()['data']['rows']
    assert result['data']['rows'] == [next(row for row in originals if row['策略'] == name) for name in expected]


@pytest.mark.parametrize('names,error', [
    (['未知','LGB'],'可用策略'), (['LGB'],'至少需要两个'),
    (['LGB','LGB融合'],'至少需要两个'),
    (['LGB','PPO','动态评分','动量'],'最多对比四个'),
])
def test_compare_errors(names, error):
    result = t.compare_strategies(names)
    assert not result['ok'] and error in result['error']


def test_compare_missing_row(monkeypatch):
    original = t.table
    monkeypatch.setattr(t, 'table', lambda *args, **kwargs: original(*args, **kwargs).query('策略 != "LGB融合"'))
    result = t.compare_strategies(['LGB','动态评分'])
    assert not result['ok'] and '指标行缺失' in result['error']


@pytest.mark.parametrize('question,name,args', [
    ('LGB 与动态评分对比','compare_strategies',{'strategies':['LGB融合','动态评分']}),
    ('LGB融合与PPO对比','compare_strategies',{'strategies':['LGB融合','PPO月度','PPO原生日频']}),
    ('PPO月度 vs LGB融合','compare_strategies',{'strategies':['PPO月度','LGB融合']}),
    ('查询2026年7月8日到7月10日，每天的市场评分','get_market_score_history',{'start':'2026-07-08','end':'2026-07-10'}),
    ('7月8日到7月10日每天评分','get_market_score_history',{'start':'2026-07-08','end':'2026-07-10'}),
    ('7月8日到7月10日','get_market_score_history',{'start':'2026-07-08','end':'2026-07-10'}),
    ('最新市场评分和状态','get_market_score',{}),
    ('LGB 夏普多少','get_strategy_metrics',{'strategy':'LGB'}),
    ('全期状态分布','get_market_state_summary',{}),
    ('2026/7/8评分','get_market_score',{'date':'2026-07-08'}),
    ('2026年7月8日到7月10日状态天数统计','get_market_state_summary',{'start':'2026-07-08','end':'2026-07-10'}),
])
def test_routes(question, name, args):
    assert g.data_fallback_calls(question) == [{'name':name,'args':args}]
    if name in ('compare_strategies','get_market_score_history'):
        assert g.router(g.intake({'question':question}))['pending'] == [{'name':name,'args':args}]


def test_compare_evidence_and_compose():
    result = t.compare_strategies(['LGB','动态评分'])
    answer = g.evidence_answer('LGB与动态评分对比', [result])
    for expected in ('LGB融合','动态评分','年化11.99%','夏普0.6462','回撤-15.02%','卡玛0.7977'):
        assert expected in answer
    for other in ('沪深300','PPO','SASF','年化波动','索提诺'):
        assert other not in answer
    only_sharpe = g.evidence_answer('LGB与动态评分比较夏普', [result])
    assert '夏普' in only_sharpe and not any(metric in only_sharpe for metric in ('年化','回撤','卡玛'))
    for names in (['LGB','动态评分'], ['LGB','PPO'], ['LGB','PPO','SASF']):
        state = g.intake({'question':'LGB融合、PPO与SASF对比'}) | {'tool_results':[t.compare_strategies(names)]}
        output = g.compose(state)
        assert len(output['answer']) <= 300
        assert all(row['策略'] in output['answer'] for row in state['tool_results'][0]['data']['rows'])
        assert g.verify(state | output)['verified']


@pytest.mark.parametrize('question', ['LGB 与动态评分对比','LGB融合与PPO对比','查询2026年7月8日到7月10日，每天的市场评分','7月8日到7月10日'])
def test_http_queries_real_graph_fake_llm(monkeypatch, tmp_path, question):
    store = ConversationStore(tmp_path/'queries.sqlite')
    monkeypatch.setattr(main,'get_conversations',lambda:store)
    try:
        result = TestClient(main.app).post('/chat', json={'question':question,'thread_id':'query'}).json()
        assert result['mode'] == 'research' and result['verified'], result
        assert len(result['answer']) <= 300 and '999999' not in result['answer']
        assert '[source:' in result['answer']
        assert len(result['tool_calls']) == 1 and result['tool_calls'][0]['ok']
        if '对比' in question:
            assert 'LGB融合' in result['answer']
            assert ('动态评分' if '动态评分' in question else 'PPO原生日频') in result['answer']
        else:
            for token in ('2026-07-08','2026-07-09','2026-07-10','32.63','39.43','33.9','估值','宏观','情绪','趋势'):
                assert token in result['answer']
            assert '%' not in result['answer']
    finally:
        store.close()


@pytest.mark.parametrize('question,error', [
    ('2026年7月4日到7月5日评分','区间内无交易日'),
    ('2026年7月8日到7月11日评分','日期越界'),
    ('7月10日到7月8日','起止日期倒置'),
    ('7月8日到七月十日评分','无法解析日期区间'),
    ('2026年2月30日到3月1日','日期无效'),
    ('2027年7月8日评分','日期越界'),
])
def test_http_date_clarification(monkeypatch, tmp_path, question, error):
    store = ConversationStore(tmp_path/'errors.sqlite')
    monkeypatch.setattr(main,'get_conversations',lambda:store)
    try:
        result = TestClient(main.app).post('/chat',json={'question':question}).json()
        assert result['mode'] == 'research' and result['verified'] is False
        assert error in result['answer'] and '请' in result['answer']
        assert '33.9' not in result['answer']
        assert not any(call['ok'] for call in result['tool_calls'])
    finally:
        store.close()


def test_new_tools_preserve_existing_refusal_policy():
    state = g.intake({'question':'对比LGB与PPO，保证收益吗？'})
    state['tool_results'] = [t.compare_strategies(['LGB','PPO'])]
    output = g.compose(state)
    assert '不能保证收益' in output['answer']
    assert g.verify(state | output)['verified']
