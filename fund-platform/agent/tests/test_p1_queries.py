"""P1 real artifact regression tests. No model, network, execution, or publication."""
import ast
import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from app.tools import registry as t


@pytest.fixture
def no_execution(monkeypatch):
    import subprocess
    before = {name for name in sys.modules if 'infer_runtime' in name}
    def forbidden(*args, **kwargs):
        pytest.fail('read-only tools must not execute subprocesses')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    yield
    assert {name for name in sys.modules if 'infer_runtime' in name} == before


def test_nav_latest_and_original_units(no_execution):
    result = t.get_strategy_nav(['LGB'])
    assert result['ok'], result['error']
    d = result['data']
    frame = t.table('results/backtest/all_strategy_nav.csv')
    assert result['as_of'] == '2026-07-10'
    assert result['source'] == 'results/backtest/all_strategy_nav.csv'
    assert d['mode'] == 'latest' and d['total_days'] == 1
    assert d['strategies'] == ['LGB融合'] and d['summary'] == {}
    assert d['rows'][0]['LGB融合'] == pytest.approx(3399752.140846137)
    assert d['multiples']['LGB融合'] == round(frame['LGB融合'].iloc[-1]/frame['LGB融合'].iloc[0], 4)
    assert d['rows'][0]['LGB融合'] > 1_000_000
    assert d['multiples']['LGB融合'] != 2.31  # distinct from C2 risk-tier sequence
    assert d['start_date'] == d['end_date'] == result['as_of']


@pytest.mark.parametrize('names,expected', [
    (None,10),([],10),(['LGB','PPO'],3),(['PPO月度','LGB','LGB融合'],2),
    (['SASF','风险平价','动量','沪深300'],4),
])
def test_nav_aliases(names, expected):
    r = t.get_strategy_nav(names, date='2026-07-08')
    assert r['ok'], r['error']
    assert r['data']['mode'] == 'date' and len(r['data']['strategies']) == expected
    assert len(r['data']['rows'][0]) == expected+1
    assert r['as_of'] == '2026-07-08'
    if names == ['LGB','PPO']:
        assert r['data']['strategies'] == ['LGB融合','PPO月度','PPO原生日频']


@pytest.mark.parametrize('count,mode', [(1,'detail'),(5,'detail'),(6,'summary'),(10,'summary')])
def test_nav_ranges(count, mode):
    frame = t.table('results/backtest/all_strategy_nav.csv').tail(count)
    r = t.get_strategy_nav(['LGB'], start=frame.date.iloc[0], end=frame.date.iloc[-1])
    assert r['ok'], r['error']
    d=r['data']
    assert d['mode'] == mode and d['total_days'] == count
    assert r['as_of'] == frame.date.iloc[-1]
    if mode == 'summary':
        assert d['rows'] == []
        s=d['summary']['LGB融合']
        assert s['start_nav'] == frame['LGB融合'].iloc[0]
        assert s['end_nav'] == frame['LGB融合'].iloc[-1]
        assert s['max']['nav'] == frame['LGB融合'].max()
        assert s['min']['date'] == frame.loc[frame['LGB融合'].idxmin(),'date']
        assert s['range_multiple'] == round(s['end_nav']/s['start_nav'],4)
    else:
        assert len(d['rows']) == count


@pytest.mark.parametrize('args,error', [
    ({'date':'2026-07-04'},'非交易日或日期越界'),
    ({'date':'2030-01-01'},'非交易日或日期越界'),
    ({'date':'2026/7/8'},'日期必须为YYYY-MM-DD'),
    ({'date':'2026-02-30'},'日期无效'),
    ({'date':'2026-07-08','start':'2026-07-08','end':'2026-07-09'},'参数冲突'),
    ({'start':'2026-07-08'},'必须同时提供'),
    ({'end':'2026-07-08'},'必须同时提供'),
    ({'start':'2026-07-10','end':'2026-07-08'},'起止日期倒置'),
    ({'start':'2026-07-04','end':'2026-07-05'},'区间内无交易日'),
    ({'start':'2015-01-04','end':'2026-07-10'},'非交易日或日期越界'),
    ({'strategies':['未知']},'未知策略'),
    ({'strategies':['LGB','PPO','动量','等权']},'一次最多查询四个策略'),
])
def test_nav_errors(args, error):
    r=t.get_strategy_nav(**args)
    assert not r['ok'] and error in r['error']
    assert r['data'] == {} and r['as_of'] is None


@pytest.mark.parametrize('day,exact', [(None,True),('2026-07-09',True),('2026-07-10',False)])
def test_cached_ppo(day, exact, no_execution):
    r=t.get_ppo_cached_weights(day)
    assert r['ok'], r['error']
    d=r['data']
    assert r['source'] == 'results/ppo_weight2.csv'
    assert r['as_of'] == d['date'] == d['latest_available'] == '2026-07-09'
    assert d['requested_date'] == day and d['exact'] is exact
    assert list(d['weights']) == ['hs300','zz500','kc50','consume','chip','gold','bond10']
    assert list(d['weights_cn']) == ['沪深300','中证500','科创50','消费','芯片','黄金','十年国债']
    assert d['weight_sum'] == 1.0 and d['min_weight'] == .122609
    assert sum(d['weights'].values()) == pytest.approx(1,abs=1e-6)


@pytest.mark.parametrize('day,error', [('2015-01-05','早于首个权重日期'),('2026/7/9','日期必须为YYYY-MM-DD'),('2026-02-30','日期无效')])
def test_ppo_dates(day, error):
    r=t.get_ppo_cached_weights(day)
    assert not r['ok'] and error in r['error'] and r['data']=={} and r['as_of'] is None


@pytest.mark.parametrize('value',[1.0,float('nan')])
def test_ppo_corrupt_weights(monkeypatch,value):
    original=t.table
    def corrupt(path,**kwargs):
        df=original(path,**kwargs)
        df.loc[df.index[-1],'hs300']=value
        return df
    monkeypatch.setattr(t,'table',corrupt)
    r=t.get_ppo_cached_weights()
    assert not r['ok'] and 'PPO 权重总和异常' in r['error']
