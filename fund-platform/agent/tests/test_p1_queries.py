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


def test_pipeline_real_artifacts(no_execution):
    r=t.get_pipeline_status()
    assert r['ok'],r['error']
    assert r['as_of']=='2026-07-10'
    d=r['data']
    assert [item['name'] for item in d['files']]==['data/raw 目录','数据汇总.xlsx','train_total_feature','train_feature_filtered','train_feature_filtered_ppo','etf_price_clean','market_score_daily','global_best.zip','feature_importance','pred_return_lgb_daily','ppo_weight','all_strategy_metrics']
    assert d['files_ready']==sum(item['ok'] for item in d['files'])==12
    assert d['files_total']==12
    assert [item['name'] for item in d['semantics']]==['PPO 两行存在','PPO 特征 44 维','PPO 权重和≈1']
    assert d['semantics_ready']==sum(item['ok'] for item in d['semantics'])==3
    assert d['semantics_total']==3
    assert d['raw_files']==69 and d['clean_csvs']==20
    for name,cols in [('etf_price_clean',15),('train_total_feature',182),('market_score_daily',7),('valuation_merged',29),('tech_factor_clean',78),('macro_merged',33)]:
        assert d['shapes'][name]==dict(rows=2798,cols=cols,encoding='GBK' if name=='train_total_feature' else 'UTF-8-SIG',shape=f'2798×{cols}')
    assert [d['feature_shapes'][name]['shape'] for name in ('train_total_feature','train_feature_filtered','train_feature_filtered_ppo')]==['2798×182','2798×38','2798×45']
    assert d['top_feature']==dict(feature='bond_10y_2y_spread',importance=.1962)
    assert d['lgb_pred_count']==7504
    assert d['annual_ic']=={'2022':.2317,'2023':.2707,'2024':.0356,'2025':-.3316,'2026':.0162}
    assert d['ppo_weight']==dict(date='2026-07-09',sum=1.0,min=.122609)
    assert d['protocol']==dict(SLIPPAGE=.0005,FEE=.001,MAX_WEIGHT=.30)
    assert '/home/' not in str(d)


@pytest.mark.parametrize('script', ['FEE=.001\nMAX_WEIGHT=.3', 'SLIPPAGE=.0005\nSLIPPAGE=.0005\nFEE=.001\nMAX_WEIGHT=.3','SLIPPAGE=unknown\nFEE=.001\nMAX_WEIGHT=.3','SLIPPAGE=True\nFEE=.001\nMAX_WEIGHT=.3'])
def test_protocol_invalid(monkeypatch,tmp_path,script):
    (tmp_path/'scripts').mkdir()
    (tmp_path/'scripts/backtest_unified.py').write_text(script)
    monkeypatch.setattr(t,'settings',dataclasses.replace(t.settings,platform_dir=tmp_path))
    r=t.get_pipeline_status()
    assert not r['ok'] and r['data']=={} and r['as_of'] is None
    assert '无法唯一解析回测成本常量' in r['error']


def test_pipeline_missing_product(monkeypatch,tmp_path):
    root=t.settings.platform_dir
    for path in ['data/raw','scripts/backtest_unified.py','models/H_equity_residual_best.zip']+[f'data/clean/{name}.csv' for name in ('train_total_feature','train_feature_filtered','train_feature_filtered_ppo','etf_price_clean','market_score_daily','valuation_merged','tech_factor_clean','macro_merged')]+['results/feature_importance.csv','results/ppo_weight2.csv','results/backtest/all_strategy_metrics.csv']:
        dest=tmp_path/path
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.symlink_to(root/path, target_is_directory=(root/path).is_dir())
    monkeypatch.setattr(t,'settings',dataclasses.replace(t.settings,platform_dir=tmp_path))
    r=t.get_pipeline_status()
    assert r['ok'],r['error']
    assert r['data']['files_ready']==11 and r['data']['lgb_pred_count']==0
    assert r['data']['annual_ic']=={}
    assert not next(item['ok'] for item in r['data']['files'] if item['name']=='pred_return_lgb_daily')


def test_pipeline_unreadable_weight(monkeypatch):
    original=t.table
    def missing(path,**kwargs):
        if path=='results/ppo_weight2.csv':raise FileNotFoundError('fixture')
        return original(path,**kwargs)
    monkeypatch.setattr(t,'table',missing)
    r=t.get_pipeline_status()
    assert r['ok'] and r['data']['ppo_weight'] is None
    assert r['data']['semantics_ready']==2


def test_ic_zero_variance_and_missing():
    import pandas as pd
    dates=pd.date_range('2022-01-01',periods=30)
    pred=pd.DataFrame({'date':dates,**{code:[1.]*30 for code in t.ETF_CN}})
    price=pd.DataFrame({'date':dates,**{code:[1.]*30 for code in t.ETF_CN}})
    assert t._annual_ic(pred,price)=={'2022':0.0}
    assert t._annual_ic(pred.head(1),price)=={'2022':0.0}
    assert t._annual_ic(None,price)=={}


@pytest.mark.parametrize('level,equity,navs',[('C2',.2,[2.44,2.31]),('C3',.5,[4.48,4.18]),('C4',.7,[4.81,4.66]),('保守',.2,[2.44,2.31]),('平衡',.5,[4.48,4.18]),('进取',.7,[4.81,4.66])])
def test_risk_tier_details(level,equity,navs):
    r=t.get_risk_tier_detail(level)
    assert r['ok'],r['error']
    d=r['data']
    assert r['source']=='results/risk_tiers.csv' and r['as_of']=='2026-07-10'
    assert [row['最终净值'] for row in d['rows']]==navs
    assert d['equity_center']==equity
    assert list(d['recommended_weights'].values())==pytest.approx([equity/5]*5+[(1-equity)/2]*2)
    assert sum(d['recommended_weights'].values())==pytest.approx(1)


@pytest.mark.parametrize('level,strategy,annual,sharpe,drawdown,nav', [
    ('C2','动态评分',.0859,.5336,-.0972,2.44),('C3','动态评分',.1614,.5243,-.2144,4.48),('C4','动态评分',.173,.5311,-.3099,4.81),
    ('C2','LGB',.079,.6263,-.1026,2.31),('C3','LGB融合',.1458,.6118,-.2183,4.18),('C4','LGB',.1596,.62,-.277,4.66)])
def test_risk_single_rows(level,strategy,annual,sharpe,drawdown,nav):
    r=t.get_risk_tier_detail(level,strategy)
    assert r['ok'],r['error']
    assert len(r['data']['rows'])==1
    row=r['data']['rows'][0]
    assert [row[key] for key in ('年化收益','夏普比率','最大回撤','最终净值')]==[annual,sharpe,drawdown,nav]


@pytest.mark.parametrize('args',[{'level':'C1'},{'level':'C22'},{'strategy':'SASF'},{'strategy':'未知'}])
def test_risk_errors(args):
    r=t.get_risk_tier_detail(**args)
    assert not r['ok'] and '可选值' in r['error'] or (not r['ok'] and '可用策略' in r['error'])
    assert r['data']=={} and r['as_of'] is None


def test_risk_multi_and_old_source():
    d=t.get_risk_tier_detail()['data']
    assert len(d['rows'])==6
    assert 'recommended_weights' not in d and 'equity_center' not in d
    r=t.get_risk_tier('C2','LGB')
    assert r['source']=='results/risk_levels_metrics.csv'
    original=t.records(t.select(t.select(t.table('results/risk_levels_metrics.csv'),'风险档','C2'),'策略源','LGB'))
    assert r['data']['rows']==original


def test_state_percentages_and_ratio_extension():
    d=t.get_market_state_summary()['data']
    assert d['counts']==dict(震荡=1317,震荡偏强=756,下行=580,上行=106,极寒=39)
    assert d['total_days']==2798
    assert d['percentages']=={state:round(count/2798*100,1) for state,count in d['counts'].items()}
    assert sum(d['percentages'].values())==pytest.approx(100,abs=.2)
    compared=t.compare_strategies(['动态评分','LGB'])['data']['rows']
    metrics=t.get_strategy_metrics()['data']['rows']
    for row in compared:
        assert row==next(item for item in metrics if item['策略']==row['策略'])
        assert row['年化收益/最大回撤']==round(row['年化收益']/abs(row['最大回撤']),4)
    assert [row['年化收益/最大回撤'] for row in compared]==[.9201,.7983]
