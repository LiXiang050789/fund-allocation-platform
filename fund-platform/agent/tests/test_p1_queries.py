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
    assert not r['ok'] and '可选值' in r['error']
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


class WrongLLM:
    def bind_tools(self, schemas):return self
    def invoke(self,messages):
        if '只选择' in messages[0].content:
            return SimpleNamespace(tool_calls=[dict(name='get_market_score',args={})])
        return SimpleNamespace(content='编造净值9999999999，今日实时。')


@pytest.fixture
def fake_llm(monkeypatch):
    from app import graph as g
    monkeypatch.setattr(g,'get_llm',lambda:WrongLLM())
    return monkeypatch


@pytest.mark.parametrize('question,name,args',[
    ('最新 PPO 权重','get_ppo_cached_weights',{}),
    ('2026年7月10日 PPO 调仓权重','get_ppo_cached_weights',{'date':'2026-07-10'}),
    ('LGB融合最新净值','get_strategy_nav',{'strategies':['LGB融合']}),
    ('LGB融合与PPO从2026年7月8日到7月10日净值','get_strategy_nav',{'start':'2026-07-08','end':'2026-07-10','strategies':['LGB融合','PPO月度','PPO原生日频']}),
    ('C2保守型最终净值','get_risk_tier_detail',{'level':'C2'}),
    ('C3推荐配置每个ETF多少','get_risk_tier_detail',{'level':'C3'}),
    ('流水线文件就绪几个','get_pipeline_status',{}),
    ('LGB 2022–2026年度IC','get_pipeline_status',{}),
    ('回测滑点/双边成本/单资产上限','get_pipeline_status',{}),
    ('十套策略收益回撤比','get_strategy_metrics',{'strategy':None}),
    ('全期各状态占比','get_market_state_summary',{}),
    ('LGB 与 PPO 净值对比','get_strategy_nav',{'strategies':['LGB融合','PPO月度','PPO原生日频']}),
])
def test_p1_routes(question,name,args,fake_llm):
    from app import graph as g
    from app.intent import classify
    assert classify(question)=='research'
    assert g.data_fallback_calls(question)==[dict(name=name,args=args)]
    if name in g.P1_TOOLS:
        assert g.router(g.intake({'question':question}))['pending']==[dict(name=name,args=args)]


def test_weight_docs_and_mixed_routes(fake_llm):
    from app import graph as g
    for q in ('PPO 权重怎么设计','评分权重规则','PPO权重怎么算'):
        assert g.document_question(q)
        assert [item['name'] for item in g.fallback_calls(q)]==['search_docs']
    for q in ('最新 PPO 权重和规则','当前PPO权重设计文档'):
        assert [item['name'] for item in g.fallback_calls(q)]==['search_docs','get_ppo_cached_weights']
        calls=g.router(g.intake({'question':q}))['pending']
        assert [call['name'] for call in calls]==['search_docs','get_ppo_cached_weights']


@pytest.mark.parametrize('question',['C3推荐配置，推荐一个ETF','请买入C2推荐配置并保证收益','推荐一只ETF','全仓C2推荐配置'])
def test_static_tier_never_bypasses_refusal(question):
    from app.intent import classify
    assert classify(question)=='refusal'


@pytest.mark.parametrize('question,tool,args,tokens',[
    ('LGB融合最新净值','get_strategy_nav',{'strategies':['LGB']},['3399752.14','3.3998','组合净值']),
    ('2026年7月10日PPO权重','get_ppo_cached_weights',{'date':'2026-07-10'},['缓存权重','2026-07-09','2026-07-10','沪深300','十年国债']),
    ('流水线文件就绪几个','get_pipeline_status',{},['12/12','3/3','69','20']),
    ('LGB 2022–2026年度IC','get_pipeline_status',{},['0.2317','0.2707','0.0356','-0.3316','0.0162']),
    ('C2保守型最终净值','get_risk_tier_detail',{'level':'C2'},['2.44','2.31','保守型C2']),
    ('C3推荐配置每个ETF多少','get_risk_tier_detail',{'level':'C3'},['10%','25%','静态推荐配置']),
    ('全期各状态占比','get_market_state_summary',{},['47.1%','27.0%','20.7%','3.8%','1.4%']),
])
def test_p1_evidence_compose(question,tool,args,tokens,fake_llm):
    from app import graph as g
    r=t.execute(tool,args)
    assert r['ok'],r['error']
    state=g.intake({'question':question})|{'tool_results':[r]}
    text=g.evidence_answer(question,[r])
    assert all(token in text for token in tokens)
    # The state distribution still accepts model composition under the old tool semantics.
    if tool=='get_market_state_summary':
        class EvidenceLLM(WrongLLM):
            def invoke(self,messages):return SimpleNamespace(content=text)
        # This fixture's monkeypatch restores the function after the test.
        fake_llm.setattr(g, 'get_llm', lambda: EvidenceLLM())
    output=g.compose(state)
    assert g.verify(state|output)['verified'],output
    assert all(token in output['answer'] for token in tokens)
    assert '[source:' in output['answer'] and len(output['answer'])<=300
    assert not any(token in output['answer'] for token in ('9999999999','实时推理','今日实时','，。'))


@pytest.mark.parametrize('count,names',[(1,['LGB']),(5,['LGB']),(6,['LGB']),(10,['LGB','PPO']), (10,['LGB','PPO','SASF']), (5,['LGB','PPO','SASF'])])
def test_nav_evidence_length(count,names,fake_llm):
    from app import graph as g
    frame=t.table('results/backtest/all_strategy_nav.csv').tail(count)
    r=t.get_strategy_nav(names,start=frame.date.iloc[0],end=frame.date.iloc[-1])
    state=g.intake({'question':'策略净值区间'})|{'tool_results':[r]}
    output=g.compose(state)
    assert g.verify(state|output)['verified'], output
    assert '组合净值' in output['answer'] and len(output['answer'])<=300
    assert '当前数据无法回答' not in output['answer']
    assert frame.date.iloc[0] in output['answer'] and frame.date.iloc[-1] in output['answer']
    if count>5:assert '缩小范围' in output['answer']


@pytest.mark.parametrize('question',['LGB融合最新净值','最新PPO权重','流水线文件就绪几个','LGB 2022–2026年度IC','C2保守型最终净值','C3推荐配置每个ETF多少','全期各状态占比'])
def test_p1_http(question,fake_llm,monkeypatch,tmp_path):
    from app import main,graph as g
    from app.conversation import ConversationStore
    from fastapi.testclient import TestClient
    # Deliberately wrong route; accurate evidence response for existing state-summary composer.
    class EvidenceLLM(WrongLLM):
        def invoke(self,messages):
            if '只选择' in messages[0].content:return super().invoke(messages)
            import json
            payload=json.loads(messages[-1].content)
            return SimpleNamespace(content=g.evidence_answer(payload['question'],payload['tool_results']))
    monkeypatch.setattr(g,'get_llm',lambda:EvidenceLLM())
    store=ConversationStore(tmp_path/'p1.sqlite')
    monkeypatch.setattr(main,'get_conversations',lambda:store)
    try:
        r=TestClient(main.app).post('/chat',json={'question':question}).json()
        assert r['mode']=='research' and r['verified'] is True,r
        assert '[source:' in r['answer'] and len(r['answer'])<=300
        assert '当前数据无法回答' not in r['answer']
        assert not any(word in r['answer'] for word in ('实时推理','今日实时','9999999999'))
    finally:store.close()


@pytest.mark.parametrize('question,error',[
    ('2026年7月4日LGB净值','非交易日或日期越界'),
    ('2026年2月30日LGB净值','日期无效'),
    ('LGB净值7月10日到7月8日','起止日期倒置'),
    ('2015年1月5日PPO权重','早于首个权重日期'),
    ('C1最终净值','未知风险档'),
    ('C22最终净值','未知风险档'),
    ('C2 PPO最终净值','未知策略'),
])
def test_p1_http_errors(question,error,fake_llm,monkeypatch,tmp_path):
    from app import main
    from app.conversation import ConversationStore
    from fastapi.testclient import TestClient
    store=ConversationStore(tmp_path/'errors.sqlite')
    monkeypatch.setattr(main,'get_conversations',lambda:store)
    try:
        r=TestClient(main.app).post('/chat',json={'question':question}).json()
        assert r['verified'] is False and error in r['answer'],r
        assert not any(word in r['answer'] for word in ('3399752','33.9','/home/','Traceback'))
    finally:store.close()


@pytest.mark.parametrize('tool,message',[
    ('get_strategy_nav','未知策略；可选值：X'),('get_risk_tier_detail','未知风险档；可选值：C2'),
    ('get_ppo_cached_weights','PPO 权重总和异常'),('get_pipeline_status','无法唯一解析回测成本常量')])
def test_readonly_error_allowlist(tool,message):
    from app.dates import readonly_error
    r=dict(tool=tool,ok=False,error='ValueError: '+message)
    assert readonly_error([r])
    r['error']='OSError: /home/private secret'
    assert readonly_error([r]) is None
    r['error']='ValueError: arbitrary /home/private'
    assert readonly_error([r]) is None
    r['error']='ValueError: 未知策略；/home/private secret'
    assert '/home/' not in readonly_error([r])


def test_p1_identity_keywords():
    from app.intent import IDENTITY
    for word in ('净值','缓存权重','流水线','年度 IC','推荐配置','9 份文档','2026-07-10','不是实时行情','人工审核'):
        assert word in IDENTITY


def test_asset_labels_are_evidence_not_unsupported_numbers():
    for result in (t.get_ppo_cached_weights(),t.get_risk_tier_detail('C3')):
        assert result['data']['asset_names']==list(t.ETF_CN.values())


@pytest.mark.parametrize('question',['流水线六主表形状','流水线特征表形状','回测滑点/双边成本/单资产上限','十套策略最新净值','十套策略收益回撤比'])
def test_additional_evidence_budgets(question,fake_llm):
    from app import graph as g
    calls=g.data_fallback_calls(question)
    results=[t.execute(call['name'],call['args']) for call in calls]
    # Existing metrics use the existing model composer; simulate an accurate short response.
    class EvidenceLLM(WrongLLM):
        def invoke(self,messages):return SimpleNamespace(content=g.evidence_answer(question,results))
    fake_llm.setattr(g,'get_llm',lambda:EvidenceLLM())
    state=g.intake({'question':question})|{'tool_results':results}
    output=g.compose(state)
    assert g.verify(state|output)['verified'],output
    assert len(output['answer'])<=300 and '当前数据无法回答' not in output['answer']


def test_mixed_docs_and_cache_preserve_both_evidence_sources(fake_llm):
    from app import graph as g
    chunk=dict(source_name='权重手册.md',section='缓存规则',text='缓存权重来自历史产物。',authority='authoritative',chunk_id='test')
    docs=dict(ok=True,tool='search_docs',data={'chunks':[chunk]},source='权重手册.md')
    state=g.intake({'question':'最新PPO权重和规则'})|{'tool_results':[docs,t.get_ppo_cached_weights()]}
    output=g.compose(state)
    assert g.verify(state|output)['verified'],output
    assert '缓存权重' in output['answer'] and '[来源: 权重手册.md §缓存规则]' in output['answer']
    assert '[source: results/ppo_weight2.csv]' in output['answer']
