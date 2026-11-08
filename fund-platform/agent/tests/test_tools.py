import dataclasses

import pytest

from app.tools import registry as t


def rows(result):
    assert result['ok'], result['error']
    return result['data']['rows']


def test_market():
    r = t.get_market_score()
    assert r['ok'] and r['as_of'] == '2026-07-10'
    assert r['data'] == dict(total_score=33.9, market_state='下行', val_score=0.0, macro_score=14.08, sent_score=10.65, trend_score=9.18)
    r = t.get_market_score(date='2020-03-23')
    assert r['data']['total_score'] == 34.38 and r['data']['market_state'] == '下行'
    assert t.get_market_score(date='2026-07-11')['as_of'] == '2026-07-10'
    assert not t.get_market_score(date='2010-01-01')['ok']
    assert not t.get_market_score(date='invalid')['ok']


def test_states():
    data = t.get_market_state_summary()['data']
    assert data['counts'] == dict(震荡=1317, 震荡偏强=756, 下行=580, 上行=106, 极寒=39)
    assert data['total_days'] == 2798
    assert not t.get_market_state_summary(start='2030-01-01')['ok']


def test_metrics():
    row = rows(t.get_strategy_metrics(strategy='动态评分'))[0]
    for key, value in dict(年化收益=.1313, 最大回撤=-.1427, 夏普比率=.5465, 卡玛比率=.9204).items():
        assert row[key] == value
    assert rows(t.get_strategy_metrics(strategy='LGB'))[0]['夏普比率'] == .6462
    row = rows(t.get_strategy_metrics(strategy='SASF'))[0]
    assert [row[k] for k in ('年化收益', '夏普比率', '最大回撤')] == [.1652, .6006, -.2656]
    assert len(rows(t.get_strategy_metrics())) == 10
    assert not t.get_strategy_metrics(strategy='未知策略')['ok']
    assert '可选值' in t.get_strategy_metrics(strategy='未知策略')['error']


def test_segments():
    data = rows(t.get_segment_performance(segment='2021后震荡'))
    for name, value in [('等权', .0751), ('动态评分', .1545), ('LGB融合', .3223)]:
        assert next(r for r in data if r['策略'] == name)['累计收益'] == value
    assert len(rows(t.get_segment_performance())) == 60


def test_crises():
    for crisis in ['2015', '2018', '2021']:
        data = rows(t.get_drawdown_recovery(crisis=crisis))
        assert len(data) == 3
        assert all(r['危机阶段'] and r['起始'] and r['结束'] for r in data)
    row = rows(t.get_drawdown_recovery(crisis='2015'))[1]
    assert row['最大回撤'] == '-13.6%' and row['修复天数'] == '239天'


def test_risk():
    row = rows(t.get_risk_tier(level='C2', strategy='动态评分'))[0]
    assert row['年化收益'] == .0859 and row['最大回撤'] == -.0972


def test_features():
    data = rows(t.get_feature_importance(top_n=5))
    assert len(data) == 5 and data[0]['importance'] >= data[-1]['importance']
    assert not t.get_feature_importance(top_n=0)['ok']
    assert not t.get_feature_importance(top_n=51)['ok']
    assert not t.get_feature_importance(top_n='5')['ok']
    assert not t.execute('unknown', {})['ok']
    assert not t.execute('get_market_score', {'path': '/etc/passwd'})['ok']


def test_caveats():
    r = t.get_data_caveats()
    assert r['ok'], r['error']
    d = r['data']
    assert [d['start'], d['end'], d['trading_days']] == ['2015-01-05', '2026-07-10', 2798]
    assert d['nan_range'] == ['2015-01-05', '2018-12-28'] and d['nan_rows'] == 975
    assert d['zero_start'] == '2024-08-19' and d['zero_rows'] == 457
    assert d['last_valid_date'] == '2024-08-16' and d['macro_lag'] == 30
    assert d['dimensions'] == {'all': {'columns': 182, 'features': 181}, 'LGB': {'columns': 38, 'features': 37}, 'PPO': {'columns': 45, 'features': 44}}


def test_missing_score_even_after_cache(monkeypatch, tmp_path):
    assert t.get_market_score()['ok']
    monkeypatch.setattr(t, 'settings', dataclasses.replace(t.settings, score_path=tmp_path/'absent.csv'))
    r = t.get_market_score()
    assert not r['ok'] and 'FileNotFoundError' in r['error']


def test_macro_parse_failure(monkeypatch, tmp_path):
    scripts = tmp_path/'scripts'
    scripts.mkdir()
    (scripts/'calc_market_score.py').write_text('MACRO_LAG = unknown\n')
    monkeypatch.setattr(t, 'settings', dataclasses.replace(t.settings, platform_dir=tmp_path))
    # Keep table inputs real while pointing the macro parser at the invalid script.
    real_settings = dataclasses.replace(t.settings, platform_dir=t.settings.score_path.parents[2])
    def read_real(relative, **kwargs):
        path = real_settings.platform_dir / relative
        stat = path.stat()
        return t._read(str(path), stat.st_mtime_ns, stat.st_size, kwargs.get('encoding', 'utf-8-sig'), kwargs.get('cols'), kwargs.get('header', False)).copy()
    monkeypatch.setattr(t, 'table', read_real)
    assert not t.get_data_caveats()['ok']
