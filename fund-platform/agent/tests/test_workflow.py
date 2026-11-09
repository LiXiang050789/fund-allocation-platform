import copy
import dataclasses
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from app import feishu
from app.config import settings
from app.tools import registry
from app.workflow import ReviewRequest, WorkflowConflict, WorkflowService, DISCLAIMER


@pytest.fixture(scope='module')
def aggregates():
    return {name: getattr(registry, name)(**args) for name, args in [
        ('get_market_score', {}), ('get_data_caveats', {}), ('get_strategy_metrics', {}),
        ('get_drawdown_recovery', {}), ('get_risk_tier', {'strategy': '动态评分'})]}


@pytest.fixture
def setup(tmp_path, monkeypatch, aggregates):
    for name, result in aggregates.items():
        assert result['ok']
        monkeypatch.setattr(registry, name, lambda result=result, **_: copy.deepcopy(result))
    config = dataclasses.replace(settings, agent_dir=tmp_path, platform_dir=tmp_path/'platform',
        reports_dir=tmp_path/'reports', checkpoint_path=tmp_path/'data/checkpoints.sqlite')
    monkeypatch.setattr(feishu, 'settings', dataclasses.replace(settings, feishu_app_id='test-app',
        feishu_app_secret='test-secret', feishu_chat_id='test-chat', feishu_webhook_url=''))
    send = Mock(return_value={'code': 0, 'data': {'message_id': 'mock-message'}})
    monkeypatch.setattr(feishu, 'notify', send)
    monkeypatch.setattr(feishu.time, 'sleep', lambda _: None)
    service = WorkflowService(config)
    yield service, send, config
    service.close()


def request(action='approve', **kwargs):
    return ReviewRequest(action=action, reviewer=' LiXiang ', **kwargs)


def test_draft_pauses_with_seven_sections_and_no_send(setup):
    service, send, config = setup
    state = service.run()
    assert state['status'] == 'awaiting_review' and state['report_date'] == state['market']['as_of']
    assert state['review_history'] == [] and state['publish_status'] == 'not_attempted'
    assert state['report_path'] is None and not config.reports_dir.exists()
    assert len([line for line in state['draft'].splitlines() if line.startswith('## ')]) == 7
    assert all(text in state['draft'] for text in ['33.90 分', '下行', '0.00 分', '14.08 分', '10.65 分', '9.18 分',
        'LGB融合', 'PPO月度', 'SASF硬切换', '2015', '2018', '2021', 'C2', 'C3', 'C4', '待审核', '未审核', '无权重缓存', DISCLAIMER])
    assert service.graph.get_state(service.graph_config(state['thread_id'])).next == ('human_review',)
    send.assert_not_called()


def test_approve_save_hash_and_duplicate(setup):
    service, send, config = setup
    state = service.review(service.run()['thread_id'], request())
    path = config.agent_dir/state['report_path']
    assert state['status'] == 'published' and state['message_id'] == 'mock-message'
    assert path.exists() and state['report_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert '已发布' in path.read_text() and 'LiXiang' in path.read_text()
    same = service.review(state['thread_id'], request())
    assert same['duplicate'] and same['report_sha256'] == state['report_sha256']
    assert send.call_count == 1
    assert send.call_args.kwargs['idempotency_key'] == state['idempotency_key']
    assert len(state['idempotency_key']) <= 50
    with pytest.raises(WorkflowConflict):
        service.review(state['thread_id'], request('edit', edited_markdown='修改'))


def test_edit_stays_pending_and_approved_file_keeps_edit(setup):
    service, send, config = setup
    state = service.run()
    edited = state['draft'].replace('以下为模型历史表现摘要', '人工修改：以下为模型历史表现摘要')
    state = service.review(state['thread_id'], request('edit', edited_markdown=edited))
    assert state['status'] == 'awaiting_review' and state['draft'] == edited
    assert state['revision'] == 1 and len(state['review_history']) == 1
    send.assert_not_called()
    state = service.review(state['thread_id'], request())
    assert state['status'] == 'published' and '人工修改' in (config.agent_dir/state['report_path']).read_text()
    assert len(state['review_history']) == 2


def test_three_rejections_stop_without_file(setup):
    service, send, config = setup
    state = service.run()
    for i in range(1, 4):
        state = service.review(state['thread_id'], request('reject', comment=f'意见{i}'))
        assert len(state['review_history']) == i
        assert state['status'] == ('rejected' if i == 3 else 'awaiting_review')
        assert f'意见{i}' in state['draft'] or i == 3
    assert state['report_path'] is None and not config.reports_dir.exists()
    send.assert_not_called()
    with pytest.raises(WorkflowConflict): service.review(state['thread_id'], request())


def test_edit_resets_consecutive_rejections(setup):
    service, _, _ = setup
    state = service.run()
    for _ in range(2): state = service.review(state['thread_id'], request('reject', comment='修改'))
    state = service.review(state['thread_id'], request('edit', edited_markdown=state['draft']))
    state = service.review(state['thread_id'], request('reject', comment='再次修改'))
    assert state['status'] == 'awaiting_review' and state['reject_streak'] == 1


def test_publish_failure_saved_sanitized_retry_same_key(setup, caplog):
    service, send, config = setup
    send.side_effect = RuntimeError('https://open.feishu.cn/secret-webhook token=credential test-secret test-chat')
    state = service.review(service.run()['thread_id'], request())
    path = config.agent_dir/state['report_path']
    assert state['status'] == 'publish_failed' and state['publish_status'] == 'failed' and path.exists()
    assert send.call_count == 3 and state['publish_attempts'] == 3
    for secret in ['secret-webhook', 'credential', 'test-secret', 'test-chat']:
        assert secret not in state['publish_error'] + path.read_text() + caplog.text
    key = state['idempotency_key']; saved_path = state['report_path']
    send.side_effect = None
    state = service.review(state['thread_id'], request())
    assert state['status'] == 'published' and state['idempotency_key'] == key and state['report_path'] == saved_path
    assert send.call_count == 4 and {call.kwargs['idempotency_key'] for call in send.call_args_list} == {key}


def test_missing_message_id_never_published(setup):
    service, send, config = setup
    send.return_value = {'code': 0, 'data': {}}
    state = service.review(service.run()['thread_id'], request())
    assert state['status'] == 'publish_failed' and state['message_id'] is None
    assert send.call_count == 3 and (config.agent_dir/state['report_path']).exists()


def test_no_feishu_saved_only_and_duplicate(setup, monkeypatch):
    service, send, config = setup
    monkeypatch.setattr(feishu, 'settings', dataclasses.replace(feishu.settings,
        feishu_app_id='', feishu_app_secret='', feishu_webhook_url=''))
    state = service.review(service.run()['thread_id'], request())
    assert state['status'] == 'saved_only' and state['publish_status'] == 'skipped' and state['publish_error'] is None
    assert (config.agent_dir/state['report_path']).exists()
    assert service.review(state['thread_id'], request())['duplicate']
    send.assert_not_called()


@pytest.mark.parametrize('tool', ['get_market_score', 'get_data_caveats', 'get_strategy_metrics', 'get_drawdown_recovery', 'get_risk_tier'])
def test_required_tool_failure_never_sends(setup, monkeypatch, tool):
    service, send, config = setup
    monkeypatch.setattr(registry, tool, lambda **_: {'ok': False, 'error': 'token=private'})
    state = service.run()
    assert state['status'] == 'failed' and not state['draft'] and not state['report_path']
    assert 'private' not in state['error'] and not config.reports_dir.exists()
    send.assert_not_called()


def test_sqlite_close_reopen_resumes_same_draft(setup):
    first, send, config = setup
    state = first.run()
    first.close()
    # Reopen after closing: only the SQLite file carries the pending review.
    second = WorkflowService(config)
    try:
        restored = second.get(state['thread_id'])
        assert restored['status'] == 'awaiting_review' and restored['draft'] == state['draft']
        final = second.review(state['thread_id'], request())
        assert final['status'] == 'published' and send.call_count == 1
    finally:
        second.close()


def test_concurrent_duplicate_approval_sends_once(setup):
    service, send, _ = setup
    tid = service.run()['thread_id']
    with ThreadPoolExecutor(max_workers=2) as pool:
        states = list(pool.map(lambda _: service.review(tid, request()), range(2)))
    assert all(s['status'] == 'published' for s in states)
    assert sum(s.get('duplicate', False) for s in states) == 1 and send.call_count == 1


@pytest.mark.parametrize('payload', [
    {'action':'invalid','reviewer':'x'}, {'action':'approve','reviewer':' '},
    {'action':'edit','reviewer':'x'}, {'action':'reject','reviewer':'x'},
    {'action':'approve','reviewer':'x','extra':True}, {'action':'approve','reviewer':123},
    {'action':'edit','reviewer':'x','edited_markdown':'x'*50001},
    {'action':'reject','reviewer':'x','comment':'x'*1001},
])
def test_review_validation(payload):
    with pytest.raises(ValidationError): ReviewRequest.model_validate(payload)


def test_optional_weight_cache_and_no_secret_or_raw_table(setup):
    service, _, config = setup
    cache = config.platform_dir/'results/ppo_weight.csv'
    cache.parent.mkdir(parents=True)
    cache.write_text('date,hs300,zz500,kc50,consume,chip,gold,bond10\n2026-07-10,0.1,0.1,0.1,0.1,0.1,0.1,0.4\n')
    state = service.run()
    assert '权重缓存日期：2026-07-10' in state['draft'] and '| bond10 | 40.00% |' in state['draft']
    assert 'test-secret' not in state['draft'] and 'test-chat' not in state['draft']
    assert 'north_net' not in state['draft'] and 'tenant_access_token' not in state['draft']


@pytest.fixture
def client(setup, monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as api
    monkeypatch.setattr(api, 'get_workflow', lambda: setup[0])
    with TestClient(api.app) as client:
        yield client


def test_api_run_get_report_and_duplicate_approval(client, setup):
    initial = client.post('/workflow/run')
    assert initial.status_code == 200
    draft = initial.json(); tid = draft['thread_id']
    assert draft['status'] == 'awaiting_review' and draft['publish']['status'] == 'not_attempted'
    assert client.get(f'/workflow/{tid}').json()['draft'] == draft['draft']
    report = client.get(f'/workflow/{tid}/report')
    assert report.status_code == 200 and report.headers['content-type'] == 'text/markdown; charset=utf-8'
    assert report.text == draft['draft']
    payload = {'action': 'approve', 'reviewer': 'LiXiang'}
    final = client.post(f'/workflow/{tid}/approve', json=payload)
    assert final.status_code == 200 and final.json()['status'] == 'published'
    duplicate = client.post(f'/workflow/{tid}/approve', json=payload).json()
    assert duplicate['duplicate'] is True and setup[1].call_count == 1
    assert client.post(f'/workflow/{tid}/approve', json={'action':'edit','reviewer':'x','edited_markdown':'x'}).status_code == 409
    assert client.post(f'/workflow/{tid}/approve', json={'action':'reject','reviewer':'x','comment':'x'}).status_code == 409
    assert client.get(f'/workflow/{tid}/report').text == (setup[2].agent_dir/final.json()['report_path']).read_text()


@pytest.mark.parametrize('payload', [
    {'action':'invalid','reviewer':'x'}, {'action':'approve','reviewer':' '},
    {'action':'edit','reviewer':'x'}, {'action':'reject','reviewer':'x'},
    {'action':'approve','reviewer':'x','extra': True},
])
def test_api_invalid_review_422(client, payload):
    assert client.post('/workflow/not-found/approve', json=payload).status_code == 422


def test_api_unknown_and_no_draft(client, setup, monkeypatch):
    assert client.get('/workflow/not-found').status_code == 404
    assert client.get('/workflow/not-found/report').status_code == 404
    assert client.post('/workflow/not-found/approve', json={'action':'approve','reviewer':'x'}).status_code == 404
    monkeypatch.setattr(registry, 'get_market_score', lambda: {'ok': False})
    state = client.post('/workflow/run').json()
    assert state['status'] == 'failed' and state['error']
    assert client.get(f"/workflow/{state['thread_id']}/report").status_code == 409
    assert client.post(f"/workflow/{state['thread_id']}/approve",json={'action':'approve','reviewer':'x'}).status_code == 409


def test_api_publish_failure_is_business_200(client, setup):
    setup[1].side_effect = RuntimeError('secret')
    tid = client.post('/workflow/run').json()['thread_id']
    response = client.post(f'/workflow/{tid}/approve',json={'action':'approve','reviewer':'x'})
    assert response.status_code == 200 and response.json()['status'] == 'publish_failed'
    assert 'secret' not in response.text


def test_atomic_save_failure_prevents_send(setup, monkeypatch):
    import app.workflow as workflow
    service, send, _ = setup
    state = service.run()
    monkeypatch.setattr(workflow, 'atomic_save', lambda *_: (_ for _ in ()).throw(OSError('token=private')))
    final = service.review(state['thread_id'], request())
    assert final['status'] == 'failed' and final['error'] == 'OSError: publish 节点失败'
    assert final['report_path'] is None
    send.assert_not_called()


def test_report_numbers_trace_to_collected_rows(setup):
    import re
    service, _, _ = setup
    state = service.run()
    body = state['draft']
    rows = sorted(state['strategies']['data']['rows'], key=lambda r: r['年化收益'], reverse=True)
    for rank, row in enumerate(rows, 1):
        expected = f"| {rank} | {row['策略']} | {row['年化收益']:.2%} | {row['最大回撤']:.2%} | {row['夏普比率']} | {row['卡玛比率']} |"
        assert expected in body
    score_section = body.split('## 二、', 1)[0]
    assert '%' not in score_section and str(state['market']['as_of']) in score_section
    assert not re.search(r'\b(?:sk-|oc_)[A-Za-z0-9]+', body)
    assert str(state['caveats']['data']['trading_days']) in body


def test_config_repr_excludes_credentials():
    config = dataclasses.replace(settings, deepseek_api_key='hidden-api-key', feishu_app_secret='hidden-secret',
        feishu_chat_id='hidden-chat', feishu_webhook_url='hidden-webhook', feishu_webhook_secret='hidden-sign')
    assert 'hidden-' not in repr(config)
