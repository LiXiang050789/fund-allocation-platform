"""HTTP contracts; no Streamlit and no external network."""
import sys
from pathlib import Path
import pytest
import requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'platform'))
import agent_client as client

class Response:
    text = '# 简报\n中文'
    def __init__(self, payload=None, status=200):
        self.payload = {} if payload is None else payload
        self.status_code = status
    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.delenv('AGENT_BASE_URL', raising=False)
    def no_network(*args, **kwargs):
        pytest.fail('unexpected network')
    monkeypatch.setattr(requests, 'request', no_network)

def capture(monkeypatch, response):
    calls = []
    def request(*args, **kwargs):
        calls.append((args, kwargs))
        return response
    monkeypatch.setattr(requests, 'request', request)
    return calls

def test_base(monkeypatch):
    assert client.get_base_url() == 'http://127.0.0.1:8001'
    monkeypatch.setenv('AGENT_BASE_URL', 'https://localhost:8001///')
    assert client.get_base_url() == 'https://localhost:8001'

@pytest.mark.parametrize('url', ['', ' ', 'file:///tmp/x', 'ftp://host', 'http://', 'http://x:bad', 'http://x:0', 'http://a b', 'https://user:my-test-secret@host', 'http://host?token=my-test-secret', 'http://host/#x'])
def test_invalid(monkeypatch, url):
    monkeypatch.setenv('AGENT_BASE_URL', url)
    with pytest.raises(client.AgentConfigError) as exc:
        client.get_base_url()
    assert 'my-test-secret' not in str(exc.value)

def test_chat(monkeypatch):
    payload = dict(answer='结果', verified=True, citations=['数据'], tool_calls=[], latency_ms=12, thread_id='id')
    calls = capture(monkeypatch, Response(payload))
    assert client.chat('市场', 'id', base_url='http://localhost:8001/', timeout=2) is payload
    assert calls == [(('POST', 'http://localhost:8001/chat'), {'timeout': 2, 'json': {'question': '市场', 'thread_id': 'id'}})]

@pytest.mark.parametrize('error', [requests.ConnectionError, requests.Timeout, requests.RequestException])
def test_unavailable(monkeypatch, error):
    def request(*args, **kwargs):
        raise error('my-test-secret http://private.host')
    monkeypatch.setattr(requests, 'request', request)
    with pytest.raises(client.AgentUnavailable) as exc:
        client.health()
    assert 'my-test-secret' not in str(exc.value)
    assert 'private.host' not in str(exc.value)
    assert exc.value.__suppress_context__

@pytest.mark.parametrize('status', [409, 422, 500])
def test_http(monkeypatch, status):
    capture(monkeypatch, Response({'detail': '无效\n状态 ' + '中'*400}, status))
    with pytest.raises(client.AgentHTTPError) as exc:
        client.health()
    assert exc.value.status_code == status
    assert len(exc.value.detail) == 300
    assert exc.value.detail.startswith('无效 状态')
    assert '\n' not in exc.value.detail

@pytest.mark.parametrize('detail', ['api_key=my-test-secret', 'Bearer my-test-secret', 'Traceback my-test-secret', 'headers my-test-secret', 'password: my-test-secret', {'input': 'my-test-secret'}, [{'msg': 'Field required', 'input': 'my-test-secret'}]])
def test_secret(monkeypatch, detail):
    capture(monkeypatch, Response({'detail': detail}, 422))
    with pytest.raises(client.AgentHTTPError) as exc:
        client.health()
    assert 'my-test-secret' not in str(exc.value)

def test_url_hidden(monkeypatch):
    capture(monkeypatch, Response({'detail': '失败 https://host/a?x=my-test-secret\n重试'}, 409))
    with pytest.raises(client.AgentHTTPError) as exc:
        client.health()
    assert exc.value.detail == '失败 [地址已隐藏] 重试'

@pytest.mark.parametrize('payload', [[], 'my-test-secret', ValueError('my-test-secret')])
def test_invalid_response(monkeypatch, payload):
    capture(monkeypatch, Response(payload))
    with pytest.raises(client.AgentError) as exc:
        client.health()
    assert 'my-test-secret' not in str(exc.value)

def test_workflow(monkeypatch):
    calls = capture(monkeypatch, Response({'status': 'awaiting_review'}))
    client.start_workflow()
    client.get_workflow('id')
    for action in ('approve', 'edit', 'reject'):
        client.review_workflow('id', action, 'tester', '意见', '正文')
    assert calls[0] == (('POST', 'http://127.0.0.1:8001/workflow/run'), {'timeout': 60.0})
    assert calls[1] == (('GET', 'http://127.0.0.1:8001/workflow/id'), {'timeout': 15.0})
    for action, (args, kwargs) in zip(('approve', 'edit', 'reject'), calls[2:]):
        assert args == ('POST', 'http://127.0.0.1:8001/workflow/id/approve')
        assert kwargs == {'timeout': 60.0, 'json': dict(action=action, reviewer='tester', comment='意见', edited_markdown='正文')}
    with pytest.raises(client.AgentError):
        client.review_workflow('id', 'invalid-my-test-secret', 'tester')
    assert len(calls) == 5

def test_report(monkeypatch):
    calls = capture(monkeypatch, Response())
    assert client.get_report('id') == '# 简报\n中文'
    assert calls == [(('GET', 'http://127.0.0.1:8001/workflow/id/report'), {'timeout': 15.0})]

def test_no_credentials(monkeypatch):
    env_get = client.os.environ.get
    def get_env(key, default=None):
        assert key != 'DEEPSEEK_API_KEY'
        return env_get(key, default)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'my-test-secret')
    monkeypatch.setattr(client.os.environ, 'get', get_env)
    calls = capture(monkeypatch, Response())
    client.chat('question', 'id')
    assert 'headers' not in calls[0][1]
    assert 'Authorization' not in repr(calls)
    assert 'my-test-secret' not in repr(calls)
