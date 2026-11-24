"""Full Streamlit AppTest plus transport tests; no LLM, HTTP, or real sends."""
import sys
from pathlib import Path
import pytest

PLATFORM = Path(__file__).resolve().parents[2]/'platform'
sys.path.insert(0, str(PLATFORM))
import agent_client


def fake_stream(question, thread, **kwargs):
    mode = 'refusal' if '推荐' in question else 'research' if '评分' in question or 'LGB' in question else 'chat'
    yield 'meta', dict(mode=mode, thread_id=thread)
    if mode == 'research':
        for stage,label in [('intake','理解'),('router','选工具'),('tools','查询/检索'),('compose','生成'),('verify','校验')]:
            yield 'status',dict(stage=stage,label=label)
    answer = '测试回答：'+question
    if mode == 'chat':
        for char in answer:
            yield 'token',dict(text=char)
    yield 'final',dict(answer=answer,mode=mode,verified=True if mode=='research' else None,
                      citations=['test evidence'] if mode=='research' else [],
                      tool_calls=[],latency_ms=10,thread_id=thread)


@pytest.fixture
def ui(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(agent_client,'health',lambda **_:dict(status='ok'))
    monkeypatch.setattr(agent_client,'stream_chat',fake_stream)
    monkeypatch.setattr(agent_client,'chat_history',lambda thread,**_:dict(messages=[]))
    monkeypatch.setattr(agent_client,'clear_chat',lambda thread,**_:dict(cleared=True))
    app=AppTest.from_file(str(PLATFORM/'app.py'),default_timeout=30).run()
    assert not app.exception
    return app


def by_label(elements,label):
    return next(item for item in elements if item.label==label)


def test_render_send_modes_switch_and_clear(ui):
    assert len(ui.chat_input)==1
    first=ui.session_state['current_conversation']
    ui.chat_input[0].set_value('你好').run()
    assert not ui.exception
    messages=ui.session_state['conversations'][first]['messages']
    assert [m['role'] for m in messages]==['assistant','user','assistant']
    assert messages[-1]['mode']=='chat'
    assert not any('校验' in str(c.value) for message in ui.get('chat_message') for c in message.get('caption'))
    ui.chat_input[0].set_value('最新评分').run()
    assert ui.session_state['conversations'][first]['messages'][-1]['mode']=='research'
    assert any('数字与引用校验' in str(c.value) for c in ui.caption)
    by_label(ui.button,'＋ 新对话').click().run()
    assert not ui.exception
    second=ui.session_state['current_conversation']
    assert first!=second and len(ui.session_state['conversations'][second]['messages'])==1
    by_label(ui.radio,'对话列表').set_value(first).run()
    assert ui.session_state['current_conversation']==first
    assert len(ui.session_state['conversations'][first]['messages'])==5
    by_label(ui.button,'清空当前对话').click().run()
    assert len(ui.session_state['conversations'][first]['messages'])==1
    assert len(ui.session_state['conversations'][second]['messages'])==1
    assert not ui.exception


def test_refusal_and_example_questions(ui):
    by_label(ui.button,'什么是夏普比率').click().run()
    assert not ui.exception
    thread=ui.session_state['current_conversation']
    assert ui.session_state['conversations'][thread]['messages'][-1]['mode']=='chat'
    ui.chat_input[0].set_value('推荐一只基金').run()
    assert ui.session_state['conversations'][thread]['messages'][-1]['mode']=='refusal'
    assert any(c.value=='合规拒答' for c in ui.caption)


def test_sse_client_parses_and_closes(monkeypatch):
    import json
    class Response:
        status_code=200
        closed=False
        def iter_lines(self,**kwargs):
            for event,payload in [('meta',{'mode':'chat'}),('token',{'text':'你好'}),('final',{'answer':'你好'})]:
                yield 'event: '+event
                yield 'data: '+json.dumps(payload,ensure_ascii=False)
                yield ''
        def close(self): self.closed=True
    response=Response()
    calls=[]
    def post(*args,**kwargs):
        calls.append(kwargs)
        return response
    monkeypatch.setattr(agent_client.requests,'post',post)
    result=list(agent_client.stream_chat('你好','a'))
    assert [event for event,_ in result]==['meta','token','final']
    assert calls[0]['stream'] and calls[0]['timeout']==(5,120)
    assert response.closed
    generator=agent_client.stream_chat('你好','a')
    next(generator)
    generator.close()
    assert response.closed


def test_sse_interruption_is_explicit(monkeypatch):
    class Response:
        status_code=200
        def iter_lines(self,**kwargs): return iter(['event: meta','data: {}',''])
        def close(self): pass
    monkeypatch.setattr(agent_client.requests,'post',lambda *a,**k:Response())
    with pytest.raises(agent_client.AgentError,match='未完成'):
        list(agent_client.stream_chat('你好','a'))


def test_all_tabs_and_width_migration(ui, caplog):
    import inspect
    import streamlit as st
    assert len(ui.tabs) == 5
    assert not ui.exception
    source = (PLATFORM/'app.py').read_text()
    assert 'use_container_width' not in source
    for name in ('plotly_chart', 'dataframe', 'button', 'image'):
        assert 'width' in inspect.signature(getattr(st, name)).parameters
    assert 'use_container_width' not in caplog.text


def test_p0_comparison_preset(ui):
    button = by_label(ui.button, 'LGB融合与PPO对比')
    button.click().run()
    assert not ui.exception
    thread = ui.session_state['current_conversation']
    messages = ui.session_state['conversations'][thread]['messages']
    assert messages[-2]['content'] == 'LGB融合与PPO对比'
    assert messages[-1]['mode'] == 'research'
