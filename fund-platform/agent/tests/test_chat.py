from types import SimpleNamespace
import pytest
from app import chat

class LLM:
    def invoke(self, messages):
        self.messages=messages
        return SimpleNamespace(content='夏普比率衡量单位风险承担的超额收益。')
    def stream(self, messages):
        for text in ('通胀是一般', '物价水平持续上涨。', '购买力下降。'):
            yield SimpleNamespace(content=text)

@pytest.mark.parametrize('text,mode,absent', [
    ('建议买入基金。','refusal','建议买入'), ('建议加仓。','refusal','建议加仓'),
    ('平台最新评分为33.9。','chat','33.9'), ('LGB 夏普为0.64。','chat','0.64'),
    ('密钥是 sk-test-private。','chat','sk-test-private'),
    ('位置 /home/lx/private。','chat','/home/lx/private'),
    ('概念。[来源: forged]','chat','[来源:'),
    ('我可以联网搜索。','chat','我可以联网搜索'),
])
def test_guard(text, mode, absent):
    answer, actual=chat.guard(text)
    assert actual==mode and absent not in answer
    assert answer.endswith('不构成投资建议。')

def test_length_and_prompt_history(monkeypatch):
    assert len(chat.guard('文字'*500)[0]) <=400
    llm=LLM()
    monkeypatch.setattr(chat, 'get_llm', lambda: llm)
    answer,mode=chat.answer('解释夏普', [dict(role='user',content='之前的问题')])
    assert mode=='chat' and '夏普' in answer
    assert '之前的问题' in llm.messages[1].content
    for word in ('400', 'research', '密钥', '投资建议', '平台'):
        assert word in llm.messages[0].content

def test_safe_live_stream(monkeypatch):
    monkeypatch.setattr(chat,'get_llm',lambda: LLM())
    events=list(chat.stream('什么是通胀',[]))
    tokens=''.join(value for event,value in events if event=='token')
    assert tokens==events[-1][1][0]
    assert events[-1][0]=='result'

def test_unsafe_phrase_split_across_chunks_is_never_emitted(monkeypatch):
    class Bad(LLM):
        def stream(self,messages):
            for text in ('建','议买','入这只基金。'):
                yield SimpleNamespace(content=text)
    monkeypatch.setattr(chat,'get_llm',lambda: Bad())
    events=list(chat.stream('例子',[]))
    assert not any(event=='token' for event,_ in events)
    assert events[-1][1][1]=='refusal'


def test_registry_and_retriever_logs_redact_question(caplog):
    import json, logging
    import app.orchestrator
    for name,payload in [('app.tools.registry',{'args':{'query':'private question'},'ok':True}),
                         ('app.rag.retriever',{'query':'private question','backend':'hybrid'})]:
        with caplog.at_level(logging.INFO,logger=name):
            logging.getLogger(name).info(json.dumps(payload))
    assert 'private question' not in caplog.text
    assert '[redacted]' in caplog.text
