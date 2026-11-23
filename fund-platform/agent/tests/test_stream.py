import json
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from app import main, chat
from app.conversation import ConversationStore
from app.orchestrator import Orchestrator

class Graph:
    def invoke(self,state):
        self.question=state['question']
        return dict(answer='评分33.9[source: data]', verified=True,
                    tool_results=[{'ok':True}],citations=['data'],tool_calls=[{'name':'tool','ok':True}])
    def stream(self,state,**kwargs):
        for node in ('intake','router','tools','compose','verify','finalize'):
            yield {node:self.invoke(state) if node=='finalize' else {}}

@pytest.fixture
def client(monkeypatch,tmp_path):
    store=ConversationStore(tmp_path/'chat.sqlite')
    monkeypatch.setattr(main,'get_conversations',lambda:store)
    monkeypatch.setattr(main,'graph',Graph())
    yield TestClient(main.app),store
    store.close()

def events(response):
    return [(block.splitlines()[0][7:],json.loads(block.splitlines()[1][6:]))
            for block in response.text.strip().split('\n\n')]

def test_sync_history_clear_and_request_validation(client):
    api,store=client
    result=api.post('/chat',json={'question':'你好','thread_id':'a'}).json()
    assert result['mode']=='chat' and result['verified'] is None
    assert result['tool_calls']==result['citations']==[]
    assert len(api.get('/chat/history/a').json()['messages'])==2
    assert api.get('/chat/history/b').json()['messages']==[]
    assert api.delete('/chat/a').json()['cleared']
    assert store.history('a')==[]
    for body in ({'question':' '},{'question':'x'*4001},{'question':'你好','thread_id':'bad id'},{'question':'你好','thread_id':''}):
        assert api.post('/chat',json=body).status_code==422

def test_chat_stream_contract(client):
    api,_=client
    response=api.post('/chat/stream',json={'question':'你好','thread_id':'chat-stream'})
    sequence=events(response)
    assert sequence[0][0]=='meta' and sequence[-1][0]=='final'
    assert any(event=='token' for event,_ in sequence)
    final=sequence[-1][1]
    assert {'answer','mode','verified','citations','tool_calls','latency_ms','thread_id'} <= final.keys()
    assert final['verified'] is None
    assert ''.join(data['text'] for event,data in sequence if event=='token')==final['answer']

def test_research_stream_has_status_but_no_token(client):
    api,_=client
    sequence=events(api.post('/chat/stream',json={'question':'最新评分'}))
    assert [data['label'] for event,data in sequence if event=='status']==['理解','选工具','查询/检索','生成','校验']
    assert not any(event=='token' for event,_ in sequence)
    assert sequence[-1][1]['mode']=='research' and sequence[-1][1]['verified'] is True

def test_no_evidence_and_failed_verify_do_not_disclose_draft(client,monkeypatch):
    api,_=client
    for state in ({'answer':'unchecked 999999','verified':True,'tool_results':[]},
                  {'answer':'unchecked 999999','verified':False,'tool_results':[{'ok':True}]}):
        monkeypatch.setattr(main,'graph',SimpleNamespace(invoke=lambda _,state=state:state))
        result=api.post('/chat',json={'question':'评分'}).json()
        assert result['verified'] is False and '999999' not in result['answer']

def test_refusal_and_llm_failure_do_not_create_partial_turns(client,monkeypatch):
    api,store=client
    result=api.post('/chat',json={'question':'推荐一只基金'}).json()
    assert result['mode']=='refusal' and result['verified'] is None
    monkeypatch.setattr(chat,'get_llm',lambda: (_ for _ in ()).throw(RuntimeError('secret')))
    sequence=events(api.post('/chat/stream',json={'question':'解释通胀','thread_id':'bad'}))
    assert sequence[-1]==('error',{'message':'服务暂时不可用，请重试'})
    assert store.history('bad')==[]

def test_cancel_before_final_writes_no_turn(tmp_path):
    store=ConversationStore(tmp_path/'chat.sqlite')
    generator=Orchestrator(store,Graph()).events('你好','cancel',True)
    assert next(generator)[0]=='meta'
    assert next(generator)[0]=='token'
    generator.close()
    assert store.history('cancel')==[]
    store.close()

def test_two_session_followups_are_isolated(client):
    api,store=client
    graph=main.graph
    api.post('/chat',json={'question':'动态评分的夏普是多少？','thread_id':'research'})
    api.post('/chat',json={'question':'你好','thread_id':'other'})
    api.post('/chat',json={'question':'那 LGB 呢','thread_id':'research'})
    assert graph.question=='LGB 的夏普是多少？'
    assert len(store.history('research'))==4 and len(store.history('other'))==2
