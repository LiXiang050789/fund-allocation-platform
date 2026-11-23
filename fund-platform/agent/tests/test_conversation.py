import pytest
from app.conversation import ConversationStore, validate_thread_id

@pytest.mark.parametrize('value', ['', '../file','a/b','空格','a b','x'*81,"'; DELETE FROM messages;--"])
def test_invalid_ids(value):
    with pytest.raises(ValueError): validate_thread_id(value)

def test_isolation_pruning_clear_and_restart(tmp_path):
    path=tmp_path/'conversations.sqlite'
    store=ConversationStore(path)
    for i in range(12):
        store.append_turn('a',str(i),dict(answer=str(i),mode='chat'))
    store.append_turn('b','other',dict(answer='other answer',mode='research'))
    assert len(store.history('a'))==20
    assert store.history('a')[0]['content']=='2'
    assert len(store.history('b'))==2
    store.close()
    store=ConversationStore(path)
    assert len(store.history('a'))==20
    store.clear('a')
    assert store.history('a')==[] and len(store.history('b'))==2
    store.close()
