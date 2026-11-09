import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from docx import Document

from app.rag.index import build, validate
from app.rag.ingest import chunks_for, parse
from app.rag.retriever import rrf, search


@pytest.fixture
def mini_corpus(tmp_path):
    root = tmp_path/'repo'
    root.mkdir()
    (root/'apple.md').write_text('# 苹果规则\n## 估值\n苹果估值低于阈值时满分。\n', encoding='utf-8')
    (root/'bond.txt').write_text('一、债券\n债券利率与宏观周期。\n', encoding='utf-8')
    doc = Document()
    doc.add_heading('风险匹配', level=1)
    doc.add_paragraph('产品等级不得超过承受能力。')
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = '等级'
    table.cell(0, 1).text = '产品'
    table.cell(1, 0).text = '谨慎'
    table.cell(1, 1).text = '债券'
    doc.save(root/'risk.docx')
    corpus = root/'corpus.yaml'
    corpus.write_text(yaml.safe_dump({'files': [
        {'path': 'apple.md', 'authority': 'authoritative'},
        {'path': 'bond.txt', 'authority': 'background'},
        {'path': 'risk.docx', 'authority': 'historical'}]}, allow_unicode=True), encoding='utf-8')
    return root, corpus, root/'index'


def fake_encode(texts):
    return np.array([[float('苹果' in text), float('债券' in text), float('风险' in text)] for text in texts], dtype='float32')


def test_parse_formats_and_metadata(mini_corpus):
    root, _, _ = mini_corpus
    for name in ['apple.md', 'bond.txt', 'risk.docx']:
        chunks = chunks_for(root/name, root, 'authoritative')
        assert chunks
        assert all({'chunk_id', 'source', 'source_name', 'section', 'doc_date', 'authority', 'text'} <= c.keys() for c in chunks)
        assert all(c['source'] == name for c in chunks)
    blocks = parse(root/'risk.docx')
    assert any(b['kind'] == 'table' and '| 等级 | 产品 |' in b['text'] for b in blocks)
    assert any(b.get('style') == 'Heading 1' for b in blocks)
    assert '苹果规则 / 估值' in chunks_for(root/'apple.md', root, 'authoritative')[0]['section']


def test_repeat_build_stable(mini_corpus):
    root, corpus, index = mini_corpus
    build(root, corpus, index, vector=False)
    first = (index/'chunks.jsonl').read_bytes()
    build(root, corpus, index, vector=False)
    assert first == (index/'chunks.jsonl').read_bytes()
    assert len(validate(root, corpus, index)['files']) == 3


def test_bm25_and_hybrid_rrf(mini_corpus):
    root, corpus, index = mini_corpus
    manifest = build(root, corpus, index, encoder=fake_encode)
    assert manifest['backend'] == 'hybrid' and (index/'faiss.index').exists()
    lexical, _ = search('苹果估值', root=root, corpus_path=corpus, index_dir=index, vector=False)
    hybrid, _ = search('苹果估值', root=root, corpus_path=corpus, index_dir=index, encoder=fake_encode)
    assert lexical['backend'] == 'bm25' and hybrid['backend'] == 'hybrid'
    assert lexical['chunks'][0]['source'] == hybrid['chunks'][0]['source'] == 'apple.md'
    assert rrf([[1, 2], [2, 1]]) == [(1, 1/61+1/62), (2, 1/62+1/61)]


def test_source_and_index_tampering(mini_corpus):
    root, corpus, index = mini_corpus
    build(root, corpus, index, vector=False)
    (root/'apple.md').write_text('changed')
    with pytest.raises(ValueError, match='索引过期，请重建'):
        search('苹果', root=root, corpus_path=corpus, index_dir=index, vector=False)
    build(root, corpus, index, vector=False)
    search('债券', root=root, corpus_path=corpus, index_dir=index, vector=False)
    (index/'chunks.jsonl').write_text('tampered')
    with pytest.raises(ValueError, match='索引过期，请重建'):
        search('债券', root=root, corpus_path=corpus, index_dir=index, vector=False)


def test_embedding_failure_falls_back(mini_corpus):
    root, corpus, index = mini_corpus
    def failed(_):
        raise RuntimeError('fake download failure')
    manifest = build(root, corpus, index, encoder=failed)
    assert manifest['backend'] == 'bm25' and not (index/'faiss.index').exists()
    assert search('苹果', root=root, corpus_path=corpus, index_dir=index)[0]['backend'] == 'bm25'


def test_long_tables_repeat_header(tmp_path):
    doc = Document()
    table = doc.add_table(rows=41, cols=2)
    table.cell(0, 0).text = '项目'
    table.cell(0, 1).text = '规则'
    for i in range(1, 41):
        table.cell(i, 0).text = str(i)
        table.cell(i, 1).text = '完整保留表格行的文字' * 5
    path = tmp_path/'table.docx'
    doc.save(path)
    chunks = chunks_for(path, tmp_path, 'authoritative')
    assert len(chunks) > 1
    assert all(c['text'].startswith('| 项目 | 规则 |\n| --- | --- |') for c in chunks)


@pytest.fixture
def indexed_tool(mini_corpus, monkeypatch):
    import dataclasses
    from app.rag import retriever
    root, corpus, index = mini_corpus
    build(root, corpus, index, vector=False)
    monkeypatch.setattr(retriever, 'settings', dataclasses.replace(retriever.settings,
        platform_dir=root/'fund-platform', index_dir=index, enable_vector_rag=False))
    monkeypatch.setattr(retriever, 'CORPUS_PATH', corpus)
    return root, corpus, index


def test_search_tool_contract(indexed_tool):
    from app.tools.registry import execute, search_docs, TOOL_SCHEMAS
    result = search_docs(query='苹果估值', top_k=2)
    assert result['ok'] and result['data']['backend'] == 'bm25'
    assert result['data']['chunks'][0]['source'] == 'apple.md'
    assert result['source'] and result['as_of']
    for args in ({'query': ''}, {'query': ' '}, {'query': 'x'*201}, {'query': '规则', 'top_k': 0},
                 {'query': '规则', 'top_k': 11}, {'query': '规则', 'path': '/etc/passwd'},
                 {'query': '规则', 'top_k': '5'}, {}):
        assert not execute('search_docs', args)['ok']
    schema = next(s['function']['parameters'] for s in TOOL_SCHEMAS if s['function']['name'] == 'search_docs')
    assert 'query' in schema['required'] and schema['additionalProperties'] is False


def test_tool_missing_and_stale_index(indexed_tool):
    from app.tools.registry import search_docs
    root, corpus, index = indexed_tool
    (root/'apple.md').write_text('changed')
    result = search_docs(query='苹果')
    assert not result['ok'] and '索引过期，请重建' in result['error']
    (index/'manifest.json').unlink()
    result = search_docs(query='苹果')
    assert not result['ok'] and 'app.rag.build_index' in result['error']


def doc_state(answer, authority='authoritative', text='规则指标为66.6。'):
    return {'question': '解释规则', 'answer': answer, 'tool_results': [
        {'ok': True, 'tool': 'search_docs', 'data': {'chunks': [
            {'source': 'rules.md', 'source_name': '规则.md', 'section': '规则 / 指标',
             'authority': authority, 'text': text}]}}]}


def test_citation_validation():
    from app.graph import verify
    assert verify(doc_state('指标66.6。[来源: 规则.md §规则]'))['verified']
    assert not verify(doc_state('指标66.6。[来源: 不存在的文件 §X]'))['verified']
    assert not verify(doc_state('指标66.6。'))['verified']
    assert not verify(doc_state('规则说明。[来源: 规则.md]'))['verified']


@pytest.mark.parametrize('authority,valid', [('authoritative', True), ('background', False), ('historical', False)])
def test_authority_numeric_boundary(authority, valid):
    from app.graph import verify
    assert verify(doc_state('指标66.6。[来源: 规则.md §规则]', authority=authority))['verified'] is valid


def test_obsolete_values_need_local_marker():
    from app.graph import verify
    assert not verify(doc_state('收益19.8%。[来源: 规则.md §规则]', text='曾写19.8。'))['verified']
    assert verify(doc_state('旧值禁用19.8%。[来源: 规则.md §规则]', authority='historical', text='曾写19.8。'))['verified']
    assert not verify(doc_state('历史说明。收益19.8%。[来源: 规则.md §规则]', text='曾写19.8。'))['verified']


def test_document_graph_offline(indexed_tool, monkeypatch):
    import app.graph as g
    monkeypatch.setattr(g, 'get_llm', lambda: (_ for _ in ()).throw(RuntimeError('offline')))
    state = g.graph.invoke({'question': '苹果估值规则是什么？'})
    assert 'search_docs' in state['route']
    assert '[来源: apple.md §' in state['answer'] and state['verified']
    assert len(state['answer']) <= 300


def test_golden_quote_check_rejects_missing_text(mini_corpus):
    from app.rag.build_index import check_golden
    root, corpus, index = mini_corpus
    build(root, corpus, index, vector=False)
    golden = root/'golden.yaml'
    golden.write_text(yaml.safe_dump({'questions': [{'id': 'Q', 'source': 'apple.md',
        'source_quote': '苹果估值低于阈值时满分。'}]}, allow_unicode=True))
    assert check_golden(index, golden)
    golden.write_text(yaml.safe_dump({'questions': [{'id': 'Q', 'source': 'apple.md',
        'source_quote': '原文没有这句话'}]}, allow_unicode=True))
    assert not check_golden(index, golden)


def test_short_paragraph_and_table_stay_together(mini_corpus):
    root, _, _ = mini_corpus
    chunks = chunks_for(root/'risk.docx', root, 'authoritative')
    assert any('产品等级不得超过承受能力。' in c['text'] and '| 等级 | 产品 |' in c['text'] for c in chunks)
    assert all(len(c['text']) <= 800 for c in chunks)


def test_score_ranges_keep_positive_bounds_and_drawdown_sign():
    from app.graph import verify
    state = doc_state('分数段21-40分，最大回撤-14.27%。[来源: 规则.md §规则]',
                      text='分数段21 - 40分，最大回撤-14.27%。')
    assert verify(state)['verified']
    state['answer'] = '分数段21-40分，最大回撤14.27%。[来源: 规则.md §规则]'
    assert not verify(state)['verified']


def test_known_source_filename_is_metadata_not_a_numeric_claim():
    from app.graph import verify
    state = {'question': '危机回撤', 'answer': '根据PPT素材/4-压力测试表.csv，最大回撤-13.6%。',
             'tool_results': [{'ok': True, 'tool': 'get_drawdown_recovery',
                 'source': 'PPT素材/4-压力测试表.csv', 'data': {'最大回撤': '-13.6%'}}]}
    assert verify(state)['verified']
    state['answer'] += '指标62.7。'
    assert not verify(state)['verified']


def test_ppo_golden_authority_and_verbatim_quotes():
    from app.config import settings
    from app.rag.ingest import load_corpus
    root = settings.platform_dir.parent
    items = yaml.safe_load((settings.agent_dir/'eval/doc_golden_set.yaml').read_text())['questions']
    assert len(items) == 16
    for item in items[-2:]:
        assert any(item['source_quote'] in c['text'] for c in chunks_for(root/item['source'], root, item['authority']))
    assert items[-2]['source'] == 'fund-platform/docs/强化学习.txt'
    assert items[-2]['source_quote'] == '④ 换手率惩罚：约束频繁无效调仓，降低交易摩擦成本，贴合实盘。'
    files = {f['path']: f['authority'] for f in load_corpus(settings.agent_dir/'app/rag/corpus.yaml')}
    assert files[items[-1]['source']] == 'authoritative'
    assert files[items[-2]['source']] == 'background'


@pytest.mark.parametrize('query', ['bond.txt', '债券', '苹果估值 bond.txt CompletelyUnrelated'])
def test_named_source_reservation(mini_corpus, query, caplog):
    import logging
    root, corpus, index = mini_corpus
    build(root, corpus, index, vector=False)
    with caplog.at_level(logging.INFO, logger='app.rag.retriever'):
        result, _ = search(query, top_k=1, root=root, corpus_path=corpus, index_dir=index, vector=False)
    assert any(c['source_name'] == 'bond.txt' for c in result['chunks'])
    assert 'named_sources' in caplog.text and 'named_reserved' in caplog.text


def test_plain_query_keeps_rrf(mini_corpus, monkeypatch):
    import app.rag.retriever as retriever
    root, corpus, index = mini_corpus
    build(root, corpus, index, vector=False)
    args = dict(root=root, corpus_path=corpus, index_dir=index, vector=False)
    actual = search('苹果估值', **args)[0]
    monkeypatch.setattr(retriever, 'named_sources', lambda *_: [])
    assert search('苹果估值', **args)[0] == actual


def test_multiple_named_sources_protect_reserved_slots(mini_corpus):
    root, corpus, index = mini_corpus
    build(root, corpus, index, vector=False)
    result, _ = search('苹果估值 bond.txt risk.docx', top_k=3, root=root, corpus_path=corpus, index_dir=index, vector=False)
    assert {'bond.txt', 'risk.docx'} <= {c['source_name'] for c in result['chunks']}
    assert len(result['chunks']) <= 3


def test_named_matching_and_generic_manual():
    from app.rag.named import named_sources
    assert named_sources(' 强化学习 设计文档 ', []) == ['强化学习.txt']
    assert named_sources(' PPO ', []) == []
    assert named_sources('手册怎么说', [], explicit=True) == []
    assert named_sources('最终 PPO 奖励', []) == ['PPO奖励口径-最终代码.md']


@pytest.mark.parametrize('cite_named,valid', [(False, False), (True, True)])
def test_named_source_verifier(cite_named, valid):
    from app.graph import verify
    state = doc_state('设计稿为收益奖励与波动、回撤、换手惩罚。[来源: 项目彻底理解手册-答辩备战版.md §规则]')
    state['question'] = '强化学习设计文档中，PPO奖励函数的正向奖励和三类惩罚是什么？'
    chunk = state['tool_results'][0]['data']['chunks'][0]
    chunk['source_name'] = '项目彻底理解手册-答辩备战版.md'
    if cite_named:
        state['tool_results'][0]['data']['chunks'].append({**chunk, 'source_name': '强化学习.txt'})
        state['answer'] += '[来源: 强化学习.txt §规则]'
    result = verify(state)
    assert result['verified'] is valid
    if not valid:
        assert '强化学习.txt' in result['correction']


def test_named_neighbor_enters_candidates(mini_corpus, monkeypatch, caplog):
    import logging
    from app.rag import retriever
    root, corpus, index = mini_corpus
    (root/'apple.md').write_text('# 苹果规则\n## 概述\n普通说明。\n## 细则\n独特指标细则。\n## 尾声\n结尾。')
    build(root, corpus, index, vector=False)
    chunks = [json.loads(line) for line in (index/'chunks.jsonl').read_text().splitlines()]
    first = next(i for i, c in enumerate(chunks) if c['source_name'] == 'apple.md')
    monkeypatch.setattr(retriever, 'rrf', lambda _: [(first, 1.0)])
    with caplog.at_level(logging.INFO, logger='app.rag.retriever'):
        result, _ = search('apple.md 独特指标细则', top_k=1, root=root, corpus_path=corpus, index_dir=index, vector=False)
    assert '独特指标' in result['chunks'][0]['text']
    assert 'named_neighbors' in caplog.text
