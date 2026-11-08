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
