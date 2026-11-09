"""BM25/FAISS 排名与 RRF 融合；运行时只读且不下载模型。"""
from __future__ import annotations

import json
import logging
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings
from .index import encode, tokens, validate
from .named import named_sources

log = logging.getLogger(__name__)
CORPUS_PATH = Path(__file__).with_name('corpus.yaml')


def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0.0) + 1/(k+rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


@lru_cache(maxsize=4)
def load_index(index_dir, signature):
    path = Path(index_dir)
    chunks = [json.loads(line) for line in (path/'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
    payload = json.loads((path/'bm25.json').read_text(encoding='utf-8'))
    if len(payload['tokens']) != len(chunks):
        raise ValueError('索引行数不一致，请重建')
    return chunks, BM25Okapi(payload['tokens'])


@lru_cache(maxsize=2)
def load_faiss(path, signature):
    import faiss
    return faiss.read_index(path)


def search(query, top_k=5, *, root=None, corpus_path=None, index_dir=None, vector=None, encoder=None):
    started = time.perf_counter()
    root = Path(root) if root is not None else settings.platform_dir.parent
    corpus_path = Path(corpus_path) if corpus_path is not None else CORPUS_PATH
    index_dir = Path(index_dir) if index_dir is not None else settings.index_dir
    vector = settings.enable_vector_rag if vector is None else vector
    manifest = validate(root, corpus_path, index_dir)  # Hash-check on every call, including cached indexes.
    signature = json.dumps(manifest['artifacts'], sort_keys=True)
    chunks, bm25 = load_index(str(index_dir), signature)
    named = named_sources(query, chunks)
    query_tokens = tokens(query)
    scores = bm25.get_scores(query_tokens)
    candidates = min(len(chunks), max(20, top_k*4))
    lexical = [i for i in sorted(range(len(chunks)), key=lambda i: (-scores[i], i)) if scores[i] > 0][:candidates]
    rankings = [lexical]
    backend = 'bm25'
    if vector and manifest['backend'] == 'hybrid':
        try:
            import faiss
            index = load_faiss(str(index_dir/'faiss.index'), signature)
            query_matrix = np.asarray((encoder or encode)([query]), dtype='float32')
            faiss.normalize_L2(query_matrix)
            _, found = index.search(query_matrix, candidates)
            rankings.append([int(i) for i in found[0] if i >= 0])
            backend = 'hybrid'
        except Exception as exc:
            log.warning('vector unavailable (%s); using BM25', type(exc).__name__)
    found, seen = [], set()
    for index, score in rrf(rankings):
        chunk = chunks[index]
        key = (chunk['source_name'], chunk['section'], chunk['text'])
        if key not in seen:
            seen.add(key)
            found.append({**chunk, 'score': score})
        if len(found) == top_k:
            break
    reserved = []
    protected = set()
    neighbors = []
    for name in named[:top_k]:
        existing = next((i for i, c in enumerate(found) if c['source_name'] == name), None)
        if existing is not None:
            # Expand adjacent chunks in corpus order, then rank this local window.
            # This retains a named source while recovering paragraphs split at headings.
            pool = [i for i, c in enumerate(chunks) if c['source_name'] == name]
            current = next(i for i in pool if chunks[i]['chunk_id'] == found[existing]['chunk_id'])
            position = pool.index(current)
            window = pool[max(0, position-1):position+2]
            neighbors.extend(chunks[i]['chunk_id'] for i in window if i != current)
            best = min(window, key=lambda i: (-scores[i], i))
            found[existing] = {**chunks[best], 'score': float(scores[best])}
            protected.add(existing)
            continue
        pool = [i for i, c in enumerate(chunks) if c['source_name'] == name]
        if not pool:
            continue
        if any(scores[i] > 0 for i in pool):
            best = min(pool, key=lambda i: (-scores[i], i))
        else:
            best = min(pool, key=lambda i: (-sum(t in chunks[i]['text'] for t in query_tokens), i))
        item = {**chunks[best], 'score': float(scores[best])}
        if len(found) < top_k:
            slot = len(found)
            found.append(item)
        else:
            slot = next((i for i in reversed(range(len(found))) if i not in protected), None)
            if slot is None:
                break
            found[slot] = item
        protected.add(slot)
        reserved.append(name)
    unique, seen = [], set()
    for chunk in found:
        key = (chunk['source_name'], chunk['section'], chunk['text'])
        if key not in seen:
            unique.append(chunk)
            seen.add(key)
    found = unique
    log.info(json.dumps(dict(query=query, top_k=top_k, backend=backend, candidates=sum(map(len, rankings)),
                  named_sources=named, named_reserved=reserved, named_neighbors=neighbors,
                  latency_ms=round((time.perf_counter()-started)*1000, 2)), ensure_ascii=False))
    return dict(backend=backend, chunks=found), manifest['built_at'].split('T')[0]
