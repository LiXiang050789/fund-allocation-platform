"""索引构建、JSON 持久化、完整性检查与 CPU embedding 懒加载。"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

from .ingest import chunks_for, digest, load_corpus

log = logging.getLogger(__name__)
MODEL = 'BAAI/bge-small-zh-v1.5'
REBUILD = '.venv/bin/python -m app.rag.build_index'
STOP = {'的', '了', '是', '和', '与', '及', '在', '有', '哪些', '什么', '多少', '如何', '请', '分别', '一个', '为', '用', '中', '对'}


def tokens(text):
    import jieba
    import re
    return [word for word in jieba.lcut(text.lower()) if word not in STOP and re.search(r'[\w\u4e00-\u9fff]', word)]


@lru_cache(maxsize=2)
def embedding_model(local_only=True):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL, device='cpu', local_files_only=local_only, trust_remote_code=False)


def encode(texts, *, local_only=True):
    return np.asarray(embedding_model(local_only).encode(texts, normalize_embeddings=True,
                      show_progress_bar=False, batch_size=16), dtype='float32')


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def missing_corpus(root, corpus_path):
    root = Path(root).resolve()
    missing = []
    for entry in load_corpus(corpus_path):
        path = (root / entry['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('语料路径越界')
        if not path.is_file():
            missing.append(entry['path'])
    return missing


def require_corpus(root, corpus_path):
    missing = missing_corpus(root, corpus_path)
    if missing:
        raise ValueError('语料缺失: ' + ', '.join(missing) +
                         '；请把文件放到仓库根目录，或从 corpus.yaml 移除。')


def build(root, corpus_path, index_dir, *, vector=True, encoder=None):
    root, index_dir = Path(root), Path(index_dir)
    require_corpus(root, corpus_path)
    files = load_corpus(corpus_path)
    chunks, manifest_files = [], []
    for entry in files:
        path = (root / entry['path']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('语料路径越界')
        before = digest(path)
        current = chunks_for(path, root.resolve(), entry['authority'])
        if digest(path) != before:
            raise ValueError('构建中语料变化，请重建')
        chunks.extend(current)
        manifest_files.append(dict(path=entry['path'], authority=entry['authority'], sha256=before,
                                   mtime=path.stat().st_mtime, chunks=len(current)))
    if not chunks:
        raise ValueError('语料为空')
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(version=1, built_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                    model=MODEL, backend='bm25', files=manifest_files, total_chunks=len(chunks),
                    corpus_sha256=digest(corpus_path), artifacts={})
    # Build all payloads before replacing manifest; runtime checks reject partial updates.
    chunks_path = index_dir/'chunks.jsonl'
    chunks_path.write_text(''.join(json.dumps(c, ensure_ascii=False)+'\n' for c in chunks), encoding='utf-8')
    write_json(index_dir/'bm25.json', {'tokens': [tokens(c['section']+'\n'+c['text']) for c in chunks]})
    faiss_path = index_dir/'faiss.index'
    if vector:
        try:
            import faiss
            matrix = (encoder or (lambda texts: encode(texts, local_only=False)))([c['text'] for c in chunks])
            matrix = np.asarray(matrix, dtype='float32')
            faiss.normalize_L2(matrix)
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            faiss.write_index(index, str(faiss_path))
            manifest['backend'] = 'hybrid'
        except Exception as exc:
            # Never serialize model/network exceptions containing tokens or request URLs.
            manifest['embedding_error'] = type(exc).__name__
            faiss_path.unlink(missing_ok=True)
            log.warning('embedding unavailable (%s); BM25-only index', type(exc).__name__)
    else:
        faiss_path.unlink(missing_ok=True)
    for name in ['chunks.jsonl', 'bm25.json'] + (['faiss.index'] if manifest['backend'] == 'hybrid' else []):
        manifest['artifacts'][name] = digest(index_dir/name)
    write_json(index_dir/'manifest.json', manifest)
    return manifest


def validate(root, corpus_path, index_dir):
    index_dir = Path(index_dir)
    if not (index_dir/'manifest.json').is_file():
        raise ValueError(f'索引缺失，请重建：{REBUILD}')
    require_corpus(root, corpus_path)
    try:
        manifest = json.loads((index_dir/'manifest.json').read_text(encoding='utf-8'))
        files = load_corpus(corpus_path)
        if manifest['corpus_sha256'] != digest(corpus_path) or files != [
                {'path': f['path'], 'authority': f['authority']} for f in manifest['files']]:
            raise ValueError('索引过期，请重建')
        for entry in manifest['files']:
            path = (Path(root)/entry['path']).resolve()
            if not path.is_relative_to(Path(root).resolve()) or digest(path) != entry['sha256']:
                raise ValueError('索引过期，请重建')
        for name, expected in manifest['artifacts'].items():
            if name not in {'chunks.jsonl', 'bm25.json', 'faiss.index'} or digest(index_dir/name) != expected:
                raise ValueError('索引过期，请重建')
        return manifest
    except FileNotFoundError as exc:
        raise ValueError(f'索引缺失，请重建：{REBUILD}') from exc
    except Exception as exc:
        raise ValueError(f'索引过期，请重建：{REBUILD} ({type(exc).__name__})') from exc
