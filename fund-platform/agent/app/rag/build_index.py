"""离线构建语料索引，并逐字检查文档评测引用。"""
import argparse
import json
from pathlib import Path

import yaml

from app.config import settings
from .index import build, missing_corpus
from .ingest import load_corpus
from .retriever import CORPUS_PATH


def normalize(text):
    return ''.join(text.split())


def check_golden(index_dir, golden_path):
    chunks = [json.loads(line) for line in (Path(index_dir)/'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
    data = yaml.safe_load(Path(golden_path).read_text(encoding='utf-8'))
    items = data['questions']
    failed = []
    for item in items:
        matched = any(c['source'] == item['source'] and item['source_quote'] in c['text'] for c in chunks)
        print(f"{item['id']} {'PASS' if matched else 'FAIL'}")
        if not matched:
            failed.append(item['id'])
    print(f'golden_quotes={len(items)-len(failed)}/{len(items)}')
    return not failed


def check_corpus(root, corpus_path):
    missing = missing_corpus(root, corpus_path)
    if missing:
        print('missing=' + ','.join(missing))
        print('请把文件放到仓库根目录，或从 corpus.yaml 移除。')
        return False
    print(f'corpus_ok={len(load_corpus(corpus_path))}')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-corpus', action='store_true')
    parser.add_argument('--check-golden', action='store_true')
    parser.add_argument('--bm25-only', action='store_true')
    args = parser.parse_args()
    if args.check_corpus:
        return 0 if check_corpus(settings.platform_dir.parent, CORPUS_PATH) else 1
    if args.check_golden:
        return 0 if check_golden(settings.index_dir, settings.agent_dir/'eval/doc_golden_set.yaml') else 1
    manifest = build(settings.platform_dir.parent, CORPUS_PATH, settings.index_dir,
                     vector=settings.enable_vector_rag and not args.bm25_only)
    print(f"documents={len(manifest['files'])} chunks={manifest['total_chunks']} backend={manifest['backend']}")
    if manifest.get('embedding_error'):
        print(f"embedding_error={manifest['embedding_error']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
