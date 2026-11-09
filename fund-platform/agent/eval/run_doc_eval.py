#!/usr/bin/env python3
"""固定16题的文档召回、引用与问答评测。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from contextlib import nullcontext

import yaml

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))
from app.config import settings
from app.graph import graph
from app.rag.build_index import normalize
from app.rag.citations import citation_matches, retrieved_chunks
from run_eval import grade

EVAL_DIR = Path(__file__).resolve().parent


def evaluate(items, offline=False):
    results = []
    context = patch('app.graph.get_llm', side_effect=RuntimeError('offline evaluation')) if offline else nullcontext()
    with context:
        for item in items:
            started = time.perf_counter()
            state = graph.invoke({'question': item['question'], 'thread_id': 'doc-eval-'+item['id']})
            chunks = retrieved_chunks(state['tool_results'])
            # Recall is measured against the actual first five retrieved chunks, never an oracle re-query.
            chunks = list({c['chunk_id']: c for c in chunks}.values())[:5]
            recall = any(c['source'] == item['source'] and normalize(item['source_quote']) in normalize(c['text']) for c in chunks)
            answer = state['answer']
            matched, invalid = citation_matches(answer, retrieved_chunks(state['tool_results']))
            hits = sum(keyword.lower() in answer.lower() for keyword in item['expected_keywords'])
            result = dict(id=item['id'], question=item['question'], answer=answer, source=item['source'],
                source_quote=item['source_quote'], golden_answer=item['golden_answer'], caveat=item.get('caveat'),
                recall_at_5=recall, citation_valid=bool(matched) and not invalid,
                expected_source_cited=any(c['source'] == item['source'] for c in matched),
                keyword_hits=hits, keyword_total=len(item['expected_keywords']),
                verified=state['verified'], tool_calls=state['tool_calls'], retrieved=chunks,
                backends=sorted({r['data']['backend'] for r in state['tool_results'] if r.get('ok') and r.get('tool') == 'search_docs'}),
                llm_events=state.get('llm_events', []), fails=grade(item, answer),
                latency_s=round(time.perf_counter()-started, 2))
            results.append(result)
            print(f"{item['id']} recall={int(recall)} citation={int(result['citation_valid'])} "
                  f"expected_source={int(result['expected_source_cited'])} verified={state['verified']} "
                  f"keywords={hits}/{len(item['expected_keywords'])} {result['latency_s']}s", flush=True)
    count = len(items)
    latency = sorted(r['latency_s'] for r in results)
    summary = dict(execution='offline_fallback' if offline else 'configured_llm',
        run_at=datetime.now().astimezone().isoformat(timespec='seconds'), model=settings.deepseek_model,
        total=count, recall_at_5=sum(r['recall_at_5'] for r in results)/count,
        citation_valid_rate=sum(r['citation_valid'] for r in results)/count,
        expected_source_cited_rate=sum(r['expected_source_cited'] for r in results)/count,
        keyword_coverage=sum(r['keyword_hits'] for r in results)/sum(r['keyword_total'] for r in results),
        verified_rate=sum(r['verified'] for r in results)/count,
        latency_p50=latency[count//2], latency_p95=latency[min(count-1, int(count*.95))],
        llm_error_count=sum(e['status'] == 'error' for r in results for e in r['llm_events']),
        backend_counts={backend: sum(backend in r['backends'] for r in results) for backend in ('hybrid', 'bm25')})
    return summary, results


def write_reports(summary, results):
    payload = dict(summary=summary, results=results)
    (EVAL_DIR/'results_doc.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# 文档问答评测报告', '', f"执行方式：`{summary['execution']}`", f"时间：{summary['run_at']}", '']
    lines += [f'- {key}: {value}' for key, value in summary.items() if key not in {'execution','run_at'}]
    lines += ['', 'Recall@5 按固定全部16题统计。引用指标要求至少一个真实引用，且全部引用可映射。',
              'keyword_coverage 按所有期望关键词统计。数字与关键词明细记录在 results_doc.json 的 fails。',
              '引用只校验片段存在，不保证语义完备。background/historical 的数字不作为当前事实依据。',
              '', '| ID | Recall@5 | 引用合法 | 期望来源 | verified | 关键词 | 失败项 |',
              '|---|---|---|---|---|---|---|']
    for r in results:
        lines.append(f"| {r['id']} | {int(r['recall_at_5'])} | {int(r['citation_valid'])} | "
                     f"{int(r['expected_source_cited'])} | {r['verified']} | {r['keyword_hits']}/{r['keyword_total']} | {'; '.join(r['fails']) or '-'} |")
    (EVAL_DIR/'report_doc.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    detail = ['# 文档题逐题答案', '', f"执行方式：`{summary['execution']}`", '']
    for r in results:
        detail += [f"## {r['id']}", '', f"问题：{r['question']}", '', '回答：', r['answer'], '',
                   f"原文引用：{r['source_quote']}", '', f"来源：{r['source']}", '',
                   '原文答案（逐字摘录）：', r['golden_answer'], '', f"原文矛盾/口径：{r['caveat'] or '无额外注记'}", '']
    (EVAL_DIR/'answers_doc.md').write_text('\n'.join(line.rstrip() for line in '\n'.join(detail).splitlines())+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    items = yaml.safe_load((EVAL_DIR/'doc_golden_set.yaml').read_text(encoding='utf-8'))['questions']
    if len(items) != 16:
        raise ValueError('文档评测必须包含固定16题')
    summary, results = evaluate(items, args.offline)
    write_reports(summary, results)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary['recall_at_5'] >= .8 and summary['citation_valid_rate'] == 1 else 1


if __name__ == '__main__':
    raise SystemExit(main())
