"""Evaluate the chat contract; only chat artifacts are written. Default uses configured LLM."""
import argparse
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import yaml
from pydantic import ValidationError

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))
from app.main import ChatRequest
from app.graph import graph
from app.conversation import ConversationStore
from app.orchestrator import Orchestrator


def run(execution="configured_llm"):
    directory = Path(__file__).resolve().parent
    items = yaml.safe_load((directory/'chat_golden_set.yaml').read_text())['questions']
    results = []
    with tempfile.TemporaryDirectory(prefix='phase6-chat-eval-') as tmp:
        store = ConversationStore(Path(tmp)/'chat.sqlite')
        orchestrator = Orchestrator(store, graph)
        try:
            for item in items:
                start = time.perf_counter()
                failures, result, first_token = [], {}, None
                try:
                    if item.get('invalid'):
                        for question in (item['question'], 'x'*4001):
                            try:
                                ChatRequest(question=question)
                                failures.append('invalid input accepted')
                            except ValidationError:
                                pass
                    else:
                        question = ChatRequest(question=item['question']).question
                        thread = 'chat-eval-' + item['id']
                        if item.get('followup'):
                            prior = orchestrator.run(question, thread)
                            if prior['mode'] != 'research':
                                failures.append('initial research route')
                            question = item['followup']
                        for event, payload in orchestrator.events(question, thread, True):
                            if event == 'token' and first_token is None:
                                first_token = round(time.perf_counter()-start, 3)
                            if event == 'error':
                                failures.append(payload['message'])
                            if event == 'final':
                                result = payload
                        if result.get('mode') != item['mode']:
                            failures.append('wrong mode: '+str(result.get('mode')))
                        if item.get('verified') and result.get('verified') is not True:
                            failures.append('research not verified')
                        if item['mode'] != 'research':
                            if result.get('verified') is not None or result.get('citations') or result.get('tool_calls'):
                                failures.append('non-research metadata')
                        for word in item.get('keywords', []):
                            if word not in result.get('answer', ''):
                                failures.append('missing keyword: '+word)
                        if item.get('date') and datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d') not in result.get('answer',''):
                            failures.append('date mismatch')
                except Exception as exc:
                    failures.append(type(exc).__name__)
                record = dict(id=item['id'], question=item['question'], result=result,
                              passed=not failures, failures=failures,
                              latency_s=round(time.perf_counter()-start,3), first_token_s=first_token)
                results.append(record)
                print(item['id'], 'PASS' if not failures else 'FAIL', failures, flush=True)
        finally:
            store.close()
    summary = dict(execution=execution, total=len(results), passed=sum(r['passed'] for r in results),
                   run_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    (directory/'results_chat.json').write_text(json.dumps(dict(summary=summary,results=results),ensure_ascii=False,indent=2)+'\n')
    lines=['# Phase 6 通用能力评测','',f"评测方式：{execution}；结果：{summary['passed']}/{summary['total']}。",'',
           '| ID | 结果 | 首 token 秒 | 总耗时秒 | 失败项 |','| --- | --- | --- | --- | --- |']
    for r in results:
        lines.append(f"| {r['id']} | {'PASS' if r['passed'] else 'FAIL'} | {r['first_token_s']} | {r['latency_s']} | {'; '.join(r['failures']) or '-'} |")
    (directory/'report_chat.md').write_text('\n'.join(lines)+'\n')
    answers=['# Phase 6 作答明细','']
    for r in results:
        answers += [f"## {r['id']}", '', r['question'] or '空输入与超长输入', '', r['result'].get('answer','输入校验'), '']
    (directory/'answers_chat.md').write_text('\n'.join(answers).rstrip()+'\n')
    print(json.dumps(summary,ensure_ascii=False))
    return 0 if summary['passed']>=10 else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true', help='假LLM契约检查，不代表真实模型质量')
    args = parser.parse_args()
    if args.offline:
        from types import SimpleNamespace
        from unittest.mock import patch
        class FakeLLM:
            def stream(self, messages):
                text = "夏普比率衡量单位风险的超额收益。通胀是一般物价持续上涨。ETF 是交易所交易基金。"
                for char in text:
                    yield SimpleNamespace(content=char)
        with patch('app.chat.get_llm', return_value=FakeLLM()), patch('app.graph.get_llm', side_effect=RuntimeError('offline evaluation')):
            raise SystemExit(run('offline_fake_llm'))
    raise SystemExit(run())
