"""One entry point for synchronous and streaming chat. Research graph is unchanged."""
import json
import logging
import time
from app import chat as general
from app.dates import history_error
from app.intent import classify, special_answer, rewrite_question, CLARIFY, NO_EVIDENCE, POLICY

log = logging.getLogger(__name__)
STAGES = {'intake': '理解', 'router': '选工具', 'tools': '查询/检索', 'compose': '生成', 'verify': '校验'}


class RedactToolArguments(logging.Filter):
    """Existing research tooling logs JSON arguments; hide them at the log boundary."""
    def filter(self, record):
        try:
            data = json.loads(record.getMessage())
            if isinstance(data, dict) and ('args' in data or 'query' in data):
                if 'args' in data:
                    data['args'] = '[redacted]'
                if 'query' in data:
                    data['query'] = '[redacted]'
                data['error'] = None if data.get('ok') else 'tool failed'
                record.msg, record.args = json.dumps(data, ensure_ascii=False), ()
        except (ValueError, TypeError):
            pass
        return True


logging.getLogger('app.tools.registry').addFilter(RedactToolArguments())
logging.getLogger('app.rag.retriever').addFilter(RedactToolArguments())


class Orchestrator:
    def __init__(self, store, graph):
        self.store, self.graph = store, graph

    @staticmethod
    def result(answer, mode, thread_id, start, **extra):
        return dict(answer=answer, mode=mode, thread_id=thread_id, verified=None,
                    citations=[], tool_calls=[], latency_ms=round((time.perf_counter()-start)*1000, 2)) | extra

    def research_result(self, state, thread_id, start):
        if not any(row.get('ok') for row in state.get('tool_results', [])):
            message = history_error(state.get('tool_results', []), state.get('question', ''))
            state = state | dict(answer=(message + "不构成投资建议。") if message else NO_EVIDENCE, verified=False, citations=[])
        elif not state.get('verified'):
            # Never disclose graph's unchecked numerical draft over HTTP/SSE.
            state = state | dict(answer='证据校验未通过，请补充具体问题或重试。不构成投资建议。', citations=[])
        return self.result(state['answer'], 'research', thread_id, start,
                           verified=bool(state.get('verified')), citations=state.get('citations', []),
                           tool_calls=state.get('tool_calls', []))

    def events(self, question, thread_id, streaming=False):
        start = time.perf_counter()
        with self.store.turn():
            history = self.store.history(thread_id)
            mode = classify(question, history)
            yield 'meta', dict(mode='chat' if mode == 'clarify' else mode, thread_id=thread_id)
            try:
                if mode == 'research':
                    rewritten = rewrite_question(question, history)
                    if streaming:
                        state = {}
                        for update in self.graph.stream({'question': rewritten, 'thread_id': thread_id}, stream_mode='updates'):
                            for node, values in update.items():
                                state.update(values or {})
                                if node in STAGES:
                                    yield 'status', dict(stage=node, label=STAGES[node])
                    else:
                        state = self.graph.invoke({'question': rewritten, 'thread_id': thread_id})
                    result = self.research_result(state, thread_id, start)
                else:
                    fixed = POLICY if mode == 'refusal' else CLARIFY if mode == 'clarify' else special_answer(question)
                    if fixed:
                        answer, final_mode = fixed, 'refusal' if mode == 'refusal' else 'chat'
                        if streaming and final_mode == 'chat':
                            for char in answer:
                                yield 'token', {'text': char}
                    elif streaming:
                        answer, final_mode = '', 'chat'
                        for event, payload in general.stream(question, history):
                            if event == 'token':
                                yield 'token', {'text': payload}
                            else:
                                answer, final_mode = payload
                    else:
                        answer, final_mode = general.answer(question, history)
                    result = self.result(answer, final_mode, thread_id, start)
                # Commit only complete turns, immediately before final; cancellation before then writes nothing.
                self.store.append_turn(thread_id, question, result)
                log.info('mode=%s route=%s latency_ms=%s token_count=%s verified=%s',
                         result['mode'], mode, result['latency_ms'], len(result['answer']), result['verified'])
                yield 'final', result
            except Exception as exc:
                log.warning('chat failed type=%s mode=%s', type(exc).__name__, mode)
                yield 'error', {'message': '服务暂时不可用，请重试'}

    def run(self, question, thread_id):
        for event, payload in self.events(question, thread_id):
            if event == 'final':
                return payload
        mode = classify(question)
        return self.result('服务暂时不可用', 'research' if mode == 'research' else 'chat', thread_id, time.perf_counter(),
                           error='service_unavailable', verified=False if mode == 'research' else None)
