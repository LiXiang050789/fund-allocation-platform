"""Phase 1 HTTP 问答入口。"""
import logging
import time
import threading
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, field_validator

from app.config import settings
from app.graph import graph
from app.rag.index import validate, REBUILD
from app.rag.retriever import CORPUS_PATH
from app.tools.registry import TOOL_SCHEMAS

logging.basicConfig(level=settings.log_level)
log = logging.getLogger(__name__)
from app.workflow import WorkflowService, WorkflowConflict, ReviewRequest, public_state

_workflow_service = None
_workflow_lock = threading.RLock()


def get_workflow():
    global _workflow_service
    with _workflow_lock:
        if _workflow_service is None:
            _workflow_service = WorkflowService(settings)
        return _workflow_service


@asynccontextmanager
async def lifespan(app):
    global _workflow_service
    try:
        yield
    finally:
        with _workflow_lock:
            if _workflow_service is not None:
                _workflow_service.close()
                _workflow_service = None


app = FastAPI(title="投研 Agent", version="1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str
    thread_id: str | None = None

    @field_validator("question")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("question 不得为空")
        return value.strip()


@app.post("/chat")
def chat(request: ChatRequest):
    started = time.perf_counter()
    thread_id = request.thread_id or str(uuid4())
    try:
        state = graph.invoke({"question": request.question, "thread_id": thread_id})
        return dict(answer=state["answer"], tool_calls=state["tool_calls"], citations=state["citations"],
                    verified=state["verified"], latency_ms=round((time.perf_counter()-started)*1000, 2), thread_id=thread_id)
    except Exception as exc:
        log.error("chat failed: %s", type(exc).__name__)
        return dict(answer="服务暂时不可用", error=type(exc).__name__, verified=False,
                    tool_calls=[], citations=[], thread_id=thread_id,
                    latency_ms=round((time.perf_counter()-started)*1000, 2))


@app.get("/healthz")
def healthz():
    backend, error, error_type = None, None, None
    try:
        manifest = validate(settings.platform_dir.parent, CORPUS_PATH, settings.index_dir)
        backend = manifest['backend'] if settings.enable_vector_rag else 'bm25'
    except Exception as exc:
        error_type = type(exc).__name__
        # Only expose known operational diagnostics; never arbitrary exception text.
        message = str(exc)
        if isinstance(exc, ValueError) and message.startswith(('语料缺失:', '索引缺失', '索引过期')):
            category = '语料缺失，请补齐 corpus.yaml 中的语料' if message.startswith('语料缺失:') else ('索引缺失' if message.startswith('索引缺失') else '索引过期')
            error = f'{category}，请重建：{REBUILD}'
        else:
            error = f'索引检查失败，请重建：{REBUILD}'
    enabled = error is None
    log.info('rag_enabled=%s rag_backend=%s rag_error=%s', enabled, backend, error_type)
    return dict(status='ok' if enabled else 'degraded', model=settings.deepseek_model,
                rag_enabled=enabled, vector_rag_enabled=settings.enable_vector_rag,
                rag_backend=backend, rag_error=error)


@app.get("/tools")
def tools():
    return [{"name": s["function"]["name"], "description": s["function"]["description"]} for s in TOOL_SCHEMAS]


@app.post('/workflow/run')
def workflow_run():
    return public_state(get_workflow().run())


def workflow_state(thread_id):
    try:
        return get_workflow().get(thread_id)
    except KeyError:
        raise HTTPException(status_code=404, detail='未知工作流') from None


@app.get('/workflow/{thread_id}')
def workflow_get(thread_id: str):
    return public_state(workflow_state(thread_id))


@app.post('/workflow/{thread_id}/approve')
def workflow_approve(thread_id: str, request: ReviewRequest):
    try:
        return public_state(get_workflow().review(thread_id, request))
    except KeyError:
        raise HTTPException(status_code=404, detail='未知工作流') from None
    except WorkflowConflict:
        raise HTTPException(status_code=409, detail='当前状态不允许该审核动作') from None


@app.get('/workflow/{thread_id}/report')
def workflow_report(thread_id: str):
    state = workflow_state(thread_id)
    if not state.get('draft'):
        raise HTTPException(status_code=409, detail='无草稿')
    text = state['draft']
    if state.get('report_path'):
        try:
            text = (get_workflow().config.agent_dir/state['report_path']).read_text(encoding='utf-8')
        except OSError:
            raise HTTPException(status_code=409, detail='报告文件不可用') from None
    return Response(content=text, media_type='text/markdown')
