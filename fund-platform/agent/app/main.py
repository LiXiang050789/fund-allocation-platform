"""Phase 1 HTTP 问答入口。"""
import logging
import time
from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel, field_validator

from app.config import settings
from app.graph import graph
from app.rag.index import validate, REBUILD
from app.rag.retriever import CORPUS_PATH
from app.tools.registry import TOOL_SCHEMAS

logging.basicConfig(level=settings.log_level)
log = logging.getLogger(__name__)
app = FastAPI(title="投研 Agent", version="1.0")


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
