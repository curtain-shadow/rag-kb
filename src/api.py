# -*- coding: utf-8 -*-
r"""FastAPI 服务：/ask（非流式 JSON）+ /ask/stream（SSE 流式）+ /health。

启动：D:\profession\venv\Scripts\python.exe -m uvicorn src.api:app --app-dir D:\profession\rag_kb --port 8000
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .parsing import get_parser
from .chunking import ParentChildChunker
from .rag import RAGPipeline

_rag: RAGPipeline | None = None


def build_rag(doc_path: str | None = None) -> RAGPipeline:
    """构建流水线：解析 + 分块 + 建索引（加载模型，较慢）。默认入库中文维基语料。"""
    global _rag
    if doc_path is None:
        doc_path = str(Path(__file__).resolve().parent.parent / "data" / "docs" / "wiki_zh_sample.md")
    doc = get_parser(Path(doc_path)).parse(Path(doc_path))
    children, _ = ParentChildChunker(child_size=512, parent_size=1536).chunk(doc)
    _rag = RAGPipeline(children)
    return _rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 服务启动时构建流水线，就绪前 /ask 返回 503
    build_rag()
    yield


app = FastAPI(title="深度RAG知识库问答系统", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok", "ready": _rag is not None}


@app.post("/ask")
def ask(req: AskRequest):
    if _rag is None:
        return JSONResponse(status_code=503, content={"error": "pipeline not built"})
    return _rag.answer(req.query)


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    if _rag is None:
        return JSONResponse(status_code=503, content={"error": "pipeline not built"})

    def event_gen():
        for ev in _rag.answer_stream(req.query):
            yield {
                "event": ev["type"],
                "data": json.dumps(ev["data"], ensure_ascii=False),
            }

    return EventSourceResponse(event_gen())
