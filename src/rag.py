# -*- coding: utf-8 -*-
"""端到端 RAG 流水线：混合检索召回 → 父块去重 → Cross-Encoder 重排 → LLM 带引用生成。

关键设计「子块检索 + 父块上下文」在这里落地：
- 检索用短子块（text），命中后去重映射回父块，把完整父块（parent_text）喂给 LLM。
- 多个命中的子块可能属于同一父块，喂给 LLM 前按 parent_chunk_id 去重，避免重复上下文。

可观测性：_retrieve() 返回 trace（各阶段候选 + 分数 + 耗时），供调试面板展示。
"""
from __future__ import annotations

import time

from .llm import LLMClient
from .retrieval import HybridRetriever
from .retrieval.reranker import CrossEncoderReranker

SYSTEM_PROMPT = (
    "你是严谨的知识库问答助手。请只依据下面【参考资料】回答，不要引入外部知识。"
    "引用某条资料时在句末标注编号，如 [1]、[2]。若资料不足以回答，请明确说明资料不足，不要编造。"
)


def _dedupe_parents(children):
    """命中的多个子块可能属于同一父块，按父块去重并保序。"""
    seen: set[str] = set()
    out = []
    for c in children:
        if c.parent_chunk_id in seen:
            continue
        seen.add(c.parent_chunk_id)
        out.append(c)
    return out


def _build_prompt(query: str, parents) -> str:
    refs = "\n\n".join(f"[{i}] {c.parent_text}" for i, c in enumerate(parents, 1))
    return f"【参考资料】\n{refs}\n\n【问题】\n{query}"


class RAGPipeline:
    def __init__(self, children, top_k: int = 12, rerank_k: int = 4):
        self.retriever = HybridRetriever(children)
        self.reranker = CrossEncoderReranker()
        self.llm = LLMClient()
        self.top_k = top_k
        self.rerank_k = rerank_k

    def _retrieve(self, query: str) -> tuple[list, dict]:
        """召回 + 重排，返回 (final_parents, trace)。"""
        t0 = time.perf_counter()
        candidates = self.retriever.retrieve(query, top_k=self.top_k)
        parents = _dedupe_parents([c for c, _ in candidates])
        t1 = time.perf_counter()

        scores = self.reranker.rerank(query, [c.parent_text for c in parents])
        ranked = sorted(zip(parents, scores), key=lambda x: -x[1])[: self.rerank_k]
        final_parents = [p for p, _ in ranked]
        t2 = time.perf_counter()

        trace = {
            "query": query,
            "retrieval": [{"id": c.chunk_id, "rrf": round(s, 4)} for c, s in candidates],
            "rerank": [{"id": c.parent_chunk_id, "score": round(s, 4)} for c, s in ranked],
            "timing_ms": {
                "retrieval": round((t1 - t0) * 1000, 1),
                "rerank": round((t2 - t1) * 1000, 1),
            },
        }
        return final_parents, trace

    def answer(self, query: str) -> dict:
        """非流式：返回完整结果 + trace。"""
        final_parents, trace = self._retrieve(query)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(query, final_parents)},
        ]
        t0 = time.perf_counter()
        answer_text = self.llm.chat(messages, temperature=0.3)
        trace["timing_ms"]["llm"] = round((time.perf_counter() - t0) * 1000, 1)
        return {"answer": answer_text, "sources": [c.parent_text for c in final_parents], "trace": trace}

    def answer_stream(self, query: str):
        """流式：先 yield sources/trace，再逐 token yield delta，最后 done（供 SSE）。"""
        final_parents, trace = self._retrieve(query)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(query, final_parents)},
        ]

        def gen():
            yield {"type": "sources", "data": {"sources": [c.parent_text for c in final_parents], "trace": trace}}
            t0 = time.perf_counter()
            for delta in self.llm.chat_stream(messages, temperature=0.3):
                yield {"type": "delta", "data": delta}
            trace["timing_ms"]["llm"] = round((time.perf_counter() - t0) * 1000, 1)
            yield {"type": "done", "data": {"trace": trace}}

        return gen()
