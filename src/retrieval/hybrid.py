# -*- coding: utf-8 -*-
"""混合检索：RRF（Reciprocal Rank Fusion）融合稀疏 + 稠密两个排序。

为什么用 RRF 而不是「加权求和」：
- BM25 分数无上界、量纲随语料变化；余弦相似度落在 [0,1]。两者量纲完全不同，
  直接加权必须先归一化到同一尺度，否则某一方天然占优，调参也脆弱。
- RRF 只依赖「排名」而非「分数绝对值」：score = Σ 1/(k + rank)，
  量纲无关、对离群分数不敏感。两个检索器「各投一票」，谁排得靠前谁贡献大，
  实现简单且稳定（业界如 Weaviate / Cohere 的混合检索也用它）。

检索命中子块（chunk.text），返回的 Chunk 携带 parent_text，供 Week3 喂给 LLM 生成。
"""
from __future__ import annotations

from ..chunking import Chunk
from .sparse import BM25Index
from .dense import DenseIndex


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        sparse: BM25Index | None = None,
        dense: DenseIndex | None = None,
        k: int = 60,
    ):
        self.chunks = {c.chunk_id: c for c in chunks}
        self.sparse = sparse or BM25Index()
        self.dense = dense or DenseIndex()
        self.k = k

        self.sparse.add_documents(chunks)
        self.dense.add_documents(chunks)

    def retrieve(self, query: str, top_k: int = 6, top_k_each: int = 20) -> list[tuple[Chunk, float]]:
        """返回 [(Chunk, rrf_score)]，按 RRF 分数降序。"""
        sparse_hits = self.sparse.search(query, top_k=top_k_each)
        dense_hits = self.dense.search(query, top_k=top_k_each)

        rrf: dict[str, float] = {}
        for rank, (cid, _) in enumerate(sparse_hits):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (self.k + rank + 1)
        for rank, (cid, _) in enumerate(dense_hits):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (self.k + rank + 1)

        ranked = sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k]
        return [(self.chunks[cid], score) for cid, score in ranked]
