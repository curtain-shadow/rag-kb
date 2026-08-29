# -*- coding: utf-8 -*-
"""BGE-M3 稠密检索：dense 头编码 + faiss 内积索引。

- BGE-M3 是三个输出头（dense / sparse / colbert）的多向量模型，本项目只用 dense 头，
  得到 1024 维稠密向量做语义检索。
- 向量先 L2 归一化，再用 faiss.IndexFlatIP（内积）索引：归一化后内积 = 余弦相似度。
- 编码走 GPU（快），索引/检索走 CPU（本阶段语料小，暴力全扫足够；Week5 再上 Milvus 做大规模 ANN）。
"""
from __future__ import annotations

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class DenseIndex:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda"):
        self.model = SentenceTransformer(model_name, device=device)
        self.chunk_ids: list[str] = []
        self.index: faiss.IndexFlatIP | None = None

    def add_documents(self, chunks: list) -> None:
        texts = [c.text for c in chunks]
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        vecs = np.asarray(vecs, dtype="float32")
        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vecs)
        self.chunk_ids = [c.chunk_id for c in chunks]

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        q = self.model.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")
        scores, idxs = self.index.search(q, top_k)
        return [(self.chunk_ids[int(i)], float(s)) for i, s in zip(idxs[0], scores[0])]
