# -*- coding: utf-8 -*-
"""Cross-Encoder 重排（bge-reranker-v2-m3）。

为什么 Cross-Encoder 不能对全库用、只能对少量候选重排：
- 双塔（Bi-Encoder，如 BGE-M3）把 query 和 doc 各自独立编码，doc 可离线预编码建索引，
  查询时只需编码 query 做一次相似度计算，对全库也很快。
- Cross-Encoder 把 query 和 doc 拼成「一对」喂进同一个 Transformer，逐 token 交互建模，
  精度明显更高，但必须对「每一对 query-doc」完整前向一次，无法离线预编码。
  对全库几万几十万文档逐对打分不可接受。
- 因此经典两段式：先轻量召回（混合检索）筛出 top-k，再 Cross-Encoder 对这少量候选精排，
  兼顾精度与耗时。
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda"):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """对 (query, text) 逐对打分，返回与 texts 同序的分数（越大越相关）。"""
        pairs = [[query, t] for t in texts]
        return [float(s) for s in self.model.predict(pairs)]
