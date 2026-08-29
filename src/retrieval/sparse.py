# -*- coding: utf-8 -*-
"""BM25 稀疏检索：jieba 分词 + 内存倒排索引（自实现，非调库）。

为什么要「向量检索 + BM25」一起用：
- 向量检索擅长语义相似（换个说法、同义词也能命中），但对「精确关键词」不够可靠，
  语义泛化可能把明确包含某术语的文档排在后面。
- BM25 基于词频统计（TF-IDF 家族），对精确词命中极其可靠。比如简历里明确写了
  "Milvus"、"LangGraph"，查询也含这些词，BM25 一定能稳命中。
- 二者互补：BM25 兜底精确匹配，向量补足语义泛化。
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict

import jieba

jieba.setLogLevel(logging.WARNING)  # 关掉 jieba 首次加载字典的日志


class BM25Index:
    """自实现 BM25 + 倒排索引。

    - 倒排索引 inverted: term -> {doc_idx: tf}，查询时只遍历命中词的 posting list。
    - BM25 公式：score(q,d) = Σ IDF(t) * tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl))
    - IDF = ln(1 + (N - df + 0.5) / (df + 0.5))，df 为出现该词的文档数。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_len: list[int] = []
        self.avgdl = 0.0
        self.inverted: dict[str, dict[int, int]] = defaultdict(dict)
        self.idf: dict[str, float] = {}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        # jieba 分词，中英文混排都保留；过滤空白 token
        return [t for t in jieba.lcut(text) if t.strip()]

    def add_documents(self, chunks: list) -> None:
        """chunks: 具有 .chunk_id 与 .text 属性的对象（如 chunking.Chunk）。"""
        for chunk in chunks:
            tokens = self.tokenize(chunk.text)
            idx = len(self.doc_ids)
            self.doc_ids.append(chunk.chunk_id)
            self.doc_tokens.append(tokens)
            self.doc_len.append(len(tokens))

            tf: dict[str, int] = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            for t, c in tf.items():
                self.inverted[t][idx] = c

        n = len(self.doc_ids)
        self.avgdl = sum(self.doc_len) / max(1, n)
        for term, postings in self.inverted.items():
            df = len(postings)
            self.idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """返回 [(chunk_id, bm25_score)]，按分数降序。"""
        scores: dict[int, float] = defaultdict(float)
        for t in self.tokenize(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for idx, tf in self.inverted[t].items():
                dl = self.doc_len[idx]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[idx] += idf * tf * (self.k1 + 1) / denom

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return [(self.doc_ids[i], float(s)) for i, s in ranked[:top_k]]
