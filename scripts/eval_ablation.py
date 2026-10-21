# -*- coding: utf-8 -*-
r"""检索消融实验：Hit Rate@k（父块级），复用 data/eval_set.json（200 条）+ 维基语料。

指标与口径：
- ground truth：eval_set.json 的 200 条，每条标注一个 gold_parent_index——语义上能回答该问题的父块，
  不是「命中某关键词」，避免 ground truth 偏向 BM25。
- Hit Rate@k = 能回答该问题的父块落在检索 top-k 里的查询占比。命中子块后按 parent_chunk_id 去重映射回父块，
  与线上「子块检索 + 父块喂 LLM」口径一致。
- 消融三组：
  1. full        = BM25 + 稠密 + RRF + Cross-Encoder 重排（top-4）
  2. no-rerank   = BM25 + 稠密 + RRF（不重排）
  3. dense-only  = 仅稠密向量（无 BM25、无重排）

跑法：D:/profession/venv/Scripts/python.exe scripts/eval_ablation.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config  # noqa: F401  # 触发 HF_ENDPOINT / HF_HOME 镜像与缓存环境变量

from src.parsing import get_parser
from src.chunking import ParentChildChunker
from src.retrieval import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker

CORPUS_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "docs" / "wiki_zh_sample.md"
EVAL_SET = Path(__file__).resolve().parent.parent / "data" / "eval_set.json"

TOP_K_DENSE = 20
TOP_K_FINAL = 4


def _dedupe_parents(chunks):
    """命中的子块按父块去重，返回每个父块一个 Chunk（携带 parent_text）。"""
    seen, out = set(), []
    for c in chunks:
        if c.parent_chunk_id not in seen:
            seen.add(c.parent_chunk_id)
            out.append(c)
    return out


def main():
    eval_data = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    items = eval_data["items"]
    corpus = Path(eval_data.get("corpus") or CORPUS_DEFAULT)

    doc = get_parser(corpus).parse(corpus)
    children, _ = ParentChildChunker(child_size=512, parent_size=1536).chunk(doc)
    idx2pid = {}
    for c in children:
        idx2pid.setdefault(c.metadata["parent_index"], c.parent_chunk_id)
    print(f"语料：{len(children)} 子块 / {len(idx2pid)} 父块，评测 {len(items)} 条", flush=True)

    # 建索引（加载 BGE-M3 + 建 BM25 + faiss + reranker）
    print("构建混合检索索引（BGE-M3 + BM25 + faiss + reranker）...", flush=True)
    retriever = HybridRetriever(children)
    reranker = CrossEncoderReranker()
    chunk_by_id = {c.chunk_id: c for c in children}

    # 收集三组配置下每个查询的「父块 id 有序列表」
    full_retrieved, norerank_retrieved, dense_retrieved = [], [], []
    gold_pids = []
    print(f"跑 {len(items)} 条查询（3 组消融）...", flush=True)
    for item in items:
        q = item["query"]
        gold_pids.append(idx2pid.get(item["gold_parent_index"], ""))

        # dense-only：仅稠密向量，映射回父块
        d_hits = retriever.dense.search(q, top_k=TOP_K_DENSE)
        dense_retrieved.append([chunk_by_id[cid].parent_chunk_id for cid, _ in d_hits])

        # hybrid（BM25 + 稠密 + RRF）：未重排，父块 id 有序列表
        h_chunks = [c for c, _ in retriever.retrieve(q, top_k=12)]
        parents = _dedupe_parents(h_chunks)
        norerank_retrieved.append([c.parent_chunk_id for c in parents])

        # full：hybrid + Cross-Encoder 重排 top-4
        scores = reranker.rerank(q, [c.parent_text for c in parents])
        ranked = sorted(zip(parents, scores), key=lambda x: -x[1])[:TOP_K_FINAL]
        full_retrieved.append([c.parent_chunk_id for c, _ in ranked])

    def hit_rate(gold, retrieved_lists, k):
        hits = sum(1 for g, ret in zip(gold, retrieved_lists) if g in ret[:k])
        return hits / len(gold)

    print("\n" + "=" * 62)
    print(f"{'配置':<18}{'Hit@1':>8}{'Hit@3':>8}{'Hit@4':>8}")
    print("=" * 62)
    for name, retrieved in [
        ("full(重排后)", full_retrieved),
        ("no-rerank", norerank_retrieved),
        ("dense-only", dense_retrieved),
    ]:
        print(f"{name:<18}{hit_rate(gold_pids, retrieved, 1):>7.1%}"
              f"{hit_rate(gold_pids, retrieved, 3):>8.1%}"
              f"{hit_rate(gold_pids, retrieved, 4):>8.1%}")
    print("=" * 62)


if __name__ == "__main__":
    main()
