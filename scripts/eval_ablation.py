# -*- coding: utf-8 -*-
r"""量化评估 + 消融实验：Hit Rate@k（父块级）。

指标与口径：
- ground truth：20 条查询，每条标注一个「gold 父块 index」——语义上能回答该问题的那个父块。
  标注依据是文档章节结构（手动核对），不是「命中某关键词」，避免 ground truth 偏向 BM25。
- Hit Rate@k = 能回答该问题的父块落在检索 top-k 里的查询占比。检索命中子块后按 parent_chunk_id 去重映射回父块，
  与线上「子块检索 + 父块喂 LLM」口径一致。
- 消融三组：
  1. full        = BM25 + 稠密 + RRF + Cross-Encoder 重排（top-4）
  2. no-rerank   = BM25 + 稠密 + RRF（不重排）
  3. dense-only  = 仅稠密向量（无 BM25、无重排）

跑法：D:\profession\venv\Scripts\python.exe scripts\eval_ablation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config  # noqa: F401  # 触发 HF_ENDPOINT / HF_HOME 镜像与缓存环境变量

from src.parsing import get_parser
from src.chunking import ParentChildChunker
from src.retrieval import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker

REAL_MD = Path(r"d:\个人\简历信息库.md")

# (查询, gold 父块 index) —— 20 条，语义级标注
GROUND_TRUTH = [
    ("求职者是什么学校什么专业？", 0),
    ("纹承堂用了什么后端框架？", 1),
    ("游戏采集工具打包了什么类型的游戏？", 1),
    ("恶意网络流量检测用了什么机器学习模型？", 1),
    ("缺失视图聚类用了什么自监督方法？", 2),
    ("跨模态语义聚类用了什么预训练模型？", 2),
    ("智联AI赛用了什么前端技术栈？", 3),
    ("MNIST 手写识别用了什么网络结构？", 3),
    ("FPGA 分频器用什么硬件描述语言？", 3),
    ("深度 RAG 项目用什么向量数据库？", 5),  # 答案在项目I正式条目(idx3)与待补2完整spec(idx5)两处都有，待补2 是更完整主源
    ("英语四级通过了吗？", 4),
    ("蓝桥杯获了什么奖？", 4),
    ("Java 秒杀系统待补项目是什么方向？", 5),
    ("推荐系统待补用什么做 CTR 预估？", 7),
    ("工业视觉检测待补检测什么？", 8),
    ("Linux 内核模块待补做什么驱动？", 9),
    ("异步 FIFO 设计解决什么时序问题？", 10),
    ("大数据平台待补用什么做实时计算？", 11),
    ("建筑 AIGC 待补用什么生成图片？", 12),
    ("智能客服多智能体 Agent 用什么框架编排？", 13),
]

TOP_K_DENSE = 20
TOP_K_FINAL = 4


def _dedupe(chunks):
    """命中的子块按父块去重，返回每个父块一个 Chunk（携带 parent_text）。"""
    seen, out = set(), []
    for c in chunks:
        if c.parent_chunk_id not in seen:
            seen.add(c.parent_chunk_id)
            out.append(c)
    return out


def _to_parents(chunks):
    """去重后返回父块 id 有序列表（与线上 _dedupe_parents 一致）。"""
    return [c.parent_chunk_id for c in _dedupe(chunks)]


def main():
    doc = get_parser(REAL_MD).parse(REAL_MD)
    children, _ = ParentChildChunker(child_size=512, parent_size=1536).chunk(doc)
    print(f"语料：{len(children)} 子块 / {len({c.parent_chunk_id for c in children})} 父块")

    # 父块 index -> parent_chunk_id
    idx2pid = {}
    for c in children:
        idx2pid.setdefault(c.metadata["parent_index"], c.parent_chunk_id)
    gold_pids = [idx2pid[idx] for _, idx in GROUND_TRUTH]
    pid2idx = {pid: idx for idx, pid in idx2pid.items()}

    # 建索引（加载 BGE-M3 + 建 BM25 + faiss）
    print("构建混合检索索引...")
    retriever = HybridRetriever(children)
    reranker = CrossEncoderReranker()
    chunk_by_id = {c.chunk_id: c for c in children}

    # 收集三组配置下每个查询的「父块 id 有序列表」
    full_retrieved, norerank_retrieved, dense_retrieved = [], [], []
    print("跑 20 条查询（3 组消融）...")
    for q, _ in GROUND_TRUTH:
        # dense-only：仅稠密向量，映射回父块
        d_hits = retriever.dense.search(q, top_k=TOP_K_DENSE)
        dense_retrieved.append(_to_parents([chunk_by_id[cid] for cid, _ in d_hits]))

        # hybrid（BM25 + 稠密 + RRF）：未重排，父块 id 有序列表
        h_chunks = [c for c, _ in retriever.retrieve(q, top_k=12)]
        parents = _dedupe(h_chunks)
        norerank_retrieved.append([c.parent_chunk_id for c in parents])

        # full：hybrid + Cross-Encoder 重排 top-4
        scores = reranker.rerank(q, [c.parent_text for c in parents])
        ranked = sorted(zip(parents, scores), key=lambda x: -x[1])[:TOP_K_FINAL]
        full_retrieved.append([c.parent_chunk_id for c, _ in ranked])

    def hit_rate(gold, retrieved_lists, k):
        hits = sum(1 for g, ret in zip(gold, retrieved_lists) if g in ret[:k])
        return hits / len(gold)

    print("\n" + "=" * 62)
    print(f"{'配置':<16}{'Hit@1':>8}{'Hit@3':>8}{'Hit@4':>8}{'Hit@5':>8}")
    print("=" * 62)
    for name, retrieved in [
        ("full(重排后)", full_retrieved),
        ("no-rerank", norerank_retrieved),
        ("dense-only", dense_retrieved),
    ]:
        print(f"{name:<16}{hit_rate(gold_pids, retrieved, 1):>7.1%}"
              f"{hit_rate(gold_pids, retrieved, 3):>8.1%}"
              f"{hit_rate(gold_pids, retrieved, 4):>8.1%}"
              f"{hit_rate(gold_pids, retrieved, 5):>8.1%}")
    print("=" * 62)

    # 逐条打印 full 配置的命中情况，便于核对
    print("\n逐条命中（full 配置，gold 父块是否进 top-4）：")
    for (q, idx), g, ret in zip(GROUND_TRUTH, gold_pids, full_retrieved):
        mark = "HIT" if g in ret else "MISS"
        print(f"  [{mark}] {q}  -> gold=idx{idx} 实际top4={[pid2idx.get(p, p) for p in ret]}")


if __name__ == "__main__":
    main()
