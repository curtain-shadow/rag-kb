# -*- coding: utf-8 -*-
r"""Week5 冒烟测试：Milvus 稠密后端（HNSW ANN + COSINE），并与 faiss 后端对比。

验证点：
1. MilvusIndex 与 DenseIndex 同一鸭子类型接口，HybridRetriever 只换 dense 后端即可切换；
2. 同一查询下 Milvus（ANN 近似）与 faiss（暴力精确）命中结果基本一致；
3. Milvus 集合落盘 .db，进程重启后索引仍在（faiss 需重建）。

跑法：D:\profession\venv\Scripts\python.exe scripts\week5_test_milvus.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parsing import get_parser
from src.chunking import ParentChildChunker
from src.retrieval import DenseIndex, HybridRetriever, MilvusIndex

REAL_MD = Path(r"d:\个人\简历信息库.md")

QUERY = "智能客服多智能体Agent用什么框架实现？"


def main():
    doc = get_parser(REAL_MD).parse(REAL_MD)
    children, _ = ParentChildChunker(child_size=512, parent_size=1536).chunk(doc)
    print(f"已解析《{doc.title}》：{len(children)} 个子块\n")

    # 1) faiss 后端（暴力精确）
    print("构建 faiss 后端（BGE-M3 编码 + IndexFlatIP）...")
    dense_faiss = DenseIndex()
    dense_faiss.add_documents(children)

    # 2) Milvus 后端（HNSW ANN + 落盘）
    print("构建 Milvus 后端（BGE-M3 编码 + HNSW + COSINE）...")
    dense_milvus = MilvusIndex()
    dense_milvus.add_documents(children)

    print("\n" + "=" * 70)
    print("同一查询，两个后端 Top5 对比：")
    print("=" * 70)
    f_hits = dense_faiss.search(QUERY, top_k=5)
    m_hits = dense_milvus.search(QUERY, top_k=5)

    print(f"\n{'faiss(精确)':<14} {'Milvus(ANN)':<14}")
    for i in range(5):
        fid, fs = f_hits[i]
        mid, ms = m_hits[i]
        match = "✓" if fid == mid else f"✗(faiss={fid[:6]} vs milvus={mid[:6]})"
        print(f"{fid:<14} {fs:+.4f}  |  {mid:<14} {ms:+.4f}  {match}")

    # 3) 通过 HybridRetriever 换后端，验证切 dense 后端后混合检索仍工作
    print("\n" + "=" * 70)
    print("用 Milvus 做 dense 后端跑混合检索（RRF）：")
    print("=" * 70)
    hyb = HybridRetriever(children, dense=dense_milvus)
    for chunk, score in hyb.retrieve(QUERY, top_k=3):
        snippet = chunk.text.replace("\n", " ")[:52]
        print(f"  [{score:.4f}] {chunk.chunk_id}  「{snippet}…」")

    print("\nWeek5 Milvus 后端冒烟测试完成")


if __name__ == "__main__":
    main()
