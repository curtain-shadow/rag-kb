# -*- coding: utf-8 -*-
r"""Week2 冒烟测试：混合检索（BM25 稀疏 + BGE-M3 稠密 + RRF 融合）。

跑法：D:\profession\venv\Scripts\python.exe scripts\week2_test_retrieval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parsing import get_parser
from src.chunking import ParentChildChunker
from src.retrieval import HybridRetriever

REAL_MD = Path(r"d:\个人\简历信息库.md")

QUERIES = [
    "深度RAG知识库项目用了哪些技术栈？",
    "智能客服多智能体Agent用什么框架实现？",
    "蓝桥杯和英语四级的获奖情况？",
]


def main():
    doc = get_parser(REAL_MD).parse(REAL_MD)
    children, _ = ParentChildChunker(child_size=512, parent_size=1536).chunk(doc)
    print(f"已解析《{doc.title}》：{len(children)} 个子块")

    print("构建混合索引（加载 BGE-M3 + 建 BM25 倒排 + faiss）...")
    retriever = HybridRetriever(children)
    print("索引构建完成\n")

    for q in QUERIES:
        print("=" * 70)
        print("查询：", q)

        # 三个视角对比：仅稀疏 / 仅稠密 / RRF 融合
        s_top1 = retriever.sparse.search(q, top_k=1)
        d_top1 = retriever.dense.search(q, top_k=1)
        fused = retriever.retrieve(q, top_k=3)

        print(f"  仅 BM25 命中: {s_top1[0][0]}  (分数 {s_top1[0][1]:.3f})")
        print(f"  仅 稠密 命中: {d_top1[0][0]}  (相似度 {d_top1[0][1]:.3f})")
        print("  RRF 融合 Top3:")
        for chunk, score in fused:
            snippet = chunk.text.replace("\n", " ")[:52]
            print(f"    - [{score:.4f}] {chunk.chunk_id}  「{snippet}…」")
        print()

    print("Week2 混合检索 冒烟测试完成")


if __name__ == "__main__":
    main()
