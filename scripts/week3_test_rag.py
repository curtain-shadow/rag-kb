# -*- coding: utf-8 -*-
r"""Week3 冒烟测试：端到端 RAG（混合检索 + 重排 + LLM 带引用生成）。

跑法：D:\profession\venv\Scripts\python.exe scripts\week3_test_rag.py
首次运行会下载 bge-reranker-v2-m3 到 D 盘缓存。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parsing import get_parser
from src.chunking import ParentChildChunker
from src.rag import RAGPipeline

REAL_MD = Path(r"d:\个人\简历信息库.md")

QUERIES = [
    "深度RAG知识库项目用了哪些技术栈？",
    "智能客服多智能体Agent用什么框架实现？",
]


def main():
    doc = get_parser(REAL_MD).parse(REAL_MD)
    children, _ = ParentChildChunker(child_size=512, parent_size=1536).chunk(doc)
    print(f"已解析《{doc.title}》：{len(children)} 个子块")

    print("构建 RAG 流水线（BGE-M3 混合检索 + bge-reranker 重排 + DeepSeek）...\n")
    rag = RAGPipeline(children)

    for q in QUERIES:
        print("=" * 70)
        print("问题：", q)
        result = rag.answer(q)
        print("回答：", result["answer"])
        print("引用来源（重排后的父块）：")
        for i, c in enumerate(result["sources"], 1):
            snippet = c.parent_text.replace("\n", " ")[:56]
            print(f"  [{i}] {c.parent_chunk_id}  「{snippet}…」")
        print()

    print("Week3 端到端 RAG 冒烟测试完成")


if __name__ == "__main__":
    main()
