# -*- coding: utf-8 -*-
r"""Week4 冒烟测试：流式生成 + 可观测 trace。

跑法：D:\profession\venv\Scripts\python.exe scripts\week4_test_stream.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parsing import get_parser
from src.chunking import ParentChildChunker
from src.rag import RAGPipeline

REAL_MD = Path(r"d:\个人\简历信息库.md")


def main():
    doc = get_parser(REAL_MD).parse(REAL_MD)
    children, _ = ParentChildChunker(child_size=512, parent_size=1536).chunk(doc)
    print(f"已解析《{doc.title}》：{len(children)} 个子块")
    print("构建流水线...\n")
    rag = RAGPipeline(children)

    q = "智能客服多智能体Agent用什么框架实现？"
    print("=" * 70)
    print("流式问题：", q)

    for ev in rag.answer_stream(q):
        if ev["type"] == "sources":
            d = ev["data"]
            print(f"\n[命中 {len(d['sources'])} 个父块] trace.timing_ms = {d['trace']['timing_ms']}")
        elif ev["type"] == "delta":
            print(ev["data"], end="", flush=True)
        elif ev["type"] == "done":
            print(f"\n\n[完成] 总耗时 trace = {ev['data']['trace']['timing_ms']}")

    print("\n" + "=" * 70)
    print("非流式 trace 对比：")
    r = rag.answer(q)
    print("回答：", r["answer"][:80], "...")
    print("trace.retrieval(前3):", r["trace"]["retrieval"][:3])
    print("trace.rerank(前3):", r["trace"]["rerank"][:3])
    print("trace.timing_ms:", r["trace"]["timing_ms"])

    print("\nWeek4 流式 + 可观测 冒烟测试完成")


if __name__ == "__main__":
    main()
