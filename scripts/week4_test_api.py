# -*- coding: utf-8 -*-
r"""Week4 API 端到端测试：对运行中的 FastAPI 服务发 /health、/ask、/ask/stream。

前提：服务已启动（见 src/api.py 头部注释）。
"""
import json
import requests

BASE = "http://127.0.0.1:8000"


def test_health():
    r = requests.get(f"{BASE}/health")
    print("=== /health ===")
    print(r.json())
    print()


def test_ask():
    print("=== /ask 非流式 ===")
    r = requests.post(f"{BASE}/ask", json={"query": "智能客服多智能体Agent用什么框架实现？"})
    d = r.json()
    print("answer:", d["answer"])
    print("timing_ms:", d["trace"]["timing_ms"])
    print("sources:", len(d["sources"]), "个父块")
    print()


def test_stream():
    print("=== /ask/stream SSE 流式 ===")
    r = requests.post(
        f"{BASE}/ask/stream",
        json={"query": "深度RAG知识库项目用了哪些技术栈？"},
        stream=True,
    )
    for line in r.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data:"):
            continue
        data = json.loads(line[5:].strip())
        if isinstance(data, str):          # delta：答案增量 token
            print(data, end="", flush=True)
        elif "sources" in data:            # sources：命中父块 + 检索 trace
            print(f"\n[sources] 命中 {len(data['sources'])} 个父块, 检索耗时 {data['trace']['timing_ms']}")
        elif "trace" in data:              # done：完整 trace
            print(f"\n[done] 总耗时 {data['trace']['timing_ms']}")
    print()


if __name__ == "__main__":
    test_health()
    test_ask()
    test_stream()
