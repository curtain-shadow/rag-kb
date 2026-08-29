# -*- coding: utf-8 -*-
r"""交互式问答 CLI：连上运行中的服务，输入问题回车，流式输出答案。

用法（两步）：
1. 起服务（另开一个终端）：
   D:\profession\venv\Scripts\python.exe -m uvicorn src.api:app --app-dir D:\profession\rag_kb --port 8000
2. 运行本脚本：
   D:\profession\venv\Scripts\python.exe scripts\ask_cli.py
3. 输入 exit / quit / 退出 结束。
"""
import json

import requests

BASE = "http://127.0.0.1:8000"


def ask_stream(q: str) -> None:
    r = requests.post(f"{BASE}/ask/stream", json={"query": q}, stream=True)
    print("答：", end="", flush=True)
    for line in r.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data:"):
            continue
        d = json.loads(line[5:].strip())
        if isinstance(d, str):              # delta：答案 token
            print(d, end="", flush=True)
        elif "sources" in d:                # sources：命中 + 检索耗时
            t = d["trace"]["timing_ms"]
            print(f"\n  └ [命中 {len(d['sources'])} 块 | 检索 {t['retrieval']}ms 重排 {t['rerank']}ms]")
    print("\n")


def main() -> None:
    print("深度 RAG 知识库问答系统（输入 exit 退出）")
    print("=" * 52)
    while True:
        try:
            q = input("\n你问：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "退出", "q"):
            break
        ask_stream(q)
    print("再见")


if __name__ == "__main__":
    main()
