# -*- coding: utf-8 -*-
"""下载中文维基子集作为公开语料。

流式加载 HuggingFace `wikipedia` 数据集的中文切片，按固定顺序过滤长度后写入
data/docs/wiki_zh_sample.md。每条维基条目写成「二级标题 + 正文」，用固定切片顺序
保证可复现（同一数据集版本/config 下 shard 顺序稳定）。

跑法：D:/profession/venv/Scripts/python.exe scripts/download_corpus.py [--limit N] [--probe]
"""
from __future__ import annotations

import argparse
import sys
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config  # noqa: F401  # 触发 HF_ENDPOINT / HF_HOME 镜像与缓存环境变量

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300, help="目标条目数（默认 300，约合数十万字符）")
    ap.add_argument("--min-len", type=int, default=200, help="过滤掉短于该字数的条目（stub/消歧义页）")
    ap.add_argument("--max-len", type=int, default=50000, help="过滤掉长于该字数的条目（仅滤极端列表）")
    ap.add_argument("--out", type=str, default="", help="输出路径（空=默认 data/docs/wiki_zh_sample.md）")
    ap.add_argument("--probe", action="store_true", help="只探测数据集结构，打印前 3 条")
    args = ap.parse_args()

    ds = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train", streaming=True)

    if args.probe:
        for ex in islice(ds, 3):
            print("keys:", sorted(ex.keys()))
            print("title:", ex.get("title"))
            print("text_len:", len(ex.get("text", "") or ""))
            print("text_head:", (ex.get("text") or "")[:120].replace("\n", " "))
            print("---")
        return

    out = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "data" / "docs" / "wiki_zh_sample.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    picked = 0
    seen = 0
    parts: list[str] = []
    for ex in ds:
        seen += 1
        text = (ex.get("text") or "").strip()
        title = (ex.get("title") or "").strip()
        if args.min_len <= len(text) <= args.max_len:
            parts.append(f"## {title}\n\n{text}")
            picked += 1
        if picked >= args.limit:
            break
        if seen >= args.limit * 30:  # 大量条目被过滤时的兜底，避免无限流式读取
            break

    out.write_text("\n\n---\n\n".join(parts), encoding="utf-8")
    total_chars = sum(len(p) for p in parts)
    print(f"已写入 {out}：{picked} 条 / 扫描 {seen} 条，总字符 {total_chars}")


if __name__ == "__main__":
    main()
