# -*- coding: utf-8 -*-
"""语料质量自检：结构清洁度 / 长度分布 / 主题偏斜 / 分块统计 / 抽样展示。

跑法：D:/profession/venv/Scripts/python.exe scripts/inspect_corpus.py
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config  # noqa: F401

from src.chunking import ParentChildChunker
from src.parsing import get_parser

CORPUS = Path(__file__).resolve().parent.parent / "data" / "docs" / "wiki_zh_sample.md"


def _pct(lens: list[int], p: float) -> int:
    i = int(len(lens) * p)
    return lens[min(i, len(lens) - 1)]


def main():
    raw = CORPUS.read_text(encoding="utf-8")
    doc = get_parser(CORPUS).parse(CORPUS)
    entries = [s.strip() for s in raw.split("\n\n---\n\n") if s.strip()]

    # ① 长度分布
    lens = sorted(len(s) for s in entries)
    print(f"条目数 {len(entries)}，总字符 {sum(lens)}")
    print(f"长度分布 min={lens[0]} p25={_pct(lens, .25)} median={_pct(lens, .5)} "
          f"p75={_pct(lens, .75)} max={lens[-1]}")
    print(f"stub(<200字)={sum(1 for x in lens if x < 200)}  超长(>20000字)={sum(1 for x in lens if x > 20000)}")

    # ② 标题 + 主题偏斜
    titles = []
    for s in entries:
        m = re.match(r"## (.+)", s)
        titles.append(m.group(1).strip() if m else "?")
    print(f"\n前 40 个标题：{titles[:40]}")
    first_chars = Counter(t[0] for t in titles)
    print(f"标题首字符分布 top12：{first_chars.most_common(12)}")

    # ③ wiki 标记残留
    marks = {
        "[[链接]]": len(re.findall(r"\[\[", raw)),
        "{{模板}}": len(re.findall(r"\{\{", raw)),
        "<ref>": len(re.findall(r"<ref", raw)),
        "==章节==": len(re.findall(r"^==[^=]+==$", raw, re.M)),
        "[n]引用": len(re.findall(r"\[\d+\]", raw)),
    }
    print(f"\nwiki 标记残留：{marks}")

    # ④ 分块统计
    children, parents = ParentChildChunker(child_size=512, parent_size=1536).chunk(doc)
    plens = [len(p.text) for p in parents]
    print(f"\n分块：{len(children)} 子块 / {len(parents)} 父块，父块平均 {sum(plens) // max(len(plens), 1)} 字")

    # ⑤ 抽样展示（人工判断可问答性）
    print(f"\n=== 抽样第 1 条前 400 字 ===\n{entries[0][:400]}")
    mid = entries[len(entries) // 2]
    print(f"\n=== 抽样第 {len(entries)//2} 条前 300 字 ===\n{mid[:300]}")


if __name__ == "__main__":
    main()
