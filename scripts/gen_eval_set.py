# -*- coding: utf-8 -*-
"""从维基语料生成 200 条评测集（query + reference_answer + gold_parent_index）。

对每个采样的父块，用 DeepSeek 构造「能靠该父块回答」的问答对；gold 即该父块的 parent_index。
等间隔采样覆盖全文，避免只取前部导致的主题偏斜。

跑法：D:/profession/venv/Scripts/python.exe scripts/gen_eval_set.py [--limit N] [--out path]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config  # noqa: F401

from openai import OpenAI

from src.chunking import ParentChildChunker
from src.config import settings
from src.parsing import get_parser

CORPUS = Path(__file__).resolve().parent.parent / "data" / "docs" / "wiki_zh_sample.md"
OUT = Path(__file__).resolve().parent.parent / "data" / "eval_set.json"

SYSTEM = (
    "你是 RAG 评测集构造专家。给定一段维基百科【资料】，构造一条问答对，用于测试检索系统能否检索到这段资料、"
    "以及模型能否只依据资料回答。要求：\n"
    "1. query 是自然的中文提问，答案必须能直接从【资料】中找到；\n"
    "2. query 具体、指向明确，偏好事实型问题（“…是什么 / …包含哪些 / …的特点 / …何时”），避免“请介绍…”这种过泛提问；\n"
    "3. reference_answer 是完整准确的参考答案，直接回答 query；\n"
    "4. 只输出 JSON。\n"
    '输出：{"query":"…","reference_answer":"…"}'
)


def _parse_json(text: str):
    text = (text or "").strip()
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        text = text[s : e + 1]
    return json.loads(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200, help="生成条数（默认 200）")
    ap.add_argument("--out", type=str, default="", help="输出路径（空=默认 data/eval_set.json）")
    args = ap.parse_args()
    out = Path(args.out) if args.out else OUT

    doc = get_parser(CORPUS).parse(CORPUS)
    _, parents = ParentChildChunker(child_size=512, parent_size=1536).chunk(doc)
    n = len(parents)
    print(f"父块总数 {n}，采样 {args.limit} 个", flush=True)

    # 等间隔采样，覆盖全文
    if args.limit > 1:
        idxs = sorted({round(i * (n - 1) / (args.limit - 1)) for i in range(args.limit)})
    else:
        idxs = [0]

    client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    items: list[dict] = []
    for k, pi in enumerate(idxs, 1):
        ptext = parents[pi].text
        user = f"【资料】\n{ptext[:2000]}"
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=settings.deepseek_model,
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                    temperature=0.5,
                    max_tokens=512,
                    response_format={"type": "json_object"},
                )
                data = _parse_json(resp.choices[0].message.content)
                query = (data.get("query") or "").strip()
                ref = (data.get("reference_answer") or "").strip()
                if query and ref:
                    items.append({"id": f"q{k:03d}", "query": query, "reference_answer": ref, "gold_parent_index": pi})
                    print(f"[{k}/{len(idxs)}] gold={pi} {query}", flush=True)
                    break
            except Exception as e:  # noqa: BLE001
                print(f"[{k}] 重试{attempt + 1}: {e}", flush=True)
                time.sleep(2)
        else:
            print(f"[{k}] 生成失败 gold={pi}", flush=True)

    payload = {
        "corpus": str(CORPUS),
        "description": f"端到端评测集（中文维基语料，{len(items)} 条）：每条含 query / 参考答案 / gold 父块 index。"
        "用于 LLM-as-judge 三指标（忠实度/相关性/引用准确率）+ 检索 Hit Rate。",
        "items": items,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写 {out}：{len(items)} 条")


if __name__ == "__main__":
    main()
