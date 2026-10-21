# -*- coding: utf-8 -*-
"""补齐 gen_eval_set.py 中因内容审核（Content Exists Risk）失败的条目。

对缺失的 id 重算等间隔采样的 gold 父块，改用相邻父块（+1 递增）重新生成，直到成功。
跑法：D:/profession/venv/Scripts/python.exe scripts/patch_eval_set.py
"""
from __future__ import annotations

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
LIMIT = 200

SYSTEM = (
    "你是 RAG 评测集构造专家。给定一段维基百科【资料】，构造一条问答对，用于测试检索系统能否检索到这段资料、"
    "以及模型能否只依据资料回答。要求：\n"
    "1. query 是自然的中文提问，答案必须能直接从【资料】中找到；\n"
    "2. query 具体、指向明确，偏好事实型问题（“…是什么 / …包含哪些 / …的特点 / …何时”），避免“请介绍…”这种过泛提问；\n"
    "3. reference_answer 是完整准确的参考答案，直接回答 query；\n"
    "4. 只输出 JSON。\n"
    '输出：{"query":"…","reference_answer":"…"}'
)


def main():
    doc = get_parser(CORPUS).parse(CORPUS)
    _, parents = ParentChildChunker(child_size=512, parent_size=1536).chunk(doc)
    n = len(parents)

    data = json.loads(OUT.read_text(encoding="utf-8"))
    items = data["items"]
    have = {i["id"] for i in items}

    # 找出缺失 id，重算其等间隔采样 gold
    missing = [(f"q{k:03d}", round((k - 1) * (n - 1) / (LIMIT - 1))) for k in range(1, LIMIT + 1)
               if f"q{k:03d}" not in have]
    print(f"缺失 {len(missing)} 条：{[m[0] for m in missing]}", flush=True)

    client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    for qid, orig_gold in missing:
        pi = orig_gold + 1  # 从相邻父块开始，避开敏感块
        ok = False
        while pi < n and not ok:
            user = f"【资料】\n{parents[pi].text[:2000]}"
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=settings.deepseek_model,
                        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                        temperature=0.5,
                        max_tokens=512,
                        response_format={"type": "json_object"},
                    )
                    txt = resp.choices[0].message.content.strip()
                    s, e = txt.find("{"), txt.rfind("}")
                    d = json.loads(txt[s : e + 1])
                    query = (d.get("query") or "").strip()
                    ref = (d.get("reference_answer") or "").strip()
                    if query and ref:
                        items.append({"id": qid, "query": query, "reference_answer": ref, "gold_parent_index": pi})
                        print(f"补 {qid}: gold={pi} {query}", flush=True)
                        ok = True
                        break
                except Exception as ex:  # noqa: BLE001
                    print(f"{qid} gold={pi} 重试{attempt + 1}: {ex}", flush=True)
                    time.sleep(2)
            pi += 1

    items.sort(key=lambda x: x["id"])
    data["items"] = items
    data["description"] = (f"端到端评测集（中文维基语料，{len(items)} 条）：每条含 query / 参考答案 / gold 父块 index。"
                           "用于 LLM-as-judge 三指标（忠实度/相关性/引用准确率）+ 检索 Hit Rate。")
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n补齐后共 {len(items)} 条")


if __name__ == "__main__":
    main()
