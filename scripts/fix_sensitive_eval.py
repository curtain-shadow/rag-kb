# -*- coding: utf-8 -*-
"""替换评测集中触发 DeepSeek 内容审核（Content Exists Risk）的敏感条目。

q019 / q040 / q163 的 gold 父块是政治历史内容，生成与评测时都会被 DeepSeek 审核拦截。
本脚本：从语料中选出不含敏感词的「安全」父块，为这三条重新生成 query + reference_answer，
替换 eval_set.json 里对应条目（gold_parent_index 一并更新）。

跑法：D:/profession/venv/Scripts/python.exe scripts/fix_sensitive_eval.py
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
TARGET_IDS = ["q019", "q040", "q163"]

# DeepSeek 内容审核常见拦截主题的关键词；命中任一即视为「不安全」，跳过不用。
BLOCKLIST = [
    "政治局", "中央委员会", "总书记", "国家主席", "国务院", "党代会", "共产党", "革命",
    "解放军", "苏联", "共产主义", "台湾", "香港", "新疆", "西藏", "文革", "六四",
    "中美", "外交", "战争", "军队", "政府", "主席", "总理", "领导人", "镇压", "暴动",
]

SYSTEM = (
    "你是 RAG 评测集构造专家。给定一段维基百科【资料】，构造一条问答对，用于测试检索系统能否检索到这段资料、"
    "以及模型能否只依据资料回答。要求：\n"
    "1. query 是自然的中文提问，答案必须能直接从【资料】中找到；\n"
    "2. query 具体、指向明确，偏好事实型问题（“…是什么 / …包含哪些 / …的特点 / …何时”），避免“请介绍…”这种过泛提问；\n"
    "3. reference_answer 是完整准确的参考答案，直接回答 query；\n"
    "4. 只输出 JSON。\n"
    '输出：{"query":"…","reference_answer":"…"}'
)


def _is_safe(text: str) -> bool:
    return not any(w in text for w in BLOCKLIST)


def main():
    doc = get_parser(CORPUS).parse(CORPUS)
    _, parents = ParentChildChunker(child_size=512, parent_size=1536).chunk(doc)

    data = json.loads(OUT.read_text(encoding="utf-8"))
    items = data["items"]
    used_gold = {i["gold_parent_index"] for i in items}

    # 找安全父块（不含敏感词，且未被现有条目用作 gold）
    safe_idx = [i for i, p in enumerate(parents) if _is_safe(p.text[:200]) and i not in used_gold]
    print(f"安全父块 {len(safe_idx)} / {len(parents)} 个", flush=True)
    # 等间隔取 3 个，尽量分散主题，避免集中在同一领域
    picks = [safe_idx[round(k * (len(safe_idx) - 1) / 2)] for k in range(3)]

    client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    by_id = {i["id"]: i for i in items}

    for qid, pi in zip(TARGET_IDS, picks):
        ptext = parents[pi].text
        user = f"【资料】\n{ptext[:2000]}"
        ok = False
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
                if query and ref and _is_safe(query) and _is_safe(ref):
                    by_id[qid] = {"id": qid, "query": query, "reference_answer": ref, "gold_parent_index": pi}
                    print(f"替换 {qid}: gold={pi} {query}", flush=True)
                    ok = True
                    break
            except Exception as ex:  # noqa: BLE001
                print(f"{qid} gold={pi} 重试{attempt + 1}: {ex}", flush=True)
                time.sleep(2)
        if not ok:
            print(f"警告：{qid} 生成失败，保持原样", flush=True)

    items = sorted(by_id.values(), key=lambda x: x["id"])
    data["items"] = items
    data["description"] = (f"端到端评测集（中文维基语料，{len(items)} 条）：每条含 query / 参考答案 / gold 父块 index。"
                           "用于 LLM-as-judge 三指标（忠实度/相关性/引用准确率）+ 检索 Hit Rate。")
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n评测集共 {len(items)} 条")


if __name__ == "__main__":
    main()
