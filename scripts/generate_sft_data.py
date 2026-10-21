# -*- coding: utf-8 -*-
r"""合成 SFT 训练数据：从简历语料生成「带引用的 RAG 问答」对，并做引用校验。

目的：微调 Qwen2.5-1.5B-Instruct 学会「只依据给定资料 + 句末标 [i] 引用」的 RAG 问答，
与线上 rag.py 的 SYSTEM_PROMPT / 参考资料格式完全一致，训练即推理。

流程：
1. 解析 + 父子分块 → 20 个父块（每个是一段完整上下文）。
2. 每个父块生成 N 条样本：参考资料 = 1 个 gold 父块 + 3 个随机干扰父块（打乱、编号 [1..4]），
   DeepSeek 同时生成「只能由 gold 回答的问题」+「引用答案」（句末标 [i]）。
3. 双重校验，不合格丢弃：
   a. 规则：答案必须引用 gold 的编号；所有 [i] 都必须落在 [1..4]；query/answer 非空。
   b. LLM 质检：答案是否忠实于资料、引用是否准确、有无编造。
4. 追加写 data/sft_data.jsonl（每行 = messages: system/user/assistant），供 train_lora.py 直接 tokenize。

注意：训练集绝不复用 data/eval_set.json 的 30 条评测 query（防泄漏）；生成的 query 与评测集逐字相同则丢弃。

跑法：D:\profession\venv\Scripts\python.exe scripts\generate_sft_data.py [--per-parent N] [--fresh]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config  # noqa: F401
from openai import OpenAI

from src.config import settings
from src.parsing import get_parser
from src.chunking import ParentChildChunker
from src.rag import SYSTEM_PROMPT

REAL_MD = Path(r"d:\个人\简历信息库.md")
EVAL_SET = Path(__file__).resolve().parent.parent / "data" / "eval_set.json"
OUT = Path(__file__).resolve().parent.parent / "data" / "sft_data.jsonl"

_CIT_RE = re.compile(r"\[(\d+)\]")

GEN_SYSTEM = (
    "你是训练数据生成器。根据【参考资料】生成一条用户可能问的问题（中文、具体、自然、口语化），"
    "并写出这条问题基于【目标资料】的引用答案。要求："
    "1. 问题必须只有【目标资料】能回答，其他干扰资料无法回答；"
    "2. 答案只依据【目标资料】，不要引入外部知识，不要编造；"
    "3. 答案在相关句末标注引用编号，如 [3]（编号是目标资料在参考资料里的编号）；"
    "4. 问题类型尽量多样（是什么 / 为什么 / 多少 / 怎么做 / 包含哪些）；"
    "5. 若目标资料不足以构成一个可回答的问题，换一个角度。"
    "输出 JSON：{\"query\":\"...\",\"answer\":\"...\"}"
)

CHECK_SYSTEM = (
    "你是数据质检员。判断【答案】是否只依据【参考资料】回答、引用编号 [i] 是否准确（引用的资料确实支撑该句结论）、"
    "有没有编造或引用不存在/不相关的编号。若答案完全忠实且引用准确返回 valid=true，否则 false。"
    "输出 JSON：{\"valid\":true,\"reason\":\"...\"}"
)


def _parse_json(text: str):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        text = text[s : e + 1]
    return json.loads(text)


def _refs_str(refs: list[str]) -> str:
    return "\n\n".join(f"[{i}] {t}" for i, t in enumerate(refs, 1))


def _valid_citations(answer: str, gold_pos: int, n_refs: int) -> bool:
    """答案必须引用 gold 编号，且所有引用编号都在 [1..n_refs]。"""
    cites = [int(x) for x in _CIT_RE.findall(answer)]
    if not cites:
        return False
    if gold_pos not in cites:
        return False
    if any(i < 1 or i > n_refs for i in cites):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-parent", type=int, default=15, help="每个父块生成的样本数")
    ap.add_argument("--fresh", action="store_true", help="清空输出文件重来")
    ap.add_argument("--limit-parents", type=int, default=0, help="只处理前 N 个父块（0=全部，调试用）")
    args = ap.parse_args()

    if args.fresh and OUT.exists():
        OUT.unlink()

    # 评测集 query 去重（防泄漏）
    eval_queries = set()
    if EVAL_SET.exists():
        eval_queries = {it["query"] for it in json.loads(EVAL_SET.read_text(encoding="utf-8"))["items"]}

    doc = get_parser(REAL_MD).parse(REAL_MD)
    _, parents = ParentChildChunker(child_size=512, parent_size=1536).chunk(doc)
    n_parents = len(parents)
    if args.limit_parents > 0:
        parents = parents[: args.limit_parents]
    print(f"语料：{n_parents} 父块，每个生成 {args.per_parent} 条", flush=True)

    client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    rng = random.Random(20261017)

    n_ok = n_drop = 0
    t0 = time.perf_counter()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        for pi, gold in enumerate(parents):
            gold_text = gold.parent_text
            others = [p.parent_text for p in parents if p.parent_chunk_id != gold.parent_chunk_id]
            for k in range(args.per_parent):
                n_distractors = min(3, len(others))
                distractors = rng.sample(others, n_distractors) if n_distractors else []
                refs = [gold_text] + distractors
                rng.shuffle(refs)
                gold_pos = refs.index(gold_text) + 1
                n_refs = len(refs)

                gen_user = f"【参考资料】\n{_refs_str(refs)}\n\n【目标资料编号】{gold_pos}"
                try:
                    data = _parse_json(client.chat.completions.create(
                        model=settings.deepseek_model,
                        messages=[{"role": "system", "content": GEN_SYSTEM},
                                  {"role": "user", "content": gen_user}],
                        temperature=0.9,
                        max_tokens=1024,
                        response_format={"type": "json_object"},
                    ).choices[0].message.content)
                    query, answer = (data.get("query") or "").strip(), (data.get("answer") or "").strip()
                except Exception as e:  # noqa: BLE001
                    n_drop += 1
                    print(f"  [{pi}-{k}] 生成解析失败，丢弃：{e}", flush=True)
                    continue

                # 规则校验
                if not query or not answer or query in eval_queries or not _valid_citations(answer, gold_pos, n_refs):
                    n_drop += 1
                    continue

                # LLM 质检
                try:
                    chk = _parse_json(client.chat.completions.create(
                        model=settings.deepseek_model,
                        messages=[{"role": "system", "content": CHECK_SYSTEM},
                                  {"role": "user", "content":
                                   f"【参考资料】\n{_refs_str(refs)}\n\n【问题】\n{query}\n\n【答案】\n{answer}"}],
                        temperature=0.0,
                        max_tokens=256,
                        response_format={"type": "json_object"},
                    ).choices[0].message.content)
                    valid = str(chk.get("valid", "")).strip().lower() in {"true", "1", "yes"}
                except Exception:  # noqa: BLE001
                    valid = False

                if not valid:
                    n_drop += 1
                    continue

                rec = {"messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"【参考资料】\n{_refs_str(refs)}\n\n【问题】\n{query}"},
                    {"role": "assistant", "content": answer},
                ]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_ok += 1

            print(f"父块 {pi} 完成，累计有效 {n_ok} / 丢弃 {n_drop}", flush=True)

    print(f"\n完成：有效 {n_ok} 条，丢弃 {n_drop} 条，耗时 {time.perf_counter()-t0:.0f}s，输出 {OUT}", flush=True)


if __name__ == "__main__":
    main()
