# -*- coding: utf-8 -*-
"""公开基准对齐：C-MTEB 中文检索 + BEIR 英文对照。

用项目同一款 BGE-M3 的 dense 头（与 src/retrieval/dense.py 一致），跑 mteb 标准检索任务：
- C-MTEB（中文，主力）：T2Retrieval / MMarcoRetrieval / CmedQAv2Retrieval / DuRetrieval
- BEIR（英文，对照）：NFCorpus / SciFact —— BGE-M3 为中文模型，英文分数仅作参考、不参与结论

输出 NDCG@10 / Recall@100 等标准指标到 data/mteb_results/，并写 summary.json。
跑法：D:/profession/venv/Scripts/python.exe scripts/eval_mteb.py [--lang cn|en|all] [--tasks T1 T2 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config  # noqa: F401  # HF_ENDPOINT 镜像 + D 盘缓存

from mteb import evaluate, get_tasks
from sentence_transformers import SentenceTransformer

OUT = Path(__file__).resolve().parent.parent / "data" / "mteb_results"

CN_TASKS = ["T2Retrieval", "MMarcoRetrievalMultilingual", "CmedqaRetrieval", "DuRetrieval"]
EN_TASKS = ["NFCorpus", "SciFact"]


def _extract_scores(result) -> dict[str, float]:
    """从单个任务结果里抽出 split=test 的指标，兼容 mteb 不同版本的返回结构。"""
    out: dict[str, float] = {}
    scores = getattr(result, "scores", None)
    if isinstance(scores, dict):
        # 优先取 test；没有 test 就取第一个 split
        test = scores.get("test") or (next(iter(scores.values())) if scores else None)
    else:
        test = scores
    for s in test or []:
        if isinstance(s, dict):  # mteb 2.x：test 是 [{metric: value}, ...]
            for k, v in s.items():
                if isinstance(v, (int, float)):
                    out[k] = float(v)
        else:  # 旧版：Score 对象（.short_name / .value）
            name = getattr(s, "short_name", None) or getattr(s, "name", None) or str(s)
            val = getattr(s, "value", None)
            if name and isinstance(val, (int, float)):
                out[name] = float(val)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--tasks", nargs="*", default=[], help="覆盖默认任务列表")
    ap.add_argument("--lang", choices=["cn", "en", "all"], default="all")
    ap.add_argument("--output-folder", default="", help="mteb 结果目录（空=默认 data/mteb_results）")
    args = ap.parse_args()

    model = SentenceTransformer(args.model)
    if args.tasks:
        tasks = args.tasks
    elif args.lang == "cn":
        tasks = CN_TASKS
    elif args.lang == "en":
        tasks = EN_TASKS
    else:
        tasks = CN_TASKS + EN_TASKS

    out_folder = Path(args.output_folder) if args.output_folder else OUT
    print(f"模型 {args.model}，任务 {tasks}，结果目录 {out_folder}", flush=True)

    t0 = time.perf_counter()
    task_objs = get_tasks(tasks=tasks)
    results = evaluate(model, tasks=task_objs, show_progress_bar=True)

    summary: dict[str, dict] = {}
    task_results = results.task_results
    if isinstance(task_results, dict):
        task_results = list(task_results.values())
    for r in task_results:
        name = getattr(r, "task_name", None) or "unknown"
        row = _extract_scores(r)
        summary[name] = row
        print(f"\n=== {name} ===", flush=True)
        for k in sorted(row):
            print(f"  {k}: {row[k]:.4f}", flush=True)

    out_folder.mkdir(parents=True, exist_ok=True)
    (out_folder / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n耗时 {time.perf_counter() - t0:.0f}s；汇总已写 {out_folder / 'summary.json'}")


if __name__ == "__main__":
    main()
