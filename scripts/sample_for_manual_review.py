# -*- coding: utf-8 -*-
"""动作 6：judge 人工抽检。

从端到端评测结果中随机抽 N 条（默认 40，约 20%），导出**盲评**模板（不含 judge 分数，仅
query / 参考答案 / 模型答案 / 引用来源）。用户填好人工三指标后，用 --score 读回，计算
人工 vs DeepSeek-judge 的一致率（简单一致率 + Cohen's kappa），回应「LLM 自评」质疑。

三指标人工标注约定：
- faith_human:    "ok"（答案有依据、无编造） / "hallucination"（有编造或无依据内容）
- relevance_human: 1-5 整数（5=完全命中，1=答非所问）
- citation_human: "accurate"（引用都准确）/ "inaccurate"（有引用不准确）/ "none"（答案无引用）

用法：
  导出模板：D:/profession/venv/Scripts/python.exe scripts/sample_for_manual_review.py --export --n 40
  算一致率：D:/profession/venv/Scripts/python.exe scripts/sample_for_manual_review.py --score
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "eval_results.json"
TEMPLATE = ROOT / "data" / "manual_review.json"
REPORT = ROOT / "data" / "manual_review_report.txt"


def cohen_kappa(a: list, b: list) -> float:
    """两个类别标签列表的 Cohen's kappa（自实现，不依赖 sklearn）。"""
    n = len(a)
    if n == 0:
        return 0.0
    pa, pb = Counter(a), Counter(b)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum(pa[k] * pb.get(k, 0) for k in pa) / (n * n)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def _export(args):
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    items = data["items"]
    # 只取 run 0（第一次运行），避免同 id 重复
    run0 = [i for i in items if i.get("run", 0) == 0]
    rng = random.Random(args.seed)
    sample = rng.sample(run0, min(args.n, len(run0)))
    sample.sort(key=lambda x: x["id"])

    out = []
    for it in sample:
        sources = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(it["sources"]))
        out.append({
            "id": it["id"],
            "query": it["query"],
            "reference_answer": it["reference_answer"],
            "answer": it["answer"],
            "sources": sources,
            # 待用户填写的三列（盲评：看不到 judge 分数）
            "faith_human": "",
            "relevance_human": "",
            "citation_human": "",
        })
    TEMPLATE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已导出 {len(out)} 条盲评模板到 {TEMPLATE}")
    print("填写约定：faith_human=ok/hallucination；relevance_human=1~5；citation_human=accurate/inaccurate/none")


def _score(args):
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    by_id = {f"{i['id']}:{i.get('run', 0)}": i for i in results["items"]}

    pairs = []  # (judge_label, human_label) 逐指标
    rows = []
    for t in template:
        it = by_id.get(f"{t['id']}:0")
        if it is None:
            print(f"警告：模板里 {t['id']} 在评测结果中不存在，跳过")
            continue
        # 1) 忠实度：judge faithfulness(0-1) 二值化 >=0.8 -> ok；human ok/hallucination
        j_faith = "ok" if it["faithfulness"] >= 0.8 else "hallucination"
        h_faith = t.get("faith_human", "").strip().lower()
        # 2) 相关性：judge relevance(1-5) vs human 1-5
        j_rel = it["relevance"]
        h_rel = t.get("relevance_human", "")
        # 3) 引用：judge 是否全准确（有引用且 acc==total）；human accurate/inaccurate/none
        has_cit = it["citation_total"] > 0
        j_cit = "accurate" if (has_cit and it["citation_accurate"] == it["citation_total"]) else ("inaccurate" if has_cit else "none")
        h_cit = t.get("citation_human", "").strip().lower()

        rows.append({"id": t["id"], "j_faith": j_faith, "h_faith": h_faith,
                     "j_rel": j_rel, "h_rel": h_rel, "j_cit": j_cit, "h_cit": h_cit})
        if h_faith in ("ok", "hallucination"):
            pairs.append(("faith", j_faith, h_faith))
        if str(h_rel).isdigit():
            pairs.append(("rel", j_rel, int(h_rel)))
        if h_cit in ("accurate", "inaccurate", "none"):
            pairs.append(("cit", j_cit, h_cit))

    lines = ["=" * 60, "人工 vs DeepSeek-judge 一致率（盲评）", "=" * 60]
    filled = {t["id"] for t in template if t.get("faith_human")}
    lines.append(f"模板 {len(template)} 条，已标注 {len(filled)} 条")
    for metric in ("faith", "rel", "cit"):
        m = [p for p in pairs if p[0] == metric]
        if not m:
            lines.append(f"\n[{metric}] 无有效标注，跳过")
            continue
        j = [p[1] for p in m]
        h = [p[2] for p in m]
        agree = sum(1 for a, b in zip(j, h) if a == b) / len(m)
        # 相关性额外算 ±1 内一致（1-5 分相邻可接受）
        if metric == "rel":
            near = sum(1 for a, b in zip(j, h) if abs(a - b) <= 1) / len(m)
            lines.append(f"\n[{metric}] 完全一致率 {agree:.1%}，±1 内一致率 {near:.1%}，kappa {cohen_kappa(j, h):.3f}（n={len(m)}）")
        else:
            lines.append(f"\n[{metric}] 一致率 {agree:.1%}，kappa {cohen_kappa(j, h):.3f}（n={len(m)}）")

    report = "\n".join(lines)
    REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n报告已写 {REPORT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true", help="导出盲评模板")
    ap.add_argument("--score", action="store_true", help="读回标注，算一致率")
    ap.add_argument("--n", type=int, default=40, help="抽样条数（默认 40）")
    ap.add_argument("--seed", type=int, default=7, help="抽样种子（默认 7）")
    ap.add_argument("--results", type=str, default="", help="评测结果路径（空=默认）")
    ap.add_argument("--template", type=str, default="", help="模板路径（空=默认）")
    args = ap.parse_args()

    global RESULTS, TEMPLATE
    if args.results:
        RESULTS = Path(args.results)
    if args.template:
        TEMPLATE = Path(args.template)

    if args.score:
        _score(args)
    else:
        _export(args)


if __name__ == "__main__":
    main()
