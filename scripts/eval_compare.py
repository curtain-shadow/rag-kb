# -*- coding: utf-8 -*-
r"""三路生成后端对比评测：DeepSeek vs 基座 Qwen2.5-1.5B vs LoRA 微调 Qwen2.5-1.5B。

复用 eval_rag.py 的端到端 LLM-as-judge 指标，裁判固定用 DeepSeek（保证口径一致），
对同一 30 条评测集、同一套检索/重排结果，只换生成后端，最后并排对比。

跑法：D:\profession\venv\Scripts\python.exe scripts\eval_compare.py [--limit N]
产出：data/eval_results_{deepseek,base,lora}.json + data/eval_report_{...}.txt + data/eval_compare.txt
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows 控制台默认 GBK，打印 ▶ 等字符会 UnicodeEncodeError；统一用 UTF-8 输出。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
EVAL_PY = ROOT / "scripts" / "eval_rag.py"
DATA = ROOT / "data"

BASE_MODEL = r"D:\profession\models\Qwen2.5-1.5B-Instruct"
LORA_ADAPTER = DATA / "lora_qwen_rag"

CONFIGS = [
    ("deepseek", "DeepSeek(云端)", {}),
    ("base", "Qwen-1.5B 基座", {"--backend": "local", "--model": BASE_MODEL}),
    ("lora", "Qwen-1.5B + LoRA", {"--backend": "local", "--model": BASE_MODEL, "--adapter": str(LORA_ADAPTER)}),
]


def run_one(name: str, extra: dict, limit: int) -> Path:
    out = DATA / f"eval_results_{name}.json"
    report = DATA / f"eval_report_{name}.txt"
    flat = [a for kv in extra.items() for a in kv]  # 展平 dict 为 --k v --k v 参数序列
    cmd = [sys.executable, str(EVAL_PY), "--fresh", "--out", str(out), "--report", str(report), *flat]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"\n{'=' * 66}\n▶ 生成后端：{name}\n{'=' * 66}", flush=True)
    subprocess.run(cmd, check=True, env=env)
    return out


def aggregate(results_path: Path) -> dict:
    items = json.loads(results_path.read_text(encoding="utf-8"))["items"]
    n = len(items)
    faith = sum(r["faithfulness"] for r in items) / n
    rel = sum(r["relevance"] for r in items) / n
    cit_acc = sum(r["citation_accurate"] for r in items)
    cit_tot = sum(r["citation_total"] for r in items)
    cit_prec = cit_acc / cit_tot if cit_tot else 0.0
    cit_cov = sum(1 for r in items if r["has_citation"]) / n
    # JSON 序列化把 hit 的 int 键变成字符串，读回时用 str(k)
    hit = {k: sum(1 for r in items if r["hit"][str(k)]) / n for k in (1, 3, 4)}
    return {"n": n, "faith": faith, "rel": rel, "cit_prec": cit_prec, "cit_cov": cit_cov, "hit": hit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    args = ap.parse_args()

    labels = []
    metrics = {}
    for name, label, extra in CONFIGS:
        out = run_one(name, extra, args.limit)
        labels.append(label)
        metrics[label] = aggregate(out)

    rows = [
        ("忠实度 Faithfulness", lambda m: f"{m['faith']:.1%}"),
        ("相关性 Relevance /5", lambda m: f"{m['rel']:.2f}"),
        ("引用准确率 Cit.Prec", lambda m: f"{m['cit_prec']:.1%}"),
        ("引用覆盖率 Cit.Cov", lambda m: f"{m['cit_cov']:.1%}"),
        ("检索 Hit@1", lambda m: f"{m['hit'][1]:.1%}"),
        ("检索 Hit@3", lambda m: f"{m['hit'][3]:.1%}"),
        ("检索 Hit@4", lambda m: f"{m['hit'][4]:.1%}"),
    ]

    col_w = max(len(r[0]) for r in rows) + 2
    lines = ["=" * 80, "三路生成后端对比（n=%d，裁判=DeepSeek）" % metrics[labels[0]]["n"], "=" * 80]
    lines.append("".ljust(col_w) + "".join(lbl.ljust(22) for lbl in labels))
    for name, fmt in rows:
        lines.append(name.ljust(col_w) + "".join(fmt(metrics[lbl]).ljust(22) for lbl in labels))
    lines.append("=" * 80)
    lines.append("说明：检索/重排完全相同，仅生成后端不同；引用与忠实度由 DeepSeek 裁判统一打分。")

    table = "\n".join(lines)
    compare_path = DATA / "eval_compare.txt"
    compare_path.write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"\n对比结果已写 {compare_path}")


if __name__ == "__main__":
    main()
