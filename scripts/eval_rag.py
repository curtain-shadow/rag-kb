# -*- coding: utf-8 -*-
r"""端到端评测：LLM-as-judge（DeepSeek 当裁判）+ 检索 Hit Rate。

三个端到端指标（自实现 judge，不依赖 ragas）：
1. 忠实度 Faithfulness —— 把答案拆成原子陈述，逐条判断是否能在【参考资料】找到依据（不编造）。
2. 答案相关性 Answer Relevance —— 对照参考答案，1-5 分判断答案是否直接、完整回答。
3. 引用准确率 Citation Accuracy —— 逐条判断 [i] 是否准确指向支撑该句结论的资料。

外加检索侧 Hit Rate@k：gold 父块是否落在 Cross-Encoder 重排后 top-k（复用 eval_ablation 口径，
但 gold 标注已按当前 20 父块切分重新核对，纠正了文档增长导致的旧标注偏移）。

评测集：data/eval_set.json（30 条，每条含 query / reference_answer / gold_parent_index）。

跑法：D:\profession\venv\Scripts\python.exe scripts\eval_rag.py [--limit N] [--fresh]
结果：增量写入 data/eval_results.json（可断点续跑），汇总打印到 stdout 并写 data/eval_report.txt。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config  # noqa: F401  # 触发 HF_ENDPOINT / HF_HOME 镜像与缓存环境变量
from openai import OpenAI

from src.config import settings
from src.parsing import get_parser
from src.chunking import ParentChildChunker
from src.rag import RAGPipeline

REAL_MD = Path(r"d:\个人\简历信息库.md")
EVAL_SET = Path(__file__).resolve().parent.parent / "data" / "eval_set.json"
RESULTS = Path(__file__).resolve().parent.parent / "data" / "eval_results.json"
REPORT = Path(__file__).resolve().parent.parent / "data" / "eval_report.txt"

_CIT_RE = re.compile(r"\[\d+\]")

FAITH_SYSTEM = (
    "你是严格的评测裁判。把【模型答案】拆成若干原子陈述，逐条判断该陈述是否能在【参考资料】中找到依据。"
    "只依据【参考资料】判断，不要引入你的外部知识；参考资料编号 [i] 只是提示，判断依据是陈述内容是否在资料中出现，"
    "而不是引用标号对不对。若答案整体声明「资料不足」且资料确实不足，则算一条 grounded=true 的陈述。"
    "输出 JSON：{\"statements\":[{\"statement\":\"...\",\"grounded\":true,\"reason\":\"...\"}]}"
)

REL_SYSTEM = (
    "你是严格的评测裁判。判断【模型答案】是否直接、完整地回答了【问题】，对照【参考答案】判断。"
    "评分 1-5：5=完全命中要点无关键遗漏无冗余；4=命中主要要点有小遗漏或小冗余；3=部分命中有较明显遗漏或偏离；"
    "2=明显偏离只沾边；1=答非所问。输出 JSON：{\"relevance\":3,\"reason\":\"...\"}"
)

CIT_SYSTEM = (
    "你是严格的评测裁判。检查【模型答案】中每个引用标注 [i] 是否准确：编号 i 对应的【参考资料】第 i 条是否确实"
    "支撑该引用所在句子的结论。引用「对」的标准是该资料内容确实支持该句结论，而不是张冠李戴；引用不存在编号则 accurate=false。"
    "若答案没有任何 [i] 引用，返回 {\"citations\":[]}。输出 JSON：{\"citations\":[{\"index\":1,\"accurate\":true,\"reason\":\"...\"}]}"
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


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"true", "1", "yes", "是", "真"}


def _to_int(v, default=0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _judge_json(client, system: str, user: str, max_tokens: int = 2000):
    resp = client.chat.completions.create(
        model=settings.deepseek_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return _parse_json(resp.choices[0].message.content)


def _refs_str(sources: list[str]) -> str:
    return "\n\n".join(f"[{i}] {t}" for i, t in enumerate(sources, 1))


def _score_faithfulness(client, query: str, answer: str, refs: str) -> tuple[float, list]:
    user = f"【参考资料】\n{refs}\n\n【问题】\n{query}\n\n【模型答案】\n{answer}"
    data = _judge_json(client, FAITH_SYSTEM, user)
    stmts = data.get("statements", []) or []
    if not stmts:
        return 0.0, []
    grounded = sum(1 for s in stmts if _to_bool(s.get("grounded")))
    return grounded / len(stmts), stmts


def _score_relevance(client, query: str, answer: str, reference: str) -> tuple[float, str]:
    user = f"【问题】\n{query}\n\n【参考答案】\n{reference}\n\n【模型答案】\n{answer}"
    data = _judge_json(client, REL_SYSTEM, user)
    rel = _to_int(data.get("relevance"), 3)
    rel = max(1, min(5, rel))
    return rel, data.get("reason", "")


def _score_citation(client, query: str, answer: str, refs: str) -> tuple[list, int, int]:
    user = f"【参考资料】\n{refs}\n\n【问题】\n{query}\n\n【模型答案】\n{answer}"
    data = _judge_json(client, CIT_SYSTEM, user)
    cites = data.get("citations", []) or []
    n_total = len(cites)
    n_acc = sum(1 for c in cites if _to_bool(c.get("accurate")))
    return cites, n_acc, n_total


def _load_done(results_path: Path) -> dict[str, dict]:
    if not results_path.exists():
        return {}
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
        out = {}
        for it in data.get("items", []):
            # JSON 把 dict 键序列化成字符串，命中标记键归一化回 int，避免聚合时 KeyError
            if isinstance(it.get("hit"), dict):
                it["hit"] = {int(k): v for k, v in it["hit"].items()}
            out[it["id"]] = it
        return out
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    ap.add_argument("--fresh", action="store_true", help="忽略已缓存结果，重跑")
    ap.add_argument("--backend", type=str, default="", help="生成后端：deepseek | local | local-http（空=用 settings）")
    ap.add_argument("--model", type=str, default="", help="local 后端基座路径")
    ap.add_argument("--adapter", type=str, default="", help="local 后端 LoRA adapter 路径")
    ap.add_argument("--http-url", type=str, default="", help="local-http 后端 base_url（如 http://127.0.0.1:8080/v1）")
    ap.add_argument("--http-model", type=str, default="", help="local-http 后端模型名")
    ap.add_argument("--out", type=str, default="", help="结果 JSON 输出路径（空=默认）")
    ap.add_argument("--report", type=str, default="", help="报告 txt 输出路径（空=默认）")
    args = ap.parse_args()

    # 生成后端切换（在构建 pipeline 前生效；judge 始终用 DeepSeek，保证评测口径一致）
    if args.backend:
        settings.llm_backend = args.backend
    if args.model:
        settings.local_model_path = args.model
    if args.adapter:
        settings.local_adapter_path = args.adapter
    if args.http_url:
        settings.local_http_base_url = args.http_url
    if args.http_model:
        settings.local_http_model = args.http_model
    results_path = Path(args.out) if args.out else RESULTS
    report_path = Path(args.report) if args.report else REPORT

    eval_data = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    items = eval_data["items"]
    if args.limit > 0:
        items = items[: args.limit]

    # 1) 建索引 + 父块映射（gold index -> parent_chunk_id）
    doc = get_parser(REAL_MD).parse(REAL_MD)
    children, _ = ParentChildChunker(child_size=512, parent_size=1536).chunk(doc)
    idx2pid = {}
    for c in children:
        idx2pid.setdefault(c.metadata["parent_index"], c.parent_chunk_id)
    print(f"语料：{len(children)} 子块 / {len(idx2pid)} 父块", flush=True)

    print("构建混合检索索引（BGE-M3 + BM25 + faiss + reranker）...", flush=True)
    pipeline = RAGPipeline(children, top_k=12, rerank_k=4)
    client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)

    # 2) 断点续跑
    done = {} if args.fresh else _load_done(results_path)

    results = []
    t_start = time.perf_counter()
    for n, item in enumerate(items, 1):
        qid = item["id"]
        if qid in done:
            results.append(done[qid])
            print(f"[{n}/{len(items)}] {qid} 已缓存，跳过", flush=True)
            continue

        query, reference, gold_idx = item["query"], item["reference_answer"], item["gold_parent_index"]
        gold_pid = idx2pid.get(gold_idx, "")

        # 端到端生成（子块检索 -> 父块去重 -> 重排 -> LLM 带引用回答）
        out = pipeline.answer(query)
        answer = out["answer"]
        sources = out["sources"]  # 与 prompt 中 [1]..[N] 一一对应
        refs = _refs_str(sources)
        final_pids = [r["id"] for r in out["trace"].get("rerank", [])]

        # 检索 Hit Rate：gold 父块是否进重排 top-k
        hit = {1: gold_pid in final_pids[:1], 3: gold_pid in final_pids[:3], 4: gold_pid in final_pids[:4]}

        # LLM-as-judge 三指标
        faith, stmts = _score_faithfulness(client, query, answer, refs)
        rel, rel_reason = _score_relevance(client, query, answer, reference)
        cites, n_acc, n_cit = _score_citation(client, query, answer, refs)
        has_cit = bool(_CIT_RE.search(answer))

        row = {
            "id": qid,
            "query": query,
            "reference_answer": reference,
            "gold_parent_index": gold_idx,
            "answer": answer,
            "sources": sources,
            "final_parent_ids": final_pids,
            "faithfulness": round(faith, 4),
            "statements": stmts,
            "relevance": rel,
            "relevance_reason": rel_reason,
            "citation_accurate": n_acc,
            "citation_total": n_cit,
            "has_citation": has_cit,
            "citations": cites,
            "hit": hit,
        }
        results.append(row)
        done[qid] = row
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(
            json.dumps({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "items": list(done.values())},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[{n}/{len(items)}] {qid} 忠实度={faith:.0%} 相关性={rel}/5 引用={n_acc}/{n_cit} "
              f"Hit@1={'Y' if hit[1] else 'N'} Hit@4={'Y' if hit[4] else 'N'}", flush=True)

    # 3) 汇总
    n = len(results)
    faith = sum(r["faithfulness"] for r in results) / n
    rel = sum(r["relevance"] for r in results) / n
    n_cit_total = sum(r["citation_total"] for r in results)
    n_cit_acc = sum(r["citation_accurate"] for r in results)
    cit_prec = n_cit_acc / n_cit_total if n_cit_total else 0.0
    cit_cov = sum(1 for r in results if r["has_citation"]) / n
    hit_rate = {k: sum(1 for r in results if r["hit"][k]) / n for k in (1, 3, 4)}

    lines = []
    lines.append("=" * 66)
    lines.append(f"端到端评测汇总（n={n}，{time.strftime('%Y-%m-%d %H:%M:%S')}）")
    lines.append("=" * 66)
    lines.append(f"忠实度 Faithfulness        {faith:6.1%}   （有依据陈述 / 总陈述）")
    lines.append(f"答案相关性 Answer Relevance {rel:5.2f}/5   （归一化 {(rel-1)/4:6.1%}）")
    lines.append(f"引用准确率 Citation Prec.  {cit_prec:6.1%}   （准确引用 / 总引用 = {n_cit_acc}/{n_cit_total}）")
    lines.append(f"引用覆盖率 Citation Cov.   {cit_cov:6.1%}   （含引用的答案 / 总答案）")
    lines.append(f"检索 Hit@1 / Hit@3 / Hit@4 {hit_rate[1]:.1%} / {hit_rate[3]:.1%} / {hit_rate[4]:.1%}")
    lines.append("=" * 66)
    lines.append("\n逐条明细（gold -> 重排 top-4 是否命中；引用 A=准确/T=总）：")
    for r in results:
        h = "".join("1" if r["hit"][1] else "·") + "".join("3" if r["hit"][3] else "·") + "".join("4" if r["hit"][4] else "·")
        lines.append(f"[{r['id']}] 忠={r['faithfulness']:.0%} 相={r['relevance']}/5 "
                     f"引={r['citation_accurate']}/{r['citation_total']} hit@134={h}  {r['query']}")

    report = "\n".join(lines)
    report_path.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n耗时 {time.perf_counter()-t_start:.0f}s；结果已写 {results_path} 与 {report_path}")


if __name__ == "__main__":
    main()
