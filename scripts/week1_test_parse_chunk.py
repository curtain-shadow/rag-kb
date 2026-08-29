# -*- coding: utf-8 -*-
r"""Week1 冒烟测试：文档解析（PDF/DOCX/MD）+ 父子分块。

跑法：D:\profession\venv\Scripts\python.exe scripts\week1_test_parse_chunk.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parsing import get_parser
from src.chunking import ParentChildChunker

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


def make_samples() -> dict[str, Path]:
    """生成测试用的 PDF / DOCX 样例（首次运行生成一次）。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample_text = (
        "深度学习是机器学习的一个分支，通过多层神经网络自动学习数据的层次化表示。"
        "其中，Transformer 架构自 2017 年提出以来，凭借自注意力机制在自然语言处理领域取得了突破。"
        "检索增强生成（RAG）结合了信息检索与大语言模型，先从知识库中检索相关文档片段，再交由模型生成答案，从而缓解幻觉问题。"
        "向量检索通过把文本编码成语义向量，用余弦相似度衡量语义相关性；而 BM25 稀疏检索基于词频统计，擅长精确词匹配。"
        "二者在混合检索中互补，能同时覆盖语义相近与关键词命中的场景。"
    )
    out: dict[str, Path] = {}

    # 生成 PDF（嵌入中文字体，否则中文会渲染成乱码）
    pdf_path = DATA_DIR / "sample.pdf"
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_font(fontname="simhei", fontfile=r"C:\Windows\Fonts\simhei.ttf")
    page.insert_textbox(pymupdf.Rect(72, 72, 523, 700), sample_text, fontsize=12, fontname="simhei")
    doc.save(str(pdf_path))
    doc.close()
    out["pdf"] = pdf_path

    # 生成 DOCX
    docx_path = DATA_DIR / "sample.docx"
    import docx as _docx
    d = _docx.Document()
    d.add_heading("深度学习与 RAG 简介", 0)
    d.add_paragraph(sample_text)
    table = d.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "技术"
    table.rows[0].cells[1].text = "作用"
    table.rows[1].cells[0].text = "BM25"
    table.rows[1].cells[1].text = "稀疏检索，精确词匹配"
    d.save(str(docx_path))
    out["docx"] = docx_path

    return out


def main():
    samples = make_samples()

    # 真实数据源：简历信息库.md（MD 解析）
    real_md = Path(r"d:\个人\简历信息库.md")

    paths = {**samples, "md": real_md}
    chunker = ParentChildChunker(child_size=512, parent_size=1536)

    for name, p in paths.items():
        if not p.exists():
            print(f"[跳过] {name}: 文件不存在 {p}")
            continue
        parser = get_parser(p)
        doc = parser.parse(p)
        children, parents = chunker.chunk(doc)

        print("=" * 60)
        print(f"[{name}] {p.name}")
        print(f"  标题: {doc.title}")
        print(f"  分段数: {len(doc.sections)}  全文字符: {len(doc.text)}")
        print(f"  父块: {len(parents)}  子块: {len(children)}")

        # 展示第一个子块及其父块映射
        if children:
            c = children[0]
            print(f"  示例子块 id={c.chunk_id}  字符={len(c.text)}")
            print(f"     → 父块 id={c.parent_chunk_id}  字符={len(c.parent_text)}")
            print(f"     → 子块文本前 40 字: {c.text[:40]}...")
            # 校验：所有子块都能映射回存在的父块
            parent_ids = {p.chunk_id for p in parents}
            ok = all(ch.parent_chunk_id in parent_ids for ch in children)
            print(f"  父块映射完整性: {'✅ 全部子块可映射回父块' if ok else '❌ 存在孤儿子块'}")
    print("=" * 60)
    print("Week1 解析 + 父子分块 冒烟测试完成")


if __name__ == "__main__":
    main()
