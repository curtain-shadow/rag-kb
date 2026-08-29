# -*- coding: utf-8 -*-
"""父子分块（Parent-Child Chunking）。

核心思路（对标工业级 RAG 的「子块检索 + 父块上下文」）：
- 父块（parent）：较长的上下文窗口，喂给 LLM 生成答案，保证上下文完整。
- 子块（child）：较短、语义聚焦的单元，用于向量检索/稀疏检索，减少噪声、提高命中精度。
- 检索时：用子块做匹配（embedding + BM25），命中的子块向上映射回其父块，把父块文本
  作为上下文交给 LLM。这样兼顾「检索精度」和「生成上下文完整」。

分块单位说明：
- chunk_size 以「字符数」计，中文下字符数与 token 数近似（BGE-M3 中文约 1~1.5 token/字），
  这里用字符数作为 token 的轻量近似，避免离线分块时加载 tokenizer。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..parsing.base import ParsedDocument


@dataclass
class Chunk:
    """子块：text 用于检索（embedding/BM25），parent_text 用于喂 LLM。"""
    chunk_id: str
    doc_id: str
    parent_chunk_id: str
    text: str                 # 子块文本（检索用）
    parent_text: str          # 父块文本（生成用，向上映射的关键）
    metadata: dict = field(default_factory=dict)


_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])")


def _split_long(text: str, max_len: int) -> list[str]:
    """单段超长时的兜底：按句子边界切，仍超长则硬切。"""
    sentences = _SENT_SPLIT.split(text)
    chunks: list[str] = []
    cur = ""
    for s in sentences:
        if not s.strip():
            continue
        if len(cur) + len(s) <= max_len:
            cur += s
        else:
            if cur.strip():
                chunks.append(cur.strip())
            cur = s if len(s) <= max_len else s[:max_len]
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def _split_by_size(text: str, max_len: int) -> list[str]:
    """把长文本切成不超过 max_len 的块，优先在段落（\n）边界断开，其次句子边界。"""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paragraphs:
        if len(cur) + len(p) + 1 <= max_len:
            cur = (cur + "\n" + p).strip() if cur else p
        else:
            if cur:
                chunks.append(cur)
            if len(p) > max_len:
                chunks.extend(_split_long(p, max_len))
                cur = ""
            else:
                cur = p
    if cur:
        chunks.append(cur)
    return chunks


class ParentChildChunker:
    """先按 parent_size 切父块，再在每个父块内按 child_size 切子块并回填 parent_chunk_id。"""

    def __init__(self, child_size: int = 512, parent_size: int = 1536):
        self.child_size = child_size
        self.parent_size = parent_size

    def chunk(self, doc: ParsedDocument) -> tuple[list[Chunk], list[Chunk]]:
        """返回 (子块列表, 父块列表)。子块携带 parent_text 供生成阶段使用。"""
        parents: list[Chunk] = []
        children: list[Chunk] = []

        parent_texts = _split_by_size(doc.text, self.parent_size)
        for pi, ptext in enumerate(parent_texts):
            parent_id = f"{doc.doc_id}-p{pi}"
            parent = Chunk(
                chunk_id=parent_id,
                doc_id=doc.doc_id,
                parent_chunk_id=parent_id,   # 父块自身无上层父块
                text=ptext,
                parent_text=ptext,
                metadata={"doc_title": doc.title, "parent_index": pi},
            )
            parents.append(parent)

            child_texts = _split_by_size(ptext, self.child_size)
            for ci, ctext in enumerate(child_texts):
                children.append(Chunk(
                    chunk_id=f"{parent_id}-c{ci}",
                    doc_id=doc.doc_id,
                    parent_chunk_id=parent_id,
                    text=ctext,
                    parent_text=ptext,
                    metadata={
                        "doc_title": doc.title,
                        "parent_index": pi,
                        "child_index": ci,
                    },
                ))

        return children, parents
