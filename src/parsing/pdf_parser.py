# -*- coding: utf-8 -*-
"""PDF 解析（PyMuPDF）：逐页抽取文本，保留页内段落结构。"""
from pathlib import Path

import pymupdf

from .base import BaseParser, ParsedDocument, make_doc_id, register


@register
class PdfParser(BaseParser):
    exts = (".pdf",)

    def parse(self, path: Path) -> ParsedDocument:
        doc = pymupdf.open(str(path))
        n_pages = doc.page_count
        sections: list[str] = []
        for page in doc:
            text = page.get_text("text").strip()
            if text:
                sections.append(text)
        doc.close()
        return ParsedDocument(
            doc_id=make_doc_id(str(path)),
            source=str(path),
            title=path.stem,
            sections=sections,
            text="\n\n".join(sections),
            metadata={"pages": n_pages, "format": "pdf"},
        )
