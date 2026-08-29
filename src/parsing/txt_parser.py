# -*- coding: utf-8 -*-
"""纯文本 / Markdown 解析：UTF-8 读取，按空行分段。"""
from pathlib import Path

from .base import BaseParser, ParsedDocument, make_doc_id, register


@register
class TxtParser(BaseParser):
    exts = (".txt", ".md", ".markdown")

    def parse(self, path: Path) -> ParsedDocument:
        raw = Path(path).read_text(encoding="utf-8", errors="ignore")
        sections = [s.strip() for s in raw.split("\n\n") if s.strip()]
        # 标题优先取首个非空 Markdown 一级标题，否则用文件名
        title = Path(path).stem
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("# "):
                title = line[2:].strip()
                break
        return ParsedDocument(
            doc_id=make_doc_id(str(path)),
            source=str(path),
            title=title,
            sections=sections,
            text="\n\n".join(sections),
            metadata={"format": path.suffix.lstrip(".")},
        )
