# -*- coding: utf-8 -*-
"""文档解析层：导入各解析器触发注册，对外暴露 get_parser。"""
from .base import ParsedDocument, get_parser, make_doc_id
from .pdf_parser import PdfParser
from .docx_parser import DocxParser
from .txt_parser import TxtParser

__all__ = ["ParsedDocument", "get_parser", "make_doc_id", "PdfParser", "DocxParser", "TxtParser"]
