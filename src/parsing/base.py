# -*- coding: utf-8 -*-
"""文档解析统一接口：把 PDF / DOCX / MD / TXT 规范化成 ParsedDocument。

设计要点：
- 解析器按「文件后缀」注册，调用方只认 get_parser()，新增格式只需加一个类。
- ParsedDocument 同时保留「分段列表 sections」和「全量文本 text」，
  分段保留原文结构（供后续父子分块在段落/句子边界切分）。
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedDocument:
    """解析后的规范化文档。"""
    doc_id: str                  # 稳定唯一 ID（源路径哈希）
    source: str                  # 源文件绝对路径
    title: str                   # 标题（文件名或首行）
    sections: list[str]          # 分段文本（保留段落结构）
    text: str                    # 全量纯文本（\n\n 拼接）
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.title:
            self.title = Path(self.source).stem


def make_doc_id(source: str) -> str:
    """用源文件绝对路径生成稳定 doc_id（同文件跨次运行不变）。"""
    return hashlib.md5(str(Path(source).resolve()).encode("utf-8")).hexdigest()[:16]


class BaseParser(ABC):
    """解析器基类：子类实现 parse()，并声明支持的 exts。"""
    exts: tuple[str, ...] = ()

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        ...


_PARSERS: dict[str, BaseParser] = {}


def register(cls):
    """注册解析器：按类声明 exts 建实例并挂到后缀表。"""
    inst = cls()
    for ext in inst.exts:
        _PARSERS[ext] = inst
    return cls


def get_parser(path: str | Path) -> BaseParser | None:
    """按文件后缀返回解析器，未知格式返回 None。"""
    return _PARSERS.get(Path(path).suffix.lower())
