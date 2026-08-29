# -*- coding: utf-8 -*-
"""LLM 调用层：DeepSeek（OpenAI 兼容协议）。后续可加 GLM 降级与重试。"""
from __future__ import annotations

from openai import OpenAI

from .config import settings


class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )

    def chat(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 1024) -> str:
        resp = self.client.chat.completions.create(
            model=settings.deepseek_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    def chat_stream(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 1024):
        """流式返回：逐个 yield 文本增量（供 SSE 推送）。"""
        resp = self.client.chat.completions.create(
            model=settings.deepseek_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
