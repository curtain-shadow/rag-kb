# -*- coding: utf-8 -*-
"""LLM 调用层抽象：统一 chat / chat_stream 接口，多后端可切换。

后端：
- DeepSeekClient：OpenAI 兼容协议（DeepSeek / GLM 均可），走云端 API。
- LocalModelClient：本地 transformers 模型（LoRA 微调后的 Qwen 等），离线 GPU 推理。
- LocalOpenAIClient：本地 OpenAI 兼容端点（Ollama / llama.cpp 起的 GGUF 服务），进程隔离。

工厂 get_llm_client() 按 settings.llm_backend 选择后端，RAGPipeline 通过它切换，业务代码零改动。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from openai import OpenAI

from .config import settings


class LLMClient(ABC):
    """统一 LLM 调用接口：chat 返回完整文本，chat_stream 逐段 yield 增量。"""

    @abstractmethod
    def chat(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 1024,
             seed: int | None = None) -> str:
        ...

    @abstractmethod
    def chat_stream(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 1024,
                    seed: int | None = None):
        ...


class DeepSeekClient(LLMClient):
    """OpenAI 兼容协议后端（DeepSeek 主；GLM 换 base_url 即可）。"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.client = OpenAI(
            api_key=api_key or settings.deepseek_api_key,
            base_url=base_url or settings.deepseek_base_url,
        )
        self.model = model or settings.deepseek_model

    def chat(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 1024,
             seed: int | None = None) -> str:
        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if seed is not None:
            kwargs["seed"] = seed
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    def chat_stream(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 1024,
                    seed: int | None = None):
        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        if seed is not None:
            kwargs["seed"] = seed
        resp = self.client.chat.completions.create(**kwargs)
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class LocalModelClient(LLMClient):
    """本地 transformers 模型（可选加载 LoRA adapter），GPU 离线推理。

    懒加载：torch / transformers / peft 在实例化时才 import，避免默认 DeepSeek 后端被拖累。
    """

    def __init__(self, model_path: str | None = None, adapter_path: str | None = None,
                 device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path or settings.local_model_path
        self.adapter_path = adapter_path or settings.local_adapter_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not self.model_path:
            raise ValueError("llm_backend=local 需要设置 local_model_path（.env）")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=dtype).to(self.device)

        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)

        self.model = model.eval()

    def _encode(self, messages: list[dict]):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tokenizer(text, return_tensors="pt")

    def chat(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 1024,
             seed: int | None = None) -> str:
        import torch

        inputs = self._encode(messages)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        gen_kwargs: dict = dict(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # seed 仅在采样（do_sample=True）时生效，固定随机性让结果可复现
        if seed is not None and temperature > 0:
            gen_kwargs["generator"] = torch.Generator(device=self.device).manual_seed(seed)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)

    def chat_stream(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 1024,
                    seed: int | None = None):
        # 本地模型先做整段生成，逐段 yield（非逐 token，SSE 仍可复用；逐 token 留待后续 TextIteratorStreamer）
        yield self.chat(messages, temperature=temperature, max_tokens=max_tokens, seed=seed)


class LocalOpenAIClient(DeepSeekClient):
    """本地 OpenAI 兼容端点（Ollama / llama.cpp server），复用 DeepSeek 的协议封装。

    llama.cpp / Ollama 都暴露 /v1/chat/completions，与 OpenAI 协议一致，换 base_url 即可；
    本地服务不校验 api_key，填占位符即可。
    """

    def __init__(self, base_url: str | None = None, model: str | None = None):
        super().__init__(
            api_key="noop",
            base_url=base_url or settings.local_http_base_url,
            model=model or settings.local_http_model,
        )


def get_llm_client() -> LLMClient:
    """按 settings.llm_backend 选择后端：
    local -> transformers 本地模型；local-http -> 本地 HTTP 服务（GGUF）；否则 DeepSeek。"""
    if settings.llm_backend == "local":
        return LocalModelClient()
    if settings.llm_backend == "local-http":
        return LocalOpenAIClient()
    return DeepSeekClient()
