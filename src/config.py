# -*- coding: utf-8 -*-
"""全局配置：从 .env 加载，统一管理 API Key / 路径 / 模型参数"""
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

# HuggingFace 兜底配置：国内直连 huggingface.co 超时，必须走镜像；缓存放 D 盘。
# 用 setdefault 只在 shell 未设置时补上，不影响用户已配置的环境变量。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT.parent / "huggingface_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT.parent / "huggingface_cache" / "hub"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM（DeepSeek 主，GLM 降级）
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # 路径
    data_dir: Path = ROOT / "data"
    docs_dir: Path = ROOT / "data" / "docs"

    # 检索参数（父子分块 / 融合）
    child_chunk_size: int = 512
    parent_chunk_size: int = 1536
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_final: int = 6


settings = Settings()
