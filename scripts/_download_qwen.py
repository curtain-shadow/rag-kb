# -*- coding: utf-8 -*-
"""临时：预下载 Qwen2.5-1.5B-Instruct 到 D 盘 HF 缓存（走 hf-mirror）。

绕过 Xet CAS：hf-mirror 对 Xet 后端支持不全（重构大文件时 401），
用 HF_HUB_DISABLE_XET=1 走普通 HTTP 下载。
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", r"D:\profession\huggingface_cache")
os.environ.setdefault("HF_HUB_CACHE", r"D:\profession\huggingface_cache\hub")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download  # noqa: E402

path = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct")
print("DONE", path)
