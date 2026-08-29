# -*- coding: utf-8 -*-
"""Day1 冒烟测试 3：测通 BGE-M3 稠密向量（本地 GPU，dense 头）
首次运行会自动从 HuggingFace 下载 BGE-M3 到 HF_HOME（D 盘）。
"""
from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-m3"

print("加载模型:", MODEL, "(首次运行需下载约 2.2GB)")
model = SentenceTransformer(MODEL, device="cuda")

queries = ["重庆邮电大学的王牌专业是什么？", "什么是检索增强生成（RAG）？"]
emb = model.encode(queries, normalize_embeddings=True)

print("dense 向量 shape:", emb.shape, "(应为 (2, 1024))")
print("向量已归一化，模长:", [round(float((e @ e) ** 0.5), 3) for e in emb])
