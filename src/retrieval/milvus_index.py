# -*- coding: utf-8 -*-
r"""Milvus 稠密检索后端：BGE-M3 dense 编码 + Milvus 向量库（HNSW ANN 索引）。

与 faiss 后端（dense.DenseIndex）的区别，也是「为什么要从 faiss 升级到 Milvus」：

- faiss.IndexFlatIP 是「暴力全扫」的精确检索：每次查询 O(N) 遍历全部向量，
  语料小没问题，但语料大了（百万级）查询延迟线性上涨，且索引只在内存里、进程一退就没了。
- Milvus 是专门的向量数据库：
  1. HNSW 图索引 → 近似最近邻（ANN），亚线性查询，能撑大规模语料；
  2. 落盘持久化（本文件 .db），进程重启索引还在，faiss 每次要重建；
  3. 生产上还能分布式部署（etcd + MinIO + 多节点）横向扩展，这里用 Milvus Lite 本地模式演示同一套 API。

与 DenseIndex 保持同一鸭子类型接口：add_documents(chunks) + search(query, top_k)，
因此 HybridRetriever 只需换 dense 后端即可在 faiss / Milvus 之间一键切换。
"""
from __future__ import annotations

import numpy as np
from pymilvus import DataType, MilvusClient
from sentence_transformers import SentenceTransformer


class MilvusIndex:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
        uri: str = r"D:\profession\rag_kb\data\index\milvus.db",
        collection_name: str = "chunks",
        dim: int = 1024,
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.client = MilvusClient(uri=uri)
        self.collection = collection_name
        self.dim = dim
        self.chunk_ids: list[str] = []

    def add_documents(self, chunks: list) -> None:
        """编码全部子块 → 建集合（HNSW + COSINE）→ 批量插入。"""
        texts = [c.text for c in chunks]
        vecs = self.model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
        vecs = np.asarray(vecs, dtype="float32")
        self.dim = int(vecs.shape[1])
        self.chunk_ids = [c.chunk_id for c in chunks]

        # 重建集合，保证 add_documents 幂等
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim)

        index_params = self.client.prepare_index_params()
        # HNSW：图结构近似最近邻，无需像 IVF 那样训练，小语料也能直接建索引
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        self.client.create_collection(self.collection, schema=schema, index_params=index_params)

        rows = [
            {"chunk_id": cid, "vector": v.tolist()}
            for cid, v in zip(self.chunk_ids, vecs)
        ]
        self.client.insert(self.collection, data=rows)

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """返回 [(chunk_id, cosine_score)]，按相似度降序（与 DenseIndex 一致）。"""
        q = self.model.encode([query], normalize_embeddings=False)
        q = np.asarray(q, dtype="float32")

        res = self.client.search(
            collection_name=self.collection,
            data=[q[0].tolist()],
            limit=top_k,
            output_fields=["chunk_id"],
            search_params={"metric_type": "COSINE", "params": {"ef": 128}},
        )
        # res[0] 为查询向量对应的命中列表；COSINE 度量下 distance 即余弦相似度（越大越像）
        return [(h["entity"]["chunk_id"], float(h["distance"])) for h in res[0]]
