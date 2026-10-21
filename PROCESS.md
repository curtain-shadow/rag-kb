# 项目实现全过程（从 0 到 1）

> 本文记录深度 RAG 知识库问答系统从零开始的完整构建过程，按**实际构建顺序**组织（不是按模块，而是按「先搭什么、后补什么」的时间线）。
> 每一步都写清：**目标 → 怎么做 → 为什么这么设计 → 踩了什么坑 → 怎么验证**。
> 适合面试前通读，回答「这个项目你是怎么做的」。

## 一句话概括

从环境搭建开始，按「检索 → 评测 → 微调 → 部署」的顺序，把中文维基语料做成了一个每一层都能讲清「为什么」、每一步都有数据支撑的工业级 RAG 系统。

---

## 第 0 步：定目标 + 搭环境（Day 1）

**目标**：做一个「能检索、能带引用回答」的 RAG 系统，语料用公开中文维基子集。定位就一句话——**每一层都不套库，能讲清为什么**。

**环境**（这步决定了后面所有技术选型，因为受「国内网络 + 8GB 显存」双重约束）：

- 全部落在 D 盘、不占 C 盘：venv、模型缓存、pip 缓存都指到 D 盘。
- 国内直连 HuggingFace 超时 → `config.py` 里 `os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")` 兜底走镜像。
- 跑三个验证脚本确认地基：GPU/CUDA 可用、DeepSeek API 连通、BGE-M3 向量能出。

**产出**：`scripts/day1_verify_gpu.py` / `day1_test_llm.py` / `day1_test_embedding.py`。

---

## 第 1 步：解析 + 分块（Week 1）

**目标**：把文档变成「能检索的最小单元 + 能给 LLM 的完整上下文」。

- **解析**：注册式多解析器，按扩展名路由到 PDF / DOCX / TXT / Markdown，统一产出 `ParsedDocument`。换语料时下游零改动。
- **分块**：核心设计是**父子分块**——子块 512 字（短、语义聚焦，用于检索），父块 1536 字（长、信息完整，用于喂 LLM），用 `parent_chunk_id` 关联。

> **为什么不用一个大块？** 向量 embedding 对长文本会稀释语义——一句话埋进一大段里就检索不到。所以「短子块检索 + 长父块喂 LLM」兼顾「检得准」与「答得全」。

**产出**：`src/parsing/`、`src/chunking/`，`week1_test_parse_chunk.py`。

---

## 第 2 步：混合检索 + RRF 融合（Week 2）

**目标**：既懂语义、又能精确匹配专有名词。

- **稠密路**：BGE-M3（只加载 dense head）+ faiss 余弦。
- **稀疏路**：自实现 BM25 + jieba 分词。
- **RRF 融合**：各取 top-20，用 `1/(k+rank)` 排名融合。

> **为什么向量 + BM25 双路？** 稠密向量擅长语义泛化（同义词、改写），但对专有名词 / 精确关键词 / 低频术语不敏感；BM25 靠词频倒排精确匹配这些词，两者召回互补。

> **为什么 RRF 不加权求和？** 稠密相似度（内积/余弦）与 BM25 分数尺度完全不一致，无法直接加权；RRF 只看各路排名、天然跨模型可比、无需归一化，且对单路异常分数鲁棒。

**产出**：`src/retrieval/`（sparse / dense / hybrid），`week2_test_retrieval.py`。

---

## 第 3 步：重排 + 带引用生成（Week 3）

**目标**：把粗召回的候选精排，再让 LLM 只依据资料、带引用回答。

- **Cross-Encoder 重排**（`bge-reranker-v2-m3`）：对父块去重后的候选精排，取 top-4。
- **RAGPipeline**：召回 → 父块去重 → 重排 → 拼成 `[i] 段落` 注入 prompt。
- **系统 prompt 定死两条铁律**：句末标 `[1][2]` 引用；资料不足时明确说明、不编造。

> **为什么 Cross-Encoder 只重排 top-k，不能全库重排？** 它把 query 与候选拼接后一起进编码器，是逐对 O(N) 推理，无法像双塔那样预编码后做 ANN 检索，全库 N 太大跑不动。所以它是「精排」，只对粗召回后的 top-k 做精排——「双塔粗排 + Cross-Encoder 精排」是工业标准组合。

**产出**：`src/rag.py`、`src/llm.py`（此阶段只有 DeepSeek 客户端），`week3_test_rag.py`。

---

## 第 4 步：服务化（Week 4）

**目标**：从脚本变成能对外提供的服务。

- FastAPI + uvicorn + SSE 流式（`/health` / `/ask` / `/ask/stream`）。
- 全链路 trace：检索、重排、生成各环节耗时可观测。

**产出**：`src/api.py`、`week4_test_stream.py`、`week4_test_api.py`。

---

## 第 5 步：向量库升级（Week 5）

**目标**：从「能跑」到「能上量」。

- faiss → Milvus（HNSW 图索引做 ANN，亚线性查询 + 集合落盘 `.db` 持久化）。
- 鸭子类型接口，`HybridRetriever(chunks, dense=MilvusIndex())` 一键切后端。

> **为什么升级？** faiss.IndexFlatIP 是暴力全扫（每次查询 O(N)、索引只在内存、进程退出即丢）；Milvus 用 HNSW 做近似最近邻，亚线性查询可撑百万级语料，且集合落盘持久化、生产可分布式扩展。小语料下两后端 Top5 命中与余弦分数完全一致（ANN 无召回损失）。

**产出**：`scripts/week5_test_milvus.py`。

---

## 第 6 步：端到端评测（Phase 1）

**目标**：用数据证明系统到底好不好，而不是「感觉挺好」。

- **自建 200 条评测集**：每条带 query + 参考答案 + gold 父块，**语义级标注、不是关键词反查**（避免评测集偏向 BM25）。
- **自实现 LLM-as-judge**（DeepSeek 当裁判），不 `pip install ragas`：忠实度（拆原子陈述判有没有编造）、相关性（对照参考答案 1–5 分）、引用准确率（`[i]` 标得对不对）。
- **检索消融**：full vs no-rerank vs dense-only，量化每一层贡献。

**结果**（DeepSeek 后端，200 条 × 3 次运行，均值 ± 标准差）：忠实度 99.8%±0.3% · 相关性 4.94/5 · 引用准确率 98.1%（795/810）· 引用覆盖率 99.8%；检索 Hit@1 93.9% / Hit@3 99.5% / Hit@4 99.5%（确定性方差 0）。3 条敏感条目（q021/q176/q185）被 DeepSeek 审核拦截跳过，有效 197 条。

**消融结论**（诚实口径，200 条维基语料）：

1. 重排是精度主力：Hit@1 80.0%→94.0%（+14pp，小语料时仅 +5pp），「双塔粗排 + 精排」在大语料上价值更大。
2. BM25 单独是负贡献（无重排阶段 Hit@1 84.5%→80.0%），配合重排才为正（full 94.0% 最优）——BM25 的价值是「多召回候选、交给重排筛选」。

**产出**：`data/eval_set.json`、`scripts/eval_rag.py`、`scripts/eval_ablation.py`。

---

## 第 7 步：本地 LoRA 微调（Phase 2，最重的一步）

**目标**：不依赖云端 DeepSeek，自己微调一个本地模型做带引用问答。

1. **合成数据**：DeepSeek 生成 294 条「带引用 RAG 问答」，**每条 `[i]` 引用都校验真实存在于给定资料**，不合格丢弃。
2. **抽象 LLM 层**：先重构 `llm.py` 出 `LLMClient` 接口，`DeepSeekClient` / `LocalModelClient` 可切换——否则本地模型接不进现有流水线。
3. **纯 LoRA bf16 微调**：r=16 / alpha=32 / dropout=0.05，可训练 18.46M（1.18%），**手写训练循环**（不依赖 bitsandbytes / trl，绕开 Windows 兼容坑）。

> **踩的最大的坑（8GB 卡 OOM 根因）**：transformers 无 padding 时默认走原生 GQA SDPA 内核（`enable_gqa=True`），torch 2.13 上把 T×T 注意力分数物化出来、显存 O(T²) 直接 OOM。修复：在 `src/finetune/train.py` monkeypatch `use_gqa_in_sdpa→False` 强制走 `repeat_kv` 展开路径，峰值显存 7.9GB → 5.2GB，`max_length=4608` 零截断。

**结果**：3 epoch，265 train / 29 val，val_loss `0.3787 → 0.3582`，adapter 存 `data/lora_qwen_rag`。

**产出**：`src/finetune/`、`scripts/train_lora.py`。

---

## 第 8 步：四路对比评测（Phase 2 收尾）

**目标**：微调到底有没有用？拉齐检索 / 重排 / 评测集，只换生成后端，对比 DeepSeek / 纯基座 / LoRA / 基座-GGUF。

> **踩的一个坑**：最初「基座」这一列因为 `config.py` 里 `local_adapter_path` 默认值误指向 LoRA 目录，实际加载的是 LoRA 模型，导致对比表失真（DeepSeek vs LoRA vs LoRA）。发现后把默认值改为空字符串（空 = 只用基座），重跑拿到纯基座真值。

**结果**（30 条，裁判=DeepSeek，检索 Hit@1/3/4=73.3/93.3/96.7% 四者一致）：

| 指标 | DeepSeek | 纯基座 | LoRA | 基座-GGUF(fp16) |
|---|---|---|---|---|
| 忠实度 | 100% | 82.2% | 98.9% | 90.4% |
| 相关性 /5 | 4.87 | 4.03 | 4.30 | 4.43 |
| 引用准确率 | 97.7% | 0% | 76.7% | 0% |
| 引用覆盖率 | 100% | 0% | 100% | 0% |

**结论**：引用能力是 LoRA 微调独有的（基座 bf16 与 GGUF 都不产引用）；LoRA 还带来忠实度 82.2%→98.9%（学会「只依据资料不编造」）。GGUF fp16 部署不损基座质量（甚至略高）。

**产出**：`scripts/eval_compare.py`。

---

## 第 9 步：本地部署（Phase 3，完成）

**目标**：让本地模型真正「一键起服务」。

- 选型 **llama.cpp**（Windows 友好、一条命令起），模型用 **Qwen2.5-1.5B GGUF fp16**（先不量化，消除「量化掉精度」的疑虑）。
- llama-server 起 OpenAI 兼容端点，`LocalOpenAIClient` 复用 DeepSeek 协议封装，业务代码零改动。
- 实测 **70 tok/s**，29/29 层全 offload GPU。
- 评测（见第 8 步）：忠实度 90.4% / 相关性 4.43，不损基座 bf16；但不产引用（基座天然行为）。部署端要带引用 → LoRA 合并导出 GGUF。

**产出**：`scripts/serve_local.bat`、`LocalOpenAIClient`。

---

## 完整时间线（一页速览）

```
Day 1   环境(墙+8GB) ──► 验证 GPU / DeepSeek / BGE-M3
Week 1  解析 + 父子分块（子块检索、父块喂 LLM）
Week 2  混合检索 + RRF（稠密 BGE-M3 + 稀疏 BM25）
Week 3  重排 + 带引用生成（Cross-Encoder + 铁律 prompt）
Week 4  FastAPI + SSE 流式 + trace
Week 5  Milvus 升级（HNSW ANN + 落盘）
Phase 1 端到端评测（200 条评测集 + LLM-as-judge + 消融）
Phase 2 LoRA 微调（294 条合成数据 + 纯 bf16 + OOM 攻坚）
        四路对比评测（DeepSeek / 基座 / LoRA / GGUF）
Phase 3 GGUF 部署（llama.cpp + fp16，70 tok/s，不损质量）
```

**贯穿全程的方法论**：先验证跑通 → 再补评测量化 → 最后工程化交付；每一层都留了「为什么」和「数据」。
