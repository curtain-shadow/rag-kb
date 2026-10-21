# 深度 RAG 知识库问答系统

一个从零实现、对标工业级的检索增强生成（RAG）项目：多格式文档解析 → 父子分块 → 混合检索（稠密 + 稀疏 + RRF 融合）→ Cross-Encoder 重排 → LLM 带引用生成 → FastAPI + SSE 流式服务 + 全链路可观测。

适合作为 AI 应用 / RAG / Agent 岗位的面试作品：**每一层都不是套库，能讲清楚「为什么这么设计」**。

---

## 架构

```
                          ┌────────────────────────────────────────────┐
  多格式文档 ──解析──▶     │  PDF / DOCX / TXT / Markdown               │
  (PyMuPDF·python-docx)   └──────────────────┬─────────────────────────┘
                                             ▼
                          父子分块（子块 512 / 父块 1536，parent_chunk_id 关联）
                                             │
                    ┌────────────────────────┼─────────────────────────┐
                    ▼                        ▼                          │
            稠密检索 BGE-M3            稀疏检索 BM25 + jieba         （离线建索引）
            （dense head · FAISS/Milvus）（自实现倒排索引）
                    └────────────┬───────────┘
                                 ▼
                          RRF 倒数排名融合（k=60）
                                 ▼
                          父块去重（多子块 → 同一父块）
                                 ▼
                      Cross-Encoder 重排（bge-reranker-v2-m3）
                                 ▼
                      LLM 带引用生成（DeepSeek / GLM 降级）
                                 ▼
                    FastAPI + SSE 流式 + trace 可观测
```

## 检索流程（核心）

1. **解析**：`src/parsing/` 注册式多解析器，按扩展名路由到 PDF / DOCX / TXT 解析器，统一产出 `ParsedDocument`。
2. **分块**：`ParentChildChunker` 把文档切成短**子块**（512 字，用于检索）与长**父块**（1536 字，用于喂 LLM），用 `parent_chunk_id` 关联。
3. **混合召回**：`HybridRetriever` 并行跑两条路——稠密（BGE-M3 语义向量 + FAISS 余弦）与稀疏（自实现 BM25 + jieba 分词），各取 top-20，用 **RRF** 融合。
4. **父块去重**：命中的多个子块可能属于同一父块，按 `parent_chunk_id` 去重，避免重复上下文。
5. **重排**：`CrossEncoderReranker` 对去重后的父块精排，取 top-4。
6. **生成**：`RAGPipeline` 把父块拼成 `[i] 段落` 参考资料注入 prompt，要求 LLM 句末标注 `[1][2]` 引用，且**资料不足时明确说明、不编造**。

## 技术栈

| 层 | 选型 |
|---|---|
| 语言 / 推理 | Python 3.13 · PyTorch 2.13.0+cu126（RTX 4060 Laptop 8GB） |
| 稠密向量 | `BAAI/bge-m3`（只加载 dense head）· faiss-cpu（`IndexFlatIP` 归一化=余弦）／ Milvus（HNSW ANN + COSINE，可切换） |
| 稀疏检索 | 自实现 BM25（k1=1.5, b=0.75）+ jieba 分词 |
| 融合 / 重排 | RRF（k=60）· `BAAI/bge-reranker-v2-m3` Cross-Encoder |
| LLM | DeepSeek（`deepseek-chat`）主，GLM 降级；本地 Qwen2.5-1.5B-Instruct（transformers / + LoRA / GGUF 三态可切换） |
| 服务 | FastAPI · uvicorn · sse-starlette（SSE 流式） |

## 四个「为什么」（面试追问必答）

1. **为什么向量 + BM25 双路，而不只用向量？**
   稠密向量擅长语义泛化（同义词、改写），但对**专有名词 / 精确关键词 / 低频术语**不敏感；BM25 靠词频倒排精确匹配这些词。两者召回有互补性，混合后召回率更稳。
2. **为什么 RRF，而不是对分数加权求和？**
   稠密相似度（内积/余弦）与 BM25 分数**尺度完全不一致**，无法直接加权；RRF 只看各路的**排名**（`1/(k+rank)`）而非绝对分，天然跨模型可比、无需归一化，且对单路异常分数鲁棒。
3. **为什么「子块检索 + 父块上下文」，而不是直接用长块检索？**
   向量 embedding 对**长文本会稀释语义**（一句话埋进一大段里就检索不到）；短子块语义聚焦、召回更准。但短块信息量不足，于是命中子块后映射回其**父块**，把完整上下文喂给 LLM，兼顾「检得准」与「答得全」。
4. **为什么 Cross-Encoder 只重排 top-k，不能全库重排？**
   Cross-Encoder 把 query 与候选**拼接后一起进编码器**，是逐对 O(N) 推理，无法像双塔那样预编码后做 ANN 检索，全库 N 太大根本跑不动。所以它是「精排」，只对粗召回后的 top-k 候选做精排，双塔粗排 + Cross-Encoder 精排是工业标准组合。

## 量化评估（检索消融 + 端到端 LLM-as-judge）

评测集在真实语料（简历信息库，当前 **74 子块 / 20 父块**）上自建，**每条 query 都带「参考答案 + gold 父块」**（语义级标注、非关键词反查，避免偏向 BM25）。两套脚本共用同一套 gold。

### 端到端三指标（`scripts/eval_rag.py`，30 条，DeepSeek 当裁判，自实现 judge 不依赖 ragas）

| 指标 | 分数 | 口径 |
|---|---|---|
| **忠实度 Faithfulness** | **100.0%** | 有依据陈述 / 总陈述（拆句逐条判定） |
| **答案相关性 Answer Relevance** | **4.93 / 5** | 对照参考答案 1–5 分 |
| **引用准确率 Citation Precision** | **97.7%** | 准确引用 / 总引用 = 42/43 |
| **引用覆盖率 Citation Coverage** | **100.0%** | 含引用答案 / 总答案 |

检索侧（同一 30 条，full 配置，Cross-Encoder 重排 top-4）：**Hit@1 73.3% / Hit@3 93.3% / Hit@4 96.7%**。

### 检索消融（`scripts/eval_ablation.py`，20 条）

| 配置 | Hit@1 | Hit@3 | Hit@4 |
|---|---|---|---|
| **full（BM25 + 稠密 + RRF + 重排）** | **80.0%** | **95.0%** | **100%** |
| no-rerank（BM25 + 稠密 + RRF） | 75.0% | 95.0% | 100% |
| dense-only（仅稠密向量） | 75.0% | 100% | 100% |

两条结论（诚实口径）：

1. **重排提精度**：Cross-Encoder 精排把 Hit@1 从 75% 提到 80%（+5pp），并把 no-rerank 掉到 95% 的 Hit@3 拉回 Hit@4=100%——「双塔粗排 + Cross-Encoder 精排」的工业组合成立。
2. **BM25 在小语料上补召回不明显**：dense-only 已 Hit@3=100%，加 BM25 后 Hit@3 反而 95%（BM25 的精确词匹配把 1 条错误父块顶进 top-3）。这是 **20 父块小语料的预期结果**——BM25 的互补价值在「精确词 / 低频术语」场景，小语料里这类词少、收益被精度损耗抵消，其价值在更大更异质的语料上才显现。

复现：

```bash
python scripts/eval_ablation.py   # 检索消融 + Hit Rate（加载 BGE-M3 + reranker，约 1–2 分钟）
python scripts/eval_rag.py        # 端到端 LLM-as-judge 三指标 + Hit Rate（30 条，约 4–6 分钟，可断点续跑）
```

## 本地 LoRA 微调（`src/finetune/` + `scripts/train_lora.py`）

用 DeepSeek 合成 **294 条「带引用 RAG 问答」SFT 数据**（每条 `[i]` 引用经校验真实存在于给定资料，不合格丢弃），对 **Qwen2.5-1.5B-Instruct** 做**纯 LoRA bf16 微调**——不量化、不依赖 bitsandbytes / trl，手写训练循环。

- **配置**：r=16 / alpha=32 / dropout=0.05，target=7 个注意力+FFN 投影层，可训练 **18.46M（1.18%）**；gradient checkpointing + 分块交叉熵（15.2w 词表 logits 分块算），8GB 卡峰值显存 **~5.2GB**。
- **训练**：3 epoch，265 train / 29 val，val_loss `0.3787 → 0.3582`，adapter 存 `data/lora_qwen_rag`。
- **权重不入库**：`data/lora_qwen_rag/` 已 gitignore，用 `scripts/train_lora.py` 一键复现。
- **关键坑（8GB 卡 OOM 根因）**：transformers 无 padding 时默认走原生 GQA SDPA 内核（`enable_gqa=True`），torch 2.13 上把 T×T 注意力分数物化出来、显存 O(T²) 直接 OOM；在 `src/finetune/train.py` monkeypatch `use_gqa_in_sdpa→False` 强制 `repeat_kv` 展开路径后，峰值 7.9GB→5.2GB。

### 四路生成后端对比（`scripts/eval_compare.py`，30 条，裁判=DeepSeek）

检索 / 重排 / 评测集完全相同，只换生成后端：

| 指标 | DeepSeek(云端) | Qwen-1.5B 基座 | Qwen-1.5B + LoRA | Qwen-1.5B GGUF(fp16) |
|---|---|---|---|---|
| 忠实度 Faithfulness | 100.0% | 82.2% | 98.9% | 90.4% |
| 相关性 Relevance /5 | 4.87 | 4.03 | 4.30 | 4.43 |
| 引用准确率 Cit.Prec | 97.7% | 0.0% | 76.7% | 0.0% |
| 引用覆盖率 Cit.Cov | 100.0% | 0.0% | 100.0% | 0.0% |
| 检索 Hit@1 / 3 / 4 | 73.3 / 93.3 / 96.7% | 同 | 同 | 同 |

> 注：基座列此前因 `config.py` 里 `local_adapter_path` 默认值误指向 LoRA 目录、导致误加载 LoRA，已修复（默认值改空）并重跑修正。

四条诚实结论：

1. **检索指标四者完全一致** —— 对比实验公平性成立（hit rate 只取决于检索/重排，与生成后端无关）。
2. **引用能力是 LoRA 微调独有的**：纯基座（bf16 与 GGUF fp16 均如此）几乎不带 `[i]` 引用（覆盖率 0%），LoRA 教会了带引用（0%→100%）。
3. **LoRA 微调还带来忠实度与相关性提升**：忠实度 82.2%→98.9%（+16.7pp，学会「只依据资料、资料不足不编造」）、相关性 4.03→4.30；引用准确率 76.7% 仍低于 DeepSeek 97.7%，小数据量对「引用标得准不准」提升有限。
4. **GGUF fp16 部署不损质量**：忠实度 90.4% / 相关性 4.43，均不低于基座 bf16（82.2% / 4.03）——「转换/量化不丢精度」成立；但作为基座它同样不产引用，部署端要带引用需走「LoRA 合并 → 导出 GGUF」。

## 本地 GGUF 部署（`scripts/serve_local.bat`，llama.cpp）

用 llama.cpp 把 Qwen2.5-1.5B 部署为本地 OpenAI 兼容服务，一键启动：

```bat
scripts\serve_local.bat   # 起 http://127.0.0.1:8080/v1，fp16 GGUF，全层 offload GPU
```

- **选型**：llama.cpp（Windows 友好、一条命令起）；模型用 **Qwen2.5-1.5B-Instruct GGUF fp16**（先不量化，消除「量化掉精度」的疑虑）。
- **零改动接入**：llama-server 暴露 OpenAI 兼容端点，`LocalOpenAIClient` 复用 DeepSeek 协议封装，`llm_backend=local-http` 一键切换，业务代码不变。
- **实测**：70 tok/s（GPU 显存带宽受限），29/29 层全 offload GPU。
- **质量**：见上「四路对比」——忠实度 90.4% / 相关性 4.43，不损基座 bf16；但不产引用（基座天然行为）。部署端要带引用 → 走「LoRA 合并 → 导出 GGUF」（`llama-export-lora`）。

复现：

```bash
D:/profession/venv/Scripts/python.exe scripts/eval_rag.py --backend local-http --out data/eval_results_local_http.json
```

## 目录结构

```
rag_kb/
├── src/
│   ├── config.py              # pydantic-settings 读 .env；HF 镜像/缓存兜底
│   ├── llm.py                 # DeepSeek OpenAI 客户端，chat / chat_stream
│   ├── rag.py                 # RAGPipeline：召回→去重→重排→带引用生成（含 trace）
│   ├── api.py                 # FastAPI：/health /ask /ask/stream
│   ├── parsing/               # 多格式解析（注册式，get_parser 路由）
│   ├── chunking/              # ParentChildChunker 父子分块
│   ├── retrieval/             # sparse / dense / milvus_index / hybrid / reranker
│   └── finetune/              # LoRA 微调：data（合成数据加载）+ train（手写训练循环）
├── scripts/                   # Day1 + Week1-4 逐周验证脚本 + eval_ablation / eval_rag / eval_compare / train_lora / serve_local.bat
├── data/docs/                 # 待入库文档（PDF/DOCX/TXT/MD）
├── data/eval_set.json         # 端到端评测集（30 条：query / 参考答案 / gold 父块）
├── requirements.txt
├── PROCESS.md                 # 项目从 0 到 1 的完整实现过程（面试前通读）
└── .env                       # API Key（已 gitignore）
```

## 快速开始

```bash
# 1. 环境（已在 D 盘 venv，见下方「环境说明」）
D:/profession/venv/Scripts/python.exe -m pip install -r requirements.txt
# torch GPU 版单独装：
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 2. 配置 .env（项目根目录）
#   DEEPSEEK_API_KEY=sk-...
#   DEEPSEEK_BASE_URL=https://api.deepseek.com
#   DEEPSEEK_MODEL=deepseek-chat

# 3. 启动服务（默认入库 d:\个人\简历信息库.md）
HF_ENDPOINT=https://hf-mirror.com \
  D:/profession/venv/Scripts/python.exe -m uvicorn src.api:app --app-dir D:/profession/rag_kb --port 8000

# 4. 调用
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d '{"query":"智能客服多智能体Agent用什么框架实现？"}'
curl -N -X POST http://127.0.0.1:8000/ask/stream -H "Content-Type: application/json" \
     -d '{"query":"深度RAG知识库项目用了哪些技术栈？"}'
```

### 接口

- `GET /health` → `{"status":"ok","ready":true}`（流水线是否已构建）
- `POST /ask` → 非流式 JSON：`{answer, sources[], trace{timing_ms}}`
- `POST /ask/stream` → SSE 流：`sources`（命中父块 + 检索耗时）→ 多个 `delta`（答案 token）→ `done`（全链路耗时）

### 逐周验证

```bash
python scripts/day1_verify_gpu.py      # GPU / CUDA
python scripts/day1_test_llm.py        # DeepSeek 连通
python scripts/day1_test_embedding.py  # BGE-M3 向量
python scripts/week1_test_parse_chunk.py  # 解析 + 分块
python scripts/week2_test_retrieval.py    # 混合检索 + RRF
python scripts/week3_test_rag.py          # 重排 + 带引用生成
python scripts/week4_test_stream.py       # 流式 + trace
python scripts/week4_test_api.py          # 服务端到端（需先起服务）
python scripts/week5_test_milvus.py       # Milvus 后端对比
python scripts/eval_ablation.py           # 检索消融 + Hit Rate@k
python scripts/eval_rag.py                # 端到端 LLM-as-judge 三指标 + Hit Rate
```

## 环境说明（D 盘 / 国内网络）

- 全部落在 D 盘，不占 C 盘：venv 在 `D:\profession\venv`，模型缓存在 `D:\profession\huggingface_cache`，pip 缓存在 `D:\profession\pip_cache`。
- 国内直连 `huggingface.co` 超时，`config.py` 里已 `setdefault("HF_ENDPOINT", "https://hf-mirror.com")` 兜底走镜像，无需每次手动 export。
- `.env` 含 API Key，已加入 `.gitignore`，**切勿提交**。

## Week5：Milvus 向量库（已完成）

稠密后端从 faiss-cpu 升级到 Milvus（Milvus Lite 本地模式，`pip install pymilvus[milvus_lite]`，免 Docker）：

- **为什么升级**：faiss.IndexFlatIP 是暴力全扫（每次查询 O(N)、索引只在内存、进程退出即丢）；Milvus 用 **HNSW 图索引做近似最近邻（ANN）**，亚线性查询可撑百万级语料，且**集合落盘 .db 持久化**、生产可分布式扩展。
- **可切换**：`MilvusIndex` 与 `DenseIndex` 同一鸭子类型接口，`HybridRetriever(chunks, dense=MilvusIndex())` 一键切后端；小语料下两后端 Top5 命中与余弦分数完全一致（ANN 无召回损失）。

接口与启动方式不变，见上文「快速开始」。验证脚本 `scripts/week5_test_milvus.py`。
