# rag-kb 升级计划：「检索 + 微调 + 评测 + 部署」闭环

> 目标：把现有 RAG 项目升级成 AI 应用岗可深挖的完整作品。
> 制定日期：2026-10-16。环境实测已通过（见「环境实测」）。

## 定位（一句话故事）

> 从零实现工业级 RAG：混合检索 + 本地微调大模型 + 端到端评测 + 一键部署，每一层都能讲清「为什么」，且都有量化数据支撑。

覆盖 AI 应用岗四个面试考点：**检索工程、微调、评测、工程化交付**。

## 现状 vs 目标

| 维度 | 现状 | 目标（本次补） |
|---|---|---|
| 检索 | ✅ 混合检索 + RRF + 重排 + Hit Rate 消融 | 保持 |
| 评测 | ⚠️ 只有检索侧 Hit Rate | 补端到端：忠实度 / 相关性 / 引用准确率（LLM-as-judge） |
| 模型 | ❌ 纯调 DeepSeek 闭源 API | 补本地 LoRA 微调 + 本地推理 |
| 部署 | ⚠️ FastAPI 裸跑 | 补 Ollama / llama.cpp serving（Docker 可选加分） |

## 环境实测（可行性已验证）

| 项 | 结果 |
|---|---|
| GPU | ✅ RTX 4060 Laptop 8GB，`cuda_available=True` |
| torch | ✅ 2.13.0+cu126 |
| 微调依赖 | ✅ peft 0.20 / transformers 5.15 / accelerate / datasets 已装 |
| bitsandbytes | ❌ 未装 → 方案改为**纯 LoRA bf16**（1.5B 无需量化也进 8GB），绕开 Windows 兼容坑 |
| trl | ❌ 未装 → **手写训练循环**（peft 已够），不依赖 SFTTrainer 黑盒 |
| Docker | ❌ 未装 → Phase 3 改用 **Ollama**，Docker 化作可选加分项 |

## 三步计划（依赖顺序，不可调换）

### Phase 1 — 端到端评测（先做，零 GPU 零风险）

- 新增 `scripts/eval_rag.py`，用 **LLM-as-judge**（DeepSeek 当裁判）度量三个端到端指标：
  1. **忠实度 Faithfulness**：答案是否都来自检索资料、有没有编造
  2. **答案相关性 Answer Relevance**：答没答到点上
  3. **引用准确率 Citation Accuracy**：`[i]` 标得对不对
- **自实现 judge**（不直接 `pip install ragas`），证明懂指标怎么算，面试加分远高于「我调了 RAGAS」。
- 把现有 20 条 ground truth 扩到 ~30 条，每条补「参考答案 + 相关父块」，存 `data/eval_set.json`。
- **交付**：评测脚本 + 评测集 + 指标表（写进 README）。

### Phase 2 — 本地 LoRA 微调（重头戏，需 GPU）

- 微调 **Qwen2.5-1.5B-Instruct** 做「带引用的 RAG 问答」，作为**本地生成模型**替换 DeepSeek。
- 训练数据：DeepSeek 从简历语料**合成 ~300 条**，**加校验**（每条 `[i]` 引用必须真实存在于给定资料，不合格丢弃）。
- **关键前置**：`llm.py` 抽象出 `LLMClient` 接口，`DeepSeekClient` / `LocalModelClient` 可切换，否则本地模型接不进现有流水线。
- **交付**：`src/finetune/` 模块 + `scripts/train_lora.py` + 保存的 LoRA adapter + **微调 vs 基座 vs DeepSeek 三路对比评测**（复用 Phase 1）。
- **预期管理**：价值 80% 在「走通 pipeline + 讲清每个环节」，20% 在涨点；评测重点量化**忠实度 / 引用准确率**的改善，不盯通用问答质量。

### Phase 3 — 部署（收尾）

- 本地模型 serving：**Ollama / llama.cpp（GGUF 量化）**，Windows 友好、一条命令起。
- Docker 化作可选加分项（需用户装 Docker Desktop）。
- **交付**：一键启动脚本 + README「部署」章节。

## 关键决策（已定）

1. **微调模型**：Qwen2.5-1.5B-Instruct（8GB 稳跑；pipeline 模型无关，可升 3B/7B）。
2. **数据来源**：DeepSeek 合成 + 引用校验过滤。

## 风险与回退

| 风险 | 回退 |
|---|---|
| 模型下载失败 | hf-mirror 多镜像重试 / 手动下载放 cache |
| 训练 OOM | 降 batch size / 开 gradient checkpointing / 换 0.5B。**已定位根因**：transformers 无 padding 时走原生 GQA SDPA 内核（`enable_gqa=True`），torch 2.13 上显存 O(T²)。修复=在 `src/finetune/train.py` monkeypatch `use_gqa_in_sdpa→False` 强制 `repeat_kv` 路径，峰值 7.9GB→5.2GB，`max_length=4608` 零截断 |
| 合成数据质量差 | 提高校验阈值 + 人工抽检；必要时改用公开中文 RAG 数据集 |
| 微调无提升 | 换子任务（query 改写）或如实报告「pipeline 能力」而非涨点 |
| Ollama 不可用 | llama.cpp 直跑 GGUF / transformers 4-bit 起本地服务 |

## 进度

- [x] Phase 1 端到端评测（2026-10-17：`scripts/eval_rag.py` + `data/eval_set.json` 30 条；忠实度 100% / 相关性 4.93/5 / 引用准确率 97.7%；检索 Hit@1 73.3%）
- [x] Phase 2 LoRA 微调（2026-10-20：`src/finetune/` + `scripts/train_lora.py` 完成；合成数据 294 条；3 epoch val_loss 0.3787→0.3582，adapter 存 `data/lora_qwen_rag`；四路对比评测完成，结论见下）
- [x] Phase 3 部署（2026-10-20：llama.cpp + GGUF fp16 一键启动 `scripts/serve_local.bat`，OpenAI 兼容端点，70 tok/s；GGUF 忠实度 90.4% / 相关性 4.43 不损基座）

### 四路生成后端对比结果（30 条，裁判=DeepSeek，检索/重排完全相同）

| 指标 | DeepSeek | Qwen-1.5B 基座 | Qwen-1.5B + LoRA | Qwen-1.5B GGUF(fp16) |
|---|---|---|---|---|
| 忠实度 Faithfulness | 100.0% | 82.2% | 98.9% | 90.4% |
| 相关性 Relevance /5 | 4.87 | 4.03 | 4.30 | 4.43 |
| 引用准确率 Cit.Prec | 97.7% | 0.0% | 76.7% | 0.0% |
| 引用覆盖率 Cit.Cov | 100.0% | 0.0% | 100.0% | 0.0% |
| 检索 Hit@1/3/4 | 73.3/93.3/96.7% | 同 | 同 | 同 |

> 注：基座列此前因 `config.py` 里 `local_adapter_path` 默认值误指向 LoRA 目录、导致误加载 LoRA，已修复（默认值改空）并重跑修正。

**结论（如实）**：① 检索指标四者一致 → 对比公平性成立；② 引用能力是 LoRA 独有——纯基座（bf16 与 GGUF fp16）都不产引用（覆盖率 0%）；③ **LoRA 微调教会带引用 + 抗编造**：引用覆盖率 0%→100%、忠实度 82.2%→98.9%（+16.7pp）、相关性 4.03→4.30；引用准确率 76.7%（< DeepSeek 97.7%）说明小数据量对「引用标得准不准」提升有限；④ **GGUF fp16 部署不损质量**：忠实度 90.4% / 相关性 4.43 ≥ 基座 bf16。
