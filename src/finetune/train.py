# -*- coding: utf-8 -*-
"""手写 LoRA 微调训练循环（不用 trl，也不依赖 accelerate Trainer）。

设计要点：
- optimizer 只吃 requires_grad 的 LoRA 参数，参数量极小 → 8GB 显存可训 1.5B。
- bf16 混合精度：前向包 autocast，权重/激活用 bfloat16，梯度与优化器状态仍 fp32。
- 分块交叉熵：Qwen 词表 15.2w，若一次性把 (T×V) logits 升 fp32 会 OOM，
  这里按序列切成小块算 loss，只保留当前 chunk 的 fp32 张量。
- 线性 warmup + 余弦退火；梯度累积；梯度裁剪；每轮可选验证集 loss。
"""
from __future__ import annotations

import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import collate_batch

# ---- 8GB 显存关键修复：禁用原生 GQA SDPA 路径 ----
# transformers 在无 padding（attention_mask=None）时会走 enable_gqa=True 的原生 GQA 内核，
# 该内核在 torch 2.13 + RTX 4060 上把 T×T 注意力分数物化出来，显存 O(T²)：seq=4096 单次
# attention 反向峰值 ~3.4GB，直接 OOM。强制走 repeat_kv 展开路径（O(T)，~0.14GB）后，
# max_length=4608（覆盖全部样本，零截断）峰值也仅 ~5.2GB。
import transformers.integrations.sdpa_attention as _sdpa

_sdpa.use_gqa_in_sdpa = lambda attention_mask, key, value: False


def _ce_chunk(hidden_chunk: torch.Tensor, labels_chunk: torch.Tensor, lm_head, V: int) -> torch.Tensor:
    """单块 logits + cross_entropy。作为 checkpoint 的重算单元，反向时 logits 临时重建即可。

    若不 checkpoint，每个 chunk 的 fp32 logits (c×V) 会被 cross_entropy 存起来用于反向，
    15 个 chunk 加起来 ≈ 完整 logits（~2.3GB），分块等于白分。这里包一层 checkpoint，
    反向时只保留 hidden_chunk（bf16，~0.8MB），logits 在反向重算（一次 matmul），峰值只留一块。
    """
    logits = lm_head(hidden_chunk).float()  # (B, c, V) fp32，仅存在于前向/重算的瞬间
    return F.cross_entropy(logits.reshape(-1, V), labels_chunk, ignore_index=-100, reduction="sum")


def compute_loss(model, batch: dict, chunk_size: int = 256) -> torch.Tensor:
    """前向只取最后一层 hidden，再分块过 lm_head 算 CE loss。

    为什么不直接 model(**batch).loss：Qwen 词表 15.2w，序列 ~3800 token 时完整
    logits (T×V) 就要 ~1.2GB(bf16)，加上 cross_entropy 的 fp32 展开，8GB 显存会 OOM。
    这里只保留 chunk 大小的 hidden，logits 用 checkpoint 反向重算，避免存全部 fp32 logits。
    """
    base = model.get_base_model()  # 剥掉 LoRA 包装，拿到底层 Qwen2ForCausalLM

    # 无 padding 时省略 2D attention_mask，让 SDPA 走 is_causal=True 的 flash 内核。
    # 否则一个非 None 的 mask 会让 SDPA 回退到 math 内核，把 T×T 的 fp32 分数矩阵物化出来
    # （seq~4000 时 ~700MB），8GB 卡直接 OOM。batch_size=1 且各样本已左截断到自身长度 → 无 padding。
    attn_mask = batch.get("attention_mask")
    if attn_mask is not None and bool(attn_mask.all().item()):
        attn_mask = None

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden = base.model(
            input_ids=batch["input_ids"],
            attention_mask=attn_mask,
            use_cache=False,
        ).last_hidden_state  # (B, T, H) bf16，仅 ~12MB

    labels = batch["labels"]
    T = hidden.shape[1]
    V = base.lm_head.out_features
    losses: list[torch.Tensor] = []
    for i in range(0, T - 1, chunk_size):
        j = min(i + chunk_size, T - 1)
        hc = hidden[:, i:j, :]  # bf16 视图，~0.8MB
        lc = labels[:, i + 1 : j + 1].reshape(-1)
        losses.append(torch.utils.checkpoint.checkpoint(_ce_chunk, hc, lc, base.lm_head, V, use_reentrant=False))
    n_tokens = int((labels[:, 1:] != -100).sum().item())
    return torch.stack(losses).sum() / max(1, n_tokens)


def _lr_at(step: int, warmup_steps: int, total_steps: int, lr: float) -> float:
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _evaluate(model, tokenizer, val_loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            total += compute_loss(model, batch).item()
            n += 1
    model.train()
    return total / max(1, n)


def train_lora(
    model,
    tokenizer,
    dataset,
    *,
    output_dir: Path,
    val_dataset=None,
    epochs: int = 3,
    batch_size: int = 1,
    grad_accum: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.03,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    log_every: int = 5,
    device: str = "cuda",
) -> None:
    torch.manual_seed(seed)
    random.seed(seed)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_batch(tokenizer, b),
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=lambda b: collate_batch(tokenizer, b),
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)

    steps_per_epoch = math.ceil(len(dataset) / (batch_size * grad_accum))
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    print(
        f"trainable={n_trainable} samples={len(dataset)} "
        f"steps_per_epoch={steps_per_epoch} total_steps={total_steps} warmup={warmup_steps}",
        flush=True,
    )

    model.train()
    global_step = 0
    t0 = time.perf_counter()

    for epoch in range(epochs):
        running_loss = 0.0
        running_steps = 0
        optimizer.zero_grad()

        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = compute_loss(model, batch) / grad_accum
            loss.backward()

            last = i + 1 == len(loader)
            if (i + 1) % grad_accum == 0 or last:
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                for pg in optimizer.param_groups:
                    pg["lr"] = _lr_at(global_step, warmup_steps, total_steps, lr)
                running_loss += loss.item() * grad_accum
                running_steps += 1

                if global_step % log_every == 0 or last:
                    avg = running_loss / running_steps
                    elapsed = time.perf_counter() - t0
                    print(
                        f"[epoch {epoch + 1}/{epochs} step {global_step}/{total_steps}] "
                        f"loss={avg:.4f} lr={pg['lr']:.2e} {elapsed:.0f}s",
                        flush=True,
                    )
                    running_loss = 0.0
                    running_steps = 0

        if val_loader is not None:
            val_loss = _evaluate(model, tokenizer, val_loader, device)
            print(f"[epoch {epoch + 1}/{epochs}] val_loss={val_loss:.4f}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"已保存 LoRA adapter → {output_dir}", flush=True)
