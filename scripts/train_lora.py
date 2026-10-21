# -*- coding: utf-8 -*-
r"""LoRA bf16 微调 Qwen2.5-1.5B-Instruct（手写训练循环，不用 trl / bitsandbytes）。

跑法：
    D:\profession\venv\Scripts\python.exe scripts\train_lora.py [--epochs 3] [--batch-size 1] ...

产出：data/lora_qwen_rag/（adapter_model.safetensors + adapter_config.json + tokenizer），
后续用 src.llm.LocalModelClient(local_model_path=基座, local_adapter_path=该目录) 加载对比评测。
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.finetune.data import SFTDataset
from src.finetune.train import train_lora

MODEL_PATH = Path(r"D:\profession\models\Qwen2.5-1.5B-Instruct")
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sft_data.jsonl"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "lora_qwen_rag"

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=str(MODEL_PATH))
    ap.add_argument("--data", type=str, default=str(DATA_PATH))
    ap.add_argument("--output", type=str, default=str(OUT_DIR))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=4608,
                    help="训练序列长度上限。修复 GQA 内核后 4608 覆盖全部样本（最大 4251），零截断")
    ap.add_argument("--val-split", type=float, default=0.1, help="验证集比例（0=不切验证集）")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--no-grad-checkpoint", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("未检测到 CUDA，本脚本需要 NVIDIA GPU")

    print(f"加载基座 {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": "cuda"}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    # 梯度检查点：大幅降低长序列激活显存；配合 peft 需 enable_input_require_grads。
    if not args.no_grad_checkpoint:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    full = SFTDataset(Path(args.data), tokenizer, max_length=args.max_length)
    val_dataset = None
    if args.val_split > 0:
        n_val = max(1, int(len(full) * args.val_split))
        n_train = len(full) - n_val
        rng = random.Random(args.seed)
        idx = list(range(len(full)))
        rng.shuffle(idx)
        train_idx, val_idx = idx[:n_train], idx[n_train:]
        train_dataset = torch.utils.data.Subset(full, train_idx)
        val_dataset = torch.utils.data.Subset(full, val_idx)
        print(f"切分：train={n_train} val={len(val_idx)}", flush=True)
    else:
        train_dataset = full

    train_lora(
        model, tokenizer, train_dataset,
        val_dataset=val_dataset,
        output_dir=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
