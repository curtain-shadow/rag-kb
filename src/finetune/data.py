# -*- coding: utf-8 -*-
"""微调数据管线：读 data/sft_data.jsonl → 套 chat template → tokenize → 标签掩码。

每条样本是 generate_sft_data.py 产出的 {"messages": [system, user, assistant]}，
user 内容与线上 rag.py 完全同构：`【参考资料】\n[i] ... \n\n【问题】\n query`。
训练时只有 assistant 回复参与 loss，其余位置 labels=-100。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def build_example(tokenizer, messages: list[dict], max_length: int) -> dict:
    """把一条 messages 转成 input_ids / attention_mask / labels。"""
    prompt = [m for m in messages if m["role"] != "assistant"]
    answer = next(m["content"] for m in messages if m["role"] == "assistant")

    prompt_ids = tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True
    )["input_ids"]
    # 回答末尾追加 eos，教模型在该处停止生成。
    answer_ids = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > max_length:
        # 超长时左截断：从参考资料头部丢弃，保留尾部的【问题】与完整回答。
        # 默认 max_length 已覆盖当前全部样本（最大 4251），此分支通常不触发。
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]
        attention_mask = attention_mask[-max_length:]

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def collate_batch(tokenizer, batch: list[dict]) -> dict:
    """右 padding 到批内最长，attention_mask 区分真实 token 与 pad。"""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attention_mask = [], [], []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_id] * pad)
        labels.append(x["labels"] + [-100] * pad)
        attention_mask.append(x["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


class SFTDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int = 4608):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: list[list[dict]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line)["messages"])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return build_example(self.tokenizer, self.examples[idx], self.max_length)
