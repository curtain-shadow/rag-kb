# -*- coding: utf-8 -*-
"""Day1 冒烟测试 1：验证 torch GPU 可用（必须走 D 盘 venv）"""
import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda version:", torch.version.cuda)
    print("device name:", torch.cuda.get_device_name(0))
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"total VRAM: {vram_gb:.1f} GB")
    x = torch.randn(1024, 1024, device="cuda")
    y = (x @ x).sum()
    print("GPU 矩阵运算完成，结果为有限值:", round(y.item()), "(非 0 / 非 NaN 即正常)")
else:
    print("!!! GPU 不可用，检查 venv 内 torch 是否 cu126 版")
