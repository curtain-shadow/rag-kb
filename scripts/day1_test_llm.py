# -*- coding: utf-8 -*-
"""Day1 冒烟测试 2：测通 DeepSeek API（OpenAI 兼容协议）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from src.config import settings

client = OpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
)

resp = client.chat.completions.create(
    model=settings.deepseek_model,
    messages=[
        {"role": "system", "content": "你是严谨的问答助手，回答要基于事实。"},
        {"role": "user", "content": "用一句话介绍重庆邮电大学。"},
    ],
    temperature=0.7,
)

print("model:", resp.model)
print("回答:", resp.choices[0].message.content)
print("usage:", resp.usage)
