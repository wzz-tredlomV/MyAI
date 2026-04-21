#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_lccc.py

下载并处理 LCCC-base 数据集，输出格式：
[
    [
        ["用户话1", "机器人话1"],
        ["用户话2", "机器人话2"],
        ...
    ],
    ...
]
"""

import json
import os
import sys
from typing import List, Tuple

# 如果尚未安装 datasets，请先执行：pip install datasets
try:
    from datasets import load_dataset
except ImportError:
    print("请先安装 datasets 库：pip install datasets")
    sys.exit(1)


def convert_dialog_to_pairs(dialog: List[str]) -> List[Tuple[str, str]]:
    """
    将单条对话（list of strings）转换为 (用户, 机器人) 对列表。
    假设对话由偶数句组成，且用户先发言。
    """
    pairs = []
    # 每次取两句，组成 (用户, 机器人) 对
    for i in range(0, len(dialog) - 1, 2):
        user_utt = dialog[i].strip()
        bot_utt = dialog[i + 1].strip()
        # 过滤掉空字符串（如果有）
        if user_utt and bot_utt:
            pairs.append([user_utt, bot_utt])
    # 如果对话总句数为奇数，丢弃最后一句
    return pairs


def process_split(dataset, split_name: str) -> List[List[Tuple[str, str]]]:
    """处理单个 split（train/validation/test），返回转换后的对话列表。"""
    conversations = []
    total_dialogs = len(dataset)
    valid_dialogs = 0

    print(f"  处理 {split_name} 集，共 {total_dialogs} 条对话...")
    for idx, sample in enumerate(dataset):
        dialog = sample["dialog"]
        pairs = convert_dialog_to_pairs(dialog)
        # 只保留至少有一轮有效对话的样本
        if pairs:
            conversations.append(pairs)
            valid_dialogs += 1

        # 每处理 10000 条输出一次进度
        if (idx + 1) % 10000 == 0:
            print(f"    已处理 {idx + 1}/{total_dialogs} 条...")

    print(f"    {split_name} 集有效对话数: {valid_dialogs}/{total_dialogs}")
    return conversations


def main():
    # 配置参数
    DATASET_NAME = "lccc"
    SUBSET = "base"               # 或 "large"
    OUTPUT_DIR = "data"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "lccc_base_conversations.json")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"正在从 Hugging Face Hub 加载 {DATASET_NAME}/{SUBSET} 数据集...")
    try:
        # 加载数据集的所有 splits
        dataset = load_dataset(DATASET_NAME, SUBSET)
    except Exception as e:
        print(f"加载失败: {e}")
        print("提示：如果网络不稳定，可以设置环境变量 HF_ENDPOINT=https://hf-mirror.com 使用镜像。")
        sys.exit(1)

    print("可用的 splits:", list(dataset.keys()))

    # 存储所有对话
    all_conversations = []

    # 按顺序处理 train, validation, test
    for split_name in ["train", "validation", "test"]:
        if split_name in dataset:
            convs = process_split(dataset[split_name], split_name)
            all_conversations.extend(convs)

    print(f"\n总计有效对话数: {len(all_conversations)}")

    # 保存为 JSON 文件
    print(f"正在保存到 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)

    print("完成！")
    # 输出文件大小
    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"文件大小: {file_size:.2f} MB")


if __name__ == "__main__":
    main()