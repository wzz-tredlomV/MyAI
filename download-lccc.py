#!/usr/bin/env python3
"""
中文对话数据集下载与处理脚本
数据集：LCCC (Large-scale Cleaned Chinese Conversation)
作者：清华大学AI实验室
来源：HuggingFace - thu-coai/lccc
格式：转换为项目所需的 conversations.json
"""

import os
import json
import argparse
from tqdm import tqdm

# 检查并安装依赖
try:
    from datasets import load_dataset
except ImportError:
    print("错误：未找到 datasets 库，请先安装：pip install datasets")
    exit(1)


def download_and_process_lccc(
    output_dir: str = "./data",
    version: str = "base",
    max_samples: int = None,
    min_turn_length: int = 5,
    max_turn_length: int = 200,
    merge_multi_turn: bool = True
):
    """
    下载并处理 LCCC 数据集
    
    Args:
        output_dir: 输出目录
        version: 数据集版本，可选 "base" 或 "large"
        max_samples: 最大样本数（None 表示全部）
        min_turn_length: 单轮对话的最小字符长度
        max_turn_length: 单轮对话的最大字符长度
        merge_multi_turn: 是否尝试合并多轮对话
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "conversations.json")
    
    print(f"正在从 HuggingFace 加载 LCCC-{version} 数据集...")
    print("首次运行会自动下载，请耐心等待...")
    
    try:
        # 加载数据集
        dataset = load_dataset("lccc", version, trust_remote_code=True)
    except Exception as e:
        print(f"加载失败: {e}")
        print("\n备选方案：")
        print("1. 手动下载 LCCC 数据集：")
        print("   - 百度网盘: https://pan.baidu.com/s/1szmNZQrwh9y994uO8DFL_A 提取码: f2ex")
        print("   - 或访问: https://github.com/thu-coai/CDial-GPT")
        print("2. 解压后将数据文件放到 data/ 目录下")
        print("3. 修改本脚本以从本地文件加载")
        return None
    
    # 数据集结构说明
    print(f"\n数据集加载成功！")
    print(f"可用 split: {list(dataset.keys())}")
    
    # LCCC 数据集通常只有 'train' split
    train_data = dataset['train']
    print(f"训练集样本数: {len(train_data)}")
    
    # 查看数据结构
    print(f"数据字段: {train_data.column_names}")
    
    # 处理数据
    conversations = []
    total_processed = 0
    total_filtered = 0
    
    # 限制样本数
    iterator = train_data
    if max_samples:
        iterator = train_data.select(range(min(max_samples, len(train_data))))
    
    print(f"\n开始处理数据...")
    for item in tqdm(iterator, desc="处理对话"):
        # LCCC 数据格式：每个样本包含多轮对话
        # 字段可能为 ['dialog'] 或 ['conversation']
        dialog = None
        if 'dialog' in item:
            dialog = item['dialog']
        elif 'conversation' in item:
            dialog = item['conversation']
        else:
            # 尝试第一个字段
            first_key = list(item.keys())[0]
            dialog = item[first_key]
        
        if not dialog or len(dialog) < 2:
            total_filtered += 1
            continue
        
        # 清理和过滤对话
        cleaned_dialog = []
        for turn in dialog:
            turn = turn.strip()
            # 过滤太短或太长的句子
            if min_turn_length <= len(turn) <= max_turn_length:
                cleaned_dialog.append(turn)
            else:
                total_filtered += 1
                break
        
        if len(cleaned_dialog) < 2:
            continue
        
        # 将多轮对话转换为问答对格式
        if merge_multi_turn and len(cleaned_dialog) >= 2:
            # 合并连续的多轮对话为完整对话段
            # 格式: [[q1, a1], [q2, a2], ...]
            conv_turns = []
            for i in range(0, len(cleaned_dialog) - 1, 2):
                if i + 1 < len(cleaned_dialog):
                    conv_turns.append([cleaned_dialog[i], cleaned_dialog[i + 1]])
            if len(conv_turns) >= 1:
                conversations.append(conv_turns)
                total_processed += len(conv_turns)
        else:
            # 不合并，每对单独作为独立对话
            for i in range(0, len(cleaned_dialog) - 1, 2):
                if i + 1 < len(cleaned_dialog):
                    conversations.append([[cleaned_dialog[i], cleaned_dialog[i + 1]]])
                    total_processed += 1
    
    # 保存为 JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    
    print(f"\n处理完成！")
    print(f"输出文件: {output_file}")
    print(f"对话段数: {len(conversations)}")
    print(f"问答对数: {total_processed}")
    print(f"过滤样本数: {total_filtered}")
    
    # 打印样例
    if conversations:
        print("\n样例对话:")
        sample = conversations[0]
        for i, (q, a) in enumerate(sample):
            print(f"  轮次 {i+1}:")
            print(f"    Q: {q[:50]}..." if len(q) > 50 else f"    Q: {q}")
            print(f"    A: {a[:50]}..." if len(a) > 50 else f"    A: {a}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="下载并处理 LCCC 中文对话数据集")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="输出目录 (默认: ./data)")
    parser.add_argument("--version", type=str, default="base",
                        choices=["base", "large"],
                        help="数据集版本: base (680万对话) 或 large (1200万对话)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大处理样本数 (默认: 全部)")
    parser.add_argument("--min_length", type=int, default=3,
                        help="单轮最小字符长度 (默认: 3)")
    parser.add_argument("--max_length", type=int, default=300,
                        help="单轮最大字符长度 (默认: 300)")
    parser.add_argument("--no_merge", action="store_true",
                        help="不合并多轮对话（每对问答独立）")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LCCC 中文对话数据集下载与处理脚本")
    print("=" * 60)
    print(f"版本: LCCC-{args.version}")
    print(f"输出目录: {args.output_dir}")
    print(f"合并多轮: {'否' if args.no_merge else '是'}")
    print("=" * 60)
    
    download_and_process_lccc(
        output_dir=args.output_dir,
        version=args.version,
        max_samples=args.max_samples,
        min_turn_length=args.min_length,
        max_turn_length=args.max_length,
        merge_multi_turn=not args.no_merge
    )


if __name__ == "__main__":
    main()