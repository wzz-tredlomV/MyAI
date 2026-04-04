"""
推理脚本 - 使用训练好的模型生成图像
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from utils.inference import create_inference


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="文本到图像生成推理")
    
    # 模型路径
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径（SavedModel目录或检查点目录）")
    parser.add_argument("--vocab_path", type=str, default=None,
                        help="词汇表路径（可选）")
    
    # 输入
    parser.add_argument("--description", type=str, default=None,
                        help="单个描述文本")
    parser.add_argument("--description_file", type=str, default=None,
                        help="描述文件路径（每行一个描述）")
    
    # 生成参数
    parser.add_argument("--target_size", type=int, default=128,
                        help="目标图像尺寸")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批次大小")
    
    # 输出
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="输出目录")
    
    return parser.parse_args()


def load_descriptions(args) -> list:
    """
    加载描述文本
    
    Args:
        args: 命令行参数
        
    Returns:
        描述文本列表
    """
    descriptions = []
    
    if args.description:
        descriptions.append(args.description)
    
    if args.description_file:
        with open(args.description_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    descriptions.append(line)
    
    if not descriptions:
        # 使用默认描述
        descriptions = [
            "a red apple on a wooden table",
            "a blue car parked on the street",
            "a white cat sleeping on a sofa",
        ]
        logger.info("使用默认描述")
    
    return descriptions


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    logger.info("=" * 50)
    logger.info("文本到图像生成推理")
    logger.info("=" * 50)
    
    # 加载描述
    descriptions = load_descriptions(args)
    logger.info(f"加载了 {len(descriptions)} 个描述")
    
    # 创建推理器
    logger.info(f"加载模型: {args.model_path}")
    inference = create_inference(
        model_path=args.model_path,
        vocab_path=args.vocab_path,
    )
    
    # 生成图像
    logger.info("生成图像...")
    images = inference.batch_generate(
        descriptions=descriptions,
        batch_size=args.batch_size,
        target_size=args.target_size,
        seed=args.seed,
        output_path=args.output_dir,
    )
    
    logger.info(f"生成完成: {len(images)} 张图像")
    logger.info(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
