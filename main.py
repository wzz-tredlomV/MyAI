#!/usr/bin/env python3
"""
Handwriting Mimic AI - 矢量字迹模仿系统
主入口程序

Usage:
    python main.py extract --image sample.jpg --style-id my_style
    python main.py generate --text "你好世界" --style-id my_style --output output/
    python main.py list-styles
    python main.py train --dataset data/train/ --epochs 100
"""
import argparse
import sys
from pathlib import Path
from typing import Optional
import json

# 确保可以导入src模块
sys.path.insert(0, str(Path(__file__).parent))

from src.vector_style_encoder import (
    HandwritingStyleEncoder, 
    StyleBank, 
    extract_and_save_style
)
from src.vector_generator import VectorHandwritingPipeline


def setup_argparse() -> argparse.ArgumentParser:
    """设置命令行参数"""
    parser = argparse.ArgumentParser(
        description='Handwriting Mimic AI - 矢量字迹模仿系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从图片提取字迹风格
  python main.py extract -i samples/my_writing.jpg -s my_style

  # 生成模仿字迹（矢量SVG输出）
  python main.py generate -t "你好世界" -s my_style -o output/

  # 列出所有保存的风格
  python main.py list-styles

  # 训练模型
  python main.py train -d data/train/ -e 100 --save-model models/model.pth
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # extract 命令
    extract_parser = subparsers.add_parser('extract', help='从图片提取字迹风格')
    extract_parser.add_argument('-i', '--image', required=True, help='手写样本图片路径')
    extract_parser.add_argument('-s', '--style-id', required=True, help='风格标识名')
    extract_parser.add_argument('--model', default=None, help='预训练模型路径（可选）')
    extract_parser.add_argument('--bank', default='data/style_bank', help='风格银行路径')
    extract_parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='计算设备')

    # generate 命令
    generate_parser = subparsers.add_parser('generate', help='生成模仿字迹')
    generate_parser.add_argument('-t', '--text', required=True, help='要生成的文字')
    generate_parser.add_argument('-s', '--style-id', required=True, help='风格ID')
    generate_parser.add_argument('-o', '--output', default='output', help='输出目录')
    generate_parser.add_argument('--model', default=None, help='生成器模型路径')
    generate_parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='计算设备')
    generate_parser.add_argument('--no-combine', action='store_true', help='不合并为单个SVG')

    # list-styles 命令
    subparsers.add_parser('list-styles', help='列出所有保存的风格')

    # train 命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('-d', '--dataset', required=True, help='训练数据集目录')
    train_parser.add_argument('-e', '--epochs', type=int, default=100, help='训练轮数')
    train_parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    train_parser.add_argument('--lr', type=float, default=0.0002, help='学习率')
    train_parser.add_argument('--save-model', default='models/handwriting_model.pth', help='模型保存路径')
    train_parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='计算设备')

    # info 命令
    info_parser = subparsers.add_parser('info', help='查看风格信息')
    info_parser.add_argument('-s', '--style-id', required=True, help='风格ID')

    return parser


def cmd_extract(args):
    """执行提取风格命令"""
    print(f"正在从 {args.image} 提取风格 '{args.style_id}'...")

    try:
        style_dict = extract_and_save_style(
            image_path=args.image,
            style_id=args.style_id,
            model_path=args.model,
            bank_path=args.bank,
            device=args.device
        )

        print(f"\n风格提取成功！")
        print(f"  风格ID: {style_dict['style_id']}")
        print(f"  全局维度: {style_dict['metadata']['global_dim']}")
        print(f"  局部维度: {style_dict['metadata']['local_dim']}")
        print(f"  笔画维度: {style_dict['metadata']['stroke_dim']}")
        print(f"  统计特征: {style_dict['statistical_features'][0][:5]}...")  # 显示前5个
        print(f"\n风格已保存到: data/style_bank/{args.style_id}.json")

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_generate(args):
    """执行生成命令"""
    print(f"正在使用风格 '{args.style_id}' 生成文字: {args.text}")

    try:
        pipeline = VectorHandwritingPipeline(
            model_path=args.model,
            device=args.device
        )

        output_paths = pipeline.generate_text(
            text=args.text,
            style_id=args.style_id,
            output_dir=args.output,
            combine=not args.no_combine
        )

        print(f"\n生成完成！")
        print(f"  输出文件:")
        for path in output_paths:
            print(f"    - {path}")

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_list_styles(args):
    """列出所有风格"""
    bank = StyleBank()
    styles = bank.list_styles()

    if not styles:
        print("风格银行为空，请先使用 'extract' 命令提取风格。")
        return

    print(f"已保存的风格 ({len(styles)} 个):")
    print("-" * 50)

    for style_id in styles:
        style_info = bank.load_style(style_id)
        meta = style_info.get('metadata', {})
        print(f"  {style_id}")
        print(f"    全局维度: {meta.get('global_dim', 'N/A')}")
        print(f"    版本: {meta.get('model_version', 'N/A')}")
        print(f"    样本图片: {style_info.get('sample_image', 'N/A')}")
        print()


def cmd_train(args):
    """执行训练命令"""
    print(f"开始训练模型...")
    print(f"  数据集: {args.dataset}")
    print(f"  轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  设备: {args.device}")

    # 这里应该实现完整的训练循环
    # 为简化，这里只输出训练配置
    print("\n[注意] 完整训练实现需要准备成对的手写数据集")
    print("训练流程:")
    print("  1. 加载数据集")
    print("  2. 训练风格编码器")
    print("  3. 训练生成器（对抗训练）")
    print("  4. 保存模型")

    # 创建模型保存目录
    model_path = Path(args.save_model)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n模型将保存到: {args.save_model}")


def cmd_info(args):
    """查看风格详细信息"""
    try:
        bank = StyleBank()
        style = bank.load_style(args.style_id)

        print(f"风格信息: {args.style_id}")
        print("=" * 50)
        print(json.dumps(style, indent=2, ensure_ascii=False))

    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """主函数"""
    parser = setup_argparse()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # 路由到对应命令
    commands = {
        'extract': cmd_extract,
        'generate': cmd_generate,
        'list-styles': cmd_list_styles,
        'train': cmd_train,
        'info': cmd_info,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        print(f"未知命令: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
