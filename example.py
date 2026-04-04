"""
示例脚本 - 演示如何使用文本到图像生成系统
"""

import os
import sys
from pathlib import Path

import tensorflow as tf
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from models.generator import create_generator
from models.discriminator import create_discriminator
from models.text_encoder import create_text_encoder
from utils.dataset_parser import DatasetParser
from utils.data_pipeline import create_data_pipeline
from utils.trainer import create_trainer
from utils.export import export_savedmodel
from utils.inference import create_inference


def example_1_build_models():
    """示例1: 构建模型"""
    print("\n" + "=" * 60)
    print("示例1: 构建模型")
    print("=" * 60)
    
    config = Config()
    vocab_size = 1000
    
    # 创建文本编码器
    text_encoder = create_text_encoder(
        encoder_type="simple",
        vocab_size=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        output_dim=config.TEXT_HIDDEN_DIM,
        max_length=config.MAX_TEXT_LENGTH,
    )
    print(f"✓ 文本编码器创建成功，参数数量: {text_encoder.count_params():,}")
    
    # 创建生成器
    generator = create_generator(
        noise_dim=config.NOISE_DIM,
        condition_dim=config.TEXT_HIDDEN_DIM,
        initial_filters=config.GENERATOR_FILTERS,
        num_layers=config.GENERATOR_LAYERS,
        output_channels=config.IMAGE_CHANNELS,
    )
    print(f"✓ 生成器创建成功，参数数量: {generator.count_params():,}")
    
    # 创建判别器
    discriminator = create_discriminator(
        condition_dim=config.TEXT_HIDDEN_DIM,
        initial_filters=config.DISCRIMINATOR_FILTERS,
        num_layers=config.DISCRIMINATOR_LAYERS,
    )
    print(f"✓ 判别器创建成功，参数数量: {discriminator.count_params():,}")
    
    return generator, discriminator, text_encoder


def example_2_parse_dataset():
    """示例2: 解析数据集"""
    print("\n" + "=" * 60)
    print("示例2: 解析数据集")
    print("=" * 60)
    
    # 首先创建示例数据
    if not os.path.exists("./data/description.txt"):
        print("创建示例数据...")
        from scripts.create_sample_data import create_sample_dataset
        create_sample_dataset("./data", num_samples=50)
    
    # 创建解析器
    parser = DatasetParser(
        root_dir="./data",
        description_file="description.txt",
        cache_dir="./cache",
    )
    
    # 构建索引
    special_tokens = {
        "PAD": "<PAD>",
        "UNK": "<UNK>",
        "START": "<START>",
        "END": "<END>",
    }
    
    dataset_index = parser.build_index(
        max_vocab_size=1000,
        max_text_length=20,
        special_tokens=special_tokens,
        use_cache=True,
    )
    
    print(f"✓ 数据集解析完成")
    print(f"  - 样本数量: {len(dataset_index)}")
    print(f"  - 词汇表大小: {parser.get_vocab_size()}")
    
    # 打印统计信息
    stats = parser.get_statistics()
    print("  - 数据集统计:")
    for key, value in stats.items():
        print(f"    {key}: {value}")
    
    return parser


def example_3_create_data_pipeline(parser):
    """示例3: 创建数据管道"""
    print("\n" + "=" * 60)
    print("示例3: 创建数据管道")
    print("=" * 60)
    
    config = Config()
    
    # 创建数据管道
    pipeline = create_data_pipeline(
        dataset_index=parser.dataset_index,
        buckets_config=config.BUCKETS,
        batch_size=4,
        max_text_length=20,
    )
    
    print(f"✓ 数据管道创建成功")
    print(f"  - 分桶数量: {len(pipeline.datasets)}")
    
    # 测试迭代
    print("  - 测试数据迭代:")
    for i, (bucket_id, batch) in enumerate(pipeline.get_dataset_iterator()):
        print(f"    Bucket {bucket_id}: image shape = {batch['image'].shape}, "
              f"text shape = {batch['text_encoded'].shape}")
        if i >= 2:
            break
    
    return pipeline


def example_4_train_one_step(generator, discriminator, text_encoder):
    """示例4: 单步训练"""
    print("\n" + "=" * 60)
    print("示例4: 单步训练")
    print("=" * 60)
    
    config = Config()
    
    # 创建训练器
    trainer = create_trainer(
        config=config,
        generator=generator,
        discriminator=discriminator,
        text_encoder=text_encoder,
    )
    
    # 创建模拟数据
    batch = {
        "image": tf.random.normal([4, 64, 64, 3]),
        "text_encoded": tf.random.uniform([4, 20], 0, 1000, dtype=tf.int32),
        "attention_mask": tf.ones([4, 20], dtype=tf.float32),
        "description": tf.constant(["test"] * 4),
    }
    
    # 执行训练步骤
    losses = trainer.train_step(batch)
    
    print(f"✓ 训练步骤完成")
    print(f"  - D Loss: {losses['d_loss']:.4f}")
    print(f"  - G Loss: {losses['g_loss']:.4f}")
    print(f"  - D Real: {losses['d_real']:.4f}")
    print(f"  - D Fake: {losses['d_fake']:.4f}")
    
    return trainer


def example_5_generate_image(generator, text_encoder):
    """示例5: 生成图像"""
    print("\n" + "=" * 60)
    print("示例5: 生成图像")
    print("=" * 60)
    
    # 创建模拟的文本编码
    text_encoded = tf.random.uniform([1, 20], 0, 1000, dtype=tf.int32)
    attention_mask = tf.ones([1, 20], dtype=tf.float32)
    
    # 编码文本
    text_condition = text_encoder([text_encoded, attention_mask], training=False)
    
    # 生成噪声
    noise = tf.random.normal([1, 100])
    
    # 生成图像
    generated_image = generator([noise, text_condition], training=False)
    
    print(f"✓ 图像生成成功")
    print(f"  - 输出形状: {generated_image.shape}")
    print(f"  - 输出范围: [{tf.reduce_min(generated_image):.3f}, {tf.reduce_max(generated_image):.3f}]")
    
    # 保存图像
    os.makedirs("./outputs", exist_ok=True)
    
    # 转换到 [0, 255]
    image = (generated_image[0] + 1.0) * 127.5
    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, tf.uint8)
    
    # 保存
    from PIL import Image
    Image.fromarray(image.numpy()).save("./outputs/example_generated.png")
    print(f"  - 图像已保存到: ./outputs/example_generated.png")


def example_6_export_model(generator, text_encoder):
    """示例6: 导出模型"""
    print("\n" + "=" * 60)
    print("示例6: 导出模型")
    print("=" * 60)
    
    config = Config()
    
    # 创建模拟词汇表
    vocab = {f"word_{i}": i for i in range(100)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    
    # 导出模型
    export_savedmodel(
        generator=generator,
        text_encoder=text_encoder,
        vocab=vocab,
        bucket_config=config.BUCKETS,
        export_dir="./saved_model_example",
        max_text_length=config.MAX_TEXT_LENGTH,
        noise_dim=config.NOISE_DIM,
    )
    
    print(f"✓ 模型导出成功")
    print(f"  - 导出目录: ./saved_model_example")


def main():
    """主函数"""
    print("=" * 60)
    print("文本到图像生成系统 - 示例演示")
    print("=" * 60)
    
    # 示例1: 构建模型
    generator, discriminator, text_encoder = example_1_build_models()
    
    # 示例2: 解析数据集
    parser = example_2_parse_dataset()
    
    # 示例3: 创建数据管道
    if parser and parser.dataset_index:
        pipeline = example_3_create_data_pipeline(parser)
    
    # 示例4: 单步训练
    trainer = example_4_train_one_step(generator, discriminator, text_encoder)
    
    # 示例5: 生成图像
    example_5_generate_image(generator, text_encoder)
    
    # 示例6: 导出模型
    example_6_export_model(generator, text_encoder)
    
    print("\n" + "=" * 60)
    print("所有示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
