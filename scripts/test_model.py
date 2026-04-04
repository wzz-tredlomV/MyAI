"""
模型测试脚本 - 测试各个组件
"""

import os
import sys
import argparse
from pathlib import Path

import tensorflow as tf
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from models.generator import create_generator
from models.discriminator import create_discriminator
from models.text_encoder import create_text_encoder
from utils.dataset_parser import DatasetParser
from utils.data_pipeline import create_data_pipeline


def test_text_encoder():
    """测试文本编码器"""
    print("\n" + "=" * 50)
    print("测试文本编码器")
    print("=" * 50)
    
    vocab_size = 1000
    batch_size = 4
    max_length = 20
    
    # 创建编码器
    encoder = create_text_encoder(
        encoder_type="simple",
        vocab_size=vocab_size,
        embedding_dim=128,
        output_dim=256,
        max_length=max_length,
    )
    
    # 创建随机输入
    text_encoded = tf.random.uniform([batch_size, max_length], 0, vocab_size, dtype=tf.int32)
    attention_mask = tf.ones([batch_size, max_length], dtype=tf.float32)
    
    # 前向传播
    output = encoder([text_encoded, attention_mask], training=False)
    
    print(f"输入形状: {text_encoded.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{tf.reduce_min(output):.3f}, {tf.reduce_max(output):.3f}]")
    print("✓ 文本编码器测试通过")
    
    return encoder


def test_generator(text_encoder):
    """测试生成器"""
    print("\n" + "=" * 50)
    print("测试生成器")
    print("=" * 50)
    
    batch_size = 2
    noise_dim = 100
    condition_dim = 256
    
    # 创建生成器
    generator = create_generator(
        noise_dim=noise_dim,
        condition_dim=condition_dim,
        initial_filters=512,
        num_layers=4,
        output_channels=3,
    )
    
    # 创建随机输入
    noise = tf.random.normal([batch_size, noise_dim])
    condition = tf.random.normal([batch_size, condition_dim])
    
    # 前向传播
    output = generator([noise, condition], training=False)
    
    print(f"噪声形状: {noise.shape}")
    print(f"条件形状: {condition.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{tf.reduce_min(output):.3f}, {tf.reduce_max(output):.3f}]")
    print("✓ 生成器测试通过")
    
    return generator


def test_discriminator():
    """测试判别器"""
    print("\n" + "=" * 50)
    print("测试判别器")
    print("=" * 50)
    
    batch_size = 2
    image_size = 64
    condition_dim = 256
    
    # 创建判别器
    discriminator = create_discriminator(
        condition_dim=condition_dim,
        initial_filters=64,
        num_layers=4,
    )
    
    # 创建随机输入
    image = tf.random.normal([batch_size, image_size, image_size, 3])
    condition = tf.random.normal([batch_size, condition_dim])
    
    # 前向传播
    output = discriminator([image, condition], training=False)
    
    print(f"图像形状: {image.shape}")
    print(f"条件形状: {condition.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{tf.reduce_min(output):.3f}, {tf.reduce_max(output):.3f}]")
    print("✓ 判别器测试通过")
    
    return discriminator


def test_dataset_parser(data_root: str = "./data"):
    """测试数据集解析器"""
    print("\n" + "=" * 50)
    print("测试数据集解析器")
    print("=" * 50)
    
    if not os.path.exists(data_root):
        print(f"数据目录不存在: {data_root}")
        print("请先运行 create_sample_data.py 创建示例数据")
        return None
    
    # 创建解析器
    parser = DatasetParser(
        root_dir=data_root,
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
    
    print(f"数据集大小: {len(dataset_index)}")
    print(f"词汇表大小: {parser.get_vocab_size()}")
    
    # 打印统计信息
    stats = parser.get_statistics()
    print("数据集统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 打印第一个样本
    if dataset_index:
        sample = dataset_index[0]
        print("\n第一个样本:")
        print(f"  路径: {sample['rel_path']}")
        print(f"  描述: {sample['description']}")
        print(f"  尺寸: {sample['image_size']}")
        print(f"  编码: {sample['encoded_text'][:10]}...")
    
    print("✓ 数据集解析器测试通过")
    
    return parser


def test_data_pipeline(parser):
    """测试数据管道"""
    print("\n" + "=" * 50)
    print("测试数据管道")
    print("=" * 50)
    
    if parser is None or not parser.dataset_index:
        print("没有可用的数据集索引")
        return
    
    config = Config()
    
    # 创建数据管道
    pipeline = create_data_pipeline(
        dataset_index=parser.dataset_index[:50],  # 使用部分数据测试
        buckets_config=config.BUCKETS,
        batch_size=4,
        max_text_length=20,
    )
    
    # 测试迭代
    print("测试数据迭代...")
    count = 0
    for bucket_id, batch in pipeline.get_dataset_iterator():
        print(f"  Bucket {bucket_id}: image shape = {batch['image'].shape}")
        count += 1
        if count >= 3:
            break
    
    print(f"✓ 数据管道测试通过 (迭代了 {count} 个批次)")


def test_training_step(generator, discriminator, text_encoder):
    """测试训练步骤"""
    print("\n" + "=" * 50)
    print("测试训练步骤")
    print("=" * 50)
    
    batch_size = 2
    image_size = 64
    noise_dim = 100
    condition_dim = 256
    max_length = 20
    
    # 创建优化器
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    
    # 创建随机数据
    real_images = tf.random.normal([batch_size, image_size, image_size, 3])
    text_encoded = tf.random.uniform([batch_size, max_length], 0, 1000, dtype=tf.int32)
    attention_mask = tf.ones([batch_size, max_length], dtype=tf.float32)
    
    # 训练判别器
    with tf.GradientTape() as tape:
        text_condition = text_encoder([text_encoded, attention_mask], training=True)
        noise = tf.random.normal([batch_size, noise_dim])
        fake_images = generator([noise, text_condition], training=True)
        
        real_logits = discriminator([real_images, text_condition], training=True)
        fake_logits = discriminator([fake_images, text_condition], training=True)
        
        d_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_logits)) + \
                 tf.reduce_mean(tf.nn.relu(1.0 + fake_logits))
    
    d_gradients = tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
    
    print(f"D Loss: {d_loss:.4f}")
    
    # 训练生成器
    with tf.GradientTape() as tape:
        text_condition = text_encoder([text_encoded, attention_mask], training=True)
        noise = tf.random.normal([batch_size, noise_dim])
        fake_images = generator([noise, text_condition], training=True)
        
        fake_logits = discriminator([fake_images, text_condition], training=True)
        
        g_loss = -tf.reduce_mean(fake_logits)
    
    g_gradients = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
    
    print(f"G Loss: {g_loss:.4f}")
    print("✓ 训练步骤测试通过")


def main():
    parser = argparse.ArgumentParser(description="测试模型组件")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="数据根目录")
    parser.add_argument("--skip_data", action="store_true",
                        help="跳过数据测试")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("文本到图像生成模型 - 组件测试")
    print("=" * 50)
    
    # 测试文本编码器
    text_encoder = test_text_encoder()
    
    # 测试生成器
    generator = test_generator(text_encoder)
    
    # 测试判别器
    discriminator = test_discriminator()
    
    # 测试数据集解析器
    if not args.skip_data:
        dataset_parser = test_dataset_parser(args.data_root)
        
        # 测试数据管道
        if dataset_parser:
            test_data_pipeline(dataset_parser)
    
    # 测试训练步骤
    test_training_step(generator, discriminator, text_encoder)
    
    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)


if __name__ == "__main__":
    main()
