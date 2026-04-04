"""
主训练脚本 - 文本到图像生成模型训练
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import tensorflow as tf

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from utils.dataset_parser import DatasetParser
from utils.data_pipeline import create_data_pipeline
from utils.trainer import create_trainer
from utils.export import export_savedmodel
from models.generator import create_generator
from models.discriminator import create_discriminator
from models.text_encoder import create_text_encoder


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练文本到图像生成模型")
    
    # 数据相关
    parser.add_argument("--data_root", type=str, default="./data",
                        help="数据根目录")
    parser.add_argument("--description_file", type=str, default="description.txt",
                        help="描述文件名")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="缓存目录")
    
    # 模型相关
    parser.add_argument("--encoder_type", type=str, default="simple",
                        choices=["simple", "transformer"],
                        help="文本编码器类型")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="检查点目录")
    
    # 训练相关
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--g_lr", type=float, default=0.0002,
                        help="生成器学习率")
    parser.add_argument("--d_lr", type=float, default=0.0002,
                        help="判别器学习率")
    parser.add_argument("--steps_per_epoch", type=int, default=None,
                        help="每轮步数")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                        help="检查点保存间隔")
    
    # 其他
    parser.add_argument("--resume", type=str, default=None,
                        help="从检查点恢复训练")
    parser.add_argument("--export", action="store_true",
                        help="训练完成后导出SavedModel")
    parser.add_argument("--export_dir", type=str, default="./saved_model",
                        help="导出目录")
    
    return parser.parse_args()


def setup_config(args):
    """
    根据命令行参数设置配置
    
    Args:
        args: 命令行参数
        
    Returns:
        配置对象
    """
    config = Config()
    
    # 更新配置
    config.DATA_ROOT = args.data_root
    config.DESCRIPTION_FILE = args.description_file
    config.CACHE_DIR = args.cache_dir
    config.CHECKPOINT_DIR = args.checkpoint_dir
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.G_LR = args.g_lr
    config.D_LR = args.d_lr
    config.CHECKPOINT_INTERVAL = args.checkpoint_interval
    config.EXPORT_DIR = args.export_dir
    
    # 创建目录
    config.create_dirs()
    
    return config


def build_models(config, vocab_size):
    """
    构建模型
    
    Args:
        config: 配置对象
        vocab_size: 词汇表大小
        
    Returns:
        (生成器, 判别器, 文本编码器)元组
    """
    logger.info("构建模型...")
    
    # 文本编码器
    text_encoder = create_text_encoder(
        encoder_type=config.encoder_type if hasattr(config, 'encoder_type') else "simple",
        vocab_size=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        output_dim=config.TEXT_HIDDEN_DIM,
        max_length=config.MAX_TEXT_LENGTH,
    )
    # 构建文本编码器
    dummy_input = tf.zeros((1, config.MAX_TEXT_LENGTH), dtype=tf.int32)
    dummy_mask = tf.ones((1, config.MAX_TEXT_LENGTH), dtype=tf.float32)
    _ = text_encoder([dummy_input, dummy_mask])
    logger.info(f"文本编码器: {text_encoder.count_params()} 参数")
    
    # 生成器
    generator = create_generator(
        noise_dim=config.NOISE_DIM,
        condition_dim=config.TEXT_HIDDEN_DIM,
        initial_filters=config.GENERATOR_FILTERS,
        num_layers=config.GENERATOR_LAYERS,
        output_channels=config.IMAGE_CHANNELS,
    )
    # 构建生成器
    dummy_noise = tf.zeros((1, config.NOISE_DIM), dtype=tf.float32)
    dummy_condition = tf.zeros((1, config.TEXT_HIDDEN_DIM), dtype=tf.float32)
    _ = generator([dummy_noise, dummy_condition])
    logger.info(f"生成器: {generator.count_params()} 参数")
    
    # 判别器
    discriminator = create_discriminator(
        condition_dim=config.TEXT_HIDDEN_DIM,
        initial_filters=config.DISCRIMINATOR_FILTERS,
        num_layers=config.DISCRIMINATOR_LAYERS,
    )
    # 构建判别器
    dummy_image = tf.zeros((1, 256, 256, 3), dtype=tf.float32)
    dummy_condition = tf.zeros((1, config.TEXT_HIDDEN_DIM), dtype=tf.float32)
    _ = discriminator([dummy_image, dummy_condition])
    logger.info(f"判别器: {discriminator.count_params()} 参数")
    
    return generator, discriminator, text_encoder


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置配置
    config = setup_config(args)
    
    logger.info("=" * 50)
    logger.info("文本到图像生成模型训练")
    logger.info("=" * 50)
    
    # 检查GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"检测到 {len(gpus)} 个GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logger.info("使用CPU训练")
    
    # 解析数据集
    logger.info("解析数据集...")
    parser = DatasetParser(
        root_dir=config.DATA_ROOT,
        description_file=config.DESCRIPTION_FILE,
        cache_dir=config.CACHE_DIR,
    )
    
    special_tokens = {
        "PAD": config.PAD_TOKEN,
        "UNK": config.UNK_TOKEN,
        "START": config.START_TOKEN,
        "END": config.END_TOKEN,
    }
    
    dataset_index = parser.build_index(
        max_vocab_size=config.MAX_VOCAB_SIZE,
        max_text_length=config.MAX_TEXT_LENGTH,
        special_tokens=special_tokens,
        use_cache=True,
    )
    
    vocab_size = parser.get_vocab_size()
    logger.info(f"词汇表大小: {vocab_size}")
    
    # 打印统计信息
    stats = parser.get_statistics()
    logger.info("数据集统计:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # 构建数据管道
    logger.info("构建数据管道...")
    data_pipeline = create_data_pipeline(
        dataset_index=dataset_index,
        buckets_config=config.BUCKETS,
        batch_size=config.BATCH_SIZE,
        max_text_length=config.MAX_TEXT_LENGTH,
    )
    
    # 构建模型
    generator, discriminator, text_encoder = build_models(config, vocab_size)
    
    # 创建训练器
    logger.info("创建训练器...")
    trainer = create_trainer(
        config=config,
        generator=generator,
        discriminator=discriminator,
        text_encoder=text_encoder,
    )
    
    # 从检查点恢复
    if args.resume:
        logger.info(f"从检查点恢复: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 训练循环
    logger.info("开始训练...")
    for epoch in range(config.EPOCHS):
        # 训练一个epoch
        losses = trainer.train_epoch(data_pipeline, args.steps_per_epoch)
        
        # 保存检查点
        if (epoch + 1) % config.CHECKPOINT_INTERVAL == 0:
            trainer.save_checkpoint(f"checkpoint_epoch_{epoch + 1}")
    
    # 保存最终检查点
    trainer.save_checkpoint("checkpoint_final")
    
    # 导出SavedModel
    if args.export:
        logger.info("导出SavedModel...")
        export_savedmodel(
            generator=generator,
            text_encoder=text_encoder,
            vocab=parser.vocab,
            bucket_config=config.BUCKETS,
            export_dir=config.EXPORT_DIR,
            max_text_length=config.MAX_TEXT_LENGTH,
            noise_dim=config.NOISE_DIM,
        )
    
    logger.info("训练完成!")


if __name__ == "__main__":
    main()
