"""
配置文件 - 文本到图像生成系统的配置参数
"""

import os


class Config:
    """配置类"""
    
    # 数据相关
    DATA_ROOT = "./data"  # 数据根目录
    DESCRIPTION_FILE = "description.txt"  # 描述文件
    IMAGE_DIR = "image"  # 图片目录
    CACHE_DIR = "./cache"  # 缓存目录
    
    # 文本编码
    MAX_VOCAB_SIZE = 10000  # 最大词汇表大小
    MAX_TEXT_LENGTH = 20  # 最大文本长度
    EMBEDDING_DIM = 128  # 词嵌入维度
    TEXT_HIDDEN_DIM = 256  # 文本编码器隐藏维度
    
    # 特殊标记
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    START_TOKEN = "<START>"
    END_TOKEN = "<END>"
    
    # 图像相关
    IMAGE_CHANNELS = 3  # 图像通道数
    MIN_IMAGE_SIZE = 64  # 最小图像尺寸
    MAX_IMAGE_SIZE = 512  # 最大图像尺寸
    
    # 分桶配置
    BUCKETS = [
        {"max_area": 64*64, "aspect_range": (0.5, 2.0), "target_size": 64},
        {"max_area": 128*128, "aspect_range": (0.5, 2.0), "target_size": 128},
        {"max_area": 256*256, "aspect_range": (0.5, 2.0), "target_size": 256},
        {"max_area": float('inf'), "aspect_range": (0.3, 3.0), "target_size": 512},
    ]
    
    # 生成器配置
    NOISE_DIM = 100  # 噪声维度
    GENERATOR_FILTERS = 512  # 生成器初始通道数
    GENERATOR_LAYERS = 4  # 生成器层数
    
    # 判别器配置
    DISCRIMINATOR_FILTERS = 64  # 判别器初始通道数
    DISCRIMINATOR_LAYERS = 4  # 判别器层数
    
    # 训练配置
    BATCH_SIZE = 32  # 批次大小
    EPOCHS = 100  # 训练轮数
    G_LR = 0.0002  # 生成器学习率
    D_LR = 0.0002  # 判别器学习率
    BETA1 = 0.5  # Adam优化器beta1
    BETA2 = 0.999  # Adam优化器beta2
    
    # 损失权重
    ADV_LOSS_WEIGHT = 1.0  # 对抗损失权重
    FM_LOSS_WEIGHT = 10.0  # 特征匹配损失权重
    
    # EMA配置
    EMA_DECAY = 0.999  # EMA衰减率
    
    # 检查点配置
    CHECKPOINT_DIR = "./checkpoints"  # 检查点目录
    CHECKPOINT_INTERVAL = 10  # 检查点保存间隔（轮数）
    
    # 导出配置
    EXPORT_DIR = "./saved_model"  # 导出目录
    
    # 日志配置
    LOG_DIR = "./logs"  # 日志目录
    SAMPLE_INTERVAL = 5  # 样本生成间隔（轮数）
    
    # 评估配置
    FID_INTERVAL = 10  # FID计算间隔（轮数）
    
    @classmethod
    def create_dirs(cls):
        """创建必要的目录"""
        dirs = [
            cls.CACHE_DIR,
            cls.CHECKPOINT_DIR,
            cls.EXPORT_DIR,
            cls.LOG_DIR,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
