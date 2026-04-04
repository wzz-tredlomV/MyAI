"""
模型导出 - 导出为SavedModel格式
"""

import os
import json
import logging
from typing import Dict, Optional

import tensorflow as tf
from tensorflow import keras

from models.generator import Generator
from models.text_encoder import create_text_encoder

from typing import Dict, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Text2ImageModel(tf.Module):
    """
    文本到图像生成模型（用于导出）
    """
    
    def __init__(self, 
                 generator: Generator,
                 text_encoder: keras.Model,
                 vocab: Dict[str, int],
                 bucket_config: list,
                 max_text_length: int = 20,
                 noise_dim: int = 100):
        """
        初始化模型
        
        Args:
            generator: 生成器
            text_encoder: 文本编码器
            vocab: 词汇表
            bucket_config: 分桶配置
            max_text_length: 最大文本长度
            noise_dim: 噪声维度
        """
        super().__init__()
        
        self.generator = generator
        self.text_encoder = text_encoder
        self.vocab = vocab
        self.bucket_config = bucket_config
        self.max_text_length = max_text_length
        self.noise_dim = noise_dim
        
        # 特殊标记
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_id = vocab.get(self.pad_token, 0)
        self.unk_id = vocab.get(self.unk_token, 1)
    
    def _tokenize(self, text: str) -> list:
        """分词"""
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = [t for t in text.split() if t]
        return tokens
    
    def _encode_text(self, text: str) -> tuple:
        """
        编码单个文本
        
        Args:
            text: 输入文本
            
        Returns:
            (编码序列, 掩码)元组
        """
        tokens = self._tokenize(text)
        
        # 词转ID
        ids = [self.vocab.get(token, self.unk_id) for token in tokens]
        
        # 截断或填充
        if len(ids) > self.max_text_length:
            ids = ids[:self.max_text_length]
            mask = [1.0] * self.max_text_length
        else:
            mask = [1.0] * len(ids) + [0.0] * (self.max_text_length - len(ids))
            ids = ids + [self.pad_id] * (self.max_text_length - len(ids))
        
        return ids, mask
    
    def _get_bucket_size(self, target_size: int) -> int:
        """
        获取最接近的分桶尺寸
        
        Args:
            target_size: 目标尺寸
            
        Returns:
            分桶尺寸
        """
        available_sizes = [config["target_size"] for config in self.bucket_config]
        
        # 找到最接近的尺寸
        closest_size = min(available_sizes, key=lambda x: abs(x - target_size))
        return closest_size
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name="descriptions"),
        tf.TensorSpec(shape=[], dtype=tf.int32, name="target_size"),
        tf.TensorSpec(shape=[], dtype=tf.int32, name="seed"),
    ])
    def serving_default(self, descriptions: tf.Tensor, 
                        target_size: tf.Tensor,
                        seed: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        标准生成接口
        
        Args:
            descriptions: 描述文本列表 [N]
            target_size: 目标图像尺寸
            seed: 随机种子
            
        Returns:
            生成的图像和使用的分辨率
        """
        batch_size = tf.shape(descriptions)[0]
        
        # 设置随机种子
        if seed > 0:
            tf.random.set_seed(seed)
        
        # 编码文本（使用py_function处理字符串）
        def encode_texts(texts):
            texts = [t.decode('utf-8') for t in texts.numpy()]
            encoded = []
            masks = []
            for text in texts:
                ids, mask = self._encode_text(text)
                encoded.append(ids)
                masks.append(mask)
            return tf.constant(encoded, dtype=tf.int32), tf.constant(masks, dtype=tf.float32)
        
        text_encoded, attention_mask = tf.py_function(
            encode_texts, [descriptions], [tf.int32, tf.float32]
        )
        text_encoded.set_shape([None, self.max_text_length])
        attention_mask.set_shape([None, self.max_text_length])
        
        # 获取文本条件
        text_condition = self.text_encoder(
            [text_encoded, attention_mask], training=False
        )
        
        # 生成噪声
        noise = tf.random.normal([batch_size, self.noise_dim])
        
        # 生成图像
        generated_images = self.generator([noise, text_condition], training=False)
        
        # 获取实际使用的分辨率
        used_resolution = tf.shape(generated_images)[1]
        
        return {
            "image": generated_images,
            "used_resolution": used_resolution,
        }
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name="descriptions"),
    ])
    def encode_text(self, descriptions: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        预编码文本接口
        
        Args:
            descriptions: 描述文本列表 [N]
            
        Returns:
            文本嵌入向量
        """
        # 编码文本
        def encode_texts(texts):
            texts = [t.decode('utf-8') for t in texts.numpy()]
            encoded = []
            masks = []
            for text in texts:
                ids, mask = self._encode_text(text)
                encoded.append(ids)
                masks.append(mask)
            return tf.constant(encoded, dtype=tf.int32), tf.constant(masks, dtype=tf.float32)
        
        text_encoded, attention_mask = tf.py_function(
            encode_texts, [descriptions], [tf.int32, tf.float32]
        )
        text_encoded.set_shape([None, self.max_text_length])
        attention_mask.set_shape([None, self.max_text_length])
        
        # 获取文本嵌入
        text_embedding = self.text_encoder(
            [text_encoded, attention_mask], training=False
        )
        
        return {"embeddings": text_embedding}
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="text_embedding"),
        tf.TensorSpec(shape=[], dtype=tf.int32, name="target_size"),
        tf.TensorSpec(shape=[], dtype=tf.int32, name="seed"),
    ])
    def generate_from_embedding(self, text_embedding: tf.Tensor,
                                 target_size: tf.Tensor,
                                 seed: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        使用预编码向量生成图像
        
        Args:
            text_embedding: 文本嵌入向量 [N, embedding_dim]
            target_size: 目标图像尺寸
            seed: 随机种子
            
        Returns:
            生成的图像
        """
        batch_size = tf.shape(text_embedding)[0]
        
        # 设置随机种子
        if seed > 0:
            tf.random.set_seed(seed)
        
        # 生成噪声
        noise = tf.random.normal([batch_size, self.noise_dim])
        
        # 生成图像
        generated_images = self.generator([noise, text_embedding], training=False)
        
        return {"image": generated_images}


def export_savedmodel(generator: Generator,
                      text_encoder: keras.Model,
                      vocab: Dict[str, int],
                      bucket_config: list,
                      export_dir: str = "./saved_model",
                      max_text_length: int = 20,
                      noise_dim: int = 100):
    """
    导出SavedModel
    
    Args:
        generator: 生成器
        text_encoder: 文本编码器
        vocab: 词汇表
        bucket_config: 分桶配置
        export_dir: 导出目录
        max_text_length: 最大文本长度
        noise_dim: 噪声维度
    """
    os.makedirs(export_dir, exist_ok=True)
    
    # 创建导出模型
    model = Text2ImageModel(
        generator=generator,
        text_encoder=text_encoder,
        vocab=vocab,
        bucket_config=bucket_config,
        max_text_length=max_text_length,
        noise_dim=noise_dim,
    )
    
    # 导出
    tf.saved_model.save(
        model,
        export_dir,
        signatures={
            "serving_default": model.serving_default,
            "encode_text": model.encode_text,
            "generate_from_embedding": model.generate_from_embedding,
        }
    )
    
    # 保存词汇表和配置
    assets_dir = os.path.join(export_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    with open(os.path.join(assets_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(assets_dir, "bucket_config.json"), "w") as f:
        json.dump(bucket_config, f, indent=2)
    
    special_tokens = {
        "PAD": "<PAD>",
        "UNK": "<UNK>",
        "START": "<START>",
        "END": "<END>",
    }
    with open(os.path.join(assets_dir, "special_tokens.json"), "w") as f:
        json.dump(special_tokens, f, indent=2)
    
    logger.info(f"SavedModel 已导出到: {export_dir}")

def load_savedmodel(export_dir: str) -> Any:
    """
    加载SavedModel
    
    Args:
        export_dir: 导出目录
        
    Returns:
        加载的模型
    """
    model = tf.saved_model.load(export_dir)
    logger.info(f"SavedModel 已从 {export_dir} 加载")
    return model
