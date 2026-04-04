"""
推理模块 - 使用训练好的模型生成图像
"""

import os
import json
import re
import logging
from typing import List, Optional, Union
from pathlib import Path

import tensorflow as tf
import numpy as np
from PIL import Image


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Text2ImageInference:
    """文本到图像推理器"""
    
    def __init__(self, 
                 model_path: str,
                 vocab_path: Optional[str] = None,
                 max_text_length: int = 20,
                 noise_dim: int = 100):
        """
        初始化推理器
        
        Args:
            model_path: 模型路径（SavedModel目录或检查点目录）
            vocab_path: 词汇表路径（可选）
            max_text_length: 最大文本长度
            noise_dim: 噪声维度
        """
        self.model_path = model_path
        self.max_text_length = max_text_length
        self.noise_dim = noise_dim
        
        # 加载词汇表
        if vocab_path is None:
            # 尝试从模型目录加载
            vocab_path = os.path.join(model_path, "assets", "vocab.json")
        
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            logger.info(f"词汇表已加载: {len(self.vocab)} 个词")
        else:
            self.vocab = None
            self.reverse_vocab = None
            logger.warning("词汇表未找到")
        
        # 特殊标记
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_id = self.vocab.get(self.pad_token, 0) if self.vocab else 0
        self.unk_id = self.vocab.get(self.unk_token, 1) if self.vocab else 1
        
        # 加载模型
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """
        加载模型
        
        Args:
            model_path: 模型路径
        """
        if os.path.isdir(model_path):
            # 检查是否是SavedModel
            if os.path.exists(os.path.join(model_path, "saved_model.pb")):
                logger.info("加载SavedModel...")
                self.model = tf.saved_model.load(model_path)
                self.model_type = "savedmodel"
            else:
                # 假设是检查点目录
                logger.info("加载检查点...")
                self._load_from_checkpoint(model_path)
                self.model_type = "checkpoint"
        else:
            raise ValueError(f"模型路径不存在: {model_path}")
    
    def _load_from_checkpoint(self, checkpoint_path: str):
        """
        从检查点加载模型
        
        Args:
            checkpoint_path: 检查点路径
        """
        from models.generator import Generator
        from models.text_encoder import create_text_encoder
        
        # 需要配置来创建模型
        # 这里简化处理，假设配置已知
        # 实际使用时应该从配置文件读取
        
        # 创建模型（使用默认配置）
        self.generator = Generator(
            noise_dim=self.noise_dim,
            condition_dim=256,
            initial_filters=512,
            num_layers=4,
        )
        
        self.text_encoder = create_text_encoder(
            encoder_type="simple",
            vocab_size=len(self.vocab) if self.vocab else 10000,
            embedding_dim=128,
            output_dim=256,
            max_length=self.max_text_length,
        )
        
        # 加载权重
        self.generator.load_weights(os.path.join(checkpoint_path, "generator.weights.h5"))
        self.text_encoder.load_weights(os.path.join(checkpoint_path, "text_encoder.weights.h5"))
        
        self.model = None
    
    def _tokenize(self, text: str) -> List[str]:
        """
        分词
        
        Args:
            text: 输入文本
            
        Returns:
            词列表
        """
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
        if self.vocab:
            ids = [self.vocab.get(token, self.unk_id) for token in tokens]
        else:
            # 如果没有词汇表，使用哈希
            ids = [hash(token) % 10000 for token in tokens]
        
        # 截断或填充
        if len(ids) > self.max_text_length:
            ids = ids[:self.max_text_length]
            mask = [1.0] * self.max_text_length
        else:
            mask = [1.0] * len(ids) + [0.0] * (self.max_text_length - len(ids))
            ids = ids + [self.pad_id] * (self.max_text_length - len(ids))
        
        return ids, mask
    
    def encode_texts(self, descriptions: List[str]) -> tuple:
        """
        编码多个文本
        
        Args:
            descriptions: 描述文本列表
            
        Returns:
            (编码序列数组, 掩码数组)元组
        """
        encoded = []
        masks = []
        
        for desc in descriptions:
            ids, mask = self._encode_text(desc)
            encoded.append(ids)
            masks.append(mask)
        
        return np.array(encoded, dtype=np.int32), np.array(masks, dtype=np.float32)
    
    def generate(self,
                 descriptions: Union[str, List[str]],
                 target_size: int = 128,
                 seed: Optional[int] = None,
                 output_path: Optional[str] = None) -> np.ndarray:
        """
        生成图像
        
        Args:
            descriptions: 描述文本或列表
            target_size: 目标图像尺寸
            seed: 随机种子
            output_path: 输出路径（可选）
            
        Returns:
            生成的图像数组 [N, H, W, 3]
        """
        # 确保是列表
        if isinstance(descriptions, str):
            descriptions = [descriptions]
        
        # 设置随机种子
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)
        
        if self.model_type == "savedmodel":
            # 使用SavedModel
            results = self.model.signatures["serving_default"](
                descriptions=tf.constant(descriptions),
                target_size=tf.constant(target_size, dtype=tf.int32),
                seed=tf.constant(seed if seed else 0, dtype=tf.int32),
            )
            generated_images = results["image"].numpy()
        else:
            # 使用检查点模型
            text_encoded, attention_mask = self.encode_texts(descriptions)
            
            text_condition = self.text_encoder(
                [text_encoded, attention_mask], training=False
            )
            
            noise = tf.random.normal([len(descriptions), self.noise_dim])
            
            generated_images = self.generator([noise, text_condition], training=False)
            generated_images = generated_images.numpy()
        
        # 转换到 [0, 255]
        generated_images = (generated_images + 1.0) * 127.5
        generated_images = np.clip(generated_images, 0, 255)
        generated_images = generated_images.astype(np.uint8)
        
        # 保存图像
        if output_path:
            self._save_images(generated_images, descriptions, output_path)
        
        return generated_images
    
    def generate_from_embedding(self,
                                 text_embeddings: np.ndarray,
                                 target_size: int = 128,
                                 seed: Optional[int] = None,
                                 output_path: Optional[str] = None) -> np.ndarray:
        """
        使用预编码的文本嵌入生成图像
        
        Args:
            text_embeddings: 文本嵌入向量 [N, embedding_dim]
            target_size: 目标图像尺寸
            seed: 随机种子
            output_path: 输出路径（可选）
            
        Returns:
            生成的图像数组 [N, H, W, 3]
        """
        # 设置随机种子
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)
        
        if self.model_type == "savedmodel":
            # 使用SavedModel
            results = self.model.signatures["generate_from_embedding"](
                text_embedding=tf.constant(text_embeddings, dtype=tf.float32),
                target_size=tf.constant(target_size, dtype=tf.int32),
                seed=tf.constant(seed if seed else 0, dtype=tf.int32),
            )
            generated_images = results["image"].numpy()
        else:
            # 使用检查点模型
            noise = tf.random.normal([len(text_embeddings), self.noise_dim])
            
            generated_images = self.generator([noise, text_embeddings], training=False)
            generated_images = generated_images.numpy()
        
        # 转换到 [0, 255]
        generated_images = (generated_images + 1.0) * 127.5
        generated_images = np.clip(generated_images, 0, 255)
        generated_images = generated_images.astype(np.uint8)
        
        # 保存图像
        if output_path:
            self._save_images(generated_images, None, output_path)
        
        return generated_images
    
    def encode_text_to_embedding(self, descriptions: Union[str, List[str]]) -> np.ndarray:
        """
        将文本编码为嵌入向量
        
        Args:
            descriptions: 描述文本或列表
            
        Returns:
            文本嵌入向量 [N, embedding_dim]
        """
        # 确保是列表
        if isinstance(descriptions, str):
            descriptions = [descriptions]
        
        if self.model_type == "savedmodel":
            # 使用SavedModel
            results = self.model.signatures["encode_text"](
                descriptions=tf.constant(descriptions)
            )
            embeddings = results["embeddings"].numpy()
        else:
            # 使用检查点模型
            text_encoded, attention_mask = self.encode_texts(descriptions)
            
            embeddings = self.text_encoder(
                [text_encoded, attention_mask], training=False
            ).numpy()
        
        return embeddings
    
    def _save_images(self, 
                     images: np.ndarray, 
                     descriptions: Optional[List[str]], 
                     output_path: str):
        """
        保存图像
        
        Args:
            images: 图像数组 [N, H, W, 3]
            descriptions: 描述文本列表
            output_path: 输出路径
        """
        os.makedirs(output_path, exist_ok=True)
        
        for i, image in enumerate(images):
            if descriptions:
                # 使用描述作为文件名
                desc = descriptions[i][:30]  # 限制长度
                desc = re.sub(r'[^\w]', '_', desc)
                filename = f"{i:03d}_{desc}.png"
            else:
                filename = f"{i:03d}.png"
            
            filepath = os.path.join(output_path, filename)
            Image.fromarray(image).save(filepath)
        
        logger.info(f"图像已保存到: {output_path}")
    
    def batch_generate(self,
                       descriptions: List[str],
                       batch_size: int = 8,
                       target_size: int = 128,
                       seed: Optional[int] = None,
                       output_path: Optional[str] = None) -> np.ndarray:
        """
        批量生成图像
        
        Args:
            descriptions: 描述文本列表
            batch_size: 批次大小
            target_size: 目标图像尺寸
            seed: 随机种子
            output_path: 输出路径（可选）
            
        Returns:
            生成的图像数组 [N, H, W, 3]
        """
        all_images = []
        
        for i in range(0, len(descriptions), batch_size):
            batch_descriptions = descriptions[i:i+batch_size]
            batch_seed = seed + i if seed is not None else None
            
            batch_images = self.generate(
                batch_descriptions,
                target_size=target_size,
                seed=batch_seed,
                output_path=None  # 不单独保存
            )
            
            all_images.append(batch_images)
        
        all_images = np.concatenate(all_images, axis=0)
        
        # 保存所有图像
        if output_path:
            self._save_images(all_images, descriptions, output_path)
        
        return all_images


def create_inference(model_path: str, **kwargs) -> Text2ImageInference:
    """
    创建推理器的工厂函数
    
    Args:
        model_path: 模型路径
        **kwargs: 其他参数
        
    Returns:
        Text2ImageInference实例
    """
    return Text2ImageInference(model_path, **kwargs)
