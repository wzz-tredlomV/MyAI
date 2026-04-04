"""
数据管道 - 分桶策略和tf.data.Dataset构建
"""

import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import tensorflow as tf
import numpy as np
from PIL import Image


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BucketManager:
    """分桶管理器"""
    
    def __init__(self, buckets_config: List[Dict]):
        """
        初始化分桶管理器
        
        Args:
            buckets_config: 分桶配置列表
                每个元素包含:
                - max_area: 最大面积
                - aspect_range: 宽高比范围 (min, max)
                - target_size: 目标输出尺寸
        """
        self.bucket_config = buckets_config
        
    def assign_bucket(self, width: int, height: int) -> int:
        """
        根据图像尺寸分配桶
        
        Args:
            width: 图像宽度
            height: 图像高度
            
        Returns:
            桶ID
        """
        area = width * height
        aspect = width / height if height > 0 else 1.0
        
        for bucket_id, config in enumerate(self.bucket_config):
            max_area = config["max_area"]
            aspect_min, aspect_max = config["aspect_range"]
            
            if area <= max_area and aspect_min <= aspect <= aspect_max:
                return bucket_id
        
        # 默认分配到最后一桶
        return len(self.bucket_config) - 1
    
    def get_target_size(self, bucket_id: int) -> int:
        """获取桶的目标尺寸"""
        return self.bucket_config[bucket_id]["target_size"]


class DataPipeline:
    """数据管道"""
    
    def __init__(self, dataset_index: List[Dict], bucket_manager: BucketManager,
                 batch_size: int = 32, max_text_length: int = 20):
        """
        初始化数据管道
        
        Args:
            dataset_index: 数据集索引
            bucket_manager: 分桶管理器
            batch_size: 批次大小
            max_text_length: 最大文本长度
        """
        self.dataset_index = dataset_index
        self.bucket_manager = bucket_manager
        self.batch_size = batch_size
        self.max_text_length = max_text_length
        
        self.bucketed_data: Dict[int, List[Dict]] = defaultdict(list)
        self.datasets: Dict[int, tf.data.Dataset] = {}
        
    def assign_buckets(self):
        """将数据分配到各个桶"""
        logger.info("开始分桶...")
        
        for entry in self.dataset_index:
            width, height = entry["image_size"]
            bucket_id = self.bucket_manager.assign_bucket(width, height)
            entry["bucket_id"] = bucket_id
            self.bucketed_data[bucket_id].append(entry)
        
        # 打印分桶统计
        for bucket_id, entries in self.bucketed_data.items():
            target_size = self.bucket_manager.get_target_size(bucket_id)
            logger.info(f"桶 {bucket_id} (目标尺寸 {target_size}): {len(entries)} 个样本")
        
        # 处理样本过少的桶（合并到相邻桶）
        self._merge_small_buckets(min_samples=self.batch_size * 2)
    
    def _merge_small_buckets(self, min_samples: int = 64):
        """
        合并样本过少的桶
        
        Args:
            min_samples: 最小样本数阈值
        """
        bucket_ids = sorted(self.bucketed_data.keys())
        
        for bucket_id in bucket_ids:
            if bucket_id not in self.bucketed_data:
                continue
            
            entries = self.bucketed_data[bucket_id]
            if len(entries) < min_samples:
                # 找到相邻的桶进行合并
                target_bucket = None
                if bucket_id > 0 and (bucket_id - 1) in self.bucketed_data:
                    target_bucket = bucket_id - 1
                elif bucket_id < len(self.bucket_manager.bucket_config) - 1:
                    target_bucket = bucket_id + 1
                
                if target_bucket is not None and target_bucket in self.bucketed_data:
                    logger.info(f"桶 {bucket_id} 样本过少 ({len(entries)}), 合并到桶 {target_bucket}")
                    self.bucketed_data[target_bucket].extend(entries)
                    del self.bucketed_data[bucket_id]
    
    def _load_and_preprocess_image(self, abs_path: str, target_size: int) -> tf.Tensor:
        """
        加载并预处理图片
        
        Args:
            abs_path: 图片绝对路径
            target_size: 目标尺寸
            
        Returns:
            预处理后的图像张量 [H, W, 3]
        """
        # 读取图片
        image = tf.io.read_file(abs_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        
        # 归一化到 [-1, 1]
        image = (image / 127.5) - 1.0
        
        # 获取原始尺寸
        orig_height = tf.shape(image)[0]
        orig_width = tf.shape(image)[1]
        
        # 计算缩放比例，保持宽高比
        scale = target_size / tf.maximum(orig_height, orig_width)
        new_height = tf.cast(tf.cast(orig_height, tf.float32) * scale, tf.int32)
        new_width = tf.cast(tf.cast(orig_width, tf.float32) * scale, tf.int32)
        
        # 缩放
        image = tf.image.resize(image, [new_height, new_width], method='bilinear')
        
        # 填充到目标尺寸（居中）
        pad_top = (target_size - new_height) // 2
        pad_bottom = target_size - new_height - pad_top
        pad_left = (target_size - new_width) // 2
        pad_right = target_size - new_width - pad_left
        
        image = tf.pad(image, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], 
                       mode='CONSTANT', constant_values=-1.0)
        
        # 确保形状正确
        image.set_shape([target_size, target_size, 3])
        
        return image
    
    def _create_generator(self, bucket_id: int):
        """
        为指定桶创建数据生成器
        
        Args:
            bucket_id: 桶ID
            
        Yields:
            (image, text_encoded, attention_mask)元组
        """
        entries = self.bucketed_data[bucket_id]
        target_size = self.bucket_manager.get_target_size(bucket_id)
        
        for entry in entries:
            abs_path = entry["abs_path"]
            encoded_text = entry["encoded_text"]
            attention_mask = entry["attention_mask"]
            
            try:
                # 加载图片
                image = self._load_and_preprocess_image_py(abs_path, target_size)
                
                yield {
                    "image": image,
                    "text_encoded": np.array(encoded_text, dtype=np.int32),
                    "attention_mask": np.array(attention_mask, dtype=np.float32),
                    "description": entry["description"],
                }
            except Exception as e:
                logger.warning(f"加载图片失败 {abs_path}: {e}")
                continue
    
    def _load_and_preprocess_image_py(self, abs_path: str, target_size: int) -> np.ndarray:
        """
        Python版本的图片加载和预处理（用于生成器）
        
        Args:
            abs_path: 图片绝对路径
            target_size: 目标尺寸
            
        Returns:
            预处理后的图像数组 [H, W, 3]
        """
        # 使用PIL加载图片
        with Image.open(abs_path) as img:
            img = img.convert('RGB')
            orig_width, orig_height = img.size
            
            # 计算缩放比例
            scale = target_size / max(orig_width, orig_height)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            
            # 缩放
            img = img.resize((new_width, new_height), Image.BILINEAR)
            
            # 创建目标尺寸画布（灰色背景）
            result = np.ones((target_size, target_size, 3), dtype=np.float32) * -1.0
            
            # 居中粘贴
            pad_top = (target_size - new_height) // 2
            pad_left = (target_size - new_width) // 2
            
            img_array = np.array(img, dtype=np.float32)
            img_array = (img_array / 127.5) - 1.0  # 归一化到 [-1, 1]
            
            result[pad_top:pad_top+new_height, pad_left:pad_left+new_width] = img_array
            
        return result
    
    def build_datasets(self) -> Dict[int, tf.data.Dataset]:
        """
        为每个桶构建tf.data.Dataset
        
        Returns:
            桶ID到数据集的映射
        """
        self.assign_buckets()
        
        for bucket_id in self.bucketed_data.keys():
            target_size = self.bucket_manager.get_target_size(bucket_id)
            entries = self.bucketed_data[bucket_id]
            
            if len(entries) == 0:
                continue
            
            # 创建数据集
            output_signature = {
                "image": tf.TensorSpec(shape=(target_size, target_size, 3), dtype=tf.float32),
                "text_encoded": tf.TensorSpec(shape=(self.max_text_length,), dtype=tf.int32),
                "attention_mask": tf.TensorSpec(shape=(self.max_text_length,), dtype=tf.float32),
                "description": tf.TensorSpec(shape=(), dtype=tf.string),
            }
            
            dataset = tf.data.Dataset.from_generator(
                lambda bid=bucket_id: self._create_generator(bid),
                output_signature=output_signature
            )
            
            # 打乱
            dataset = dataset.shuffle(buffer_size=min(len(entries), 1000))
            
            # 批次化（同桶样本，动态填充）
            dataset = dataset.batch(self.batch_size, drop_remainder=True)
            
            # 预取
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            self.datasets[bucket_id] = dataset
            logger.info(f"桶 {bucket_id} 数据集构建完成: {len(entries)} 个样本")
        
        return self.datasets
    
    def get_dataset_iterator(self):
        """
        获取所有桶的数据集迭代器
        
        Yields:
            (bucket_id, batch)元组
        """
        # 创建迭代器列表
        iterators = {
            bucket_id: iter(dataset) 
            for bucket_id, dataset in self.datasets.items()
        }
        
        # 循环 yield 各桶的数据
        while iterators:
            for bucket_id in list(iterators.keys()):
                try:
                    batch = next(iterators[bucket_id])
                    yield bucket_id, batch
                except StopIteration:
                    del iterators[bucket_id]


def create_data_pipeline(dataset_index: List[Dict], 
                         buckets_config: List[Dict],
                         batch_size: int = 32,
                         max_text_length: int = 20) -> DataPipeline:
    """
    创建数据管道的便捷函数
    
    Args:
        dataset_index: 数据集索引
        buckets_config: 分桶配置
        batch_size: 批次大小
        max_text_length: 最大文本长度
        
    Returns:
        DataPipeline实例
    """
    bucket_manager = BucketManager(buckets_config)
    pipeline = DataPipeline(dataset_index, bucket_manager, batch_size, max_text_length)
    pipeline.build_datasets()
    return pipeline
