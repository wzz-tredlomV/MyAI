"""
数据集解析器 - 解析description.txt并构建索引
"""

import os
import re
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

import numpy as np
from PIL import Image


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetParser:
    """数据集解析器"""
    
    def __init__(self, root_dir: str, description_file: str = "description.txt", 
                 cache_dir: str = "./cache"):
        """
        初始化数据集解析器
        
        Args:
            root_dir: 数据根目录
            description_file: 描述文件名
            cache_dir: 缓存目录
        """
        self.root_dir = Path(root_dir).resolve()
        self.description_file = self.root_dir / description_file
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_index: List[Dict] = []
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        
    def parse_description_file(self) -> List[Tuple[str, str]]:
        """
        解析description.txt文件
        
        Returns:
            列表，每个元素为(相对路径, 描述文本)的元组
        """
        if not self.description_file.exists():
            raise FileNotFoundError(f"描述文件不存在: {self.description_file}")
        
        entries = []
        separator = " --- "
        
        logger.info(f"正在解析描述文件: {self.description_file}")
        
        with open(self.description_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # 严格匹配分隔符
                if separator not in line:
                    logger.warning(f"第{line_num}行格式错误，跳过: {line[:50]}...")
                    continue
                
                parts = line.split(separator, 1)
                if len(parts) != 2:
                    logger.warning(f"第{line_num}行分割失败，跳过")
                    continue
                
                rel_path, description = parts
                rel_path = rel_path.strip()
                description = description.strip()
                
                # 过滤空描述
                if not description:
                    logger.warning(f"第{line_num}行描述为空，跳过")
                    continue
                
                entries.append((rel_path, description))
        
        logger.info(f"成功解析 {len(entries)} 条记录")
        return entries
    
    def validate_paths(self, entries: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
        """
        验证路径并转换为绝对路径
        
        Args:
            entries: (相对路径, 描述文本)列表
            
        Returns:
            (相对路径, 绝对路径, 描述文本)列表
        """
        validated = []
        missing_count = 0
        
        for rel_path, description in entries:
            # 统一路径分隔符
            rel_path = os.path.normpath(rel_path)
            abs_path = self.root_dir / rel_path
            
            if not abs_path.exists():
                logger.warning(f"图片文件不存在，跳过: {abs_path}")
                missing_count += 1
                continue
            
            validated.append((rel_path, str(abs_path), description))
        
        logger.info(f"路径验证完成: {len(validated)} 条有效, {missing_count} 条缺失")
        return validated
    
    def scan_image_metadata(self, abs_path: str) -> Optional[Dict]:
        """
        扫描图片元数据
        
        Args:
            abs_path: 图片绝对路径
            
        Returns:
            元数据字典，失败返回None
        """
        try:
            with Image.open(abs_path) as img:
                return {
                    "size": img.size,  # (width, height)
                    "mode": img.mode,
                    "format": img.format,
                }
        except Exception as e:
            logger.warning(f"读取图片元数据失败 {abs_path}: {e}")
            return None
    
    def tokenize(self, text: str) -> List[str]:
        """
        分词：小写、去标点、分割
        
        Args:
            text: 输入文本
            
        Returns:
            词列表
        """
        # 转小写
        text = text.lower()
        # 去除标点，保留字母数字和空格
        text = re.sub(r'[^\w\s]', ' ', text)
        # 分割并过滤空字符串
        tokens = [t for t in text.split() if t]
        return tokens
    
    def build_vocabulary(self, descriptions: List[str], max_vocab_size: int = 10000,
                         special_tokens: Dict[str, str] = None) -> Dict[str, int]:
        """
        构建词汇表
        
        Args:
            descriptions: 描述文本列表
            max_vocab_size: 最大词汇表大小
            special_tokens: 特殊标记字典
            
        Returns:
            词汇表字典
        """
        # 统计词频
        counter = Counter()
        for desc in descriptions:
            tokens = self.tokenize(desc)
            counter.update(tokens)
        
        # 保留高频词
        most_common = counter.most_common(max_vocab_size - len(special_tokens or {}))
        
        # 构建词汇表
        vocab = {}
        idx = 0
        
        # 先添加特殊标记
        if special_tokens:
            for token_name, token_str in special_tokens.items():
                vocab[token_str] = idx
                idx += 1
        
        # 添加普通词
        for word, _ in most_common:
            if word not in vocab:
                vocab[word] = idx
                idx += 1
        
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        
        logger.info(f"词汇表构建完成: {len(vocab)} 个词")
        return vocab
    
    def encode_text(self, text: str, max_length: int = 20, 
                    pad_token: str = "<PAD>", unk_token: str = "<UNK>") -> Tuple[List[int], List[int]]:
        """
        编码文本为ID序列
        
        Args:
            text: 输入文本
            max_length: 最大序列长度
            pad_token: 填充标记
            unk_token: 未知词标记
            
        Returns:
            (编码序列, 注意力掩码)元组
        """
        tokens = self.tokenize(text)
        
        # 词转ID
        pad_id = self.vocab.get(pad_token, 0)
        unk_id = self.vocab.get(unk_token, 1)
        
        ids = [self.vocab.get(token, unk_id) for token in tokens]
        
        # 截断或填充
        if len(ids) > max_length:
            ids = ids[:max_length]
            mask = [1] * max_length
        else:
            mask = [1] * len(ids) + [0] * (max_length - len(ids))
            ids = ids + [pad_id] * (max_length - len(ids))
        
        return ids, mask
    
    def build_index(self, max_vocab_size: int = 10000, max_text_length: int = 20,
                    special_tokens: Dict[str, str] = None,
                    use_cache: bool = True) -> List[Dict]:
        """
        构建数据集索引
        
        Args:
            max_vocab_size: 最大词汇表大小
            max_text_length: 最大文本长度
            special_tokens: 特殊标记
            use_cache: 是否使用缓存
            
        Returns:
            数据集索引列表
        """
        cache_file = self.cache_dir / "dataset_index.pkl"
        vocab_file = self.cache_dir / "vocab.json"
        
        # 尝试加载缓存
        if use_cache and cache_file.exists() and vocab_file.exists():
            logger.info("加载缓存索引...")
            with open(cache_file, 'rb') as f:
                self.dataset_index = pickle.load(f)
            with open(vocab_file, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            logger.info(f"缓存加载完成: {len(self.dataset_index)} 条记录")
            return self.dataset_index
        
        # 解析描述文件
        entries = self.parse_description_file()
        
        # 验证路径
        validated = self.validate_paths(entries)
        
        # 去重（以相对路径为键）
        seen_paths = set()
        unique_entries = []
        for rel_path, abs_path, description in validated:
            if rel_path not in seen_paths:
                seen_paths.add(rel_path)
                unique_entries.append((rel_path, abs_path, description))
        
        logger.info(f"去重后: {len(unique_entries)} 条记录")
        
        # 构建词汇表
        descriptions = [desc for _, _, desc in unique_entries]
        self.build_vocabulary(descriptions, max_vocab_size, special_tokens)
        
        # 构建索引
        self.dataset_index = []
        for rel_path, abs_path, description in unique_entries:
            # 扫描图片元数据
            metadata = self.scan_image_metadata(abs_path)
            if metadata is None:
                continue
            
            # 编码文本
            encoded_text, attention_mask = self.encode_text(
                description, max_text_length, 
                special_tokens.get("PAD", "<PAD>"),
                special_tokens.get("UNK", "<UNK>")
            )
            
            entry = {
                "rel_path": rel_path,
                "abs_path": abs_path,
                "description": description,
                "description_tokens": self.tokenize(description),
                "encoded_text": encoded_text,
                "attention_mask": attention_mask,
                "image_size": metadata["size"],
                "image_mode": metadata["mode"],
                "image_format": metadata["format"],
            }
            self.dataset_index.append(entry)
        
        logger.info(f"索引构建完成: {len(self.dataset_index)} 条有效记录")
        
        # 保存缓存
        if use_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.dataset_index, f)
            with open(vocab_file, 'w', encoding='utf-8') as f:
                json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            logger.info(f"缓存已保存到: {self.cache_dir}")
        
        return self.dataset_index
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return len(self.vocab)
    
    def decode_text(self, encoded: List[int]) -> str:
        """
        解码ID序列为文本
        
        Args:
            encoded: ID序列
            
        Returns:
            解码后的文本
        """
        tokens = []
        for idx in encoded:
            if idx == 0:  # PAD
                break
            token = self.reverse_vocab.get(idx, "<UNK>")
            if token not in ["<PAD>", "<START>", "<END>"]:
                tokens.append(token)
        return " ".join(tokens)
    
    def get_statistics(self) -> Dict:
        """
        获取数据集统计信息
        
        Returns:
            统计信息字典
        """
        if not self.dataset_index:
            return {}
        
        total = len(self.dataset_index)
        sizes = [entry["image_size"] for entry in self.dataset_index]
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        areas = [w * h for w, h in sizes]
        
        desc_lengths = [len(entry["description_tokens"]) for entry in self.dataset_index]
        
        return {
            "total_samples": total,
            "avg_width": np.mean(widths),
            "avg_height": np.mean(heights),
            "avg_area": np.mean(areas),
            "min_width": min(widths),
            "max_width": max(widths),
            "min_height": min(heights),
            "max_height": max(heights),
            "avg_description_length": np.mean(desc_lengths),
            "max_description_length": max(desc_lengths),
            "vocab_size": len(self.vocab),
        }
