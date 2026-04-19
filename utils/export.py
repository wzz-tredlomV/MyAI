"""
模型导出 - 导出为SavedModel格式
"""

import os
import json
import logging
from typing import Dict, Optional, Any

import tensorflow as tf
from tensorflow import keras

from models.generator import Generator
from models.text_encoder import create_text_encoder

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
        super().__init__()
        self.generator = generator
        self.text_encoder = text_encoder
        self.vocab = vocab
        self.bucket_config = bucket_config
        self.max_text_length = max_text_length
        self.noise_dim = noise_dim
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_id = vocab.get(self.pad_token, 0)
        self.unk_id = vocab.get(self.unk_token, 1)
        self._vocab_keys = tf.constant(list(vocab.keys()))
        self._vocab_vals = tf.constant(list(vocab.values()), dtype=tf.int32)
        self._vocab_tbl = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self._vocab_keys, self._vocab_vals),
            default_value=self.unk_id
        )

    def _tf_tokenize(self, texts):
        texts = tf.strings.lower(texts)
        texts = tf.strings.regex_replace(texts, r'[^a-zA-Z0-9 ]', ' ')
        tokenized = tf.strings.split(texts)
        tokens = tokenized.to_tensor(default_value="", shape=[None,self.max_text_length])
        tokens = tokens[:, :self.max_text_length]
        return tokens

    def _tf_encode(self, tokens):
        ids = self._vocab_tbl.lookup(tokens)
        pad = tf.fill([tf.shape(tokens)[0], self.max_text_length], tf.constant(self.pad_id, dtype=tf.int32))
        masks = tf.cast(tf.not_equal(tokens, ""), tf.float32)
        ids = tf.where(tf.equal(tokens, ""), pad, ids)
        ids = tf.cast(ids, tf.int32)
        return ids, masks

    def _get_bucket_size(self, target_size: int) -> int:
        available_sizes = [config["target_size"] for config in self.bucket_config]
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
        batch_size = tf.shape(descriptions)[0]
        if seed > 0:
            tf.random.set_seed(seed)
        tokens = self._tf_tokenize(descriptions)
        ids, mask = self._tf_encode(tokens)
        text_condition = self.text_encoder([ids, mask], training=False)
        noise = tf.random.normal([batch_size, self.noise_dim])
        generated_images = self.generator([noise, text_condition], training=False)
        used_resolution = tf.shape(generated_images)[1]
        return {
            "image": generated_images,
            "used_resolution": used_resolution,
        }

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name="descriptions"),
    ])
    def encode_text(self, descriptions: tf.Tensor) -> Dict[str, tf.Tensor]:
        tokens = self._tf_tokenize(descriptions)
        ids, mask = self._tf_encode(tokens)
        text_embedding = self.text_encoder([ids, mask], training=False)
        return {"embeddings": text_embedding}

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="text_embedding"),
        tf.TensorSpec(shape=[], dtype=tf.int32, name="target_size"),
        tf.TensorSpec(shape=[], dtype=tf.int32, name="seed"),
    ])
    def generate_from_embedding(self, text_embedding: tf.Tensor,
                                 target_size: tf.Tensor,
                                 seed: tf.Tensor) -> Dict[str, tf.Tensor]:
        batch_size = tf.shape(text_embedding)[0]
        if seed > 0:
            tf.random.set_seed(seed)
        noise = tf.random.normal([batch_size, self.noise_dim])
        generated_images = self.generator([noise, text_embedding], training=False)
        return {"image": generated_images}


def export_savedmodel(generator: Generator,
                      text_encoder: keras.Model,
                      vocab: Dict[str, int],
                      bucket_config: list,
                      export_dir: str = "./saved_model",
                      max_text_length: int = 20,
                      noise_dim: int = 100):
    os.makedirs(export_dir, exist_ok=True)
    model = Text2ImageModel(
        generator=generator,
        text_encoder=text_encoder,
        vocab=vocab,
        bucket_config=bucket_config,
        max_text_length=max_text_length,
        noise_dim=noise_dim,
    )
    tf.saved_model.save(
        model,
        export_dir,
        signatures={
            "serving_default": model.serving_default,
            "encode_text": model.encode_text,
            "generate_from_embedding": model.generate_from_embedding,
        }
    )
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
    model = tf.saved_model.load(export_dir)
    logger.info(f"SavedModel 已从 {export_dir} 加载")
    return model
