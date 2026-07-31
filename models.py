"""
models.py - 所有自定义层和 LiteratureTransformer 模型定义（改进版）
改进点：
  1. 新增 SinusoidalPositionalEmbedding：可学习位置编码
  2. CustomTransformerBlock 新增 Stochastic Depth（随机层丢弃）
  3. CustomMultiHeadAttention 新增输出投影 dropout
  4. LiteratureTransformer 新增 final LayerNorm（Pre-LN 标准架构）
  5. 所有层添加 get_config / from_config 支持 Keras 序列化
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from config import ModelConfig

register_keras_serializable = tf.keras.utils.register_keras_serializable


# ============================================================
# 安全工具
# ============================================================
def safe_gelu(x):
    dtype = x.dtype
    x_f32 = tf.cast(x, tf.float32)
    cdf = 0.5 * (1.0 + tf.tanh(
        tf.sqrt(2.0 / tf.constant(np.pi, dtype=tf.float32)) *
        (x_f32 + 0.044715 * tf.pow(x_f32, 3))
    ))
    return tf.cast(x_f32 * cdf, dtype)


# ============================================================
# 位置编码嵌入（新增）
# ============================================================
@register_keras_serializable(package="MyAI")
class SinusoidalPositionalEmbedding(layers.Layer):
    """
    Token 嵌入 + 可学习位置嵌入
    替换原来的纯 Embedding，让模型感知词序
    """
    def __init__(self, vocab_size, embed_dim, max_len=2048, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len

    def build(self, input_shape):
        self.token_embed = layers.Embedding(self.vocab_size, self.embed_dim, name="token_embedding")
        self.pos_embed = layers.Embedding(self.max_len, self.embed_dim, name="position_embedding")
        super().build(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(seq_len)
        # 广播: [seq_len] -> [1, seq_len] -> [batch, seq_len, embed_dim]
        return self.token_embed(x) + self.pos_embed(positions)

    @property
    def embeddings(self):
        """暴露 token 嵌入矩阵，供输出层共享权重"""
        return self.token_embed.embeddings

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "max_len": self.max_len
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ============================================================
# 自定义层
# ============================================================
@register_keras_serializable(package="MyAI")
class RotaryEmbedding(layers.Layer):
    def __init__(self, head_dim, max_len=2048, base=10000.0, **kwargs):
        super().__init__(**kwargs)
        self.head_dim = head_dim
        self.max_len = max_len
        self.base = base

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, seq_len=None):
        if seq_len is None:
            seq_len = tf.shape(x)[2]
        positions = tf.range(seq_len, dtype=tf.float32)
        inv_freq = 1.0 / (self.base ** (
            tf.range(0, self.head_dim, 2, dtype=tf.float32) /
            tf.cast(self.head_dim, tf.float32)
        ))
        angles = tf.einsum('i,j->ij', positions, inv_freq)
        angles = tf.repeat(angles, repeats=2, axis=-1)
        cos = tf.cos(angles)
        sin = tf.sin(angles)
        cos = tf.cast(cos, x.dtype)
        sin = tf.cast(sin, x.dtype)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated = tf.stack([-x2, x1], axis=-1)
        rotated = tf.reshape(rotated, tf.shape(x))
        cos = tf.reshape(cos, [1, 1, seq_len, self.head_dim])
        sin = tf.reshape(sin, [1, 1, seq_len, self.head_dim])
        return x * cos + rotated * sin

    def get_config(self):
        config = super().get_config()
        config.update({"head_dim": self.head_dim, "max_len": self.max_len, "base": self.base})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="MyAI")
class CustomLayerNorm(layers.Layer):
    supports_masking = True
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, x):
        dtype = x.dtype
        x_f32 = tf.cast(x, tf.float32)
        mean = tf.reduce_mean(x_f32, axis=-1, keepdims=True)
        var = tf.reduce_mean(tf.square(x_f32 - mean), axis=-1, keepdims=True)
        normalized = (x_f32 - mean) * tf.math.rsqrt(var + self.epsilon)
        normalized = tf.cast(normalized, dtype)
        return self.gamma * normalized + self.beta

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="MyAI")
class CustomMultiHeadAttention(layers.Layer):
    supports_masking = True
    def __init__(self, embed_dim, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = tf.sqrt(tf.cast(self.head_dim, tf.float32))
        self.dropout_rate = dropout

    def build(self, input_shape):
        init = keras.initializers.GlorotUniform()
        self.wq = self.add_weight(name='wq', shape=(self.embed_dim, self.embed_dim), initializer=init, trainable=True)
        self.wk = self.add_weight(name='wk', shape=(self.embed_dim, self.embed_dim), initializer=init, trainable=True)
        self.wv = self.add_weight(name='wv', shape=(self.embed_dim, self.embed_dim), initializer=init, trainable=True)
        self.wo = self.add_weight(name='wo', shape=(self.embed_dim, self.embed_dim), initializer=init, trainable=True)
        self.dropout = layers.Dropout(self.dropout_rate)
        self.out_dropout = layers.Dropout(self.dropout_rate)  # 新增：输出投影后的 dropout
        self.rotary = RotaryEmbedding(self.head_dim)
        super().build(input_shape)

    def call(self, x, training=False, attention_mask=None, use_causal_mask=False):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        q = tf.matmul(x, self.wq)
        k = tf.matmul(x, self.wk)
        v = tf.matmul(x, self.wv)
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch_size, seq_len, self.num_heads, self.head_dim])
        v = tf.reshape(v, [batch_size, seq_len, self.num_heads, self.head_dim])
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        q = self.rotary(q, seq_len)
        k = self.rotary(k, seq_len)
        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores / tf.cast(self.scale, scores.dtype)
        combined_mask = None
        if use_causal_mask:
            causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.bool), -1, 0)
            causal_mask = tf.logical_not(causal_mask)
            causal_mask = tf.reshape(causal_mask, [1, 1, seq_len, seq_len])
            combined_mask = causal_mask
        if attention_mask is not None:
            padding_mask = tf.cast(tf.equal(attention_mask, 0), tf.bool)
            padding_mask = tf.reshape(padding_mask, [batch_size, 1, 1, seq_len])
            if combined_mask is None:
                combined_mask = padding_mask
            else:
                combined_mask = tf.logical_or(combined_mask, padding_mask)
        if combined_mask is not None:
            neg_inf = tf.cast(-1e9, scores.dtype)
            scores = tf.where(combined_mask, neg_inf, scores)
        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)
        attn_output = tf.matmul(attn_weights, v)
        attn_output = tf.transpose(attn_output, [0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, [batch_size, seq_len, self.embed_dim])
        output = tf.matmul(attn_output, self.wo)
        output = self.out_dropout(output, training=training)  # 新增 dropout
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "dropout": self.dropout_rate})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="MyAI")
class CustomFFN(layers.Layer):
    supports_masking = True
    def __init__(self, embed_dim, hidden_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

    def build(self, input_shape):
        init = keras.initializers.GlorotUniform()
        self.w1 = self.add_weight(name='w1', shape=(self.embed_dim, self.hidden_dim), initializer=init, trainable=True)
        self.b1 = self.add_weight(name='b1', shape=(self.hidden_dim,), initializer='zeros', trainable=True)
        self.w2 = self.add_weight(name='w2', shape=(self.hidden_dim, self.embed_dim), initializer=init, trainable=True)
        self.b2 = self.add_weight(name='b2', shape=(self.embed_dim,), initializer='zeros', trainable=True)
        self.dropout = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, x, training=False):
        hidden = tf.matmul(x, self.w1) + self.b1
        hidden = safe_gelu(hidden)
        hidden = self.dropout(hidden, training=training)
        output = tf.matmul(hidden, self.w2) + self.b2
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "hidden_dim": self.hidden_dim, "dropout": self.dropout_rate})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="MyAI")
class CustomTransformerBlock(layers.Layer):
    supports_masking = True
    def __init__(self, embed_dim, num_heads, dropout=0.1, survival_prob=1.0, **kwargs):
        """
        survival_prob: Stochastic Depth 存活概率。
                       训练时以 (1 - survival_prob) 概率跳过整个层。
        """
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.survival_prob = survival_prob

    def build(self, input_shape):
        self.ln1 = CustomLayerNorm()
        self.attn = CustomMultiHeadAttention(self.embed_dim, self.num_heads, self.dropout_rate)
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.ln2 = CustomLayerNorm()
        self.ffn = CustomFFN(self.embed_dim, self.embed_dim * 4, self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, x, training=False, attention_mask=None, use_causal_mask=False):
        # Stochastic Depth：训练时随机跳过整个层
        if training and self.survival_prob < 1.0:
            # 使用 tf.cond 确保在 graph mode 下工作
            keep = tf.random.uniform([]) < self.survival_prob
            # 即使跳过，也按比例缩放残差（Deep Networks with Stochastic Depth 论文做法）
            scale = 1.0 / self.survival_prob
        else:
            keep = True
            scale = 1.0

        def _forward():
            normed = self.ln1(x)
            attn_out = self.attn(normed, training=training, attention_mask=attention_mask, use_causal_mask=use_causal_mask)
            attn_out = self.dropout1(attn_out, training=training)
            h = x + attn_out * scale
            normed = self.ln2(h)
            ffn_out = self.ffn(normed, training=training)
            ffn_out = self.dropout2(ffn_out, training=training)
            return h + ffn_out * scale

        def _skip():
            return x

        if training and self.survival_prob < 1.0:
            return tf.cond(keep, _forward, _skip)
        return _forward()

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
            "survival_prob": self.survival_prob
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="MyAI")
class LiteratureTransformer(keras.Model):
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # ✅ 改进1：使用带位置编码的嵌入
        self.token_embed = SinusoidalPositionalEmbedding(
            config.vocab_size, config.embed_dim, config.max_len
        )
        self.dropout = layers.Dropout(config.dropout)
        # ✅ 改进2：Stochastic Depth，最后一层不丢弃（survival_prob=1.0）
        self.blocks = [
            CustomTransformerBlock(
                config.embed_dim,
                config.num_heads,
                config.dropout,
                survival_prob=1.0 - config.stochastic_depth_rate if i < config.num_layers - 1 else 1.0
            )
            for i in range(config.num_layers)
        ]
        # ✅ 改进3：输出前添加 Final LayerNorm（Pre-LN 标准架构）
        self.final_ln = CustomLayerNorm()

    def build(self, input_shape):
        self.token_embed.build(input_shape)
        for block in self.blocks:
            block.build([None, self.config.seq_len, self.config.embed_dim])
        self.final_ln.build([None, self.config.seq_len, self.config.embed_dim])
        super().build(input_shape)

    def call(self, x, training=False):
        inputs = x
        seq_len = tf.shape(x)[1]
        x = self.token_embed(x)
        x = self.dropout(x, training=training)
        attention_mask = tf.cast(tf.not_equal(inputs, 0), tf.float32)
        for block in self.blocks:
            x = block(x, training=training, attention_mask=attention_mask, use_causal_mask=True)
        x = self.final_ln(x)  # ✅ 最终 LayerNorm
        # 使用 token_embed.embeddings 共享输入输出权重
        logits = tf.matmul(x, self.token_embed.embeddings, transpose_b=True)
        return logits

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config.to_dict()})
        return config

    @classmethod
    def from_config(cls, config):
        if "config" in config and isinstance(config["config"], dict):
            return cls(ModelConfig.from_dict(config["config"]))
        return cls(ModelConfig(**config))

    def get_build_config(self):
        return {"input_shape": [None, self.config.seq_len]}

    def build_from_config(self, config):
        if config and "input_shape" in config:
            dummy_input = tf.keras.Input(shape=config["input_shape"][1:], dtype=tf.int32)
            self(dummy_input)
