"""
文本编码器 - 将文本描述编码为条件向量
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class SimpleTextEncoder(keras.Model):
    """
    简单文本编码器
    Embedding -> GlobalAveragePooling -> Dense + LayerNorm
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 output_dim: int = 256, max_length: int = 20, **kwargs):
        """
        初始化简单文本编码器
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            output_dim: 输出维度
            max_length: 最大序列长度
        """
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.max_length = max_length
        
        # 词嵌入层
        self.embedding = layers.Embedding(
            vocab_size, embedding_dim, 
            mask_zero=True,
            name="text_embedding"
        )
        
        # 全局池化
        self.global_pool = layers.GlobalAveragePooling1D(name="global_pool")
        
        # 投影层
        self.projection = layers.Dense(output_dim, name="projection")
        self.layer_norm = layers.LayerNormalization(name="layer_norm")
        
    def call(self, inputs, training=None):
        """
        前向传播
        
        Args:
            inputs: 可以是 (encoded_text, attention_mask) 元组或仅 encoded_text
            
        Returns:
            文本条件向量 [B, output_dim]
        """
        if isinstance(inputs, (list, tuple)):
            encoded_text, attention_mask = inputs
        else:
            encoded_text = inputs
            attention_mask = None
        
        # 词嵌入 [B, L] -> [B, L, embedding_dim]
        x = self.embedding(encoded_text)
        
        # 应用掩码
        if attention_mask is not None:
            # 扩展掩码维度 [B, L] -> [B, L, 1]
            mask = tf.expand_dims(attention_mask, -1)
            x = x * mask
        
        # 全局池化 [B, L, embedding_dim] -> [B, embedding_dim]
        x = self.global_pool(x)
        
        # 投影和归一化
        x = self.projection(x)
        x = self.layer_norm(x)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "output_dim": self.output_dim,
            "max_length": self.max_length,
        })
        return config


class TransformerTextEncoder(keras.Model):
    """
    Transformer文本编码器（进阶方案）
    Embedding + 位置编码 -> Transformer Encoder -> [CLS]输出或全局池化
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 output_dim: int = 256, max_length: int = 20,
                 num_layers: int = 4, num_heads: int = 4, 
                 ff_dim: int = 512, dropout_rate: float = 0.1, **kwargs):
        """
        初始化Transformer文本编码器
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            output_dim: 输出维度
            max_length: 最大序列长度
            num_layers: Transformer层数
            num_heads: 注意力头数
            ff_dim: 前馈网络维度
            dropout_rate: Dropout率
        """
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.max_length = max_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        # 词嵌入
        self.embedding = layers.Embedding(
            vocab_size, embedding_dim,
            mask_zero=True,
            name="text_embedding"
        )
        
        # 可学习位置编码
        self.position_embedding = layers.Embedding(
            max_length, embedding_dim,
            name="position_embedding"
        )
        
        # Transformer编码器层
        self.transformer_layers = []
        for i in range(num_layers):
            self.transformer_layers.append(
                TransformerEncoderBlock(
                    embedding_dim, num_heads, ff_dim, dropout_rate,
                    name=f"transformer_block_{i}"
                )
            )
        
        # 全局池化
        self.global_pool = layers.GlobalAveragePooling1D(name="global_pool")
        
        # 投影层
        self.projection = layers.Dense(output_dim, name="projection")
        self.layer_norm = layers.LayerNormalization(name="output_norm")
        
    def call(self, inputs, training=None):
        """
        前向传播
        
        Args:
            inputs: (encoded_text, attention_mask) 元组或仅 encoded_text
            
        Returns:
            文本条件向量 [B, output_dim]
        """
        if isinstance(inputs, (list, tuple)):
            encoded_text, attention_mask = inputs
        else:
            encoded_text = inputs
            attention_mask = None
        
        batch_size = tf.shape(encoded_text)[0]
        seq_len = tf.shape(encoded_text)[1]
        
        # 词嵌入
        x = self.embedding(encoded_text)  # [B, L, D]
        
        # 位置编码
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = tf.expand_dims(positions, 0)
        positions = tf.tile(positions, [batch_size, 1])
        pos_embeddings = self.position_embedding(positions)
        
        x = x + pos_embeddings
        
        # 创建注意力掩码
        if attention_mask is not None:
            # 转换掩码格式 [B, L] -> [B, 1, 1, L] 用于多头注意力
            attention_mask = tf.expand_dims(tf.expand_dims(attention_mask, 1), 1)
            attention_mask = (1.0 - attention_mask) * -1e9
        
        # Transformer层
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, attention_mask, training=training)
        
        # 全局池化
        x = self.global_pool(x)
        
        # 投影
        x = self.projection(x)
        x = self.layer_norm(x)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "output_dim": self.output_dim,
            "max_length": self.max_length,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


class TransformerEncoderBlock(layers.Layer):
    """Transformer编码器块"""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, 
                 dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        # 多头自注意力
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim,
            dropout=dropout_rate, name="mha"
        )
        
        # 前馈网络
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu", name="ffn_dense1"),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim, name="ffn_dense2"),
            layers.Dropout(dropout_rate),
        ], name="ffn")
        
        # LayerNorm
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, name="ln1")
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, name="ln2")
        
        # Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, attention_mask=None, training=None):
        # 自注意力 + 残差连接
        attn_output = self.attention(
            inputs, inputs, inputs,
            attention_mask=attention_mask,
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # 前馈网络 + 残差连接
        ffn_output = self.ffn(out1, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


def create_text_encoder(encoder_type: str = "simple", **kwargs) -> keras.Model:
    """
    创建文本编码器的工厂函数
    
    Args:
        encoder_type: "simple" 或 "transformer"
        **kwargs: 编码器参数
        
    Returns:
        文本编码器模型
    """
    if encoder_type == "simple":
        return SimpleTextEncoder(**kwargs)
    elif encoder_type == "transformer":
        return TransformerTextEncoder(**kwargs)
    else:
        raise ValueError(f"未知的编码器类型: {encoder_type}")
