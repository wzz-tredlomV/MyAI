import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import os
import glob
import re
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import shutil
register_keras_serializable = tf.keras.utils.register_keras_serializable

# ============================================================
# 注册所有自定义对象（Keras 3 推荐做法，加载时无需手动传 custom_objects）
# ============================================================

@dataclass
class ModelConfig:
    vocab_size: int = 5000
    embed_dim: int = 384
    num_heads: int = 6
    num_layers: int = 6
    max_len: int = 512
    dropout: float = 0.1
    seq_len: int = 256
    batch_size: int = 4
    pretrain_lr: float = 5e-4
    sft_lr: float = 1e-6          # 降低学习率防止 NaN
    rl_lr: float = 1e-7           # 降低学习率防止 NaN
    weight_decay: float = 0.01
    pretrain_epochs: int = 5
    sft_epochs: int = 10
    rl_epochs: int = 10
    steps_per_epoch: int = 300

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@register_keras_serializable(package="MyAI")
class RotaryEmbedding(layers.Layer):
    def __init__(self, head_dim, max_len=2048, base=10000.0, **kwargs):
        super().__init__(**kwargs)
        self.head_dim = head_dim
        self.max_len = max_len
        self.base = base

    def build(self, input_shape):
        positions = tf.range(self.max_len, dtype=tf.float32)
        inv_freq = 1.0 / (self.base ** (tf.range(0, self.head_dim, 2, dtype=tf.float32) / tf.cast(self.head_dim, tf.float32)))
        angles = tf.einsum('i,j->ij', positions, inv_freq)
        angles = tf.repeat(angles, repeats=2, axis=-1)
        super().build(input_shape)

    def call(self, x, seq_len=None):
        if seq_len is None:
            seq_len = tf.shape(x)[2]

        positions = tf.range(seq_len, dtype=tf.float32)
        inv_freq = 1.0 / (self.base ** (tf.range(0, self.head_dim, 2, dtype=tf.float32) / tf.cast(self.head_dim, tf.float32)))
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
        config.update({
            "head_dim": self.head_dim,
            "max_len": self.max_len,
            "base": self.base,
        })
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
        self.gamma = self.add_weight(
            name='gamma',
            shape=input_shape[-1:],
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        var = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        normalized = (x - mean) / tf.sqrt(var + self.epsilon)
        return self.gamma * normalized + self.beta

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(tf.sqrt(2.0 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
    return x * cdf


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
        self.wq = self.add_weight(
            name='wq',
            shape=(self.embed_dim, self.embed_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.wk = self.add_weight(
            name='wk',
            shape=(self.embed_dim, self.embed_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.wv = self.add_weight(
            name='wv',
            shape=(self.embed_dim, self.embed_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.wo = self.add_weight(
            name='wo',
            shape=(self.embed_dim, self.embed_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.dropout = layers.Dropout(self.dropout_rate)
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

        scores = tf.matmul(q, k, transpose_b=True) / self.scale

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
            scores = tf.where(combined_mask, tf.float32.min, scores)

        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)
        attn_output = tf.matmul(attn_weights, v)

        attn_output = tf.transpose(attn_output, [0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, [batch_size, seq_len, self.embed_dim])
        output = tf.matmul(attn_output, self.wo)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
        })
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
        self.w1 = self.add_weight(
            name='w1',
            shape=(self.embed_dim, self.hidden_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b1 = self.add_weight(
            name='b1',
            shape=(self.hidden_dim,),
            initializer='zeros',
            trainable=True
        )
        self.w2 = self.add_weight(
            name='w2',
            shape=(self.hidden_dim, self.embed_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b2 = self.add_weight(
            name='b2',
            shape=(self.embed_dim,),
            initializer='zeros',
            trainable=True
        )
        self.dropout = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, x, training=False):
        hidden = tf.matmul(x, self.w1) + self.b1
        hidden = gelu(hidden)
        hidden = self.dropout(hidden, training=training)
        output = tf.matmul(hidden, self.w2) + self.b2
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="MyAI")
class CustomTransformerBlock(layers.Layer):
    supports_masking = True

    def __init__(self, embed_dim, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.ln1 = CustomLayerNorm()
        self.attn = CustomMultiHeadAttention(self.embed_dim, self.num_heads, self.dropout_rate)
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.ln2 = CustomLayerNorm()
        self.ffn = CustomFFN(self.embed_dim, self.embed_dim * 4, self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, x, training=False, attention_mask=None, use_causal_mask=False):
        normed = self.ln1(x)
        attn_out = self.attn(normed, training=training, attention_mask=attention_mask, use_causal_mask=use_causal_mask)
        attn_out = self.dropout1(attn_out, training=training)
        x = x + attn_out

        normed = self.ln2(x)
        ffn_out = self.ffn(normed, training=training)
        ffn_out = self.dropout2(ffn_out, training=training)
        x = x + ffn_out
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
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
        self.token_embed = layers.Embedding(config.vocab_size, config.embed_dim)
        self.dropout = layers.Dropout(config.dropout)
        self.blocks = [CustomTransformerBlock(config.embed_dim, config.num_heads, config.dropout) for _ in range(config.num_layers)]

    def build(self, input_shape):
        self.token_embed.build(input_shape)
        for block in self.blocks:
            block.build([None, self.config.seq_len, self.config.embed_dim])
        super().build(input_shape)

    def call(self, x, training=False):
        inputs = x
        seq_len = tf.shape(x)[1]
        x = self.token_embed(x)
        x = self.dropout(x, training=training)
        attention_mask = tf.cast(tf.not_equal(inputs, 0), tf.float32)
        for block in self.blocks:
            x = block(x, training=training, attention_mask=attention_mask, use_causal_mask=True)
        return tf.matmul(x, self.token_embed.embeddings, transpose_b=True)

    def get_config(self):
        config = super().get_config()
        config.update({
            "config": self.config.to_dict(),
        })
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


# ==================== 模型保存/加载工具函数 ====================

def save_model_dual_format(model, save_dir, is_best=False):
    os.makedirs(save_dir, exist_ok=True)

    keras_path = os.path.join(save_dir, "model.keras")
    model.save(keras_path)

    savedmodel_path = os.path.join(save_dir, "savedmodel")
    if os.path.exists(savedmodel_path):
        shutil.rmtree(savedmodel_path)

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(model.config.to_dict(), f, ensure_ascii=False, indent=2)

    if is_best:
        print(f"  ✓ 新的最佳模型已保存到 {save_dir}")
        print(f"    - .keras: {keras_path}")
        print(f"    - SavedModel: {savedmodel_path}")
    else:
        print(f"  模型已保存到 {save_dir}")


def load_model_keras(load_dir, vocab_size=None):
    keras.mixed_precision.set_global_policy("float32")
    keras_path = os.path.join(load_dir, "model.keras")
    if not os.path.exists(keras_path):
        raise FileNotFoundError(f"找不到 .keras 模型文件: {keras_path}")

    model = keras.models.load_model(keras_path)
    print(f"  .keras 模型已从 {keras_path} 加载")
    return model


def save_best_model(model, save_dir, is_best=False):
    if is_best:
        save_model_dual_format(model, save_dir, is_best=True)


# ==================== 数据生成器 ====================

class PretrainDataGenerator:
    def __init__(self, text, char_to_idx, config: ModelConfig, start_ratio=0.0, end_ratio=1.0):
        self.char_to_idx = char_to_idx
        self.config = config
        self.pad_id = char_to_idx.get('<|pad|>', 0)
        self.unk_id = char_to_idx.get('<|unk|>', 3)
        self.ids = self._encode_text(text)
        stride = config.seq_len // 4

        total_len = len(self.ids)
        start_idx = int(total_len * start_ratio)
        end_idx = int(total_len * end_ratio)
        self.ids = self.ids[start_idx:end_idx]

        self.indices = list(range(0, len(self.ids) - config.seq_len, stride))
        self.num_samples = len(self.indices)

    def _encode_text(self, text):
        ids = []
        for ch in text:
            ids.append(self.char_to_idx.get(ch, self.unk_id))
        return np.array(ids, dtype=np.int32)

    def __len__(self):
        return self.num_samples // self.config.batch_size

    def __call__(self):
        batch_x, batch_y = [], []
        count = 0
        for start_idx in self.indices:
            segment = self.ids[start_idx:start_idx + self.config.seq_len + 1]
            if len(segment) < self.config.seq_len + 1:
                segment = np.pad(segment, (0, self.config.seq_len + 1 - len(segment)), constant_values=self.pad_id)
            x = segment[:-1]
            y = segment[1:]
            batch_x.append(x)
            batch_y.append(y)
            count += 1
            if count == self.config.batch_size:
                yield np.array(batch_x, dtype=np.int32), np.array(batch_y, dtype=np.int32)
                batch_x, batch_y = [], []
                count = 0


# ==================== 评估函数 ====================

def evaluate_pretrain(model, val_dataset, loss_fn, max_val_steps=50):
    val_losses = []
    step = 0
    for x, y in val_dataset:
        if step >= max_val_steps:
            break
        logits = model(x, training=False)
        loss = loss_fn(y, logits)
        val_losses.append(loss.numpy())
        step += 1
    avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
    val_perplexity = np.exp(avg_val_loss)
    return avg_val_loss, val_perplexity


# ==================== 训练函数 ====================

def pretrain(model, train_data_generator, val_data_generator, vocab_size, config: ModelConfig):
    try:
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)

        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=config.pretrain_lr,
            decay_steps=config.steps_per_epoch * config.pretrain_epochs,
            alpha=1e-6
        )
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
            global_clipnorm=1.0
        )

        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0)

        train_dataset = tf.data.Dataset.from_generator(
            train_data_generator,
            output_signature=(
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.int32),
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.int32)
            )
        )
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_generator(
            val_data_generator,
            output_signature=(
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.int32),
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.int32)
            )
        )
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        total_samples = len(train_data_generator)
        actual_steps_per_epoch = min(config.steps_per_epoch, total_samples)
        total_steps = actual_steps_per_epoch * config.pretrain_epochs

        best_val_loss = float('inf')
        best_epoch = 0

        with tqdm(total=total_steps, desc="预训练总进度", unit="step", position=0, leave=True) as total_pbar:
            for epoch in range(config.pretrain_epochs):
                epoch_pbar = tqdm(
                    total=actual_steps_per_epoch,
                    desc=f"Epoch {epoch+1}/{config.pretrain_epochs}",
                    unit="step",
                    position=1,
                    leave=False
                )
                epoch_losses = []
                step = 0
                for x, y in train_dataset:
                    if step >= actual_steps_per_epoch:
                        break
                    with tf.GradientTape() as tape:
                        logits = model(x, training=True)
                        loss = loss_fn(y, logits)
                    grads = tape.gradient(loss, model.trainable_variables)

                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    loss_val = loss.numpy()
                    epoch_losses.append(loss_val)
                    step += 1

                    epoch_pbar.update(1)
                    epoch_pbar.set_postfix(loss=f"{loss_val:.4f}")
                    total_pbar.update(1)
                    total_pbar.set_postfix(epoch=f"{epoch+1}", loss=f"{loss_val:.4f}")

                epoch_pbar.close()
                avg_loss = np.mean(epoch_losses) if epoch_losses else 0
                train_perplexity = np.exp(avg_loss)

                print(f"\n  正在验证 Epoch {epoch+1}...")
                val_loss, val_perplexity = evaluate_pretrain(model, val_dataset, loss_fn)
                print(f"  [验证] Loss: {val_loss:.4f}, 困惑度: {val_perplexity:.2f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    save_best_model(model, "saved_model/pretrain_best", is_best=True)
                    print(f"  ★ 新的最佳模型！Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
                else:
                    print(f"  当前最佳: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")

                print(f"\n[Epoch {epoch+1}/{config.pretrain_epochs}] 训练Loss: {avg_loss:.4f}, 训练困惑度: {train_perplexity:.2f}")

        print(f"\n预训练完成！最佳模型来自 Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
        if os.path.exists("saved_model/pretrain_best"):
            if os.path.exists("saved_model/pretrain_final"):
                shutil.rmtree("saved_model/pretrain_final")
            shutil.copytree("saved_model/pretrain_best", "saved_model/pretrain_final")
            print("最佳模型已保存为 pretrain_final")

    except KeyboardInterrupt:
        print("\n训练被用户中断，保存当前模型...")
        save_model_dual_format(model, "saved_model/pretrain_interrupted")
        raise
    except Exception as e:
        print(f"训练出错: {e}")
        save_model_dual_format(model, "saved_model/pretrain_error")
        raise


class SFTDataGenerator:
    def __init__(self, jsonl_path, vocab, config: ModelConfig, max_samples=None):
        self.vocab = vocab
        self.config = config
        self.pad_id = vocab.get('<|pad|>', 0)
        self.unk_id = vocab.get('<|unk|>', 3)
        self.bos_id = vocab.get('<|bos|>', 1)
        self.eos_id = vocab.get('<|eos|>', 2)
        self.user_id = vocab.get('<|user|>', 4)
        self.bot_id = vocab.get('<|bot|>', 5)
        self.samples = self._load_data(jsonl_path, max_samples)

    def _load_data(self, path, max_samples=None):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    prompt_ids, response_ids, full_ids = self._encode(item['prompt'], item['response'])
                    if len(full_ids) >= 2:
                        samples.append({'x': full_ids[:-1], 'y': full_ids[1:], 'prompt_len': len(prompt_ids)})
                    if max_samples and len(samples) >= max_samples:
                        break
        return samples

    def _encode(self, prompt, response):
        prompt_ids = [self.bos_id, self.user_id]
        for ch in prompt:
            prompt_ids.append(self.vocab.get(ch, self.unk_id))
        prompt_ids.append(self.bot_id)
        response_ids = []
        for ch in response:
            response_ids.append(self.vocab.get(ch, self.unk_id))
        response_ids.append(self.eos_id)
        return prompt_ids, response_ids, prompt_ids + response_ids

    def __len__(self):
        return len(self.samples) // self.config.batch_size

    def __call__(self):
        batch_x, batch_y, batch_mask = [], [], []
        count = 0
        for sample in self.samples:
            x = sample['x']
            y = sample['y']
            prompt_len = sample['prompt_len']
            loss_mask = [0.0] * prompt_len + [1.0] * (len(x) - prompt_len)
            if len(x) < self.config.seq_len:
                pad_len = self.config.seq_len - len(x)
                x = x + [self.pad_id] * pad_len
                y = y + [self.pad_id] * pad_len
                loss_mask = loss_mask + [0.0] * pad_len
            else:
                x = x[:self.config.seq_len]
                y = y[:self.config.seq_len]
                loss_mask = loss_mask[:self.config.seq_len]
            batch_x.append(x)
            batch_y.append(y)
            batch_mask.append(loss_mask)
            count += 1
            if count == self.config.batch_size:
                yield np.array(batch_x, dtype=np.int32), np.array(batch_y, dtype=np.int32), np.array(batch_mask, dtype=np.float32)
                batch_x, batch_y, batch_mask = [], [], []
                count = 0


def evaluate_sft(model, val_dataset, sft_loss_fn, max_val_steps=50):
    val_losses = []
    step = 0
    for x, y, mask in val_dataset:
        if step >= max_val_steps:
            break
        logits = model(x, training=False)
        loss = sft_loss_fn(y, logits, mask)
        val_losses.append(loss.numpy())
        step += 1
    return np.mean(val_losses) if val_losses else float('inf')


def sft_train(model, train_jsonl_path, val_jsonl_path, vocab, config: ModelConfig):
    try:
        print(f"\n📊 SFT训练配置: lr={config.sft_lr}, epochs={config.sft_epochs}, clipnorm=0.5")

        train_data_gen = SFTDataGenerator(train_jsonl_path, vocab, config)
        if val_jsonl_path and os.path.exists(val_jsonl_path):
            val_data_gen = SFTDataGenerator(val_jsonl_path, vocab, config)
        else:
            all_samples = SFTDataGenerator(train_jsonl_path, vocab, config).samples
            split_idx = int(len(all_samples) * 0.8)
            train_data_gen = SFTDataGenerator(train_jsonl_path, vocab, config, max_samples=split_idx)
            val_data_gen = SFTDataGenerator(train_jsonl_path, vocab, config)

        def sft_loss(y_true, y_pred, loss_mask):
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
            loss = loss * loss_mask
            total_mask = tf.reduce_sum(loss_mask) + 1e-8
            return tf.reduce_sum(loss) / total_mask

        optimizer = keras.optimizers.AdamW(
            learning_rate=config.sft_lr,
            weight_decay=config.weight_decay,
            global_clipnorm=0.5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )

        train_dataset = tf.data.Dataset.from_generator(
            train_data_gen,
            output_signature=(
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.int32),
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.int32),
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.float32)
            )
        )
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_generator(
            val_data_gen,
            output_signature=(
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.int32),
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.int32),
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.float32)
            )
        )
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        total_steps_estimate = len(train_data_gen)
        total_steps = total_steps_estimate * config.sft_epochs

        best_val_loss = float('inf')
        best_epoch = 0

        with tqdm(total=total_steps, desc="SFT总进度", unit="step", position=0, leave=True) as total_pbar:
            for epoch in range(config.sft_epochs):
                epoch_pbar = tqdm(
                    total=total_steps_estimate,
                    desc=f"SFT Epoch {epoch+1}/{config.sft_epochs}",
                    unit="step",
                    position=1,
                    leave=False
                )
                epoch_losses = []
                step = 0
                for x, y, mask in train_dataset:
                    with tf.GradientTape() as tape:
                        logits = model(x, training=True)
                        loss = sft_loss(y, logits, mask)
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    loss_val = loss.numpy()
                    epoch_losses.append(loss_val)
                    step += 1

                    epoch_pbar.update(1)
                    epoch_pbar.set_postfix(loss=f"{loss_val:.4f}")
                    total_pbar.update(1)
                    total_pbar.set_postfix(epoch=f"{epoch+1}", loss=f"{loss_val:.4f}")

                epoch_pbar.close()
                avg_loss = np.mean(epoch_losses) if epoch_losses else 0

                print(f"\n  正在验证 SFT Epoch {epoch+1}...")
                val_loss = evaluate_sft(model, val_dataset, sft_loss)
                print(f"  [验证] Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    save_best_model(model, "saved_model/sft_best", is_best=True)
                    print(f"  ★ 新的最佳模型！Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
                else:
                    print(f"  当前最佳: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")

                print(f"\n[SFT Epoch {epoch+1}/{config.sft_epochs}] 训练Loss: {avg_loss:.4f}")

        print(f"\nSFT完成！最佳模型来自 Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
        if os.path.exists("saved_model/sft_best"):
            if os.path.exists("saved_model/sft_final"):
                shutil.rmtree("saved_model/sft_final")
            shutil.copytree("saved_model/sft_best", "saved_model/sft_final")
            print("最佳模型已保存为 sft_final")

    except KeyboardInterrupt:
        print("\n训练被用户中断，保存当前模型...")
        save_model_dual_format(model, "saved_model/sft_interrupted")
        raise
    except Exception as e:
        print(f"训练出错: {e}")
        save_model_dual_format(model, "saved_model/sft_error")
        raise


class DPODataGenerator:
    def __init__(self, jsonl_path, vocab, config: ModelConfig, max_pairs=None):
        self.vocab = vocab
        self.config = config
        self.pad_id = vocab.get('<|pad|>', 0)
        self.unk_id = vocab.get('<|unk|>', 3)
        self.bos_id = vocab.get('<|bos|>', 1)
        self.eos_id = vocab.get('<|eos|>', 2)
        self.user_id = vocab.get('<|user|>', 4)
        self.bot_id = vocab.get('<|bot|>', 5)
        self.pairs = self._load_data(jsonl_path, max_pairs)

    def _load_data(self, path, max_pairs=None):
        pairs = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    chosen = self._encode(item['prompt'], item['chosen'])
                    rejected = self._encode(item['prompt'], item['rejected'])
                    if len(chosen) > 0 and len(rejected) > 0:
                        pairs.append((chosen, rejected))
                    if max_pairs and len(pairs) >= max_pairs:
                        break
        return pairs

    def _encode(self, prompt, response):
        ids = [self.bos_id, self.user_id]
        for ch in prompt:
            ids.append(self.vocab.get(ch, self.unk_id))
        ids.append(self.bot_id)
        for ch in response:
            ids.append(self.vocab.get(ch, self.unk_id))
        ids.append(self.eos_id)
        return ids

    def __call__(self):
        for chosen, rejected in self.pairs:
            max_len = max(len(chosen), len(rejected))
            chosen = chosen + [self.pad_id] * (max_len - len(chosen))
            rejected = rejected + [self.pad_id] * (max_len - len(rejected))
            yield np.array(chosen, dtype=np.int32), np.array(rejected, dtype=np.int32)


def compute_logprob(logits, tokens, mask):
    token_neg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tokens, logits=logits)
    token_logprob = -token_neg_loss
    return tf.reduce_sum(token_logprob * mask) / tf.reduce_sum(mask)


def evaluate_dpo(model, ref_model, val_dataset, beta=0.1, max_val_steps=30):
    @tf.function
    def eval_step(chosen, rejected):
        policy_chosen_logits = model(chosen, training=False)
        policy_rejected_logits = model(rejected, training=False)
        ref_chosen_logits = ref_model(chosen, training=False)
        ref_rejected_logits = ref_model(rejected, training=False)
        chosen_mask = tf.cast(chosen != 0, tf.float32)
        rejected_mask = tf.cast(rejected != 0, tf.float32)
        policy_chosen_logprob = compute_logprob(policy_chosen_logits, chosen, chosen_mask)
        policy_rejected_logprob = compute_logprob(policy_rejected_logits, rejected, rejected_mask)
        ref_chosen_logprob = compute_logprob(ref_chosen_logits, chosen, chosen_mask)
        ref_rejected_logprob = compute_logprob(ref_rejected_logits, rejected, rejected_mask)
        policy_ratio = policy_chosen_logprob - policy_rejected_logprob
        ref_ratio = ref_chosen_logprob - ref_rejected_logprob
        loss = -tf.math.log(tf.sigmoid(beta * (policy_ratio - ref_ratio)))
        acc = tf.cast(policy_ratio > ref_ratio, tf.float32)
        return loss, acc

    val_losses = []
    val_accs = []
    step = 0
    for chosen, rejected in val_dataset:
        if step >= max_val_steps:
            break
        loss, acc = eval_step(chosen, rejected)
        val_losses.append(loss.numpy())
        val_accs.append(acc.numpy())
        step += 1

    return np.mean(val_losses) if val_losses else float('inf'), np.mean(val_accs) if val_accs else 0.0


def dpo_train(model, train_jsonl_path, val_jsonl_path, vocab, config: ModelConfig, beta=0.1):
    try:
        print(f"\n📊 DPO训练配置: lr={config.rl_lr}, epochs={config.rl_epochs}, clipnorm=0.5")

        ref_model = build_model(config)
        ref_model.set_weights(model.get_weights())

        @tf.function
        def ref_call(x):
            return ref_model(x, training=False)

        train_data_gen = DPODataGenerator(train_jsonl_path, vocab, config)
        if val_jsonl_path and os.path.exists(val_jsonl_path):
            val_data_gen = DPODataGenerator(val_jsonl_path, vocab, config)
        else:
            all_pairs = DPODataGenerator(train_jsonl_path, vocab, config).pairs
            split_idx = int(len(all_pairs) * 0.8)
            train_data_gen = DPODataGenerator(train_jsonl_path, vocab, config, max_pairs=split_idx)
            val_data_gen = DPODataGenerator(train_jsonl_path, vocab, config)

        optimizer = keras.optimizers.AdamW(
            learning_rate=config.rl_lr,
            weight_decay=config.weight_decay,
            global_clipnorm=0.5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )

        train_dataset = tf.data.Dataset.from_generator(
            train_data_gen,
            output_signature=(
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )
        train_dataset = train_dataset.padded_batch(
            config.batch_size,
            padded_shapes=([None], [None]),
            padding_values=(vocab.get('<|pad|>', 0), vocab.get('<|pad|>', 0))
        )
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_generator(
            val_data_gen,
            output_signature=(
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )
        val_dataset = val_dataset.padded_batch(
            config.batch_size,
            padded_shapes=([None], [None]),
            padding_values=(vocab.get('<|pad|>', 0), vocab.get('<|pad|>', 0))
        )
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        total_pairs = len(train_data_gen.pairs)
        steps_per_epoch = (total_pairs + config.batch_size - 1) // config.batch_size
        total_steps = steps_per_epoch * config.rl_epochs

        @tf.function
        def dpo_step(chosen, rejected):
            with tf.GradientTape() as tape:
                policy_chosen_logits = model(chosen, training=True)
                policy_rejected_logits = model(rejected, training=True)
                ref_chosen_logits = ref_call(chosen)
                ref_rejected_logits = ref_call(rejected)
                chosen_mask = tf.cast(chosen != 0, tf.float32)
                rejected_mask = tf.cast(rejected != 0, tf.float32)
                policy_chosen_logprob = compute_logprob(policy_chosen_logits, chosen, chosen_mask)
                policy_rejected_logprob = compute_logprob(policy_rejected_logits, rejected, rejected_mask)
                ref_chosen_logprob = compute_logprob(ref_chosen_logits, chosen, chosen_mask)
                ref_rejected_logprob = compute_logprob(ref_rejected_logits, rejected, rejected_mask)
                policy_ratio = policy_chosen_logprob - policy_rejected_logprob
                ref_ratio = ref_chosen_logprob - ref_rejected_logprob
                loss = -tf.math.log(tf.sigmoid(beta * (policy_ratio - ref_ratio)))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        best_val_loss = float('inf')
        best_epoch = 0

        with tqdm(total=total_steps, desc="DPO总进度", unit="step", position=0, leave=True) as total_pbar:
            for epoch in range(config.rl_epochs):
                epoch_pbar = tqdm(
                    total=steps_per_epoch,
                    desc=f"DPO Epoch {epoch+1}/{config.rl_epochs}",
                    unit="step",
                    position=1,
                    leave=False
                )
                epoch_losses = []
                step = 0
                for chosen, rejected in train_dataset:
                    loss = dpo_step(chosen, rejected)
                    loss_val = loss.numpy()
                    epoch_losses.append(loss_val)
                    step += 1

                    epoch_pbar.update(1)
                    epoch_pbar.set_postfix(loss=f"{loss_val:.4f}")
                    total_pbar.update(1)
                    total_pbar.set_postfix(epoch=f"{epoch+1}", loss=f"{loss_val:.4f}")

                epoch_pbar.close()
                avg_loss = np.mean(epoch_losses) if epoch_losses else 0

                print(f"\n  正在验证 DPO Epoch {epoch+1}...")
                val_loss, val_acc = evaluate_dpo(model, ref_model, val_dataset, beta)
                print(f"  [验证] Loss: {val_loss:.4f}, 准确率: {val_acc:.2%}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    save_best_model(model, "saved_model/rl_best", is_best=True)
                    print(f"  ★ 新的最佳模型！Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
                else:
                    print(f"  当前最佳: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")

                print(f"\n[DPO Epoch {epoch+1}/{config.rl_epochs}] 训练Loss: {avg_loss:.4f}")

        print(f"\nDPO完成！最佳模型来自 Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
        if os.path.exists("saved_model/rl_best"):
            if os.path.exists("saved_model/rl_final"):
                shutil.rmtree("saved_model/rl_final")
            shutil.copytree("saved_model/rl_best", "saved_model/rl_final")
            print("最佳模型已保存为 rl_final")

    except KeyboardInterrupt:
        print("\n训练被用户中断，保存当前模型...")
        save_model_dual_format(model, "saved_model/rl_interrupted")
        raise
    except Exception as e:
        print(f"训练出错: {e}")
        save_model_dual_format(model, "saved_model/rl_error")
        raise


def load_vocab(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        char_to_idx = json.load(f)

    required = ['<|pad|>', '<|unk|>', '<|bos|>', '<|eos|>', '<|user|>', '<|bot|>']
    for token in required:
        if token not in char_to_idx:
            char_to_idx[token] = len(char_to_idx)
            print(f"自动添加特殊 token: {token} = {char_to_idx[token]}")

    return char_to_idx


def stream_corpus(folder_path, chunk_size=100000):
    for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                text = f.read(chunk_size)
                if not text:
                    break
                text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
                yield text, os.path.basename(file_path)


def load_corpus(folder_path):
    all_text = []
    for text, filename in stream_corpus(folder_path):
        all_text.append(text)
    return "\n".join(all_text)


def build_model(config):
    model = LiteratureTransformer(config)
    dummy_len = max(config.seq_len, 16)
    dummy_input = tf.constant([list(range(min(config.vocab_size, dummy_len)))], dtype=tf.int32)
    _ = model(dummy_input, training=False)
    return model


if __name__ == "__main__":
    keras.mixed_precision.set_global_policy("float32")
    VOCAB_PATH = "vocab.json"
    CORPUS_FOLDER = "./corpus"
    SFT_DATA_PATH = "sft_data.jsonl"
    SFT_VAL_PATH = "sft_val.jsonl"
    RL_DATA_PATH = "rl_data.jsonl"
    RL_VAL_PATH = "rl_val.jsonl"

    if not os.path.exists(VOCAB_PATH):
        print(f"找不到词汇表: {VOCAB_PATH}")
        exit()
    if not os.path.exists(CORPUS_FOLDER):
        print(f"找不到语料文件夹: {CORPUS_FOLDER}")
        exit()

    vocab = load_vocab(VOCAB_PATH)
    config = ModelConfig(
        vocab_size=len(vocab),
        embed_dim=384,
        num_heads=6,
        num_layers=6,
        max_len=512,
        seq_len=256,
        batch_size=4,
        pretrain_epochs=5,
        sft_epochs=20,          # GitHub Actions 安全范围
        rl_epochs=15,           # GitHub Actions 安全范围
        steps_per_epoch=300
    )
    print(f"词汇表大小: {config.vocab_size}")
    print(f"📊 训练配置: SFT={config.sft_epochs}轮, RL={config.rl_epochs}轮, 预计耗时~5-6小时")

    corpus_text = load_corpus(CORPUS_FOLDER)
    print(f"总语料长度: {len(corpus_text)} 字符")

    model = build_model(config)
    total_params = sum([tf.size(v).numpy() for v in model.trainable_variables])
    print(f"模型参数量: {total_params:,}")

    os.makedirs("saved_model", exist_ok=True)

    print("\n" + "="*50)
    print("开始预训练")
    print("="*50)
    train_gen = PretrainDataGenerator(corpus_text, vocab, config, start_ratio=0.0, end_ratio=0.8)
    val_gen = PretrainDataGenerator(corpus_text, vocab, config, start_ratio=0.8, end_ratio=1.0)
    pretrain(model, train_gen, val_gen, config.vocab_size, config)

    if os.path.exists(SFT_DATA_PATH):
        print("\n" + "="*50)
        print("开始 SFT 微调")
        print("="*50)
        model = load_model_keras("saved_model/pretrain_final")
        val_path = SFT_VAL_PATH if os.path.exists(SFT_VAL_PATH) else None
        sft_train(model, SFT_DATA_PATH, val_path, vocab, config)

    if os.path.exists(RL_DATA_PATH):
        print("\n" + "="*50)
        print("开始 DPO 强化学习")
        print("="*50)
        model = load_model_keras("saved_model/sft_final")
        val_path = RL_VAL_PATH if os.path.exists(RL_VAL_PATH) else None
        dpo_train(model, RL_DATA_PATH, val_path, vocab, config)

    print("\n" + "="*50)
    print("保存配置文件")
    print("="*50)

    with open("saved_model/config.json", "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
    print("配置已保存到 saved_model/config.json")

    with open("saved_model/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print("词汇表已保存到 saved_model/vocab.json")

    print("\n" + "="*50)
    print("训练全部完成！最终可用模型：")
    print("  - 预训练: saved_model/pretrain_final/")
    print("    ├── model.keras      (Keras 3 格式，用于调试)")
    print("    ├── savedmodel/      (SavedModel 格式，用于部署)")
    print("    └── config.json      (模型配置)")
    print("  - SFT微调: saved_model/sft_final/")
    print("  - DPO强化: saved_model/rl_final/")
    print("="*50)