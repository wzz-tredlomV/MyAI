import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import os
import glob
import re
import numpy as np
from dataclasses import dataclass, asdict
from tqdm import tqdm
import shutil
import time
from datetime import datetime

register_keras_serializable = tf.keras.utils.register_keras_serializable

# ============================================================
# 全局配置与自动环境修复
# ============================================================

def setup_environment():
    """自动配置 GPU 内存和随机种子，避免 OOM 和增强可复现性"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU 内存设置失败: {e}")
    tf.random.set_seed(42)
    np.random.seed(42)

setup_environment()

# ============================================================
# 配置类（新增多个自动调节参数）
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

    # 学习率配置
    pretrain_lr: float = 5e-4
    sft_lr: float = 1e-5      # 从 1e-6 调高，1e-6 太小容易平原
    rl_lr: float = 1e-6       # 从 1e-7 调高
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1 # 新增：预热步数比例
    min_lr_ratio: float = 0.1 # 新增：最小学习率比例

    # 训练配置
    pretrain_epochs: int = 5
    sft_epochs: int = 20
    rl_epochs: int = 15
    steps_per_epoch: int = 300

    # 新增：自动处理参数
    gradient_accumulation_steps: int = 4  # 等效 batch_size = 16
    early_stop_patience: int = 3          # 早停耐心值
    plateau_patience: int = 5             # 学习率衰减耐心值（按 epoch）
    auto_lr_reduce_factor: float = 0.5  # 学习率衰减系数
    enable_mixed_precision: bool = True   # 是否启用混合精度
    max_nan_tolerance: int = 3            # 连续 NaN 容忍次数
    checkpoint_freq: int = 1              # 每 N 个 epoch 保存 checkpoint
    log_dir: str = "logs"                 # TensorBoard 日志目录

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


# ============================================================
# 学习率调度（Warmup + Cosine + 平原自动衰减）
# ============================================================

class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """带预热的余弦退火，支持外部乘数（用于自动降 LR）"""
    def __init__(self, initial_lr, warmup_steps, total_steps, alpha=0.1, multiplier=1.0):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.alpha = alpha
        self.multiplier = multiplier

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)

        warmup_lr = self.initial_lr * (step / warmup)

        progress = tf.clip_by_value((step - warmup) / tf.maximum(total - warmup, 1.0), 0.0, 1.0)
        cosine = 0.5 * (1.0 + tf.cos(np.pi * progress))
        decayed = (1.0 - self.alpha) * cosine + self.alpha
        decay_lr = self.initial_lr * decayed

        lr = tf.cond(step < warmup, lambda: warmup_lr, lambda: decay_lr)
        return lr * self.multiplier

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "alpha": self.alpha,
            "multiplier": self.multiplier,
        }


class AdaptiveLRManager:
    """自动检测 Loss 平原并降低学习率"""
    def __init__(self, optimizer, config: ModelConfig, total_steps):
        self.optimizer = optimizer
        self.config = config
        self.total_steps = total_steps
        self.loss_history = []
        self.plateau_counter = 0
        self.nan_counter = 0
        self.current_multiplier = 1.0

    def on_step_end(self, loss_val):
        self.loss_history.append(float(loss_val))
        if len(self.loss_history) > 100:
            self.loss_history.pop(0)

    def on_epoch_end(self, val_loss):
        if len(self.loss_history) < 20:
            return False

        recent = self.loss_history[-50:]
        recent_mean = np.mean(recent)
        older = self.loss_history[:-50] if len(self.loss_history) > 50 else self.loss_history[:20]
        older_mean = np.mean(older)

        no_improvement = (older_mean - recent_mean) / max(abs(older_mean), 1e-8) < 0.01
        increasing = recent_mean > older_mean * 1.02

        if no_improvement or increasing:
            self.plateau_counter += 1
            print(f"  ⚠️ 检测到 Loss {'上升' if increasing else '平原'} (计数: {self.plateau_counter}/{self.config.plateau_patience})")
        else:
            self.plateau_counter = max(0, self.plateau_counter - 1)

        if self.plateau_counter >= self.config.plateau_patience:
            self._reduce_lr()
            self.plateau_counter = 0
            return True
        return False

    def _reduce_lr(self):
        self.current_multiplier *= self.config.auto_lr_reduce_factor
        old_lr = self.optimizer.learning_rate
        if isinstance(old_lr, WarmupCosineDecay):
            new_lr = WarmupCosineDecay(
                initial_lr=old_lr.initial_lr,
                warmup_steps=old_lr.warmup_steps,
                total_steps=old_lr.total_steps,
                alpha=old_lr.alpha,
                multiplier=self.current_multiplier
            )
            self.optimizer.learning_rate = new_lr
        print(f"🔧 自动降低学习率! 当前乘数: {self.current_multiplier:.3f}")

    def on_nan_detected(self):
        self.nan_counter += 1
        print(f"  ⚠️ 检测到 NaN/Inf (计数: {self.nan_counter}/{self.config.max_nan_tolerance})")
        if self.nan_counter >= self.config.max_nan_tolerance:
            self._reduce_lr()
            self.nan_counter = 0
            return True
        return False

    def reset_nan_counter(self):
        self.nan_counter = 0

# ============================================================
# 安全工具函数
# ============================================================

def safe_gelu(x):
    """数值稳定的 GELU，支持混合精度"""
    dtype = x.dtype
    x_f32 = tf.cast(x, tf.float32)
    cdf = 0.5 * (1.0 + tf.tanh(
        tf.sqrt(2.0 / tf.constant(np.pi, dtype=tf.float32)) * 
        (x_f32 + 0.044715 * tf.pow(x_f32, 3))
    ))
    return tf.cast(x_f32 * cdf, dtype)


def check_gradients_nan_inf(grads):
    """检查梯度是否包含 NaN 或 Inf"""
    for g in grads:
        if g is None:
            continue
        if tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g)):
            return True
    return False


def get_gradient_norm(grads):
    """计算全局梯度范数"""
    norms = []
    for g in grads:
        if g is not None:
            norms.append(tf.reduce_sum(tf.square(g)))
    if not norms:
        return 0.0
    return tf.sqrt(tf.add_n(norms)).numpy()


# ============================================================
# 自定义层（修复数值稳定性）
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

# ============================================================
# 模型保存/加载/Checkpoint 工具（增强版）
# ============================================================

def save_model_dual_format(model, save_dir, is_best=False):
    os.makedirs(save_dir, exist_ok=True)
    keras_path = os.path.join(save_dir, "model.keras")
    savedmodel_path = os.path.join(save_dir, "savedmodel")
    if os.path.exists(savedmodel_path):
        shutil.rmtree(savedmodel_path)

    try:
        model.save(keras_path)
    except Exception as e:
        print(f"  ⚠️ .keras 保存失败: {e}, 尝试 SavedModel 格式...")

    try:
        tf.saved_model.save(model, savedmodel_path)
    except Exception as e:
        print(f"  ⚠️ SavedModel 保存失败: {e}")

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(model.config.to_dict(), f, ensure_ascii=False, indent=2)

    if is_best:
        print(f"  ✓ 新的最佳模型已保存到 {save_dir}")


def save_checkpoint(model, optimizer, epoch, step, save_dir):
    """保存训练状态（模型权重 + 优化器状态 + 训练进度）"""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_path = os.path.join(save_dir, f"ckpt_epoch{epoch}_step{step}")
    checkpoint.write(checkpoint_path)

    state = {"epoch": epoch, "step": step}
    with open(os.path.join(save_dir, "training_state.json"), "w") as f:
        json.dump(state, f)
    return checkpoint_path


def load_checkpoint_if_exists(model, optimizer, save_dir):
    """自动恢复最近的 checkpoint"""
    state_path = os.path.join(save_dir, "training_state.json")
    if not os.path.exists(state_path):
        return None, 0, 0

    with open(state_path, "r") as f:
        state = json.load(f)

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest = tf.train.latest_checkpoint(save_dir)
    if latest:
        checkpoint.restore(latest)
        print(f"  ✅ 自动恢复训练状态: Epoch {state['epoch']}, Step {state['step']}")
        return state, state["epoch"], state["step"]
    return None, 0, 0


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


# ============================================================
# 数据生成器（增强版：支持 Shuffle）
# ============================================================

class PretrainDataGenerator:
    def __init__(self, text, char_to_idx, config: ModelConfig, start_ratio=0.0, end_ratio=1.0, shuffle=True):
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
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _encode_text(self, text):
        ids = []
        for ch in text:
            ids.append(self.char_to_idx.get(ch, self.unk_id))
        return np.array(ids, dtype=np.int32)

    def __len__(self):
        return self.num_samples // self.config.batch_size

    def __call__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
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


class SFTDataGenerator:
    def __init__(self, jsonl_path, vocab, config: ModelConfig, max_samples=None, samples=None):
        self.vocab = vocab
        self.config = config
        self.pad_id = vocab.get('<|pad|>', 0)
        self.unk_id = vocab.get('<|unk|>', 3)
        self.bos_id = vocab.get('<|bos|>', 1)
        self.eos_id = vocab.get('<|eos|>', 2)
        self.user_id = vocab.get('<|user|>', 4)
        self.bot_id = vocab.get('<|bot|>', 5)

        if samples is not None:
            self.samples = samples
            self.num_samples = len(samples)
        else:
            self._analyze_data(jsonl_path)
            self.samples = self._load_data(jsonl_path, max_samples)

        print(f"  ✅ 加载了 {len(self.samples)} 条样本")
        if self.samples:
            print(f"  ✅ 样本平均长度: {np.mean([len(s['x']) for s in self.samples]):.0f}")

    def _analyze_data(self, jsonl_path, max_samples=500):
        print(f"\n📊 分析数据: {jsonl_path}")
        prompt_lens = []
        response_lens = []
        total_lens = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                if line.strip():
                    item = json.loads(line)
                    prompt_len = len(item.get('prompt', ''))
                    response_len = len(item.get('response', ''))
                    prompt_lens.append(prompt_len)
                    response_lens.append(response_len)
                    total_lens.append(prompt_len + response_len)

        if prompt_lens:
            p95_total = int(np.percentile(total_lens, 95))
            p90_prompt = int(np.percentile(prompt_lens, 90))

            self.seq_len = min(max(p95_total + 10, 128), 512)
            self.seq_len = (self.seq_len // 8) * 8
            self.max_prompt_len = max(self.seq_len - 20, 100)
            self.max_prompt_len = min(self.max_prompt_len, p90_prompt + 20)

            if self.seq_len != self.config.seq_len:
                print(f"  ✅ 自动调整 seq_len: {self.config.seq_len} -> {self.seq_len}")
                self.config.seq_len = self.seq_len

            print(f"  ✅ 调整后: seq_len={self.seq_len}, max_prompt_len={self.max_prompt_len}")

    def _encode(self, prompt, response):
        prompt_ids = [self.bos_id, self.user_id]
        for ch in prompt:
            prompt_ids.append(self.vocab.get(ch, self.unk_id))
        prompt_ids.append(self.bot_id)

        response_ids = []
        for ch in response:
            response_ids.append(self.vocab.get(ch, self.unk_id))
        response_ids.append(self.eos_id)

        return prompt_ids, response_ids

    def _load_data(self, path, max_samples=None):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    prompt_ids, response_ids = self._encode(item['prompt'], item['response'])

                    if len(prompt_ids) > self.max_prompt_len:
                        prompt_ids = prompt_ids[:self.max_prompt_len]

                    full_ids = prompt_ids + response_ids
                    if len(full_ids) > self.config.seq_len + 1:
                        full_ids = full_ids[:self.config.seq_len + 1]

                    if len(full_ids) >= 2:
                        x = full_ids[:-1]
                        y = full_ids[1:]
                        prompt_len = len(prompt_ids)
                        samples.append({'x': x, 'y': y, 'prompt_len': prompt_len})

                    if max_samples and len(samples) >= max_samples:
                        break
        return samples

    def __len__(self):
        return len(self.samples) // self.config.batch_size

    def __call__(self):
        indices = list(range(len(self.samples)))
        np.random.shuffle(indices)
        batch_x, batch_y, batch_mask = [], [], []
        count = 0
        for idx in indices:
            sample = self.samples[idx]
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
                yield (np.array(batch_x, dtype=np.int32), 
                       np.array(batch_y, dtype=np.int32), 
                       np.array(batch_mask, dtype=np.float32))
                batch_x, batch_y, batch_mask = [], [], []
                count = 0


class DPODataGenerator:
    def __init__(self, jsonl_path, vocab, config: ModelConfig, max_pairs=None, pairs=None):
        self.vocab = vocab
        self.config = config
        self.pad_id = vocab.get('<|pad|>', 0)
        self.unk_id = vocab.get('<|unk|>', 3)
        self.bos_id = vocab.get('<|bos|>', 1)
        self.eos_id = vocab.get('<|eos|>', 2)
        self.user_id = vocab.get('<|user|>', 4)
        self.bot_id = vocab.get('<|bot|>', 5)
        if pairs is not None:
            self.pairs = pairs
        else:
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
        indices = list(range(len(self.pairs)))
        np.random.shuffle(indices)
        for idx in indices:
            chosen, rejected = self.pairs[idx]
            max_len = max(len(chosen), len(rejected))
            chosen = chosen + [self.pad_id] * (max_len - len(chosen))
            rejected = rejected + [self.pad_id] * (max_len - len(rejected))
            yield np.array(chosen, dtype=np.int32), np.array(rejected, dtype=np.int32)


# ============================================================
# 评估函数（增强错误处理）
# ============================================================

def evaluate_pretrain(model, val_dataset, loss_fn, max_val_steps=50):
    val_losses = []
    step = 0
    for x, y in val_dataset:
        if step >= max_val_steps:
            break
        try:
            logits = model(x, training=False)
            loss = loss_fn(y, logits)
            val_losses.append(loss.numpy())
        except Exception as e:
            print(f"  ⚠️ 验证步出错: {e}")
        step += 1
    avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
    val_perplexity = np.exp(avg_val_loss)
    return avg_val_loss, val_perplexity


def evaluate_sft(model, val_dataset, sft_loss_fn, max_val_steps=50):
    val_losses = []
    step = 0
    for x, y, mask in val_dataset:
        if step >= max_val_steps:
            break
        try:
            logits = model(x, training=False)
            loss = sft_loss_fn(y, logits, mask)
            val_losses.append(loss.numpy())
        except Exception as e:
            print(f"  ⚠️ SFT 验证步出错: {e}")
        step += 1
    return np.mean(val_losses) if val_losses else float('inf')

# ============================================================
# 预训练（增强版：梯度累积 + 自动恢复 + 早停 + TensorBoard）
# ============================================================

def pretrain(model, train_data_generator, val_data_generator, vocab_size, config: ModelConfig):
    try:
        # 自动设置混合精度
        if config.enable_mixed_precision:
            try:
                policy = keras.mixed_precision.Policy('mixed_float16')
                keras.mixed_precision.set_global_policy(policy)
                print("  ✅ 已启用 mixed_float16")
            except Exception as e:
                print(f"  ⚠️ 混合精度不可用: {e}，使用 float32")
                keras.mixed_precision.set_global_policy('float32')
        else:
            keras.mixed_precision.set_global_policy('float32')

        total_samples = len(train_data_generator)
        actual_steps_per_epoch = min(config.steps_per_epoch, total_samples)
        total_steps = actual_steps_per_epoch * config.pretrain_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)

        lr_schedule = WarmupCosineDecay(
            initial_learning_rate=config.pretrain_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            alpha=config.min_lr_ratio
        )
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
            global_clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )

        # 自适应学习率管理器
        lr_manager = AdaptiveLRManager(optimizer, config, total_steps)

        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0)

        train_dataset = tf.data.Dataset.from_generator(
            train_data_generator,
            output_signature=(
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.int32),
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.int32)
            )
        ).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_generator(
            val_data_generator,
            output_signature=(
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.int32),
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.int32)
            )
        ).prefetch(tf.data.AUTOTUNE)

        # TensorBoard
        log_dir = os.path.join(config.log_dir, "pretrain", datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        summary_writer = tf.summary.create_file_writer(log_dir)

        # 尝试恢复 checkpoint
        ckpt_dir = "saved_model/pretrain_ckpt"
        state, start_epoch, global_step = load_checkpoint_if_exists(model, optimizer, ckpt_dir)

        best_val_loss = float('inf')
        best_epoch = 0
        early_stop_counter = 0
        accumulated_grads = None
        accum_step = 0

        with tqdm(total=total_steps, desc="预训练总进度", unit="step", position=0, leave=True) as total_pbar:
            total_pbar.update(global_step)

            for epoch in range(start_epoch, config.pretrain_epochs):
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

                    try:
                        with tf.GradientTape() as tape:
                            logits = model(x, training=True)
                            loss = loss_fn(y, logits)
                            # 梯度累积：损失除以累积步数
                            if config.gradient_accumulation_steps > 1:
                                loss = loss / config.gradient_accumulation_steps

                        grads = tape.gradient(loss, model.trainable_variables)

                        # NaN/Inf 检查
                        if check_gradients_nan_inf(grads):
                            lr_manager.on_nan_detected()
                            epoch_pbar.update(1)
                            total_pbar.update(1)
                            step += 1
                            global_step += 1
                            continue
                        else:
                            lr_manager.reset_nan_counter()

                        # 梯度累积逻辑
                        if config.gradient_accumulation_steps > 1:
                            if accumulated_grads is None:
                                accumulated_grads = [tf.zeros_like(v) for v in model.trainable_variables]
                            accumulated_grads = [acc + g for acc, g in zip(accumulated_grads, grads)]
                            accum_step += 1

                            if accum_step % config.gradient_accumulation_steps == 0:
                                optimizer.apply_gradients(zip(accumulated_grads, model.trainable_variables))
                                accumulated_grads = None
                        else:
                            optimizer.apply_gradients(zip(grads, model.trainable_variables))

                        loss_val = loss.numpy() * (config.gradient_accumulation_steps if config.gradient_accumulation_steps > 1 else 1)
                        epoch_losses.append(loss_val)
                        lr_manager.on_step_end(loss_val)

                        # 记录 TensorBoard
                        with summary_writer.as_default():
                            tf.summary.scalar('train_loss', loss_val, step=global_step)
                            tf.summary.scalar('learning_rate', optimizer.learning_rate(global_step), step=global_step)
                            grad_norm = get_gradient_norm(grads)
                            tf.summary.scalar('grad_norm', grad_norm, step=global_step)

                        step += 1
                        global_step += 1
                        epoch_pbar.update(1)
                        epoch_pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{float(optimizer.learning_rate(global_step)):.2e}")
                        total_pbar.update(1)
                        total_pbar.set_postfix(epoch=f"{epoch+1}", loss=f"{loss_val:.4f}")

                    except tf.errors.ResourceExhaustedError as e:
                        print(f"\n  ⚠️ OOM 错误，跳过该 batch: {e}")
                        tf.keras.backend.clear_session()
                        continue

                epoch_pbar.close()
                avg_loss = np.mean(epoch_losses) if epoch_losses else 0
                train_perplexity = np.exp(avg_loss)

                print(f"\n  正在验证 Epoch {epoch+1}...")
                val_loss, val_perplexity = evaluate_pretrain(model, val_dataset, loss_fn)
                print(f"  [验证] Loss: {val_loss:.4f}, 困惑度: {val_perplexity:.2f}")

                with summary_writer.as_default():
                    tf.summary.scalar('val_loss', val_loss, step=global_step)
                    tf.summary.scalar('val_perplexity', val_perplexity, step=global_step)

                # 自动学习率调整
                lr_reduced = lr_manager.on_epoch_end(val_loss)
                if lr_reduced:
                    early_stop_counter = 0  # 降 LR 后重置早停计数

                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    early_stop_counter = 0
                    save_best_model(model, "saved_model/pretrain_best", is_best=True)
                    print(f"  ★ 新的最佳模型！Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
                else:
                    early_stop_counter += 1
                    print(f"  当前最佳: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
                    print(f"  早停计数: {early_stop_counter}/{config.early_stop_patience}")

                # 早停检查
                if early_stop_counter >= config.early_stop_patience:
                    print(f"\n🛑 触发早停！验证集连续 {config.early_stop_patience} 轮未改善。")
                    break

                # 定期保存 checkpoint
                if (epoch + 1) % config.checkpoint_freq == 0:
                    save_checkpoint(model, optimizer, epoch + 1, global_step, ckpt_dir)

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

# ============================================================
# SFT 训练（增强版）
# ============================================================

def sft_train(model, train_jsonl_path, val_jsonl_path, vocab, config: ModelConfig):
    try:
        print(f"\n📊 SFT训练配置: lr={config.sft_lr}, epochs={config.sft_epochs}, clipnorm=0.5")

        # 只加载一次数据
        full_data_gen = SFTDataGenerator(train_jsonl_path, vocab, config)
        all_samples = full_data_gen.samples

        if val_jsonl_path and os.path.exists(val_jsonl_path):
            train_data_gen = SFTDataGenerator(train_jsonl_path, vocab, config, samples=all_samples)
            val_data_gen = SFTDataGenerator(val_jsonl_path, vocab, config)
        else:
            split_idx = int(len(all_samples) * 0.8)
            train_samples = all_samples[:split_idx]
            val_samples = all_samples[split_idx:]
            train_data_gen = SFTDataGenerator(train_jsonl_path, vocab, config, samples=train_samples)
            val_data_gen = SFTDataGenerator(train_jsonl_path, vocab, config, samples=val_samples)
            print(f"  ✅ 自动划分: 训练 {len(train_samples)} / 验证 {len(val_samples)}")

        def sft_loss(y_true, y_pred, loss_mask):
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
            loss = loss * loss_mask
            total_mask = tf.reduce_sum(loss_mask)
            total_mask = tf.maximum(total_mask, 1.0)
            return tf.reduce_sum(loss) / total_mask

        total_steps_estimate = len(train_data_gen)
        total_steps = total_steps_estimate * config.sft_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)

        lr_schedule = WarmupCosineDecay(
            initial_learning_rate=config.sft_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            alpha=config.min_lr_ratio
        )
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
            global_clipnorm=0.5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )

        lr_manager = AdaptiveLRManager(optimizer, config, total_steps)

        train_dataset = tf.data.Dataset.from_generator(
            train_data_gen,
            output_signature=(
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.int32),
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.int32),
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.float32)
            )
        ).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_generator(
            val_data_gen,
            output_signature=(
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.int32),
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.int32),
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.float32)
            )
        ).prefetch(tf.data.AUTOTUNE)

        log_dir = os.path.join(config.log_dir, "sft", datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        summary_writer = tf.summary.create_file_writer(log_dir)

        ckpt_dir = "saved_model/sft_ckpt"
        state, start_epoch, global_step = load_checkpoint_if_exists(model, optimizer, ckpt_dir)

        best_val_loss = float('inf')
        best_epoch = 0
        early_stop_counter = 0
        accumulated_grads = None
        accum_step = 0

        with tqdm(total=total_steps, desc="SFT总进度", unit="step", position=0, leave=True) as total_pbar:
            total_pbar.update(global_step)

            for epoch in range(start_epoch, config.sft_epochs):
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
                    if step >= total_steps_estimate:
                        break

                    try:
                        with tf.GradientTape() as tape:
                            logits = model(x, training=True)
                            loss = sft_loss(y, logits, mask)
                            if config.gradient_accumulation_steps > 1:
                                loss = loss / config.gradient_accumulation_steps

                        grads = tape.gradient(loss, model.trainable_variables)

                        if check_gradients_nan_inf(grads):
                            lr_manager.on_nan_detected()
                            step += 1
                            global_step += 1
                            continue
                        else:
                            lr_manager.reset_nan_counter()

                        if config.gradient_accumulation_steps > 1:
                            if accumulated_grads is None:
                                accumulated_grads = [tf.zeros_like(v) for v in model.trainable_variables]
                            accumulated_grads = [acc + g for acc, g in zip(accumulated_grads, grads)]
                            accum_step += 1

                            if accum_step % config.gradient_accumulation_steps == 0:
                                optimizer.apply_gradients(zip(accumulated_grads, model.trainable_variables))
                                accumulated_grads = None
                        else:
                            optimizer.apply_gradients(zip(grads, model.trainable_variables))

                        loss_val = loss.numpy() * (config.gradient_accumulation_steps if config.gradient_accumulation_steps > 1 else 1)
                        epoch_losses.append(loss_val)
                        lr_manager.on_step_end(loss_val)

                        with summary_writer.as_default():
                            tf.summary.scalar('train_loss', loss_val, step=global_step)
                            tf.summary.scalar('learning_rate', optimizer.learning_rate(global_step), step=global_step)
                            tf.summary.scalar('grad_norm', get_gradient_norm(grads), step=global_step)

                        step += 1
                        global_step += 1
                        epoch_pbar.update(1)
                        epoch_pbar.set_postfix(loss=f"{loss_val:.4f}")
                        total_pbar.update(1)
                        total_pbar.set_postfix(epoch=f"{epoch+1}", loss=f"{loss_val:.4f}")

                    except tf.errors.ResourceExhaustedError as e:
                        print(f"\n  ⚠️ OOM 错误，跳过该 batch: {e}")
                        tf.keras.backend.clear_session()
                        continue

                epoch_pbar.close()
                avg_loss = np.mean(epoch_losses) if epoch_losses else 0

                print(f"\n  正在验证 SFT Epoch {epoch+1}...")
                val_loss = evaluate_sft(model, val_dataset, sft_loss)
                print(f"  [验证] Loss: {val_loss:.4f}")

                with summary_writer.as_default():
                    tf.summary.scalar('val_loss', val_loss, step=global_step)

                lr_reduced = lr_manager.on_epoch_end(val_loss)
                if lr_reduced:
                    early_stop_counter = 0

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    early_stop_counter = 0
                    save_best_model(model, "saved_model/sft_best", is_best=True)
                    print(f"  ★ 新的最佳模型！Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
                else:
                    early_stop_counter += 1
                    print(f"  当前最佳: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
                    print(f"  早停计数: {early_stop_counter}/{config.early_stop_patience}")

                if early_stop_counter >= config.early_stop_patience:
                    print(f"\n🛑 触发早停！")
                    break

                if (epoch + 1) % config.checkpoint_freq == 0:
                    save_checkpoint(model, optimizer, epoch + 1, global_step, ckpt_dir)

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

# ============================================================
# DPO 相关（增强版）
# ============================================================

def compute_logprob(logits, tokens, mask):
    token_neg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tokens, logits=logits)
    token_logprob = -token_neg_loss
    return tf.reduce_sum(token_logprob * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


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
        loss = -tf.math.log(tf.sigmoid(beta * (policy_ratio - ref_ratio)) + 1e-8)
        acc = tf.cast(policy_ratio > ref_ratio, tf.float32)
        return loss, acc

    val_losses = []
    val_accs = []
    step = 0
    for chosen, rejected in val_dataset:
        if step >= max_val_steps:
            break
        try:
            loss, acc = eval_step(chosen, rejected)
            val_losses.append(loss.numpy())
            val_accs.append(acc.numpy())
        except Exception as e:
            print(f"  ⚠️ DPO 验证步出错: {e}")
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

        # 只加载一次数据
        full_data_gen = DPODataGenerator(train_jsonl_path, vocab, config)
        all_pairs = full_data_gen.pairs

        if val_jsonl_path and os.path.exists(val_jsonl_path):
            train_data_gen = DPODataGenerator(train_jsonl_path, vocab, config, pairs=all_pairs)
            val_data_gen = DPODataGenerator(val_jsonl_path, vocab, config)
        else:
            split_idx = int(len(all_pairs) * 0.8)
            train_pairs = all_pairs[:split_idx]
            val_pairs = all_pairs[split_idx:]
            train_data_gen = DPODataGenerator(train_jsonl_path, vocab, config, pairs=train_pairs)
            val_data_gen = DPODataGenerator(train_jsonl_path, vocab, config, pairs=val_pairs)
            print(f"  ✅ 自动划分: 训练 {len(train_pairs)} / 验证 {len(val_pairs)}")

        total_pairs = len(train_data_gen.pairs)
        steps_per_epoch = (total_pairs + config.batch_size - 1) // config.batch_size
        total_steps = steps_per_epoch * config.rl_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)

        lr_schedule = WarmupCosineDecay(
            initial_learning_rate=config.rl_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            alpha=config.min_lr_ratio
        )
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
            global_clipnorm=0.5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )

        lr_manager = AdaptiveLRManager(optimizer, config, total_steps)

        train_dataset = tf.data.Dataset.from_generator(
            train_data_gen,
            output_signature=(
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        ).padded_batch(
            config.batch_size,
            padded_shapes=([None], [None]),
            padding_values=(vocab.get('<|pad|>', 0), vocab.get('<|pad|>', 0))
        ).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_generator(
            val_data_gen,
            output_signature=(
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        ).padded_batch(
            config.batch_size,
            padded_shapes=([None], [None]),
            padding_values=(vocab.get('<|pad|>', 0), vocab.get('<|pad|>', 0))
        ).prefetch(tf.data.AUTOTUNE)

        log_dir = os.path.join(config.log_dir, "dpo", datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        summary_writer = tf.summary.create_file_writer(log_dir)

        ckpt_dir = "saved_model/rl_ckpt"
        state, start_epoch, global_step = load_checkpoint_if_exists(model, optimizer, ckpt_dir)

        best_val_loss = float('inf')
        best_epoch = 0
        early_stop_counter = 0
        accumulated_grads = None
        accum_step = 0

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
                loss = -tf.math.log(tf.sigmoid(beta * (policy_ratio - ref_ratio)) + 1e-8)
                if config.gradient_accumulation_steps > 1:
                    loss = loss / config.gradient_accumulation_steps
            grads = tape.gradient(loss, model.trainable_variables)
            return loss, grads

        with tqdm(total=total_steps, desc="DPO总进度", unit="step", position=0, leave=True) as total_pbar:
            total_pbar.update(global_step)

            for epoch in range(start_epoch, config.rl_epochs):
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
                    if step >= steps_per_epoch:
                        break

                    try:
                        loss, grads = dpo_step(chosen, rejected)

                        if check_gradients_nan_inf(grads):
                            lr_manager.on_nan_detected()
                            step += 1
                            global_step += 1
                            continue
                        else:
                            lr_manager.reset_nan_counter()

                        if config.gradient_accumulation_steps > 1:
                            if accumulated_grads is None:
                                accumulated_grads = [tf.zeros_like(v) for v in model.trainable_variables]
                            accumulated_grads = [acc + g for acc, g in zip(accumulated_grads, grads)]
                            accum_step += 1

                            if accum_step % config.gradient_accumulation_steps == 0:
                                optimizer.apply_gradients(zip(accumulated_grads, model.trainable_variables))
                                accumulated_grads = None
                        else:
                            optimizer.apply_gradients(zip(grads, model.trainable_variables))

                        loss_val = loss.numpy() * (config.gradient_accumulation_steps if config.gradient_accumulation_steps > 1 else 1)
                        epoch_losses.append(loss_val)
                        lr_manager.on_step_end(loss_val)

                        with summary_writer.as_default():
                            tf.summary.scalar('train_loss', loss_val, step=global_step)
                            tf.summary.scalar('learning_rate', optimizer.learning_rate(global_step), step=global_step)
                            tf.summary.scalar('grad_norm', get_gradient_norm(grads), step=global_step)

                        step += 1
                        global_step += 1
                        epoch_pbar.update(1)
                        epoch_pbar.set_postfix(loss=f"{loss_val:.4f}")
                        total_pbar.update(1)
                        total_pbar.set_postfix(epoch=f"{epoch+1}", loss=f"{loss_val:.4f}")

                    except tf.errors.ResourceExhaustedError as e:
                        print(f"\n  ⚠️ OOM 错误，跳过该 batch: {e}")
                        tf.keras.backend.clear_session()
                        continue

                epoch_pbar.close()
                avg_loss = np.mean(epoch_losses) if epoch_losses else 0

                print(f"\n  正在验证 DPO Epoch {epoch+1}...")
                val_loss, val_acc = evaluate_dpo(model, ref_model, val_dataset, beta)
                print(f"  [验证] Loss: {val_loss:.4f}, 准确率: {val_acc:.2%}")

                with summary_writer.as_default():
                    tf.summary.scalar('val_loss', val_loss, step=global_step)
                    tf.summary.scalar('val_accuracy', val_acc, step=global_step)

                lr_reduced = lr_manager.on_epoch_end(val_loss)
                if lr_reduced:
                    early_stop_counter = 0

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    early_stop_counter = 0
                    save_best_model(model, "saved_model/rl_best", is_best=True)
                    print(f"  ★ 新的最佳模型！Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
                else:
                    early_stop_counter += 1
                    print(f"  当前最佳: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
                    print(f"  早停计数: {early_stop_counter}/{config.early_stop_patience}")

                if early_stop_counter >= config.early_stop_patience:
                    print(f"\n🛑 触发早停！")
                    break

                if (epoch + 1) % config.checkpoint_freq == 0:
                    save_checkpoint(model, optimizer, epoch + 1, global_step, ckpt_dir)

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

# ============================================================
# 工具函数
# ============================================================

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


# ============================================================
# 主程序（增强版）
# ============================================================

if __name__ == "__main__":
    keras.mixed_precision.set_global_policy("float32")
    VOCAB_PATH = "vocab.json"
    CORPUS_FOLDER = "./corpus"
    SFT_DATA_PATH = "sft_train.jsonl"
    SFT_VAL_PATH = "sft_val.jsonl"
    RL_DATA_PATH = "rl_train.jsonl"
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
        pretrain_lr=5e-4,
        sft_lr=1e-5,      # 已调高
        rl_lr=1e-6,       # 已调高
        pretrain_epochs=5,
        sft_epochs=20,
        rl_epochs=15,
        steps_per_epoch=300,
        gradient_accumulation_steps=4,  # 新增：等效 batch_size=16
        early_stop_patience=3,
        plateau_patience=5,
        enable_mixed_precision=True,
    )

    print(f"词汇表大小: {config.vocab_size}")
    print(f"📊 训练配置: SFT={config.sft_epochs}轮, RL={config.rl_epochs}轮")
    print(f"📊 自动处理: 梯度累积={config.gradient_accumulation_steps}, 早停耐心={config.early_stop_patience}")
    print(f"📊 学习率: 预训练={config.pretrain_lr}, SFT={config.sft_lr}, RL={config.rl_lr}")

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
    print("    ├── model.keras      (Keras 3 格式)")
    print("    ├── savedmodel/      (SavedModel 格式)")
    print("    └── config.json      (模型配置)")
    print("  - SFT微调: saved_model/sft_final/")
    print("  - DPO强化: saved_model/rl_final/")
    print("="*50)
