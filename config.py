"""
config.py - 共享配置和训练工具（改进版）
改进点：
  1. 降低默认学习率，提升 weight_decay
  2. 增加 label_smoothing 配置
  3. 增加 stochastic_depth_rate 配置
  4. 优化早停逻辑，增加 min_delta 防止微小波动触发
"""
import tensorflow as tf
from tensorflow import keras
import json
import os
import numpy as np
from dataclasses import dataclass, asdict


# ============================================================
# 配置类
# ============================================================
@dataclass
class ModelConfig:
    vocab_size: int = 5000
    embed_dim: int = 384
    num_heads: int = 6
    num_layers: int = 6
    max_len: int = 512
    dropout: float = 0.25          # 从 0.1 提升到 0.25，增强正则化
    seq_len: int = 512
    batch_size: int = 4

    pretrain_lr: float = 5e-5      # 从 3e-4 降低到 5e-5，更稳定
    sft_lr: float = 1e-5
    rl_lr: float = 1e-6
    weight_decay: float = 0.1      # 从 0.01 提升到 0.1，Transformer 标准值
    warmup_ratio: float = 0.05     # 从 0.1 缩短到 0.05，更快进入稳定期
    min_lr_ratio: float = 0.05

    label_smoothing: float = 0.1   # 新增：防止过度自信
    stochastic_depth_rate: float = 0.1  # 新增：Stochastic Depth 丢弃率

    pretrain_epochs: int = 12
    sft_epochs: int = 8
    rl_epochs: int = 5
    steps_per_epoch: int = 300

    gradient_accumulation_steps: int = 4
    early_stop_patience: int = 10
    plateau_patience: int = 8      # 从 10 缩短到 8，更快响应
    auto_lr_reduce_factor: float = 0.5
    enable_mixed_precision: bool = True
    max_nan_tolerance: int = 3
    checkpoint_freq: int = 1
    log_dir: str = "logs"

    # 验证集评估步数（新增）
    val_max_steps: int = 500

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


# ============================================================
# 学习率调度
# ============================================================
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_steps, total_steps, alpha=0.1, multiplier=1.0):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.alpha = alpha
        self.multiplier = multiplier

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)
        warmup_lr = self.initial_learning_rate * (step / warmup)
        progress = tf.clip_by_value((step - warmup) / tf.maximum(total - warmup, 1.0), 0.0, 1.0)
        cosine = 0.5 * (1.0 + tf.cos(np.pi * progress))
        decayed = (1.0 - self.alpha) * cosine + self.alpha
        decay_lr = self.initial_learning_rate * decayed
        lr = tf.cond(step < warmup, lambda: warmup_lr, lambda: decay_lr)
        return lr * self.multiplier

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "alpha": self.alpha,
            "multiplier": self.multiplier,
        }


# ============================================================
# 自适应学习率管理器（改进版）
# ============================================================
class AdaptiveLRManager:
    def __init__(self, optimizer, lr_schedule, config: ModelConfig, total_steps, initial_lr, verbose=True):
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.config = config
        self.total_steps = total_steps
        self.initial_lr = initial_lr
        self.loss_history = []
        self.val_loss_history = []
        self.plateau_counter = 0
        self.nan_counter = 0
        self.current_multiplier = 1.0
        self.lr_reduce_count = 0
        self.max_lr_reduces = 3
        self.verbose = verbose
        self.min_delta = 0.001  # 新增：最小改善阈值，忽略微小波动

    def on_step_end(self, loss_val):
        self.loss_history.append(float(loss_val))
        if len(self.loss_history) > 100:
            self.loss_history.pop(0)

    def on_epoch_end(self, val_loss):
        """改进：基于 min_delta 的早停判断，避免微小波动"""
        should_stop = False
        self.val_loss_history.append(float(val_loss))

        if len(self.val_loss_history) < 3:
            return should_stop

        # 检查连续上升（带 min_delta 阈值）
        last_3 = self.val_loss_history[-3:]
        is_rising = all(
            last_3[i] + self.min_delta < last_3[i+1] 
            for i in range(len(last_3)-1)
        )

        if is_rising:
            self.plateau_counter += 1
            if self.verbose:
                print(f"  ⚠️ Val Loss 连续上升 {self.plateau_counter}/{self.config.plateau_patience}")
        else:
            self.plateau_counter = max(0, self.plateau_counter - 1)

        if self.plateau_counter >= self.config.plateau_patience:
            if self.verbose:
                print(f"⏹️ 早停触发: 连续 {self.config.plateau_patience} 个 epoch Val Loss 上升")
            should_stop = True

        # 新增：如果最近 5 个 epoch 都没有改善超过 min_delta，也降低学习率
        if len(self.val_loss_history) >= 5:
            recent_5 = self.val_loss_history[-5:]
            best_recent = min(recent_5[:-1]) if len(recent_5) > 1 else float('inf')
            if recent_5[-1] > best_recent - self.min_delta and recent_5[-1] > 3.0:
                self._reduce_lr()
                self.plateau_counter = 0

        return should_stop

    def _reduce_lr(self):
        if self.lr_reduce_count >= self.max_lr_reduces:
            if self.verbose:
                print(f"🔧 学习率已达最大降低次数 ({self.max_lr_reduces})，不再降低")
            return
        self.current_multiplier *= self.config.auto_lr_reduce_factor
        self.lr_reduce_count += 1
        new_lr = WarmupCosineDecay(
            initial_learning_rate=self.lr_schedule.initial_learning_rate,
            warmup_steps=self.lr_schedule.warmup_steps,
            total_steps=self.lr_schedule.total_steps,
            alpha=self.lr_schedule.alpha,
            multiplier=self.current_multiplier
        )
        self.optimizer.learning_rate = new_lr
        self.lr_schedule = new_lr
        if self.verbose:
            print(f"🔧 自动降低学习率! 当前乘数: {self.current_multiplier:.3f} (第 {self.lr_reduce_count} 次)")

    def on_nan_detected(self):
        self.nan_counter += 1
        if self.verbose:
            print(f"  ⚠️ NaN/Inf ({self.nan_counter}/{self.config.max_nan_tolerance})")
        if self.nan_counter >= self.config.max_nan_tolerance:
            self._reduce_lr()
            self.nan_counter = 0
            return True
        return False

    def reset_nan_counter(self):
        self.nan_counter = 0

    def get_current_lr(self, step):
        return float(self.lr_schedule(step))
