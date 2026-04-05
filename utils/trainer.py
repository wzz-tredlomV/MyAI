"""
训练器 - 条件GAN训练循环
"""

import os
import time
import logging
from typing import Dict, List, Optional, Callable
from collections import defaultdict

import tensorflow as tf
from tensorflow import keras
import numpy as np

from models.generator import Generator, GeneratorEMA
from models.discriminator import Discriminator, hinge_loss_d, hinge_loss_g
from models.text_encoder import create_text_encoder


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """条件GAN训练器"""

    def __init__(self,
                 generator: Generator,
                 discriminator: Discriminator,
                 text_encoder: keras.Model,
                 g_optimizer: keras.optimizers.Optimizer,
                 d_optimizer: keras.optimizers.Optimizer,
                 noise_dim: int = 100,
                 adv_loss_weight: float = 1.0,
                 fm_loss_weight: float = 10.0,
                 ema_decay: float = 0.999,
                 checkpoint_dir: str = "./checkpoints",
                 log_dir: str = "./logs"):
        """
        初始化训练器

        Args:
            generator: 生成器
            discriminator: 判别器
            text_encoder: 文本编码器
            g_optimizer: 生成器优化器
            d_optimizer: 判别器优化器
            noise_dim: 噪声维度
            adv_loss_weight: 对抗损失权重
            fm_loss_weight: 特征匹配损失权重
            ema_decay: EMA衰减率
            checkpoint_dir: 检查点目录
            log_dir: 日志目录
        """
        self.generator = generator
        self.discriminator = discriminator
        self.text_encoder = text_encoder
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        self.noise_dim = noise_dim
        self.adv_loss_weight = adv_loss_weight
        self.fm_loss_weight = fm_loss_weight

        # 创建EMA生成器
        self.generator_ema = GeneratorEMA(generator, ema_decay)

        # 检查点和日志
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # TensorBoard
        self.summary_writer = tf.summary.create_file_writer(log_dir)

        # 训练状态
        self.global_step = 0
        self.epoch = 0

        # 损失历史
        self.loss_history = defaultdict(list)

    @tf.function
    def train_step_d(self, real_images: tf.Tensor, text_encoded: tf.Tensor,
                     attention_mask: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        判别器训练步骤
        """
        batch_size = tf.shape(real_images)[0]

        with tf.GradientTape() as tape:
            # 编码文本
            text_condition = self.text_encoder(
                [text_encoded, attention_mask], training=True
            )

            # 生成噪声
            noise = tf.random.normal([batch_size, self.noise_dim])

            # 生成假图
            fake_images = self.generator([noise, text_condition], training=True)
            # 确保生成图像与真实图像尺寸一致
            target_h = tf.shape(real_images)[1]
            target_w = tf.shape(real_images)[2]
            fake_images = tf.image.resize(fake_images, [target_h, target_w], method='bilinear')

            # 判别真实图
            real_logits = self.discriminator([real_images, text_condition], training=True)

            # 判别假图
            fake_logits = self.discriminator([fake_images, text_condition], training=True)

            # Hinge损失
            d_loss = hinge_loss_d(real_logits, fake_logits)

        # 计算梯度并更新
        gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(gradients, self.discriminator.trainable_variables)
        )

        return {
            "d_loss": d_loss,
            "d_real": tf.reduce_mean(real_logits),
            "d_fake": tf.reduce_mean(fake_logits),
        }

    @tf.function
    def train_step_g(self, real_images: tf.Tensor, text_encoded: tf.Tensor,
                     attention_mask: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        生成器训练步骤
        """
        batch_size = tf.shape(real_images)[0]

        # 编码文本（需要更新文本编码器）
        text_condition = self.text_encoder([text_encoded, attention_mask], training=True)

        # 生成噪声
        noise = tf.random.normal([batch_size, self.noise_dim])

        with tf.GradientTape(persistent=True) as tape:
            # 生成假图
            fake_images = self.generator([noise, text_condition], training=True)
            # 确保生成图像与真实图像尺寸一致
            target_h = tf.shape(real_images)[1]
            target_w = tf.shape(real_images)[2]
            fake_images = tf.image.resize(fake_images, [target_h, target_w], method='bilinear')

            # 判别假图
            fake_logits = self.discriminator([fake_images, text_condition], training=True)

            # 对抗损失
            g_adv_loss = hinge_loss_g(fake_logits)

            # 特征匹配损失（L1损失）
            g_fm_loss = tf.reduce_mean(tf.abs(fake_images - real_images))

            # 总损失
            g_loss = self.adv_loss_weight * g_adv_loss + self.fm_loss_weight * g_fm_loss

        # 分别计算生成器和文本编码器的梯度
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        text_gradients = tape.gradient(g_loss, self.text_encoder.trainable_variables)
        
        # 删除持久化梯度带
        del tape

        # 应用梯度
        if g_gradients is not None:
            self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        if text_gradients is not None:
            self.g_optimizer.apply_gradients(zip(text_gradients, self.text_encoder.trainable_variables))

        # 更新EMA
        self.generator_ema.update()

        return {
            "g_loss": g_loss,
            "g_adv_loss": g_adv_loss,
            "g_fm_loss": g_fm_loss,
        }

    def train_step(self, batch: Dict[str, tf.Tensor]) -> Dict[str, float]:
        """
        完整的训练步骤（D + G）
        """
        real_images = batch["image"]
        text_encoded = batch["text_encoded"]
        attention_mask = batch["attention_mask"]

        # 训练判别器
        d_losses = self.train_step_d(real_images, text_encoded, attention_mask)

        # 训练生成器
        g_losses = self.train_step_g(real_images, text_encoded, attention_mask)

        # 合并损失
        losses = {**d_losses, **g_losses}

        # 记录损失
        for key, value in losses.items():
            self.loss_history[key].append(float(value))

        self.global_step += 1

        return {k: float(v) for k, v in losses.items()}

    def train_epoch(self, data_pipeline, steps_per_epoch: Optional[int] = None):
        """
        训练一个epoch
        """
        logger.info(f"开始 Epoch {self.epoch + 1}")

        epoch_start_time = time.time()
        epoch_losses = defaultdict(list)

        step = 0
        for bucket_id, batch in data_pipeline.get_dataset_iterator():
            if steps_per_epoch and step >= steps_per_epoch:
                break

            step_start_time = time.time()

            # 训练步骤
            losses = self.train_step(batch)

            # 累积损失
            for key, value in losses.items():
                epoch_losses[key].append(value)

            step_time = time.time() - step_start_time

            # 打印进度
            if step % 10 == 0:
                logger.info(
                    f"Epoch {self.epoch + 1}, Step {step}, "
                    f"Bucket {bucket_id}, "
                    f"D Loss: {losses['d_loss']:.4f}, "
                    f"G Loss: {losses['g_loss']:.4f}, "
                    f"Time: {step_time:.3f}s"
                )

            step += 1

        # 计算平均损失
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}

        epoch_time = time.time() - epoch_start_time
        logger.info(
            f"Epoch {self.epoch + 1} 完成, "
            f"时间: {epoch_time:.2f}s, "
            f"平均 D Loss: {avg_losses.get('d_loss', 0):.4f}, "
            f"平均 G Loss: {avg_losses.get('g_loss', 0):.4f}"
        )

        # 记录到TensorBoard
        with self.summary_writer.as_default():
            for key, value in avg_losses.items():
                tf.summary.scalar(f"epoch/{key}", value, step=self.epoch)
            tf.summary.scalar("epoch/time", epoch_time, step=self.epoch)

        self.epoch += 1

        return avg_losses

    def save_checkpoint(self, filename: Optional[str] = None):
        """
        保存检查点
        """
        if filename is None:
            filename = f"checkpoint_epoch_{self.epoch}"

        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        os.makedirs(checkpoint_path, exist_ok=True)

        # 保存模型权重
        self.generator.save_weights(os.path.join(checkpoint_path, "generator.weights.h5"))
        self.discriminator.save_weights(os.path.join(checkpoint_path, "discriminator.weights.h5"))
        self.text_encoder.save_weights(os.path.join(checkpoint_path, "text_encoder.weights.h5"))

        # 保存优化器状态
        checkpoint = tf.train.Checkpoint(
            g_optimizer=self.g_optimizer,
            d_optimizer=self.d_optimizer,
            generator=self.generator,
            discriminator=self.discriminator,
            text_encoder=self.text_encoder,
        )
        checkpoint.save(os.path.join(checkpoint_path, "ckpt"))

        logger.info(f"检查点已保存到: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        """
        # 加载模型权重
        self.generator.load_weights(os.path.join(checkpoint_path, "generator.weights.h5"))
        self.discriminator.load_weights(os.path.join(checkpoint_path, "discriminator.weights.h5"))
        self.text_encoder.load_weights(os.path.join(checkpoint_path, "text_encoder.weights.h5"))

        # 加载优化器状态
        checkpoint = tf.train.Checkpoint(
            g_optimizer=self.g_optimizer,
            d_optimizer=self.d_optimizer,
            generator=self.generator,
            discriminator=self.discriminator,
            text_encoder=self.text_encoder,
        )
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

        logger.info(f"检查点已从 {checkpoint_path} 加载")

    def generate_samples(self, descriptions: List[str],
                         text_encoder_preprocess: Callable,
                         target_size: int = 128,
                         seed: Optional[int] = None) -> np.ndarray:
        """
        生成样本图像
        """
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        # 编码文本
        text_encoded, attention_mask = text_encoder_preprocess(descriptions)
        text_encoded = tf.constant(text_encoded, dtype=tf.int32)
        attention_mask = tf.constant(attention_mask, dtype=tf.float32)

        # 获取文本条件
        text_condition = self.text_encoder(
            [text_encoded, attention_mask], training=False
        )

        # 生成噪声
        batch_size = len(descriptions)
        noise = tf.random.normal([batch_size, self.noise_dim])

        # 生成图像（使用EMA生成器）
        generated_images = self.generator_ema([noise, text_condition], training=False)

        # 转换到 [0, 255]
        generated_images = (generated_images + 1.0) * 127.5
        generated_images = tf.clip_by_value(generated_images, 0, 255)
        generated_images = tf.cast(generated_images, tf.uint8)

        return generated_images.numpy()


def create_trainer(config, generator, discriminator, text_encoder):
    """
    创建训练器的工厂函数
    """
    # 创建优化器
    g_optimizer = keras.optimizers.Adam(
        learning_rate=config.G_LR,
        beta_1=config.BETA1,
        beta_2=config.BETA2
    )

    d_optimizer = keras.optimizers.Adam(
        learning_rate=config.D_LR,
        beta_1=config.BETA1,
        beta_2=config.BETA2
    )

    return Trainer(
        generator=generator,
        discriminator=discriminator,
        text_encoder=text_encoder,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        noise_dim=config.NOISE_DIM,
        adv_loss_weight=config.ADV_LOSS_WEIGHT,
        fm_loss_weight=config.FM_LOSS_WEIGHT,
        ema_decay=config.EMA_DECAY,
        checkpoint_dir=config.CHECKPOINT_DIR,
        log_dir=config.LOG_DIR,
    )
