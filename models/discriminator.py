"""
判别器网络 - PatchGAN判别器（所有层在 __init__ 中创建，兼容 tf.function）
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Discriminator(keras.Model):
    def __init__(self, condition_dim: int = 256, initial_filters: int = 64,
                 num_layers: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.condition_dim = condition_dim
        self.initial_filters = initial_filters
        self.num_layers = num_layers

        # 条件投影层
        self.condition_projection = layers.Dense(
            condition_dim, name="condition_projection"
        )

        # 构建下采样卷积层列表
        self.conv_layers = []
        filters = initial_filters
        for i in range(num_layers):
            strides = 2 if i < num_layers - 1 else 1
            conv = layers.Conv2D(
                filters, kernel_size=4, strides=strides, padding="same",
                use_bias=False, name=f"conv_{i}"
            )
            lrelu = layers.LeakyReLU(0.2, name=f"lrelu_{i}")
            self.conv_layers.append((conv, lrelu, strides))
            filters = min(filters * 2, 512)

        # 最终输出层
        self.final_conv = layers.Conv2D(
            1, kernel_size=4, strides=1, padding="same",
            use_bias=False, name="final_conv"
        )

    def call(self, inputs, training=None):
        image, condition = inputs
        batch_size = tf.shape(image)[0]
        height = tf.shape(image)[1]
        width = tf.shape(image)[2]

        # 投影条件并扩展到空间维度
        proj = self.condition_projection(condition)           # [B, cond_dim]
        proj = tf.reshape(proj, [batch_size, 1, 1, self.condition_dim])
        proj = tf.tile(proj, [1, height, width, 1])

        # 拼接图像和条件
        x = tf.concat([image, proj], axis=-1)                 # [B, H, W, 3+cond_dim]

        # 依次通过每个卷积块
        for conv, lrelu, strides in self.conv_layers:
            x = conv(x)
            x = lrelu(x)

        # 最终输出
        output = self.final_conv(x)
        return output

    def discriminate(self, image, condition, training=False):
        return self([image, condition], training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "condition_dim": self.condition_dim,
            "initial_filters": self.initial_filters,
            "num_layers": self.num_layers,
        })
        return config


class MultiScaleDiscriminator(keras.Model):
    def __init__(self, condition_dim: int = 256, num_discriminators: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.condition_dim = condition_dim
        self.num_discriminators = num_discriminators
        self.discriminators = [
            Discriminator(condition_dim, name=f"discriminator_{i}")
            for i in range(num_discriminators)
        ]
        self.downsample = layers.AveragePooling2D(pool_size=3, strides=2, padding="same")

    def call(self, inputs, training=None):
        image, condition = inputs
        outputs = []
        x = image
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc([x, condition], training=training))
            if i < len(self.discriminators) - 1:
                x = self.downsample(x)
        return outputs


def hinge_loss_d(real_logits, fake_logits):
    real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_logits))
    fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_logits))
    return real_loss + fake_loss


def hinge_loss_g(fake_logits):
    return -tf.reduce_mean(fake_logits)


def create_discriminator(use_multiscale: bool = False, **kwargs) -> keras.Model:
    if use_multiscale:
        return MultiScaleDiscriminator(**kwargs)
    else:
        return Discriminator(**kwargs)
