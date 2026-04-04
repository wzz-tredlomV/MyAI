"""
生成器网络 - 条件GAN生成器，使用ResBlock-Up和AdaIN
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class InstanceNormalization(layers.Layer):
    """自定义 Instance Normalization 层（功能完整，支持 center/scale 可训练参数）"""
    def __init__(self, axis=-1, epsilon=1e-5, center=True, scale=True, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

    def build(self, input_shape):
        # 归一化的参数形状：通道数
        if self.axis == -1:
            self.param_shape = (input_shape[-1],)
        else:
            self.param_shape = (input_shape[self.axis],)

        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=self.param_shape,
                initializer='zeros',
                trainable=True
            )
        else:
            self.beta = None

        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=self.param_shape,
                initializer='ones',
                trainable=True
            )
        else:
            self.gamma = None
        super().build(input_shape)

    def call(self, inputs, training=None):
        # 计算均值和方差的轴：除 axis 指定的通道轴外，所有其他轴（batch, height, width）
        reduction_axes = [i for i in range(inputs.shape.rank) if i != self.axis]
        mean = tf.reduce_mean(inputs, axis=reduction_axes, keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=reduction_axes, keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)

        if self.scale:
            normalized = normalized * self.gamma
        if self.center:
            normalized = normalized + self.beta
        return normalized

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class AdaIN(layers.Layer):
    """
    Adaptive Instance Normalization (AdaIN)
    使用文本条件学习gamma和beta参数
    """

    def __init__(self, num_channels: int, condition_dim: int = 256, **kwargs):
        """
        初始化AdaIN

        Args:
            num_channels: 特征图通道数
            condition_dim: 条件向量维度
        """
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.condition_dim = condition_dim

        # 实例归一化（不使用可训练的beta/gamma，因为后面会从condition学习）
        self.instance_norm = InstanceNormalization(
            axis=-1, center=False, scale=False,
            name="instance_norm"
        )

        # 从条件向量学习gamma和beta
        self.gamma_mlp = layers.Dense(
            num_channels, name="gamma_mlp",
            kernel_initializer="zeros"
        )
        self.beta_mlp = layers.Dense(
            num_channels, name="beta_mlp",
            kernel_initializer="zeros"
        )

    def call(self, inputs, training=None):
        """
        前向传播

        Args:
            inputs: (feature_map, condition) 元组
                feature_map: [B, H, W, C]
                condition: [B, condition_dim]

        Returns:
            AdaIN后的特征图 [B, H, W, C]
        """
        feature_map, condition = inputs

        # 实例归一化
        normalized = self.instance_norm(feature_map)

        # 学习gamma和beta
        gamma = self.gamma_mlp(condition)  # [B, C]
        beta = self.beta_mlp(condition)    # [B, C]

        # 扩展维度以广播
        gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)  # [B, 1, 1, C]
        beta = tf.expand_dims(tf.expand_dims(beta, 1), 1)    # [B, 1, 1, C]

        # 应用AdaIN
        output = gamma * normalized + beta

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_channels": self.num_channels,
            "condition_dim": self.condition_dim,
        })
        return config


class ResBlockUp(layers.Layer):
    """
    上采样残差块
    包含上采样、卷积和AdaIN
    """

    def __init__(self, out_channels: int, condition_dim: int = 256,
                 use_upsample: bool = True, **kwargs):
        """
        初始化上采样残差块

        Args:
            out_channels: 输出通道数
            condition_dim: 条件向量维度
            use_upsample: 是否使用上采样
        """
        super().__init__(**kwargs)

        self.out_channels = out_channels
        self.condition_dim = condition_dim
        self.use_upsample = use_upsample

        # 主路径
        if use_upsample:
            self.upsample = layers.UpSampling2D(size=2, interpolation="nearest", name="upsample")

        self.conv1 = layers.Conv2D(
            out_channels, 3, padding="same", use_bias=False,
            name="conv1"
        )
        self.adain1 = AdaIN(out_channels, condition_dim, name="adain1")
        self.activation1 = layers.ReLU(name="relu1")

        self.conv2 = layers.Conv2D(
            out_channels, 3, padding="same", use_bias=False,
            name="conv2"
        )
        self.adain2 = AdaIN(out_channels, condition_dim, name="adain2")
        self.activation2 = layers.ReLU(name="relu2")

        # 捷径路径
        self.shortcut_conv = layers.Conv2D(
            out_channels, 1, padding="same", use_bias=False,
            name="shortcut_conv"
        )
        if use_upsample:
            self.shortcut_upsample = layers.UpSampling2D(size=2, interpolation="nearest", name="shortcut_upsample")

    def call(self, inputs, training=None):
        """
        前向传播

        Args:
            inputs: (feature_map, condition) 元组
                feature_map: [B, H, W, C_in]
                condition: [B, condition_dim]

        Returns:
            输出特征图 [B, H*2, W*2, out_channels] (如果上采样)
        """
        feature_map, condition = inputs

        # 主路径
        x = feature_map
        if self.use_upsample:
            x = self.upsample(x)

        x = self.conv1(x)
        x = self.adain1([x, condition], training=training)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.adain2([x, condition], training=training)

        # 捷径路径
        shortcut = feature_map
        if self.use_upsample:
            shortcut = self.shortcut_upsample(shortcut)
        shortcut = self.shortcut_conv(shortcut)

        # 残差连接
        output = self.activation2(x + shortcut)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_channels": self.out_channels,
            "condition_dim": self.condition_dim,
            "use_upsample": self.use_upsample,
        })
        return config


class Generator(keras.Model):
    """
    条件GAN生成器
    噪声 + 文本条件 -> 图像
    """

    def __init__(self, noise_dim: int = 100, condition_dim: int = 256,
                 initial_filters: int = 512, num_layers: int = 4,
                 output_channels: int = 3, **kwargs):
        """
        初始化生成器

        Args:
            noise_dim: 噪声向量维度
            condition_dim: 条件向量维度
            initial_filters: 初始通道数
            num_layers: 上采样层数
            output_channels: 输出图像通道数
        """
        super().__init__(**kwargs)

        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.initial_filters = initial_filters
        self.num_layers = num_layers
        self.output_channels = output_channels

        # 计算初始空间尺寸
        self.initial_size = 4

        # 条件融合层（噪声 + 文本条件）
        self.condition_fusion = layers.Dense(
            noise_dim + condition_dim, name="condition_fusion"
        )

        # 全连接层，将融合后的向量映射到初始特征图
        self.fc = layers.Dense(
            self.initial_size * self.initial_size * initial_filters,
            name="fc"
        )
        self.reshape = layers.Reshape(
            (self.initial_size, self.initial_size, initial_filters),
            name="reshape"
        )

        # 上采样残差块
        self.res_blocks = []
        filters = initial_filters
        for i in range(num_layers):
            filters = filters // 2 if i > 0 else initial_filters
            self.res_blocks.append(
                ResBlockUp(
                    filters, condition_dim,
                    use_upsample=True,
                    name=f"resblock_up_{i}"
                )
            )

        # 输出层
        self.output_conv = layers.Conv2D(
            output_channels, 3, padding="same",
            activation="tanh", name="output_conv"
        )

    def call(self, inputs, training=None):
        """
        前向传播

        Args:
            inputs: (noise, condition) 元组
                noise: [B, noise_dim]
                condition: [B, condition_dim]

        Returns:
            生成图像 [B, H, W, 3]
        """
        noise, condition = inputs

        # 融合噪声和条件
        fused = tf.concat([noise, condition], axis=-1)
        fused = self.condition_fusion(fused)

        # 映射到初始特征图
        x = self.fc(fused)
        x = self.reshape(x)

        # 通过上采样残差块
        for res_block in self.res_blocks:
            x = res_block([x, condition], training=training)

        # 输出层
        output = self.output_conv(x)

        return output

    def generate(self, noise: tf.Tensor, condition: tf.Tensor,
                 training: bool = False) -> tf.Tensor:
        """
        生成图像的便捷方法

        Args:
            noise: 噪声向量 [B, noise_dim]
            condition: 文本条件向量 [B, condition_dim]
            training: 是否训练模式

        Returns:
            生成图像 [B, H, W, 3]
        """
        return self([noise, condition], training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "noise_dim": self.noise_dim,
            "condition_dim": self.condition_dim,
            "initial_filters": self.initial_filters,
            "num_layers": self.num_layers,
            "output_channels": self.output_channels,
        })
        return config


class GeneratorEMA(keras.Model):
    """
    生成器的EMA版本，用于推理
    """

    def __init__(self, generator: Generator, decay: float = 0.999, **kwargs):
        """
        初始化EMA生成器

        Args:
            generator: 原始生成器
            decay: EMA衰减率
        """
        super().__init__(**kwargs)

        self.generator = generator
        self.decay = decay

        # 创建EMA变量
        self.ema_weights = []
        for var in generator.trainable_variables:
            ema_var = tf.Variable(var, trainable=False, name=f"{var.name}_ema")
            self.ema_weights.append(ema_var)

    def update(self):
        """更新EMA权重"""
        for var, ema_var in zip(self.generator.trainable_variables, self.ema_weights):
            ema_var.assign(self.decay * ema_var + (1 - self.decay) * var)

    def call(self, inputs, training=None):
        """使用EMA权重进行前向传播"""
        # 临时替换权重
        original_weights = []
        for var, ema_var in zip(self.generator.trainable_variables, self.ema_weights):
            original_weights.append(var.value())
            var.assign(ema_var.value())

        output = self.generator(inputs, training=False)

        # 恢复原始权重
        for var, orig_val in zip(self.generator.trainable_variables, original_weights):
            var.assign(orig_val)

        return output


def create_generator(**kwargs) -> Generator:
    """
    创建生成器的工厂函数

    Args:
        **kwargs: 生成器参数

    Returns:
        生成器模型
    """
    return Generator(**kwargs)
