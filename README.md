# 文本到图像生成系统 (Text-to-Image Generation)

基于TensorFlow 2.x实现的条件GAN（cGAN）文本到图像生成系统。

## 项目结构

```
.
├── config.py                 # 配置文件
├── train.py                  # 主训练脚本
├── inference.py              # 推理脚本
├── models/                   # 模型定义
│   ├── __init__.py
│   ├── generator.py          # 生成器（ResBlock-Up + AdaIN）
│   ├── discriminator.py      # 判别器（PatchGAN）
│   └── text_encoder.py       # 文本编码器
├── utils/                    # 工具模块
│   ├── __init__.py
│   ├── dataset_parser.py     # 数据集解析器
│   ├── data_pipeline.py      # 数据管道（分桶策略）
│   ├── trainer.py            # 训练器
│   ├── export.py             # 模型导出
│   └── inference.py          # 推理模块
├── scripts/                  # 辅助脚本
│   ├── create_sample_data.py # 创建示例数据
│   └── test_model.py         # 模型测试
└── README.md                 # 本文件
```

## 功能特性

### 1. 数据集解析
- 支持 `description.txt` 格式（`路径 --- 描述`）
- 自动处理相对路径和子目录
- 路径验证和异常处理
- 词汇表构建和文本编码
- 缓存机制加速重复加载

### 2. 数据管道
- **分桶策略**：根据图像尺寸自动分桶
  - 桶0: 面积 < 64×64
  - 桶1: 64×64 ≤ 面积 < 128×128
  - 桶2: 128×128 ≤ 面积 < 256×256
  - 桶3: 面积 ≥ 256×256
- 动态批次构建
- 图像预处理（缩放、填充、归一化）

### 3. 文本编码器
- **简单方案**：Embedding + GlobalAveragePooling + Dense
- **进阶方案**：Transformer Encoder（2-4层，多头注意力）
- 支持注意力掩码处理变长序列

### 4. 生成器网络
- 噪声 + 文本条件融合
- ResBlock-Up上采样块
- AdaIN（自适应实例归一化）注入文本条件
- 输出尺寸：64×64, 128×128, 256×256, 512×512

### 5. 判别器网络
- PatchGAN架构
- 谱归一化（Spectral Normalization）
- 条件判别（图像 + 文本条件拼接）

### 6. 训练
- Hinge Loss对抗损失
- 特征匹配损失
- EMA（指数移动平均）生成器
- TensorBoard日志记录
- 检查点保存和恢复

### 7. 导出与推理
- SavedModel格式导出
- 多签名支持：
  - `serving_default`: 标准生成
  - `encode_text`: 预编码文本
  - `generate_from_embedding`: 使用预编码向量生成
- 批量推理支持

## 快速开始

### 安装依赖

```bash
pip install tensorflow pillow numpy
```

### 1. 创建示例数据

```bash
python scripts/create_sample_data.py --output_dir ./data --num_samples 100
```

### 2. 测试模型组件

```bash
python scripts/test_model.py --data_root ./data
```

### 3. 训练模型

```bash
# 基础训练
python train.py --data_root ./data --epochs 100 --batch_size 32

# 使用Transformer编码器
python train.py --data_root ./data --encoder_type transformer --epochs 100

# 从检查点恢复
python train.py --data_root ./data --resume ./checkpoints/checkpoint_epoch_50

# 训练并导出
python train.py --data_root ./data --epochs 100 --export
```

### 4. 推理生成

```bash
# 单个描述
python inference.py --model_path ./saved_model \
    --description "a red apple on a wooden table" \
    --target_size 128 \
    --output_dir ./outputs

# 批量生成
python inference.py --model_path ./saved_model \
    --description_file ./descriptions.txt \
    --target_size 128 \
    --batch_size 8 \
    --output_dir ./outputs
```

## 数据集格式

### 目录结构

```
data/
├── image/
│   ├── image1.png
│   ├── image2.png
│   └── subfolder/
│       └── image3.png
└── description.txt
```

### description.txt 格式

```
image/image1.png --- A dog sitting on grass
image/image2.png --- A red pen on white desk
image/subfolder/image3.png --- A car in the street
```

- 使用 ` --- `（空格+三个连字符+空格）作为分隔符
- 支持子目录嵌套
- 相对路径以 `description.txt` 所在目录为根

## 配置参数

主要配置参数在 `config.py` 中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 32 | 批次大小 |
| `EPOCHS` | 100 | 训练轮数 |
| `G_LR` | 0.0002 | 生成器学习率 |
| `D_LR` | 0.0002 | 判别器学习率 |
| `NOISE_DIM` | 100 | 噪声维度 |
| `MAX_VOCAB_SIZE` | 10000 | 最大词汇表大小 |
| `MAX_TEXT_LENGTH` | 20 | 最大文本长度 |
| `TEXT_HIDDEN_DIM` | 256 | 文本编码维度 |

## 模型架构详情

### 生成器

```
输入:
    噪声 z: [B, 100]
    文本条件 c: [B, 256]
    │
    ▼
条件融合: z 与 c 拼接 → [B, 356]
    ↓
Dense → Reshape 至 4×4×512
    ↓
ResBlock-Up × 4（每层上采样2×，通道减半）
    - 每层注入文本条件（AdaIN）
    - 分辨率: 4→8→16→32→64
    ↓
Conv3×3 → Tanh → 输出图像 [B, H, W, 3]
```

### 判别器

```
输入:
    图像: [B, H, W, 3]
    文本条件: [B, 256]（复制为空间尺寸后拼接）
    │
    ▼
拼接 → [B, H, W, 259]
    ↓
Conv4×4-stride2-SN-LReLU × 3
    ↓
Conv4×4-stride1-SN-LReLU
    ↓
Conv4×4-stride1-SN（输出 logits）
    ↓
输出: [B, H/8, W/8, 1]
```

## 训练技巧

1. **分桶训练**：不同尺寸的图像分到不同桶，减少填充浪费
2. **谱归一化**：稳定GAN训练
3. **EMA生成器**：用于推理，生成更稳定的图像
4. **特征匹配损失**：帮助生成器学习真实数据分布
5. **梯度惩罚**：可选的WGAN-GP变体

## 监控与评估

训练过程中会自动记录：
- 损失曲线（TensorBoard）
- 生成样本
- 检查点

查看TensorBoard：
```bash
tensorboard --logdir ./logs
```

## 许可证

MIT License
