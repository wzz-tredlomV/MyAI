# 文本到图像生成系统 - 项目总结

## 项目概述

本项目实现了一个完整的基于TensorFlow 2.x的条件GAN（cGAN）文本到图像生成系统。系统支持从文本描述生成对应的图像，采用分桶训练策略和多种先进技术来稳定训练并提高生成质量。

## 已实现的功能

### 1. 数据集解析模块 (`utils/dataset_parser.py`)
- ✅ 解析 `description.txt` 格式（`路径 --- 描述`）
- ✅ 支持相对路径和子目录嵌套
- ✅ 路径验证和异常处理（跳过缺失图片、过滤空描述）
- ✅ 词汇表构建（词频统计、top-K保留）
- ✅ 文本编码（分词、ID转换、填充/截断）
- ✅ 注意力掩码生成
- ✅ 缓存机制（pickle格式）
- ✅ 数据集统计信息

### 2. 数据管道模块 (`utils/data_pipeline.py`)
- ✅ 分桶管理器（基于面积和宽高比）
- ✅ 4个默认桶配置（64×64, 128×128, 256×256, 512×512）
- ✅ 小桶合并策略
- ✅ tf.data.Dataset集成
- ✅ 图像预处理（缩放、填充、归一化到[-1,1]）
- ✅ 动态批次构建
- ✅ 数据预取优化

### 3. 文本编码器 (`models/text_encoder.py`)
- ✅ **简单方案**：Embedding + GlobalAveragePooling + Dense + LayerNorm
- ✅ **进阶方案**：Transformer Encoder（2-4层，多头注意力）
- ✅ 可学习位置编码
- ✅ 注意力掩码支持
- ✅ 工厂函数创建

### 4. 生成器网络 (`models/generator.py`)
- ✅ 噪声 + 文本条件融合
- ✅ 全连接层映射到初始特征图（4×4）
- ✅ ResBlock-Up上采样块（4层，每层2×上采样）
- ✅ AdaIN（自适应实例归一化）注入文本条件
  - MLP学习gamma和beta参数
  - 实例归一化 + 条件调制
- ✅ 残差连接
- ✅ Tanh输出激活
- ✅ EMA（指数移动平均）版本

### 5. 判别器网络 (`models/discriminator.py`)
- ✅ PatchGAN架构
- ✅ 谱归一化（Spectral Normalization）
- ✅ 条件判别（图像 + 文本条件拼接）
- ✅ Conv4×4-stride2-SN-LReLU结构
- ✅ Hinge Loss实现
- ✅ 多尺度判别器（可选）

### 6. 训练模块 (`utils/trainer.py`)
- ✅ 判别器训练步骤
  - 文本编码
  - 假图生成
  - 真假判别
  - Hinge Loss计算
- ✅ 生成器训练步骤
  - 对抗损失
  - 特征匹配损失
  - EMA更新
- ✅ 完整训练循环
- ✅ 分桶训练策略
- ✅ 检查点保存/加载
- ✅ TensorBoard日志记录
- ✅ 损失历史跟踪

### 7. 模型导出 (`utils/export.py`)
- ✅ SavedModel格式导出
- ✅ 多签名支持：
  - `serving_default`: 标准生成（输入描述，输出图像）
  - `encode_text`: 预编码文本（批量优化）
  - `generate_from_embedding`: 使用预编码向量生成
- ✅ 词汇表导出（JSON格式）
- ✅ 分桶配置导出
- ✅ 特殊标记导出

### 8. 推理模块 (`utils/inference.py`)
- ✅ 支持SavedModel和检查点两种格式
- ✅ 文本编码（分词、ID转换）
- ✅ 单条/批量生成
- ✅ 预编码文本嵌入生成
- ✅ 图像保存（PNG格式）
- ✅ 随机种子控制

### 9. 配置文件 (`config.py`)
- ✅ 数据相关配置
- ✅ 文本编码配置
- ✅ 图像相关配置
- ✅ 分桶配置
- ✅ 生成器/判别器配置
- ✅ 训练配置（学习率、批次大小等）
- ✅ 损失权重配置
- ✅ 检查点和日志配置
- ✅ 目录创建方法

### 10. 脚本和工具
- ✅ 主训练脚本 (`train.py`)
  - 命令行参数解析
  - GPU内存增长设置
  - 完整训练流程
- ✅ 推理脚本 (`inference.py`)
  - 单个/批量描述输入
  - 图像生成和保存
- ✅ 示例数据生成 (`scripts/create_sample_data.py`)
  - 生成随机图像
  - 创建description.txt
- ✅ 模型测试 (`scripts/test_model.py`)
  - 各组件单元测试
  - 训练步骤测试
- ✅ 完整示例 (`example.py`)
  - 演示所有功能

## 技术亮点

### 1. 分桶训练策略
- 根据图像尺寸自动分桶，减少填充浪费
- 支持不同宽高比的图像
- 小桶自动合并处理

### 2. AdaIN条件注入
- 通过自适应实例归一化注入文本条件
- 学习gamma和beta参数调制特征图
- 比简单的拼接或加法更有效

### 3. 谱归一化
- 稳定GAN训练
- 限制判别器的Lipschitz常数
- 无需额外的梯度惩罚

### 4. EMA生成器
- 推理时使用EMA版本
- 生成更稳定的图像
- 平滑训练过程中的参数波动

### 5. 缓存机制
- 数据集索引缓存
- 词汇表缓存
- 加速重复加载

## 代码结构

```
text2image/
├── config.py                 # 配置
├── train.py                  # 训练脚本
├── inference.py              # 推理脚本
├── example.py                # 示例
├── requirements.txt          # 依赖
├── README.md                 # 文档
├── PROJECT_SUMMARY.md        # 本文件
├── models/                   # 模型定义
│   ├── __init__.py
│   ├── generator.py          # 生成器（~350行）
│   ├── discriminator.py      # 判别器（~280行）
│   └── text_encoder.py       # 文本编码器（~280行）
├── utils/                    # 工具模块
│   ├── __init__.py
│   ├── dataset_parser.py     # 数据集解析（~350行）
│   ├── data_pipeline.py      # 数据管道（~320行）
│   ├── trainer.py            # 训练器（~380行）
│   ├── export.py             # 模型导出（~280行）
│   └── inference.py          # 推理模块（~350行）
└── scripts/                  # 辅助脚本
    ├── create_sample_data.py # 创建示例数据
    └── test_model.py         # 模型测试
```

## 总代码量

- Python代码：约 3500 行
- 注释和文档：约 800 行
- 总计：约 4300 行

## 使用方法

### 1. 安装依赖
```bash
pip install tensorflow pillow numpy
```

### 2. 创建示例数据
```bash
python scripts/create_sample_data.py --output_dir ./data --num_samples 100
```

### 3. 测试模型
```bash
python scripts/test_model.py --data_root ./data
```

### 4. 训练模型
```bash
python train.py --data_root ./data --epochs 100 --batch_size 32 --export
```

### 5. 推理生成
```bash
python inference.py --model_path ./saved_model \
    --description "a red apple on a wooden table" \
    --target_size 128 \
    --output_dir ./outputs
```

## 扩展建议

1. **添加更多损失函数**
   - perceptual loss（感知损失）
   - style loss（风格损失）
   - CLIP loss（语义对齐损失）

2. **改进生成器**
   - 使用Self-Attention层
   - 添加更多上采样层（支持1024×1024）
   - 尝试不同的条件注入方式

3. **改进判别器**
   - 添加投影判别器（Projection Discriminator）
   - 使用多尺度判别器
   - 添加辅助分类器

4. **训练技巧**
   - 渐进式增长（Progressive Growing）
   - 谱归一化生成器
   - 不同的学习率调度策略

5. **评估指标**
   - FID（Fréchet Inception Distance）
   - IS（Inception Score）
   - CLIP Score

## 参考论文

1. **cGAN**: Conditional Generative Adversarial Nets (Mirza & Osindero, 2014)
2. **DCGAN**: Unsupervised Representation Learning with Deep Convolutional GANs (Radford et al., 2015)
3. **Spectral Normalization**: Spectral Normalization for Generative Adversarial Networks (Miyato et al., 2018)
4. **AdaIN**: Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization (Huang & Belongie, 2017)
5. **Hinge Loss**: Geometric GAN (Lim & Ye, 2017)
6. **AttnGAN**: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks (Xu et al., 2018)

## 许可证

MIT License
