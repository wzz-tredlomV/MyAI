# ✍️ Handwriting Mimic AI

基于深度学习的矢量字迹模仿系统。从单张手写样本中提取风格特征，生成可无限缩放的 SVG 矢量字迹。

## 🎯 核心功能

- **风格提取**：从手写图片提取可复用的风格向量（JSON格式保存）
- **矢量生成**：输出 SVG 格式，支持无限缩放不失真
- **风格银行**：管理多个字迹风格，随时切换使用
- **笔画级控制**：支持8种笔画类型（直线、曲线、圆弧、钩、顿笔等）

## 📁 项目结构

```
handwriting-mimic-ai/
├── src/
│   ├── vector_style_encoder.py   # 风格编码器
│   └── vector_generator.py       # 矢量生成器 + SVG渲染
├── main.py                        # CLI主入口
├── train.py                       # 训练脚本
├── tests/                         # 单元测试
├── data/
│   └── style_bank/               # 风格银行存储
├── models/                        # 模型检查点
├── output/                        # 生成结果
└── .github/workflows/ci.yml      # CI/CD配置
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 提取字迹风格

```bash
python main.py extract   --image samples/my_writing.jpg   --style-id my_style
```

提取的风格将保存为 `data/style_bank/my_style.json`，包含：
- 全局风格向量（256维）
- 笔画微观特征（64维）
- 统计特征（倾斜度、笔画密度等）
- 局部空间风格图

### 3. 生成模仿字迹

```bash
python main.py generate   --text "你好世界"   --style-id my_style   --output output/
```

输出为 SVG 矢量文件，可直接用于印刷、网页或设计软件。

### 4. 管理风格

```bash
# 列出所有风格
python main.py list-styles

# 查看风格详情
python main.py info --style-id my_style
```

## 🧠 技术架构

### 风格编码器
- **多尺度卷积**：捕获不同粗细的笔画特征
- **统计特征提取**：倾斜度、笔画密度、边缘粗糙度等可解释特征
- **残差连接**：深层特征提取，保留细节

### 生成器
- **AdaIN（自适应实例归一化）**：将风格向量注入内容特征
- **笔画参数预测**：输出10维笔画参数（类型、坐标、粗细、压力、曲率）
- **存在性图**：预测笔画位置热力图

### SVG渲染器
- 支持8种笔画类型：直线、二次/三次贝塞尔曲线、圆弧、点、钩、按笔、提笔
- 笔画粗细和压力映射到 SVG stroke-width 和 opacity
- 保留风格元数据在 SVG 的 `<metadata>` 中

## 🏋️ 训练模型

准备成对数据集（同一个人的手写样本 + 对应标准字体骨架）：

```bash
python train.py   --data data/train/   --epochs 100   --batch-size 16   --save-dir models/
```

数据集结构：
```
data/train/
├── writer_001/
│   ├── char_001.png
│   ├── char_002.png
│   └── ...
├── writer_002/
│   └── ...
└── skeletons/
    ├── char_001.png
    └── ...
```

## 🧪 测试

```bash
pytest tests/ -v --cov=src
```

## 🐳 Docker 使用

```bash
docker build -t handwriting-mimic-ai .
docker run -v $(pwd)/samples:/app/samples handwriting-mimic-ai   python main.py extract --image samples/my.jpg --style-id docker_style
```

## 📄 许可证

MIT License
