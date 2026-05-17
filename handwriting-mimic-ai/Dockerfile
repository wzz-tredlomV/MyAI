FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY src/ ./src/
COPY main.py .
COPY train.py .

# 创建数据目录
RUN mkdir -p data/style_bank output models

# 设置入口点
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
