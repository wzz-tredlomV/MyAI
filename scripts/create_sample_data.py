"""
创建示例数据脚本 - 生成测试用的description.txt和示例图片
"""

import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# 示例描述模板
SAMPLE_DESCRIPTIONS = [
    "a red apple on a wooden table",
    "a blue car parked on the street",
    "a white cat sleeping on a sofa",
    "a green tree in the park",
    "a yellow flower in a vase",
    "a black dog running on grass",
    "a brown horse in a field",
    "a purple butterfly on a leaf",
    "an orange sunset over the ocean",
    "a pink rose in a garden",
    "a gray building in the city",
    "a silver airplane in the sky",
    "a golden sunset behind mountains",
    "a white cloud in the blue sky",
    "a red balloon floating in air",
    "a green frog sitting on a lily pad",
    "a brown bear in the forest",
    "a colorful bird on a branch",
    "a bright star in the night sky",
    "a small boat on a calm lake",
]


def create_sample_image(description: str, size: tuple = (128, 128)) -> Image.Image:
    """
    创建示例图片
    
    Args:
        description: 描述文本
        size: 图片尺寸
        
    Returns:
        PIL图像
    """
    # 创建随机颜色的背景
    bg_color = tuple(np.random.randint(50, 200, 3).tolist())
    image = Image.new('RGB', size, bg_color)
    
    # 添加一些随机形状
    draw = ImageDraw.Draw(image)
    
    # 添加随机矩形
    for _ in range(3):
        x1 = np.random.randint(0, size[0] // 2)
        y1 = np.random.randint(0, size[1] // 2)
        x2 = np.random.randint(size[0] // 2, size[0])
        y2 = np.random.randint(size[1] // 2, size[1])
        color = tuple(np.random.randint(0, 255, 3).tolist())
        draw.rectangle([x1, y1, x2, y2], fill=color)
    
    # 添加随机圆形
    for _ in range(2):
        x = np.random.randint(20, size[0] - 20)
        y = np.random.randint(20, size[1] - 20)
        r = np.random.randint(10, 30)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
    
    return image


def create_sample_dataset(output_dir: str, num_samples: int = 100):
    """
    创建示例数据集
    
    Args:
        output_dir: 输出目录
        num_samples: 样本数量
    """
    output_path = Path(output_dir)
    image_dir = output_path / "image"
    
    # 创建目录
    image_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    (image_dir / "animals").mkdir(exist_ok=True)
    (image_dir / "nature").mkdir(exist_ok=True)
    (image_dir / "objects").mkdir(exist_ok=True)
    
    # 生成描述文件
    descriptions = []
    
    for i in range(num_samples):
        # 随机选择描述模板
        template = SAMPLE_DESCRIPTIONS[i % len(SAMPLE_DESCRIPTIONS)]
        
        # 添加一些变化
        if i >= len(SAMPLE_DESCRIPTIONS):
            template = f"{template} with beautiful lighting"
        
        # 确定子目录
        if "cat" in template or "dog" in template or "horse" in template or "bear" in template or "frog" in template or "bird" in template or "butterfly" in template:
            subdir = "animals"
        elif "tree" in template or "flower" in template or "rose" in template or "grass" in template or "forest" in template or "mountain" in template or "ocean" in template or "lake" in template or "sunset" in template or "sky" in template or "cloud" in template or "star" in template:
            subdir = "nature"
        else:
            subdir = "objects"
        
        # 创建图片
        # 随机尺寸
        sizes = [(64, 64), (128, 128), (256, 256), (128, 192), (192, 128)]
        size = sizes[i % len(sizes)]
        
        image = create_sample_image(template, size)
        
        # 保存图片
        image_filename = f"{subdir}/image_{i:04d}.png"
        image_path = image_dir / image_filename
        image.save(image_path)
        
        # 记录描述
        descriptions.append(f"image/{image_filename} --- {template}")
    
    # 保存描述文件
    description_file = output_path / "description.txt"
    with open(description_file, "w", encoding="utf-8") as f:
        f.write("\n".join(descriptions))
    
    print(f"示例数据集已创建:")
    print(f"  目录: {output_dir}")
    print(f"  图片: {image_dir}")
    print(f"  描述文件: {description_file}")
    print(f"  样本数: {num_samples}")


def main():
    parser = argparse.ArgumentParser(description="创建示例数据集")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="输出目录")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="样本数量")
    
    args = parser.parse_args()
    
    create_sample_dataset(args.output_dir, args.num_samples)


if __name__ == "__main__":
    main()
