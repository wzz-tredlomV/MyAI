"""
矢量字迹风格编码器
从手写样本图片中提取可复用的风格向量，并保存为JSON格式
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


class ResidualBlock(nn.Module):
    """残差块，用于深层特征提取"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class StrokeFeatureExtractor(nn.Module):
    """
    笔画级特征提取器
    提取笔画粗细、曲率、倾斜角度等微观特征
    """
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.out_dim = out_dim

        # 多尺度卷积捕获不同粗细的笔画
        self.conv_1x1 = nn.Conv2d(1, 32, 1)
        self.conv_3x3 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv_5x5 = nn.Conv2d(1, 32, 5, padding=2)

        self.fusion = nn.Sequential(
            nn.Conv2d(96, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, out_dim, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        c1 = F.relu(self.conv_1x1(x))
        c3 = F.relu(self.conv_3x3(x))
        c5 = F.relu(self.conv_5x5(x))
        fused = torch.cat([c1, c3, c5], dim=1)
        return self.fusion(fused).view(x.size(0), -1)


class HandwritingStyleEncoder(nn.Module):
    """
    完整的字迹风格编码器
    输出结构化的风格描述，可序列化为JSON保存
    """
    def __init__(self, 
                 global_dim: int = 256,
                 local_dim: int = 128,
                 stroke_dim: int = 64):
        super().__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.stroke_dim = stroke_dim

        # 主干特征提取
        self.backbone = nn.Sequential(
            # 输入: [B, 1, 128, 128]
            nn.Conv2d(1, 64, 4, 2, 1),    # -> [B, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),   # -> [B, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            ResidualBlock(128),

            nn.Conv2d(128, 256, 4, 2, 1),  # -> [B, 256, 16, 16]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            ResidualBlock(256),

            nn.Conv2d(256, 512, 4, 2, 1),  # -> [B, 512, 8, 8]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            ResidualBlock(512),
        )

        # 全局风格向量（整体书写特征）
        self.global_style_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, global_dim)
        )

        # 局部空间风格图（空间分布特征）
        self.local_style_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, local_dim, 3, 1, 1),
        )

        # 笔画微观特征
        self.stroke_extractor = StrokeFeatureExtractor(stroke_dim)

        # 统计特征提取（倾斜度、笔画密度等）
        self.statistical_fc = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

    def extract_statistical_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取可解释的统计特征"""
        batch_size = x.size(0)
        stats = []

        for i in range(batch_size):
            img = x[i, 0].cpu().numpy()

            # 二值化
            _, binary = cv2.threshold((img * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

            # 笔画密度
            density = np.sum(binary > 0) / binary.size

            # 水平/垂直投影方差（反映结构分布）
            h_proj = np.sum(binary, axis=1)
            v_proj = np.sum(binary, axis=0)
            h_var = np.var(h_proj) / (np.max(h_proj) + 1e-5)
            v_var = np.var(v_proj) / (np.max(v_proj) + 1e-5)

            # 倾斜估计（使用矩）
            moments = cv2.moments(binary)
            if moments['mu02'] != 0:
                skew = moments['mu11'] / (moments['mu02'] + 1e-5)
            else:
                skew = 0

            # 笔画宽度估计
            dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            avg_width = np.mean(dist_transform[dist_transform > 0]) * 2 if np.any(dist_transform > 0) else 0

            # 边缘粗糙度
            edges = cv2.Canny(binary, 50, 150)
            edge_ratio = np.sum(edges > 0) / edges.size

            # 连通区域数量（反映断笔/连笔特征）
            num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary)

            # 宽高比
            h, w = binary.shape
            aspect_ratio = w / h if h > 0 else 1

            # 重心位置
            cy = moments['m01'] / (moments['m00'] + 1e-5) / h if moments['m00'] > 0 else 0.5
            cx = moments['m10'] / (moments['m00'] + 1e-5) / w if moments['m00'] > 0 else 0.5

            stat_vec = [
                density, h_var, v_var, skew, avg_width,
                edge_ratio, num_labels / 10, aspect_ratio, cy, cx
            ]
            stats.append(stat_vec)

        return torch.tensor(stats, dtype=torch.float32, device=x.device)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播，返回多层级风格特征
        """
        features = self.backbone(x)

        # 全局风格
        global_style = self.global_style_head(features)

        # 局部风格图
        local_style = self.local_style_head(features)

        # 笔画特征
        stroke_features = self.stroke_extractor(x)

        # 统计特征
        stat_features = self.extract_statistical_features(x)
        stat_encoded = self.statistical_fc(stat_features)

        return {
            'global': global_style,
            'local': local_style,
            'stroke': stroke_features,
            'statistical': stat_encoded,
            'raw_stats': stat_features
        }

    def encode_to_dict(self, x: torch.Tensor) -> Dict:
        """
        将风格特征转换为可序列化的字典
        """
        with torch.no_grad():
            features = self.forward(x)

        result = {
            'global_style': features['global'].cpu().numpy().tolist(),
            'stroke_features': features['stroke'].cpu().numpy().tolist(),
            'statistical_features': features['raw_stats'].cpu().numpy().tolist(),
            'local_style_shape': list(features['local'].shape),
            'local_style': features['local'].cpu().numpy().tolist(),
            'metadata': {
                'global_dim': self.global_dim,
                'local_dim': self.local_dim,
                'stroke_dim': self.stroke_dim,
                'model_version': '1.0.0'
            }
        }

        return result


class StyleBank:
    """
    风格银行：管理和保存提取的字迹风格
    """
    def __init__(self, bank_path: str = "data/style_bank"):
        self.bank_path = Path(bank_path)
        self.bank_path.mkdir(parents=True, exist_ok=True)
        self.styles: Dict[str, Dict] = {}
        self._load_existing()

    def _load_existing(self):
        """加载已保存的风格"""
        for style_file in self.bank_path.glob("*.json"):
            style_id = style_file.stem
            with open(style_file, 'r', encoding='utf-8') as f:
                self.styles[style_id] = json.load(f)

    def save_style(self, style_id: str, style_dict: Dict, sample_image_path: Optional[str] = None):
        """保存风格到银行"""
        style_dict['style_id'] = style_id
        style_dict['sample_image'] = sample_image_path

        save_path = self.bank_path / f"{style_id}.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(style_dict, f, ensure_ascii=False, indent=2)

        self.styles[style_id] = style_dict
        print(f"风格 '{style_id}' 已保存到 {save_path}")

    def load_style(self, style_id: str) -> Dict:
        """从银行加载风格"""
        if style_id in self.styles:
            return self.styles[style_id]

        style_path = self.bank_path / f"{style_id}.json"
        if style_path.exists():
            with open(style_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        raise ValueError(f"风格 '{style_id}' 不存在")

    def list_styles(self) -> List[str]:
        """列出所有可用风格"""
        return list(self.styles.keys())

    def delete_style(self, style_id: str):
        """删除风格"""
        style_path = self.bank_path / f"{style_id}.json"
        if style_path.exists():
            style_path.unlink()
        self.styles.pop(style_id, None)


def extract_and_save_style(
    image_path: str,
    style_id: str,
    model_path: Optional[str] = None,
    bank_path: str = "data/style_bank",
    device: str = "cpu"
) -> Dict:
    """
    便捷函数：从图片提取风格并保存

    Args:
        image_path: 手写样本图片路径
        style_id: 风格标识名
        model_path: 预训练模型路径（可选）
        bank_path: 风格银行路径
        device: 计算设备

    Returns:
        风格字典
    """
    # 初始化编码器
    encoder = HandwritingStyleEncoder().to(device)

    if model_path and Path(model_path).exists():
        encoder.load_state_dict(torch.load(model_path, map_location=device))
        encoder.eval()

    # 预处理图片
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    img = Image.open(image_path).convert('L')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 提取风格
    encoder.eval()
    style_dict = encoder.encode_to_dict(img_tensor)

    # 保存到风格银行
    bank = StyleBank(bank_path)
    bank.save_style(style_id, style_dict, image_path)

    return style_dict


if __name__ == "__main__":
    # 测试
    print("HandwritingStyleEncoder 模块加载完成")
    print(f"模型参数量: {sum(p.numel() for p in HandwritingStyleEncoder().parameters()):,}")
