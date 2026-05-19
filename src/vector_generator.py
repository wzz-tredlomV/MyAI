"""
矢量字迹生成器
将风格特征与文字内容融合，生成SVG矢量输出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom


class AdaIN(nn.Module):
    """自适应实例归一化 - 风格注入核心"""

    def __init__(self, style_dim: int, num_features: int):
        super().__init__()
        self.style_dim = style_dim
        self.num_features = num_features

        # 将风格向量转换为缩放和平移参数
        self.fc = nn.Sequential(
            nn.Linear(style_dim, 256), nn.ReLU(), nn.Linear(256, num_features * 2)
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        content: [B, C, H, W] 内容特征
        style: [B, style_dim] 风格向量
        """
        # 计算内容特征的均值和方差
        content_mean = content.mean(dim=[2, 3], keepdim=True)
        content_std = content.std(dim=[2, 3], keepdim=True) + 1e-5

        # 归一化内容
        normalized = (content - content_mean) / content_std

        # 生成风格参数
        style_params = self.fc(style)
        gamma = style_params[:, : self.num_features].view(-1, self.num_features, 1, 1)
        beta = style_params[:, self.num_features:].view(-1, self.num_features, 1, 1)

        # 应用风格
        return gamma * normalized + beta


class ResBlock(nn.Module):
    """带AdaIN的残差块"""

    def __init__(self, channels: int, style_dim: int):
        super().__init__()
        self.adain1 = AdaIN(style_dim, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.adain2 = AdaIN(style_dim, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x, style):
        residual = x
        out = F.relu(self.conv1(self.adain1(x, style)))
        out = self.conv2(self.adain2(out, style))
        return F.relu(out + residual)


class VectorHandwritingGenerator(nn.Module):
    """
    矢量字迹生成器
    生成笔画参数，后续转为SVG路径
    """

    def __init__(
        self,
        style_dim: int = 256,
        content_dim: int = 128,
        num_stroke_types: int = 8,
        max_strokes_per_char: int = 20,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.content_dim = content_dim
        self.num_stroke_types = num_stroke_types
        self.max_strokes = max_strokes_per_char

        # 内容编码器（骨架图 -> 内容特征）
        self.content_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, content_dim, 3, 1, 1),
        )

        # 风格-内容融合解码器
        self.decoder = nn.ModuleList(
            [
                self._make_decoder_block(content_dim, 256, style_dim),
                self._make_decoder_block(256, 128, style_dim),
                self._make_decoder_block(128, 64, style_dim),
                self._make_decoder_block(64, 32, style_dim),
            ]
        )

        # 笔画参数预测头
        # 每个笔画: [type, x1, y1, x2, y2, cx, cy, width, pressure, curvature]
        self.stroke_predictor = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, num_stroke_types * 10, 1),  # 10 params per stroke type
        )

        # 笔画存在性预测（哪些位置有笔画）
        self.stroke_presence = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 1, 1), nn.Sigmoid()
        )

    def _make_decoder_block(self, in_ch: int, out_ch: int, style_dim: int):
        return nn.ModuleDict(
            {
                "upsample": nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                "adain": AdaIN(style_dim, out_ch),
                "conv": nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                "resblock": ResBlock(out_ch, style_dim),
            }
        )

    def forward(
        self, skeleton: torch.Tensor, style: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        skeleton: [B, 1, H, W] 文字骨架图
        style: [B, style_dim] 风格向量

        Returns:
            stroke_params: [B, num_strokes, 10] 笔画参数
            presence_map: [B, 1, H, W] 笔画存在热力图
            feature_map: [B, 32, H, W] 特征图（用于后续细化）
        """
        # 编码内容
        x = self.content_encoder(skeleton)

        # 解码融合
        for block in self.decoder:
            x = block["upsample"](x)
            x = block["adain"](x, style)
            x = F.relu(block["conv"](x))
            x = block["resblock"](x, style)

        # 预测笔画参数
        stroke_logits = self.stroke_predictor(x)  # [B, num_stroke_types*10, H, W]
        B, _, H, W = stroke_logits.shape

        # 重塑为笔画参数
        stroke_params = stroke_logits.view(B, self.num_stroke_types, 10, H, W)

        # 笔画存在性
        presence = self.stroke_presence(x)

        return {
            "stroke_params": stroke_params,
            "presence_map": presence,
            "feature_map": x,
        }


class SVGRenderer:
    """
    将模型输出的笔画参数渲染为SVG矢量图形
    """

    def __init__(self, canvas_size: Tuple[int, int] = (512, 512)):
        self.width, self.height = canvas_size
        self.stroke_types = {
            0: "line",
            1: "quadratic",
            2: "cubic",
            3: "arc",
            4: "dot",
            5: "hook",
            6: "press",
            7: "lift",
        }

    def strokes_to_svg(
        self,
        stroke_params: np.ndarray,
        presence_map: np.ndarray,
        char: str = "",
        style_info: Optional[Dict] = None,
    ) -> str:
        """
        将笔画参数转换为SVG字符串

        Args:
            stroke_params: [num_strokes, 10] 笔画参数数组
            presence_map: [H, W] 存在性热力图
            char: 对应的字符
            style_info: 风格信息（用于设置颜色等）
        """
        # 创建SVG根元素
        svg = ET.Element("svg")
        svg.set("xmlns", "http://www.w3.org/2000/svg")
        svg.set("width", str(self.width))
        svg.set("height", str(self.height))
        svg.set("viewBox", f"0 0 {self.width} {self.height}")

        # 添加元数据
        metadata = ET.SubElement(svg, "metadata")
        if style_info:
            style_meta = ET.SubElement(metadata, "handwriting-style")
            style_meta.set("id", style_info.get("style_id", "unknown"))
            style_meta.set("version", "1.0")

        # 添加字符信息
        title = ET.SubElement(svg, "title")
        title.text = f"Handwriting: {char}"

        # 创建笔画组
        g = ET.SubElement(svg, "g")
        g.set("id", f"char-{char}")
        g.set("fill", "none")
        g.set("stroke", "#000000")
        g.set("stroke-linecap", "round")
        g.set("stroke-linejoin", "round")

        # 根据存在性图过滤笔画
        threshold = 0.5
        active_strokes = presence_map > threshold

        # 提取笔画并生成路径
        for i, params in enumerate(stroke_params):
            if i >= len(active_strokes) or not np.any(active_strokes[i]):
                continue

            stroke_type_idx = (
                int(params[0] * self.num_stroke_types) % self.num_stroke_types
            )
            stroke_type = self.stroke_types.get(stroke_type_idx, "line")

            # 解析参数（归一化到画布坐标）
            x1 = params[1] * self.width
            y1 = params[2] * self.height
            x2 = params[3] * self.width
            y2 = params[4] * self.height
            cx = params[5] * self.width
            cy = params[6] * self.height
            width = max(1, params[7] * 10)  # 笔画粗细
            pressure = params[8]  # 压力（影响透明度）
            curvature = params[9]  # 曲率

            # 创建路径元素
            path = ET.SubElement(g, "path")

            if stroke_type == "line":
                d = f"M {x1:.1f} {y1:.1f} L {x2:.1f} {y2:.1f}"
            elif stroke_type == "quadratic":
                d = f"M {x1:.1f} {y1:.1f} Q {cx:.1f} {cy:.1f} {x2:.1f} {y2:.1f}"
            elif stroke_type == "cubic":
                # 使用曲率生成控制点
                cp1x = x1 + (cx - x1) * curvature
                cp1y = y1 + (cy - y1) * curvature
                cp2x = x2 + (cx - x2) * curvature
                cp2y = y2 + (cy - y2) * curvature
                d = f"M {x1:.1f} {y1:.1f} C {cp1x:.1f} {cp1y:.1f} {cp2x:.1f} {cp2y:.1f} {x2:.1f} {y2:.1f}"
            elif stroke_type == "arc":
                rx = abs(x2 - x1) / 2
                ry = abs(y2 - y1) / 2
                d = f"M {x1:.1f} {y1:.1f} A {rx:.1f} {ry:.1f} 0 0 1 {x2:.1f} {y2:.1f}"
            else:
                d = f"M {x1:.1f} {y1:.1f} L {x2:.1f} {y2:.1f}"

            path.set("d", d)
            path.set("stroke-width", str(width))
            path.set("opacity", str(0.5 + pressure * 0.5))
            path.set("data-stroke-type", stroke_type)
            path.set("data-pressure", str(pressure))

        # 美化输出
        rough_string = ET.tostring(svg, encoding="unicode")
        reparsed = minidom.parseString(rough_string)
        pretty = reparsed.toprettyxml(indent="  ")

        # 移除空行
        lines = [line for line in pretty.split("\n") if line.strip()]
        return "\n".join(lines)

    def save_svg(self, svg_string: str, output_path: str):
        """保存SVG到文件"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(svg_string)
        print(f"SVG已保存: {output_path}")


class VectorHandwritingPipeline:
    """
    完整的矢量字迹生成流水线
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        canvas_size: Tuple[int, int] = (512, 512),
    ):
        self.device = torch.device(device)
        self.canvas_size = canvas_size

        # 初始化模型
        self.generator = VectorHandwritingGenerator().to(self.device)
        if model_path and Path(model_path).exists():
            self.generator.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
        self.generator.eval()

        self.renderer = SVGRenderer(canvas_size)
        self.style_bank_path = Path("data/style_bank")

    def load_style(self, style_id: str) -> Dict:
        """从风格银行加载风格"""
        style_path = self.style_bank_path / f"{style_id}.json"
        if not style_path.exists():
            raise ValueError(f"风格 '{style_id}' 不存在于风格银行")

        with open(style_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_character(
        self, char: str, style_id: str, output_path: Optional[str] = None
    ) -> str:
        """
        生成单个字符的矢量字迹

        Args:
            char: 要生成的字符
            style_id: 风格ID
            output_path: 输出SVG路径（可选）

        Returns:
            SVG字符串
        """
        # 加载风格
        style_dict = self.load_style(style_id)

        # 解析风格向量
        global_style = torch.tensor(style_dict["global_style"], dtype=torch.float32).to(
            self.device
        )

        # 生成骨架图（简化版：使用标准字体渲染）
        skeleton = self._char_to_skeleton(char)
        skeleton_tensor = (
            torch.from_numpy(skeleton).float().unsqueeze(0).unsqueeze(0).to(self.device)
        )

        # 生成笔画参数
        with torch.no_grad():
            output = self.generator(skeleton_tensor, global_style)

        # 提取笔画参数
        stroke_params = (
            output["stroke_params"][0].cpu().numpy()
        )  # [num_stroke_types, 10, H, W]
        presence_map = output["presence_map"][0, 0].cpu().numpy()

        # 简化为 [num_strokes, 10] 格式
        # 取每个stroke type在presence最高的位置的参数
        num_types, _, H, W = stroke_params.shape
        strokes = []
        for t in range(num_types):
            # 找到该类型笔画最可能存在的位置
            type_presence = presence_map
            max_idx = np.unravel_index(np.argmax(type_presence), type_presence.shape)
            h, w = max_idx
            params = stroke_params[t, :, h, w]
            strokes.append(params)

        strokes = np.array(strokes)

        # 渲染SVG
        svg = self.renderer.strokes_to_svg(strokes, presence_map, char, style_dict)

        if output_path:
            self.renderer.save_svg(svg, output_path)

        return svg

    def generate_text(
        self, text: str, style_id: str, output_dir: str = "output", combine: bool = True
    ) -> List[str]:
        """
        生成一段文字的矢量字迹

        Args:
            text: 要生成的文本
            style_id: 风格ID
            output_dir: 输出目录
            combine: 是否合并为单个SVG

        Returns:
            SVG文件路径列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        svgs = []
        for i, char in enumerate(text):
            if char.strip() == "":
                continue

            char_path = output_dir / f"char_{i:03d}_{char}.svg"
            self.generate_character(char, style_id, str(char_path))
            svgs.append(str(char_path))

        if combine and svgs:
            combined_path = output_dir / f"text_{text[:10]}.svg"
            self._combine_svgs(svgs, str(combined_path), text)
            svgs.append(str(combined_path))

        return svgs

    def _char_to_skeleton(self, char: str) -> np.ndarray:
        """将字符转换为骨架图（简化实现）"""
        import cv2

        # 创建空白画布
        img = np.zeros((128, 128), dtype=np.uint8)

        # 使用OpenCV渲染标准字体作为骨架引导
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(char, font, 3, 2)[0]
        x = (128 - text_size[0]) // 2
        y = (128 + text_size[1]) // 2

        cv2.putText(img, char, (x, y), font, 3, 255, 2)

        # 归一化到 [-1, 1]
        return (img.astype(np.float32) / 127.5) - 1.0

    def _combine_svgs(self, svg_paths: List[str], output_path: str, text: str):
        """将多个字符SVG合并为一个"""
        svg = ET.Element("svg")
        svg.set("xmlns", "http://www.w3.org/2000/svg")

        total_width = len(svg_paths) * 512
        svg.set("width", str(total_width))
        svg.set("height", "512")
        svg.set("viewBox", f"0 0 {total_width} 512")

        title = ET.SubElement(svg, "title")
        title.text = f"Handwriting: {text}"

        for i, path in enumerate(svg_paths):
            try:
                tree = ET.parse(path)
                root = tree.getroot()

                # 提取所有路径
                for g in root.findall(".//{http://www.w3.org/2000/svg}g"):
                    new_g = ET.SubElement(svg, "g")
                    new_g.set("transform", f"translate({i * 512}, 0)")

                    for path_elem in g.findall("{http://www.w3.org/2000/svg}path"):
                        new_g.append(path_elem)
            except Exception as e:
                print(f"合并 {path} 时出错: {e}")

        rough_string = ET.tostring(svg, encoding="unicode")
        reparsed = minidom.parseString(rough_string)
        pretty = reparsed.toprettyxml(indent="  ")

        lines = [line for line in pretty.split("\n") if line.strip()]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"合并SVG已保存: {output_path}")


if __name__ == "__main__":
    print("VectorHandwritingGenerator 模块加载完成")
    print(
        f"模型参数量: {sum(p.numel() for p in VectorHandwritingGenerator().parameters()):,}"
    )
