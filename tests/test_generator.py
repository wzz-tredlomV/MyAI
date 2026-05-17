"""
测试矢量生成器
"""
import unittest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_generator import VectorHandwritingGenerator, SVGRenderer


class TestVectorHandwritingGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = VectorHandwritingGenerator()
        self.batch_size = 2
        self.skeleton = torch.randn(self.batch_size, 1, 128, 128)
        self.style = torch.randn(self.batch_size, 256)

    def test_forward(self):
        """测试生成器前向传播"""
        output = self.generator(self.skeleton, self.style)

        self.assertIn('stroke_params', output)
        self.assertIn('presence_map', output)
        self.assertIn('feature_map', output)

        self.assertEqual(output['presence_map'].shape[0], self.batch_size)


class TestSVGRenderer(unittest.TestCase):
    def setUp(self):
        self.renderer = SVGRenderer()

    def test_strokes_to_svg(self):
        """测试SVG渲染"""
        strokes = np.random.rand(8, 10)
        presence = np.random.rand(128, 128)

        svg = self.renderer.strokes_to_svg(strokes, presence, "测")

        self.assertIn('<svg', svg)
        self.assertIn('</svg>', svg)
        self.assertIn('handwriting-style', svg)


if __name__ == '__main__':
    unittest.main()
