"""
测试风格编码器
"""
import unittest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_style_encoder import HandwritingStyleEncoder, StyleBank


class TestHandwritingStyleEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = HandwritingStyleEncoder()
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, 1, 128, 128)

    def test_forward_shape(self):
        """测试前向传播输出形状"""
        output = self.encoder(self.input_tensor)

        self.assertEqual(output['global'].shape, (self.batch_size, 256))
        self.assertEqual(output['local'].shape[0], self.batch_size)
        self.assertEqual(output['stroke'].shape, (self.batch_size, 64))
        self.assertEqual(output['statistical'].shape, (self.batch_size, 32))

    def test_encode_to_dict(self):
        """测试编码为字典"""
        style_dict = self.encoder.encode_to_dict(self.input_tensor)

        self.assertIn('global_style', style_dict)
        self.assertIn('stroke_features', style_dict)
        self.assertIn('statistical_features', style_dict)
        self.assertIn('metadata', style_dict)

        # 检查维度
        self.assertEqual(len(style_dict['global_style'][0]), 256)
        self.assertEqual(len(style_dict['stroke_features'][0]), 64)
        self.assertEqual(len(style_dict['statistical_features'][0]), 10)


class TestStyleBank(unittest.TestCase):
    def setUp(self):
        self.bank = StyleBank(bank_path="data/test_style_bank")
        self.test_style = {
            'global_style': [[0.1] * 256],
            'metadata': {'global_dim': 256}
        }

    def test_save_and_load(self):
        """测试保存和加载风格"""
        self.bank.save_style("test_style", self.test_style)
        loaded = self.bank.load_style("test_style")

        self.assertEqual(loaded['global_style'], self.test_style['global_style'])

    def test_list_styles(self):
        """测试列出风格"""
        self.bank.save_style("style1", self.test_style)
        self.bank.save_style("style2", self.test_style)

        styles = self.bank.list_styles()
        self.assertIn("style1", styles)
        self.assertIn("style2", styles)


if __name__ == '__main__':
    unittest.main()
