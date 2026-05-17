"""
Handwriting Mimic AI - 矢量字迹模仿系统
"""

__version__ = "1.0.0"
__author__ = "Handwriting AI Team"

from src.vector_style_encoder import HandwritingStyleEncoder, StyleBank
from src.vector_generator import VectorHandwritingGenerator, VectorHandwritingPipeline

__all__ = [
    "HandwritingStyleEncoder",
    "StyleBank", 
    "VectorHandwritingGenerator",
    "VectorHandwritingPipeline",
]
