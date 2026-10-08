"""Simulator-independent SMP data and shared offline/online features.

Preprocessing has optional MuJoCo/SciPy dependencies. Runtime imports use only
NumPy and PyTorch; no FAR-Holosoma or MimicKit environment imports are required.
"""

from .dataset import CarryWindowDataset, MotionCollator, make_dataloader
from .features import FeatureSchema, encode_motion

__all__ = ["CarryWindowDataset", "FeatureSchema", "MotionCollator", "encode_motion", "make_dataloader"]
