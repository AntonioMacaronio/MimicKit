"""Simulator-independent SMP data and shared offline/online features.

Preprocessing has optional MuJoCo/SciPy dependencies. Runtime imports use only
NumPy and PyTorch; no FAR-Holosoma or MimicKit environment imports are required.
"""

from .dataset import CarryWindowDataset, MotionCollator, make_dataloader
from .features import FeatureSchema, encode_motion, encode_window

__all__ = [
    "CarryWindowDataset",
    "FeatureSchema",
    "MotionCollator",
    "encode_motion",
    "encode_window",
    "make_dataloader",
]
