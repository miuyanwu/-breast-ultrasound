"""
乳腺超声图像智能诊断系统 — 核心源码包
"""

from .config import (
    MODEL_REGISTRY,
    INFERENCE_PRESETS,
    WEIGHTS_DIR,
    DATA_DIR,
    BIRADS_MAP,
    resolve_weight_path,
    resolve_device,
)

__all__ = [
    "MODEL_REGISTRY",
    "INFERENCE_PRESETS",
    "WEIGHTS_DIR",
    "DATA_DIR",
    "BIRADS_MAP",
    "resolve_weight_path",
    "resolve_device",
]
