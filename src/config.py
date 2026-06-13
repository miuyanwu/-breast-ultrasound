"""
集中配置 — 模型注册表、推理预设、权重路径。
inference_engine 和 test_ensemble 共用此注册表，消除重复。
"""
import os

import torch

from src.models.base import BreastCancerMultiTaskNet
from src.models.v5 import V5Model, V5ModelUNetPP
from src.models.v6 import BreastCancerMultiTaskNetV6

# ============================================================
# 路径
# ============================================================
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(_PROJECT_ROOT, 'weights')
DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')


def _weight_path(*names):
    """Return first existing weight file from candidates."""
    for name in names:
        path = os.path.join(WEIGHTS_DIR, name)
        if os.path.exists(path):
            return path
    # Fallback: search root (legacy)
    for name in names:
        path = os.path.join(_PROJECT_ROOT, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"找不到模型权重: {', '.join(names)}")


# ============================================================
# 模型注册表 (统一 inference_engine + test_ensemble)
# ============================================================
MODEL_REGISTRY = {
    'V2.3': {
        'cls': lambda: BreastCancerMultiTaskNet(encoder_name="resnet34"),
        'weight': ['best_model_v2.3.pth'],
        'arch': 'v2',
        'has_birads': False,
        'label': 'V2.3 基础特征引擎',
    },
    'V5.1': {
        'cls': lambda: V5Model(),
        'weight': ['best_model_v5.1.pth'],
        'arch': 'v5',
        'has_birads': True,
        'label': 'V5.1 BI-RADS 多任务引擎',
    },
    'V5.2': {
        'cls': lambda: V5ModelUNetPP(),
        'weight': ['best_model_v5.2.pth'],
        'arch': 'v5',
        'has_birads': False,
        'label': 'V5.2 深度特征引擎',
    },
    'V6': {
        'cls': lambda: BreastCancerMultiTaskNetV6(),
        'weight': ['best_model_v6.pth'],
        'arch': 'v6',
        'has_birads': False,
        'label': 'V6 LoMix 引擎',
    },
    'V7.4': {
        'cls': lambda: BreastCancerMultiTaskNetV6(),
        'weight': ['best_model_v7.4.pth'],
        'arch': 'v6',
        'has_birads': False,
        'label': 'V7.4 多尺度引擎',
    },
    'S123': {
        'cls': lambda: BreastCancerMultiTaskNet(encoder_name="resnet34"),
        'weight': ['best_model_v7.5_seed123.pth'],
        'arch': 'v2',
        'has_birads': False,
        'label': 'V7.5 Seed 123',
    },
    'S456': {
        'cls': lambda: BreastCancerMultiTaskNet(encoder_name="resnet34"),
        'weight': ['best_model_v7.5_seed456.pth'],
        'arch': 'v2',
        'has_birads': False,
        'label': 'V7.5 Seed 456',
    },
    'S789': {
        'cls': lambda: BreastCancerMultiTaskNet(encoder_name="resnet34"),
        'weight': ['best_model_v7.5_seed789.pth'],
        'arch': 'v2',
        'has_birads': False,
        'label': 'V7.5 Seed 789',
    },
    'SAM': {
        'cls': lambda: BreastCancerMultiTaskNet(encoder_name="resnet34"),
        'weight': ['best_model_v7.6.pth'],
        'arch': 'v2',
        'has_birads': False,
        'label': 'SAM 泛化优化引擎',
    },
    'DISTILL': {
        'cls': lambda: BreastCancerMultiTaskNet(encoder_name="resnet34"),
        'weight': [
            'best_model_distill.pth',
            'best_model_distill_seed42.pth',
            'best_model_distill_seed123.pth',
            'best_model_distill_seed456.pth',
            'best_model_distill_seed789.pth',
        ],
        'arch': 'v2',
        'has_birads': False,
        'label': 'DISTILL 单模型蒸馏引擎',
    },
    'SOUP': {
        'cls': lambda: BreastCancerMultiTaskNet(encoder_name="resnet34"),
        'weight': ['best_model_soup.pth'],
        'arch': 'v2',
        'has_birads': False,
        'label': 'SOUP 单模型权重汤引擎',
    },
}


# ============================================================
# 推理预设
# ============================================================
ENSEMBLE4_MODELS = ('V2.3', 'V5.1', 'V5.2', 'SAM')
COMPACT_WEIGHTED_MODELS = ('DISTILL', 'SOUP', 'V2.3')
COMPACT_WEIGHTED_LOGIT_WEIGHTS = {
    'DISTILL': 0.043422758938255444,
    'SOUP': 0.8786080776229839,
    'V2.3': 0.07796916343876055,
}
INFERENCE_PRESETS = {
    'ensemble4': ENSEMBLE4_MODELS,
    'compact_weighted': COMPACT_WEIGHTED_MODELS,
    'distill': ('DISTILL',),
    'soup': ('SOUP',),
    'v23': ('V2.3',),
}


def resolve_weight_path(weight):
    """candidates → first existing path (searches weights/ then root)."""
    candidates = weight if isinstance(weight, (list, tuple)) else [weight]
    for candidate in candidates:
        wpath = os.path.join(WEIGHTS_DIR, candidate)
        if os.path.exists(wpath):
            return wpath
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"找不到模型权重: {', '.join(candidates)}")


# ============================================================
# BI-RADS 映射
# ============================================================
BIRADS_MAP = {
    0: "BI-RADS 2类 (良性)",
    1: "BI-RADS 3类 (可能良性)",
    2: "BI-RADS 4类 (可疑恶性)",
    3: "BI-RADS 5类 (高度恶性)",
}


# ============================================================
# 分割管线配置 (队友方案: 分割 → V9 分类)
# ============================================================
# 粗分割模型 (2× MiT-B2 UNet 平均)
SEG_COARSE_WEIGHTS = [
    "seg_unet_mit_b2_fold0_s384_seed3101_best.pth",
    "seg_unet_mit_b2_all_s384_seed6201_epoch12.pth",
]
# ROI 精修模型
SEG_ROI_WEIGHT = "seg_roi_unet_mit_b2_all_best.pth"

# 分割→分类融合参数
SEG_THEN_CLASSIFY_CONFIG = {
    "original_weight": 0.5,     # 原图概率权重
    "roi_weight": 0.5,          # ROI 概率权重
    "roi_margin": 1.0,          # 分类阶段 ROI 扩边
    "threshold": 0.55,          # 分类阈值
}

# 新增推理预设: "seg_then_classify"
#   - 先跑两阶段分割管线获取病灶掩膜
#   - 对原图和 ROI 分别跑 V9 分类
#   - 0.5:0.5 融合两路概率
INFERENCE_PRESETS["seg_then_classify"] = COMPACT_WEIGHTED_MODELS


def resolve_device(device_arg):
    """'auto' | 'cuda' | 'cpu' → torch.device."""
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device_arg == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA GPU is available. Use --device cpu or --device auto.")
    return torch.device(device_arg)
