"""
推理引擎模块
  load_models():         @st.cache_resource — 加载分类模型，常驻 GPU
  load_seg_pipeline():   @st.cache_resource — 加载分割管线模型
  infer_single():        @st.cache_data — 单图 TTA 融合推理
  infer_with_seg():      @st.cache_data — 分割→分类联合推理 (V10)
  overlay_mask():        半透明染色 + 轮廓线，全程 RGB 空间
"""
import io
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import albumentations as A
import cv2
import streamlit as st

from src.models import BreastCancerMultiTaskNet, V5Model, V5ModelUNetPP
from src.config import (
    MODEL_REGISTRY, BIRADS_MAP, INFERENCE_PRESETS,
    ENSEMBLE4_MODELS, COMPACT_WEIGHTED_MODELS,
    COMPACT_WEIGHTED_LOGIT_WEIGHTS,
    resolve_weight_path,
)

# V10 分割→分类配置 — 延迟导入，避免 segmentation_models_pytorch 缺失时影响 V9 推理
_SEG_CONFIG_LOADED = False
_SEG_COARSE_WEIGHTS = None
_SEG_ROI_WEIGHT = None
_SEG_THEN_CLASSIFY_CONFIG = None


def _ensure_seg_config():
    """延迟加载 V10 分割管线配置。仅在首次调用 seg 功能时导入。"""
    global _SEG_CONFIG_LOADED, _SEG_COARSE_WEIGHTS, _SEG_ROI_WEIGHT, _SEG_THEN_CLASSIFY_CONFIG
    if _SEG_CONFIG_LOADED:
        return
    try:
        from src.config import (
            SEG_COARSE_WEIGHTS,
            SEG_ROI_WEIGHT,
            SEG_THEN_CLASSIFY_CONFIG,
        )
        _SEG_COARSE_WEIGHTS = SEG_COARSE_WEIGHTS
        _SEG_ROI_WEIGHT = SEG_ROI_WEIGHT
        _SEG_THEN_CLASSIFY_CONFIG = SEG_THEN_CLASSIFY_CONFIG
    except ImportError as e:
        raise ImportError(
            "V10 分割→分类需要 segmentation_models_pytorch。"
            "请使用 conda 环境运行: D:\\Anaconda\\envs\\pytorch\\python.exe -m streamlit run app/main.py\n"
            f"原始错误: {e}"
        )
    _SEG_CONFIG_LOADED = True


def _has_weight(name: str) -> bool:
    try:
        resolve_weight_path(MODEL_REGISTRY[name]['weight'])
        return True
    except FileNotFoundError:
        return False


def resolve_inference_models(preset: str = 'compact'):
    """Return model names for an inference preset.

    compact prefers the best compact setup available from local weights:
    DISTILL+SOUP, then DISTILL, then SOUP, and finally the existing 4-model
    ensemble so the app remains usable before new compact weights are produced.
    """
    if preset == 'compact':
        has_distill = _has_weight('DISTILL')
        has_soup = _has_weight('SOUP')
        has_v23 = _has_weight('V2.3')
        if has_distill and has_soup and has_v23:
            return COMPACT_WEIGHTED_MODELS
        if has_distill and has_soup:
            return ('DISTILL', 'SOUP')
        if has_distill:
            return ('DISTILL',)
        if has_soup:
            return ('SOUP',)
        return ENSEMBLE4_MODELS
    if preset not in INFERENCE_PRESETS:
        raise ValueError(f"Unknown inference preset: {preset}")
    model_names = INFERENCE_PRESETS[preset]
    # 验证所有权重文件都存在
    for name in model_names:
        if not _has_weight(name):
            raise FileNotFoundError(
                f"推理模式 '{preset}' 需要 '{name}' 的权重文件，但未找到。"
                f"如需使用完整历史模型，请联系管理员获取权重文件。"
            )
    return model_names


def resolve_fusion_config(preset: str, model_names):
    """Return the classification fusion rule for loaded model names."""
    model_names = tuple(model_names)
    if model_names == COMPACT_WEIGHTED_MODELS and preset in ('compact', 'compact_weighted'):
        weights = np.array([COMPACT_WEIGHTED_LOGIT_WEIGHTS[name] for name in model_names], dtype=np.float64)
        weights = weights / weights.sum()
        return {'mode': 'weighted_logit', 'weights': weights}
    return {'mode': 'prob_mean', 'weights': None}


# 预处理 transform: 固定 Resize(256,256)，与训练测试一致
_test_transform = A.Compose([A.Resize(256, 256)])


# ============================================================
# 1. 模型加载 (缓存)
# ============================================================
@st.cache_resource
def load_models(device_str: str = 'cuda', preset: str = 'compact'):
    """加载推理模型。@st.cache_resource 保证同一 preset 只加载一次。"""
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    models = {}
    model_names = resolve_inference_models(preset)

    for name in model_names:
        cfg = MODEL_REGISTRY[name]
        model = cfg['cls']().to(device)
        weight_path = resolve_weight_path(cfg['weight'])
        model.load_state_dict(
            torch.load(weight_path, map_location=device, weights_only=True)
        )
        model.eval()
        models[name] = {'model': model, 'cfg': cfg}

    return models, device


# ============================================================
# 2. 图像预处理
# ============================================================
def _preprocess_image(image_bytes: bytes):
    """bytes → PIL → np.array(RGB) → Resize(256,256) → tensor [1,3,256,256]"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image)
    resized = _test_transform(image=image_np)
    image_np = resized['image']  # uint8 [256,256,3]
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # [1, 3, 256, 256]
    return image_tensor, image_np


# ============================================================
# 3. 模型输出提取
# ============================================================
def _extract_outputs(model, images, arch: str, has_birads: bool):
    """统一提取 mask_logits, cls_logits, birads_logits。"""
    out = model(images)
    if arch == 'v2':
        mask_logits, cls_logits = out
        birads_logits = None
    elif arch == 'v5':
        mask_logits, cls_logits, birads_logits = out
        if not has_birads:
            birads_logits = None
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return mask_logits, cls_logits, birads_logits


# ============================================================
# 4. 单图融合推理 (缓存)
# ============================================================
@st.cache_data
def infer_single(image_bytes: bytes, device_str: str = 'cuda', preset: str = 'compact'):
    """TTA + preset 融合推理。@st.cache_data 按图像内容缓存结果。"""
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    models, _ = load_models(device_str, preset)

    image_tensor, image_np = _preprocess_image(image_bytes)
    image_tensor = image_tensor.to(device)

    # TTA 对: 原图 + 水平翻转
    images_orig = image_tensor
    images_flip = torch.flip(image_tensor, dims=[3])

    cls_probs_all = []
    cls_logits_all = []
    mask_probs_all = []
    birads_logits_v51 = None
    per_model_probs = {}

    for name, entry in models.items():
        model = entry['model']
        cfg = entry['cfg']
        arch = cfg['arch']
        has_birads = cfg.get('has_birads', False)

        with torch.no_grad():
            mask_orig, cls_orig, birads_orig = _extract_outputs(
                model, images_orig, arch, has_birads)
            mask_flip, cls_flip, birads_flip = _extract_outputs(
                model, images_flip, arch, has_birads)

        # --- 分类 TTA: 平均 logits → sigmoid ---
        cls_avg = (cls_orig + cls_flip) / 2.0
        cls_prob = torch.sigmoid(cls_avg).item()
        cls_probs_all.append(cls_prob)
        cls_logits_all.append(cls_avg.item())
        per_model_probs[name] = cls_prob

        # --- Mask TTA: 翻转回 → 平均 → sigmoid ---
        mask_orig_prob = torch.sigmoid(mask_orig)
        mask_flip_prob = torch.sigmoid(mask_flip)
        mask_flip_back = torch.flip(mask_flip_prob, dims=[3])
        mask_avg_prob = (mask_orig_prob + mask_flip_back) / 2.0
        mask_probs_all.append(mask_avg_prob.squeeze().cpu().numpy())

        # --- BI-RADS: 仅从 V5.1 取 ---
        if has_birads and birads_orig is not None:
            birads_logits_v51 = birads_orig.detach().cpu()

    # --- 融合 ---
    model_names = list(models.keys())
    fusion_config = resolve_fusion_config(preset, model_names)
    if fusion_config['mode'] == 'weighted_logit':
        fused_logit = float(np.dot(np.array(cls_logits_all, dtype=np.float64), fusion_config['weights']))
        fused_prob = float(1.0 / (1.0 + np.exp(-fused_logit)))
    else:
        fused_prob = float(np.mean(cls_probs_all))

    # Mask: 模型 TTA mask 概率平均 → 二值化
    fused_mask_prob = np.mean(mask_probs_all, axis=0)
    mask_binary = fused_mask_prob > 0.5

    # BI-RADS: V5.1 argmax
    birads_idx = 0
    birads_label = BIRADS_MAP[0]
    if birads_logits_v51 is not None:
        birads_idx = int(birads_logits_v51.argmax(dim=1).item())
        birads_label = BIRADS_MAP.get(birads_idx, BIRADS_MAP[0])

    return {
        'prob': fused_prob,
        'birads_idx': birads_idx,
        'birads_label': birads_label,
        'mask_binary': mask_binary,
        'image_original': image_np,
        'cls_probs_per_model': per_model_probs,
        'inference_models': model_names,
        'fusion_mode': fusion_config['mode'],
    }


# ============================================================
# 5. Mask 叠加可视化
# ============================================================
def overlay_mask(image_rgb: np.ndarray,
                 mask_binary: np.ndarray,
                 is_malignant: bool) -> np.ndarray:
    """半透明染色 + 轮廓线。全程 RGB 空间操作。

    Args:
        image_rgb:   [H, W, 3] uint8 RGB
        mask_binary: [H, W] bool
        is_malignant: True→医疗警告红, False→安全绿

    Returns:
        [H, W, 3] uint8 RGB 叠加图
    """
    if is_malignant:
        color = (231, 76, 60)   # 医疗警告红
    else:
        color = (46, 204, 113)  # 安全绿

    if image_rgb.dtype != np.uint8:
        image_rgb = (image_rgb * 255).astype(np.uint8)
    mask_uint8 = mask_binary.astype(np.uint8)

    color_layer = np.zeros_like(image_rgb)
    color_layer[mask_uint8 > 0] = color

    blended = cv2.addWeighted(image_rgb, 0.6, color_layer, 0.4, 0)

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, color, thickness=2)

    return blended


# ============================================================
# 6. 分割管线加载 (V10: 分割→分类)
# ============================================================
@st.cache_resource
def load_seg_pipeline(device_str: str = "cuda"):
    """加载两阶段 MiT-B2 UNet 分割管线。@st.cache_resource 保证只加载一次。"""
    import sys
    from pathlib import Path

    _project_root = Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    from src.segmentation.roi_pipeline import SegmentationPipeline

    _ensure_seg_config()
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    weights_dir = _project_root / "weights"

    pipeline = SegmentationPipeline(
        coarse_weights=[
            weights_dir / _SEG_COARSE_WEIGHTS[0],
            weights_dir / _SEG_COARSE_WEIGHTS[1],
        ],
        roi_weight=weights_dir / _SEG_ROI_WEIGHT,
        device=device,
    )
    return pipeline


# ============================================================
# 7. 分割→分类联合推理 (V10)
# ============================================================
@st.cache_data
def infer_with_seg(
    image_bytes: bytes,
    device_str: str = "cuda",
    preset: str = "compact_weighted",
):
    """分割→分类联合推理: 原图 V9 + 分割 ROI V9 → 0.5:0.5 融合。

    管线:
        1. 原图 → V9 分类 → prob_original
        2. 原图 → 两阶段分割管线 → 病灶掩膜
        3. ROI (margin=1.0) → V9 分类 → prob_roi
        4. 融合: 0.5 * prob_original + 0.5 * prob_roi

    Args:
        image_bytes: 上传图像 bytes
        device_str: 'cuda' 或 'cpu'
        preset: 分类推理预设

    Returns:
        dict 同 infer_single() 结构，额外包含:
          - seg_mask: 分割掩膜 [H,W] bool
          - prob_original: 原图 V9 概率
          - prob_roi: ROI V9 概率
    """
    import sys
    from pathlib import Path

    _project_root = Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    from src.segmentation.roi_pipeline import crop_roi_pil

    _ensure_seg_config()
    config = _SEG_THEN_CLASSIFY_CONFIG

    # 1. 原图 V9 推理
    result_original = infer_single(image_bytes, device_str, preset)
    prob_original = result_original["prob"]

    # 2. 分割管线 → 掩膜
    seg_pipeline = load_seg_pipeline(device_str)
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = image_pil.size
    seg_mask = seg_pipeline.predict(image_pil)  # [H,W] bool (原始尺寸)

    # 3. ROI 裁剪 + V9 推理
    mask_pil = Image.fromarray(seg_mask.astype(np.uint8) * 255, mode="L")
    roi_pil = crop_roi_pil(image_pil, mask_pil, margin=config["roi_margin"])

    # 将 ROI 编码为 bytes → infer_single
    roi_buf = io.BytesIO()
    roi_pil.save(roi_buf, format="PNG")
    roi_bytes = roi_buf.getvalue()

    result_roi = infer_single(roi_bytes, device_str, preset)
    prob_roi = result_roi["prob"]

    # 4. 融合
    fused_prob = (
        config["original_weight"] * prob_original
        + config["roi_weight"] * prob_roi
    )
    threshold = config["threshold"]
    prediction = "恶性" if fused_prob >= threshold else "良性"
    is_malignant = prediction == "恶性"

    # BI-RADS 使用原图结果
    birads_idx = result_original.get("birads_idx", 0)
    birads_label = result_original.get("birads_label", BIRADS_MAP[0])

    # 使用分割掩膜做叠加可视化
    image_np = result_original["image_original"]  # [256,256,3] uint8
    # 将 seg_mask resize 到 256×256 用于 overlay
    seg_mask_256 = np.array(
        Image.fromarray(seg_mask).resize((256, 256), Image.Resampling.NEAREST)
    )
    overlay = overlay_mask(image_np, seg_mask_256, is_malignant)

    return {
        "prob": fused_prob,
        "prob_original": prob_original,
        "prob_roi": prob_roi,
        "birads_idx": birads_idx,
        "birads_label": birads_label,
        "mask_binary": seg_mask_256,
        "seg_mask_full": seg_mask,
        "image_original": image_np,
        "image_overlay": overlay,
        "cls_probs_per_model": result_original.get("cls_probs_per_model", {}),
        "inference_models": result_original.get("inference_models", []),
        "fusion_mode": "seg_then_classify",
        "prediction": prediction,
    }
