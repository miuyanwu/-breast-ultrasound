"""
分割推理管线模块 — MiT-B2 UNet 粗分割 + ROI 精修。

用法:
    from src.segmentation import SegmentationPipeline
    pipeline = SegmentationPipeline(coarse_weights=[...], roi_weight=...)
    mask = pipeline.predict(image_pil)  # np.ndarray bool [H,W]
"""

from src.segmentation.roi_pipeline import (
    SegmentationPipeline,
    load_seg_model,
    crop_roi,
    postprocess_mask,
)

__all__ = [
    "SegmentationPipeline",
    "load_seg_model",
    "crop_roi",
    "postprocess_mask",
]
