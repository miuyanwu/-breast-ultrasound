"""
两阶段分割推理管线: 粗分割 (2× MiT-B2 UNet 平均) → ROI 精修 (1× MiT-B2 UNet)。

适配队友方案: 粗分割平均权重 0.75 + ROI 精修 0.25。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

# ============================================================
# 常量
# ============================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 384  # 分割模型输入尺寸
COARSE_WEIGHT = 0.75  # 粗分割在最终概率中的权重
ROI_WEIGHT = 0.25  # ROI 精修在最终概率中的权重
COARSE_THRESHOLD = 0.5  # 粗分割二值化阈值 (用于 ROI 提取)
FINAL_THRESHOLD = 0.38  # 最终掩膜二值化阈值
ROI_MARGIN = 0.35  # ROI 裁剪边距
MIN_COMPONENT_AREA = 400  # 后处理最小连通域面积


# ============================================================
# 模型加载
# ============================================================
def load_seg_model(
    weight_path: Union[str, Path],
    device: torch.device,
) -> Tuple[torch.nn.Module, bool, str]:
    """加载 MiT-B2 UNet 分割模型。

    Args:
        weight_path: .pth 权重文件路径
        device: torch device

    Returns:
        (model, aux_enabled, normalization_name)
        aux_enabled 恒为 False (纯分割模型无辅助分类头)
        normalization_name 恒为 'imagenet'

    Raises:
        ImportError: 如果 segmentation_models_pytorch 未安装
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            "segmentation_models_pytorch 未安装。请运行: pip install segmentation-models-pytorch"
        )

    # 从 checkpoint 读取 args 确认架构
    ckpt = torch.load(str(weight_path), map_location=device, weights_only=True)
    args = ckpt.get("args", {})
    encoder = args.get("encoder", "mit_b2")
    arch = args.get("arch", "unet")
    norm = args.get("normalize", "imagenet")

    if arch == "unet":
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,  # 从 checkpoint 加载
            in_channels=3,
            classes=1,
        )
    else:
        raise ValueError(f"不支持的分割架构: {arch} (仅支持 'unet')")

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # 纯分割模型，无 aux 分类头
    return model, False, norm


# ============================================================
# 预处理 & 推理
# ============================================================
def _normalize_batch(
    images: torch.Tensor,
    normalization: str = "imagenet",
) -> torch.Tensor:
    """对 [B, 3, H, W] float32 [0,1] 张量做归一化。"""
    if normalization == "imagenet":
        mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
        return (images - mean) / std
    # 无归一化
    return images


@torch.no_grad()
def _seg_probability(
    model: torch.nn.Module,
    aux: bool,
    normalization: str,
    images: torch.Tensor,
) -> torch.Tensor:
    """TTA 分割推理: 原图 + 水平翻转 → 平均 sigmoid 概率。

    Args:
        model: SMP UNet 模型
        aux: 是否使用 aux 分类头 (恒为 False)
        normalization: 归一化方案名
        images: [1, 3, H, W] float32 [0,1]

    Returns:
        [1, 1, H, W] sigmoid 概率
    """
    x = _normalize_batch(images, normalization)
    direct = torch.sigmoid(model(x))
    flipped = torch.sigmoid(model(torch.flip(x, dims=[3])))
    return (direct + torch.flip(flipped, dims=[3])) / 2.0


# ============================================================
# ROI 裁剪
# ============================================================
def crop_roi(
    mask: np.ndarray,
    margin: float = 0.35,
) -> Tuple[int, int, int, int]:
    """从二值掩膜中计算 ROI 边界框 (正方形扩边)。

    Args:
        mask: [H, W] bool 二值掩膜
        margin: 扩边比例 (0.35 = 35% 边长扩展到四周)

    Returns:
        (x1, y1, x2, y2) ROI 坐标 (含边距)
    """
    ys, xs = np.where(mask)
    h, w = mask.shape
    if len(xs) == 0:
        return 0, 0, w, h
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    side = int(max(x2 - x1, y2 - y1) * (1.0 + 2.0 * margin))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    xa = max(0, cx - side // 2)
    ya = max(0, cy - side // 2)
    xb = min(w, xa + side)
    yb = min(h, ya + side)
    # 修正左/上边界
    xa = max(0, xb - side)
    ya = max(0, yb - side)
    return xa, ya, xb, yb


def crop_roi_pil(
    image: Image.Image,
    mask: Image.Image,
    margin: float = 1.0,
) -> Image.Image:
    """从 PIL 图像中按掩膜 ROI 裁剪 (用于分类阶段)。

    Args:
        image: RGB PIL 原图
        mask: L 模式 PIL 掩膜
        margin: 扩边比例 (分类阶段默认=1.0)

    Returns:
        ROI 裁剪后的 RGB PIL 图像
    """
    binary = np.asarray(mask) > 127
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return image
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    width, height = x2 - x1, y2 - y1
    pad = int(max(width, height) * margin)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image.width, x2 + pad)
    y2 = min(image.height, y2 + pad)
    return image.crop((x1, y1, x2, y2))


def dim_background(
    image: Image.Image,
    mask: Image.Image,
    background_scale: float = 0.3,
) -> Image.Image:
    """将非病灶区域压暗至 background_scale 倍亮度。

    Args:
        image: RGB PIL 原图
        mask: L 模式 PIL 掩膜
        background_scale: 背景亮度缩放因子

    Returns:
        RGB PIL 图像
    """
    rgb = np.asarray(image, dtype=np.float32)
    binary = np.asarray(mask.resize(image.size, Image.Resampling.NEAREST)) > 127
    output = rgb * background_scale
    output[binary] = rgb[binary]
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))


# ============================================================
# 后处理
# ============================================================
def postprocess_mask(
    mask: np.ndarray,
    min_area: int = MIN_COMPONENT_AREA,
) -> np.ndarray:
    """掩膜后处理: 填充孔洞 → 连通域分析 → 保留面积 ≥ min_area 的区域。

    Args:
        mask: [H, W] bool 二值掩膜
        min_area: 最小连通域面积 (像素)

    Returns:
        [H, W] bool 后处理掩膜
    """
    filled = ndimage.binary_fill_holes(mask)
    labeled, count = ndimage.label(filled)
    if count == 0:
        return filled
    sizes = ndimage.sum(filled, labeled, index=np.arange(1, count + 1))
    keep = np.where(sizes >= min_area)[0] + 1
    if len(keep) == 0:
        keep = np.array([int(np.argmax(sizes)) + 1])
    return np.isin(labeled, keep)


# ============================================================
# 完整推理管线
# ============================================================
class SegmentationPipeline:
    """两阶段 MiT-B2 UNet 分割管线。

    管线流程:
        1. 粗分割: 2 个粗模型各自 TTA → 平均
        2. ROI 提取: 粗分割 > 0.5 → 正方形扩边裁剪 (margin=0.35)
        3. ROI 精修: 1 个精修模型 TTA → 放回全图
        4. 融合: 0.75 * 粗 + 0.25 * 精 → 阈值 0.38
        5. 后处理: 填充孔洞 + 去小连通域 (<400px)
    """

    def __init__(
        self,
        coarse_weights: List[Union[str, Path]],
        roi_weight: Union[str, Path],
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            coarse_weights: 两个粗分割模型权重路径
            roi_weight: ROI 精修模型权重路径
            device: torch device (默认 auto)
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._coarse_models = [
            load_seg_model(Path(w), self.device) for w in coarse_weights
        ]
        self._roi_model = load_seg_model(Path(roi_weight), self.device)

    @torch.no_grad()
    def predict(
        self,
        image: Union[Image.Image, np.ndarray],
    ) -> np.ndarray:
        """对单张图像运行完整分割管线。

        Args:
            image: RGB PIL Image 或 [H,W,3] uint8 numpy

        Returns:
            [H, W] bool 二值分割掩膜 (原始图像尺寸)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        original_size = image.size  # (W, H)

        # 预处理: resize → tensor
        array = np.array(
            image.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
        )
        tensor = (
            torch.from_numpy(array)
            .permute(2, 0, 1)
            .float()
            .div(255)
            .unsqueeze(0)
            .to(self.device)
        )

        # 1. 粗分割: 两模型平均
        coarse_probs = torch.stack([
            _seg_probability(model, aux, norm, tensor)
            for model, aux, norm in self._coarse_models
        ]).mean(dim=0)

        # 2. ROI 提取
        xa, ya, xb, yb = crop_roi(
            coarse_probs[0, 0].cpu().numpy() >= COARSE_THRESHOLD,
            ROI_MARGIN,
        )

        # 3. ROI 精修
        crop = F.interpolate(
            tensor[:, :, ya:yb, xa:xb],
            size=(INPUT_SIZE, INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        model, aux, norm = self._roi_model
        roi_prob = _seg_probability(model, aux, norm, crop)

        # 放回全图
        refined = torch.zeros_like(coarse_probs)
        refined[:, :, ya:yb, xa:xb] = F.interpolate(
            roi_prob,
            size=(yb - ya, xb - xa),
            mode="bilinear",
            align_corners=False,
        )

        # 4. 融合 + 后处理
        fused = (
            COARSE_WEIGHT * coarse_probs + ROI_WEIGHT * refined
        )[0, 0].cpu().numpy()
        mask = postprocess_mask(fused >= FINAL_THRESHOLD)

        # 5. 还原到原始尺寸
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_img = mask_img.resize(original_size, Image.Resampling.NEAREST)
        return np.asarray(mask_img) > 127

    def predict_pil(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """与 predict 相同，但返回 PIL Image (L 模式)。"""
        mask = self.predict(image)
        return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")


# ============================================================
# CLI 批量推理
# ============================================================
def main():
    """命令行批量分割推理入口。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="两阶段 MiT-B2 UNet 分割管线 — 批量推理"
    )
    parser.add_argument("--input-dir", required=True, help="输入图像目录")
    parser.add_argument("--output-dir", required=True, help="输出掩膜目录")
    parser.add_argument(
        "--coarse", nargs=2, required=True,
        help="两个粗分割模型权重路径",
    )
    parser.add_argument("--roi", required=True, help="ROI 精修模型权重路径")
    args = parser.parse_args()

    pipeline = SegmentationPipeline(
        coarse_weights=args.coarse,
        roi_weight=args.roi,
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    paths = [
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        and "_mask" not in p.stem.lower()
    ]

    for i, path in enumerate(paths, start=1):
        image = Image.open(path).convert("RGB")
        mask = pipeline.predict(image)
        dest = (output_dir / path.relative_to(input_dir)).with_suffix(".png")
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(dest)
        print(f"{i}/{len(paths)} {path.name}", flush=True)


if __name__ == "__main__":
    main()
