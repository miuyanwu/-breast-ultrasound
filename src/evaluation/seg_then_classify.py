"""
分割→分类联合评估: 原图 V9 + 分割 ROI V9 融合评测。

用法:
    # 测试集评估 (benign/malignant 子目录)
    python -m src.evaluation.seg_then_classify \
        --image-dir data/测试集 \
        --mask-dir outputs/predicted_masks/test \
        --output-dir outputs/seg_then_classify/test

    # 训练集评估 (CSV + Images/Masks 目录)
    python -m src.evaluation.seg_then_classify \
        --image-dir data/训练集 \
        --mask-dir outputs/predicted_masks/train \
        --output-dir outputs/seg_then_classify/train \
        --csv data/训练集/bus_data.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# 确保项目根目录在 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import MODEL_REGISTRY, resolve_weight_path
from src.segmentation.roi_pipeline import crop_roi_pil, dim_background

# ============================================================
# 常量
# ============================================================
V9_MODELS = ["DISTILL", "SOUP", "V2.3"]
V9_WEIGHTS = np.array(
    [0.043422758938255444, 0.8786080776229839, 0.07796916343876055],
    dtype=np.float64,
)
METRIC_WEIGHTS = {
    "auc": 15,
    "accuracy": 10,
    "sensitivity": 15,
    "specificity": 10,
    "precision": 5,
    "f1": 5,
}
CLASS_SIZE = 256  # 分类模型输入尺寸


# ============================================================
# 分类模型加载与推理
# ============================================================
def load_v9_models(device: torch.device) -> List[torch.nn.Module]:
    """加载 V9 三个分类模型。"""
    models = []
    for name in V9_MODELS:
        cfg = MODEL_REGISTRY[name]
        model = cfg["cls"]().to(device)
        wpath = resolve_weight_path(cfg["weight"])
        model.load_state_dict(
            torch.load(wpath, map_location=device, weights_only=True)
        )
        model.eval()
        models.append(model)
    return models


@torch.no_grad()
def classify_batch(
    models: List[torch.nn.Module],
    arrays: List[np.ndarray],
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """对图像数组列表做 V9 加权 logit 融合推理。

    Args:
        models: V9 三模型列表
        arrays: [N] 个 uint8 [256,256,3] numpy 数组
        device: torch device
        batch_size: 批大小

    Returns:
        [N] 恶性概率数组
    """
    probabilities = []
    for start in range(0, len(arrays), batch_size):
        batch_np = np.stack(arrays[start : start + batch_size])
        batch = (
            torch.from_numpy(batch_np)
            .permute(0, 3, 1, 2)
            .float()
            .div_(255)
            .to(device)
        )
        logits = []
        for model in models:
            _, direct = model(batch)
            _, flipped = model(torch.flip(batch, dims=[3]))
            logits.append(((direct + flipped) / 2.0).cpu().numpy())
        fused = np.column_stack(logits) @ V9_WEIGHTS
        probabilities.extend((1.0 / (1.0 + np.exp(-fused))).tolist())
    return np.asarray(probabilities, dtype=np.float64)


# ============================================================
# 指标计算
# ============================================================
def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.55,
) -> Dict:
    """计算固定阈值的全部指标。"""
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    result = {
        "auc": float(roc_auc_score(labels, probs)),
        "accuracy": float(accuracy_score(labels, pred)),
        "sensitivity": float(recall_score(labels, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp) if (tn + fp) > 0 else 0),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }
    result["weighted_points"] = float(
        sum(METRIC_WEIGHTS[k] * result[k] for k in METRIC_WEIGHTS)
    )
    result["weighted_score"] = result["weighted_points"] / 60.0
    return result


def find_best_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
) -> Dict:
    """搜索最佳阈值 (最大化加权总分)。"""
    candidates = np.unique(
        np.r_[np.linspace(0.02, 0.98, 193), probs]
    )
    best = None
    best_score = -1.0
    for t in candidates:
        m = compute_metrics(labels, probs, float(t))
        if m["weighted_points"] > best_score:
            best_score = m["weighted_points"]
            best = m
    return best


# ============================================================
# 数据加载
# ============================================================
def load_from_subdirs(
    image_dir: Path,
    mask_dir: Path,
) -> Tuple[List[np.ndarray], np.ndarray, pd.DataFrame]:
    """从 benign/malignant 子目录加载图像和掩膜。

    Returns:
        (variants dict, labels array, records DataFrame)
    """
    variants: Dict[str, List[np.ndarray]] = {
        "original": [],
        "roi_m025": [],
        "roi_m050": [],
        "roi_m100": [],
        "dim_background": [],
    }
    records = []

    malignant_folder = (
        "malignant" if (image_dir / "malignant").is_dir() else "malign"
    )
    for class_name, label in [("benign", 0), (malignant_folder, 1)]:
        class_image_dir = image_dir / class_name
        class_mask_dir = mask_dir / class_name
        for path in sorted(
            p for p in class_image_dir.glob("*.png")
            if "_mask" not in p.stem.lower()
        ):
            image = Image.open(path).convert("RGB")
            mask = Image.open(class_mask_dir / path.name).convert("L")

            variants["original"].append(
                np.asarray(
                    image.resize((CLASS_SIZE, CLASS_SIZE), Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                )
            )
            for margin, key in [(0.25, "roi_m025"), (0.50, "roi_m050"), (1.00, "roi_m100")]:
                variants[key].append(
                    np.asarray(
                        crop_roi_pil(image, mask, margin).resize(
                            (CLASS_SIZE, CLASS_SIZE), Image.Resampling.BILINEAR
                        ),
                        dtype=np.uint8,
                    )
                )
            variants["dim_background"].append(
                np.asarray(
                    dim_background(image, mask).resize(
                        (CLASS_SIZE, CLASS_SIZE), Image.Resampling.BILINEAR
                    ),
                    dtype=np.uint8,
                )
            )
            records.append({
                "id": path.name,
                "class": class_name,
                "true_label": label,
            })

    records_df = pd.DataFrame(records)
    labels = records_df["true_label"].values.astype(int)
    return variants, labels, records_df


def load_from_csv(
    data_dir: Path,
    mask_dir: Path,
) -> Tuple[Dict[str, List[np.ndarray]], np.ndarray, pd.DataFrame]:
    """从 bus_data.csv + Images/Masks 目录加载 (训练集格式)。

    Returns:
        (variants dict, labels array, records DataFrame)
    """
    table = pd.read_csv(data_dir / "bus_data.csv")
    originals, rois, ids, label_list = [], [], [], []
    for row in table.itertuples(index=False):
        image = Image.open(data_dir / "Images" / f"{row.ID}.png").convert("RGB")
        mask = Image.open(mask_dir / f"{row.ID}.png").convert("L")
        originals.append(
            np.asarray(
                image.resize((CLASS_SIZE, CLASS_SIZE), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
        )
        rois.append(
            np.asarray(
                crop_roi_pil(image, mask, margin=1.0).resize(
                    (CLASS_SIZE, CLASS_SIZE), Image.Resampling.BILINEAR
                ),
                dtype=np.uint8,
            )
        )
        label_list.append(1 if row.Pathology == "malignant" else 0)
        ids.append(row.ID)

    variants = {"original": originals, "roi_m100": rois}
    labels = np.asarray(label_list, dtype=int)
    records_df = pd.DataFrame({"id": ids, "true_label": label_list})
    return variants, labels, records_df


# ============================================================
# 主入口
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="分割→分类联合评估 (原图 V9 + ROI V9 融合)"
    )
    parser.add_argument("--image-dir", required=True, help="图像目录")
    parser.add_argument("--mask-dir", required=True, help="预测掩膜目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument(
        "--csv",
        default=None,
        help="训练集 CSV 路径 (bus_data.csv)。提供则使用 CSV 模式，否则使用子目录模式。",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="'cuda', 'cpu', 或 'auto' (默认)",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    if args.csv:
        variants, labels, records_df = load_from_csv(
            image_dir, mask_dir
        )
        # CSV 模式: 只融合原图 + roi_m100
        fusion_pairs = [("roi_m100",)]
    else:
        variants, labels, records_df = load_from_subdirs(
            image_dir, mask_dir
        )
        # 子目录模式: 测试所有 ROI 变体和融合
        fusion_pairs = [
            ("roi_m025",), ("roi_m050",), ("roi_m100",), ("dim_background",),
        ]

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"  Device: {device}")
    print(f"  Samples: {len(labels)}")
    print(f"  Benign: {(labels == 0).sum()}, Malignant: {(labels == 1).sum()}")

    # 加载分类模型
    print("  Loading V9 classification models...")
    models = load_v9_models(device)

    # 推理各变体
    probs: Dict[str, np.ndarray] = {}
    for name in variants:
        print(f"  Inferring {name} ({len(variants[name])} images)...")
        probs[name] = classify_batch(models, variants[name], device)

    # 构建候选融合方案
    candidates: Dict[str, np.ndarray] = {}
    # 纯变体
    for name, prob in probs.items():
        candidates[name] = prob
    # 原图 + ROI 融合
    for roi_name, *_ in fusion_pairs:
        if roi_name not in probs:
            continue
        for original_weight in np.linspace(0.1, 0.9, 9):
            name = f"fusion_original_{original_weight:.1f}_{roi_name}"
            candidates[name] = (
                original_weight * probs["original"]
                + (1.0 - original_weight) * probs[roi_name]
            )

    # 评估所有候选方案
    print(f"  Evaluating {len(candidates)} candidates...")
    results = []
    for name, probability in candidates.items():
        results.append({
            "name": name,
            "fixed_threshold_0_55": compute_metrics(labels, probability, 0.55),
            "best_threshold": find_best_threshold(labels, probability),
        })

    # 按固定阈值总分排序
    results.sort(
        key=lambda r: r["fixed_threshold_0_55"]["weighted_points"],
        reverse=True,
    )

    # 输出
    # (a) 逐图预测 CSV
    pred_df = records_df.copy()
    for name, probability in probs.items():
        pred_df[f"prob_{name}"] = probability
    pred_df.to_csv(
        output_dir / "classification_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # (b) 候选方案排名 CSV
    ranking_rows = []
    for r in results:
        row = {"name": r["name"]}
        row.update(r["fixed_threshold_0_55"])
        ranking_rows.append(row)
    pd.DataFrame(ranking_rows).to_csv(
        output_dir / "classification_candidate_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # (c) 完整报告 JSON
    report = {
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "sample_count": int(len(labels)),
        "benign_count": int((labels == 0).sum()),
        "malignant_count": int((labels == 1).sum()),
        "classification_models": V9_MODELS,
        "v9_weights": V9_WEIGHTS.tolist(),
        "metric_weights": METRIC_WEIGHTS,
        "best_fixed": results[0],
        "original_baseline": next(
            (r for r in results if r["name"] == "original"), None
        ),
        "all_results": results,
    }
    (output_dir / "classification_all_candidates.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 终端摘要
    print(f"\n{'='*60}")
    print("  最佳方案 (固定阈值 0.55)")
    print(f"{'='*60}")
    best = results[0]
    print(f"  方案: {best['name']}")
    m = best["fixed_threshold_0_55"]
    print(f"  AUC={m['auc']:.4f}  Acc={m['accuracy']:.4f}  "
          f"Sens={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}  "
          f"Prec={m['precision']:.4f}  F1={m['f1']:.4f}")
    print(f"  加权总分: {m['weighted_points']:.2f}/60")

    orig = next((r for r in results if r["name"] == "original"), None)
    if orig:
        om = orig["fixed_threshold_0_55"]
        delta_auc = m["auc"] - om["auc"]
        delta_f1 = m["f1"] - om["f1"]
        print(f"\n  对比纯原图基线:")
        print(f"    ΔAUC = {delta_auc:+.4f}  ΔF1 = {delta_f1:+.4f}")

    print(f"\n  完整结果已保存至: {output_dir}")


if __name__ == "__main__":
    main()
