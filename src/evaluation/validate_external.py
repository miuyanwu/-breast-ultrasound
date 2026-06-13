"""
External dataset validation for V9 weighted compact.

用法:
  python scripts/validate_external.py --dataset breast   # BrEaST-Lesions-USG
  python scripts/validate_external.py --dataset uclm     # BUS-UCLM
  python scripts/validate_external.py --dataset all      # both

数据集说明:
  BrEaST-Lesions-USG: 256 cases, benign=154, malignant=98, normal=4
    → normal 样本另有 benign 标签，作为良性评估
  BUS-UCLM: benign=174, malign=90, normal=419
    → 有大量 normal（无病灶）样本，模型未训练过此类别
    → 默认 normal 不参与良性/恶性 AUC 计算，但会单独报告模型对其的预测分布
"""
import sys, os
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score,
    precision_score, f1_score, confusion_matrix,
)

from src.config import MODEL_REGISTRY, resolve_weight_path
from src.models.base import BreastCancerMultiTaskNet

# ============================================================
# V9 配置
# ============================================================
V9_MODELS = ['DISTILL', 'SOUP', 'V2.3']
V9_WEIGHTS = np.array([0.043422758938255444, 0.8786080776229839, 0.07796916343876055],
                       dtype=np.float64)
V9_WEIGHTS = V9_WEIGHTS / V9_WEIGHTS.sum()


def load_v9_models(device):
    """Load V9 three models: DISTILL, SOUP, V2.3."""
    models = {}
    for name in V9_MODELS:
        cfg = MODEL_REGISTRY[name]
        model = cfg['cls']().to(device)
        path = resolve_weight_path(cfg['weight'])
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        models[name] = model
        print(f"  Loaded {name}: {os.path.basename(path)}")
    return models


@torch.no_grad()
def v9_infer(models, image_tensor, device):
    """TTA flip → extract logits → weighted logit fusion → prob."""
    logits_all = []
    for name in V9_MODELS:
        model = models[name]
        # Original
        _, logit_orig = model(image_tensor)
        # Flip TTA
        flipped = torch.flip(image_tensor, dims=[3])
        _, logit_flip = model(flipped)
        avg_logit = (logit_orig + logit_flip) / 2.0
        logits_all.append(avg_logit.cpu().numpy())

    logits_matrix = np.column_stack(logits_all)  # [B, 3]
    fused_logit = logits_matrix @ V9_WEIGHTS
    fused_prob = 1.0 / (1.0 + np.exp(-fused_logit))
    return fused_prob


# ============================================================
# Dataset loaders
# ============================================================
def load_breast_dataset():
    """Load BrEaST-Lesions-USG from classification folder.
    Returns: (images_uint8 [N,256,256,3], labels [N], filenames [N])
    """
    base = os.path.join(_PROJECT_ROOT, '外部数据1', 'breast-lesions-usg',
                         'breast-lesions-usg', 'classification')
    images, labels, filenames = [], [], []
    for cls_name, cls_label in [('benign', 0), ('malignant', 1), ('normal', 0)]:
        cls_dir = os.path.join(base, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.lower().endswith('.png'):
                continue
            img_path = os.path.join(cls_dir, fname)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((256, 256), Image.BILINEAR)
            images.append(np.array(img, dtype=np.uint8))
            labels.append(cls_label)
            filenames.append(f'{cls_name}/{fname}')
    return images, np.array(labels), filenames


def load_uclm_dataset():
    """Load BUS-UCLM from separated folder.
    Returns: (images, labels, filenames, normals)
    """
    base = os.path.join(_PROJECT_ROOT, '外部数据2', 'bus_uclm_separated')
    images, labels, filenames = [], [], []
    normal_images, normal_fnames = [], []

    for cls_name, cls_label in [('benign', 0), ('malign', 1)]:
        cls_dir = os.path.join(base, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.lower().endswith('.png'):
                continue
            img_path = os.path.join(cls_dir, fname)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((256, 256), Image.BILINEAR)
            images.append(np.array(img, dtype=np.uint8))
            labels.append(cls_label)
            filenames.append(f'{cls_name}/{fname}')

    # Normal samples: not in benign/malignant training distribution
    normal_dir = os.path.join(base, 'normal')
    if os.path.isdir(normal_dir):
        for fname in sorted(os.listdir(normal_dir)):
            if not fname.lower().endswith('.png'):
                continue
            img_path = os.path.join(normal_dir, fname)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((256, 256), Image.BILINEAR)
            normal_images.append(np.array(img, dtype=np.uint8))
            normal_fnames.append(f'normal/{fname}')

    return images, np.array(labels), filenames, normal_images, normal_fnames


# ============================================================
# Evaluation
# ============================================================
def evaluate(y_true, y_score, dataset_name):
    """Print multi-threshold metrics."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    auc = roc_auc_score(y_true, y_score)
    n_benign = int((y_true == 0).sum())
    n_malig = int((y_true == 1).sum())

    print(f"\n{'='*60}")
    print(f"  {dataset_name}  —  {n_benign+n_malig} samples (benign={n_benign}, malignant={n_malig})")
    print(f"{'='*60}")

    # Best F1 sweep
    best = {'thresh': None, 'f1': -1.0}
    for thresh in np.linspace(0.05, 0.95, 181):
        preds = (y_score > thresh).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best['f1']:
            best = {'thresh': float(thresh), 'f1': float(f1)}

    for thresh in [0.40, 0.45, 0.50, 0.55]:
        preds = (y_score > thresh).astype(int)
        acc = accuracy_score(y_true, preds)
        rec = recall_score(y_true, preds)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = precision_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds)
        print(f"  thresh={thresh:.2f}: AUC={auc:.4f} Acc={acc:.4f} "
              f"Rec={rec:.4f} Spec={spec:.4f} Prec={prec:.4f} F1={f1:.4f}")

    print(f"  best F1={best['f1']:.4f} @{best['thresh']:.3f}")
    return auc


def evaluate_normal(models, normal_images, normal_fnames, device):
    """Report model probability distribution on normal (no-lesion) samples."""
    if not normal_images:
        return
    batch = torch.from_numpy(np.stack(normal_images)).permute(0, 3, 1, 2).float() / 255.0
    probs = []
    for i in range(0, len(batch), 16):
        chunk = batch[i:i+16].to(device)
        probs.extend(v9_infer(models, chunk, device).tolist())

    probs = np.array(probs)
    print(f"\n--- Normal samples (n={len(probs)}) — 模型未训练过此类别 ---")
    print(f"  Prob mean={probs.mean():.4f}  median={np.median(probs):.4f}  "
          f"std={probs.std():.4f}")
    print(f"  Prob > 0.50: {(probs > 0.50).sum()}/{len(probs)} "
          f"({(probs > 0.50).mean()*100:.1f}% 被判恶性)")
    print(f"  Prob > 0.45: {(probs > 0.45).sum()}/{len(probs)}")
    print(f"  Prob > 0.40: {(probs > 0.40).sum()}/{len(probs)}")
    # Show some examples
    idxs = np.argsort(probs)[::-1]
    print(f"  Top-5 highest prob (most 'malignant-looking'):")
    for i in idxs[:5]:
        print(f"    {normal_fnames[i]:30s}  prob={probs[i]:.4f}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['breast', 'uclm', 'all'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("Loading V9 models (DISTILL + SOUP + V2.3)...")
    models = load_v9_models(device)

    @torch.no_grad()
    def run_inference(images_list):
        """Convert uint8 list → tensor → V9 inference."""
        batch = torch.from_numpy(np.stack(images_list)).permute(0, 3, 1, 2).float() / 255.0
        probs = []
        for i in range(0, len(batch), 16):
            chunk = batch[i:i+16].to(device)
            probs.extend(v9_infer(models, chunk, device).tolist())
        return np.array(probs)

    if args.dataset in ('breast', 'all'):
        imgs, labels, fnames = load_breast_dataset()
        probs = run_inference(imgs)
        evaluate(labels, probs, 'BrEaST-Lesions-USG (外部数据1)')

    if args.dataset in ('uclm', 'all'):
        imgs, labels, fnames, normal_imgs, normal_fnames = load_uclm_dataset()
        probs = run_inference(imgs)
        evaluate(labels, probs, 'BUS-UCLM (外部数据2)')
        evaluate_normal(models, normal_imgs, normal_fnames, device)


if __name__ == '__main__':
    main()
