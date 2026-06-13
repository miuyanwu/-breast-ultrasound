"""
集成测试: multi-model fusion + optional multi-scale TTA + stacking.

默认融合: mean(sigmoid(logit_i))

用法: python -m src.evaluation.test_ensemble --models V2.3,V5.1,V5.2,SAM
"""
import sys, os
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch, torch.nn as nn, numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

from src.data import BreastUltrasoundDataset, get_case_split_train_val
from src.config import MODEL_REGISTRY, resolve_weight_path, resolve_device


def load_models(device, enabled):
    loaded = {}
    for name in enabled:
        if name not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model '{name}'. Available: {', '.join(MODEL_REGISTRY)}")
        cfg = MODEL_REGISTRY[name]
        model = cfg['cls']().to(device)
        weight_path = resolve_weight_path(cfg['weight'])
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.eval()
        loaded[name] = {'model': model, 'cfg': cfg}
        print(f"  Loaded {name}: {weight_path}")
    return loaded


def extract_cls_logits(model, images, arch):
    """Extract classification logits from model output, handling different architectures."""
    out = model(images)
    if arch == 'v2':
        cls_logits = out[1]
    elif arch == 'v5':
        cls_logits = out[1]
    elif arch == 'v6':
        cls_logits = out[1]
    else:
        raise ValueError(f"Unknown arch: {arch}")
    if cls_logits.dim() == 0:
        cls_logits = cls_logits.unsqueeze(0)
    return cls_logits


@torch.no_grad()
def ensemble_test(enabled, use_tta_scales=False, stacking_models=None,
                  device_arg='auto', fusion='prob_mean', batch_size=16,
                  fusion_weights=None):
    device = resolve_device(device_arg)
    scales = [0.75, 1.0, 1.25] if use_tta_scales else [1.0]

    print(f"\n=== V7 Ensemble: {' + '.join(enabled)} ===")
    print(f"Device: {device} | TTA Scales: {scales} | Stacking: {stacking_models is not None} | Fusion: {fusion}")
    weights_arr = None
    if fusion_weights is not None:
        weights_arr = np.asarray(fusion_weights, dtype=np.float64)
        if len(weights_arr) != len(enabled):
            raise ValueError(f"--weights length {len(weights_arr)} does not match --models length {len(enabled)}")
        if np.any(weights_arr < 0):
            raise ValueError("--weights must be non-negative")
        weight_sum = weights_arr.sum()
        if weight_sum <= 0:
            raise ValueError("--weights must sum to a positive value")
        weights_arr = weights_arr / weight_sum
        print(f"Fusion weights: {dict(zip(enabled, weights_arr.tolist()))}")

    models = load_models(device, enabled)

    test_ds = BreastUltrasoundDataset('./data', mode='test')
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- Stacking: fit logistic regression on validation set ---
    stacking_coefs = None
    stacking_intercept = None
    if stacking_models:
        print("\n[Stacking] Fitting Logistic Regression on validation set...")
        _, val_dataset = get_case_split_train_val(data_dir='./data', val_ratio=0.2, seed=42)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        val_all_logits = {name: [] for name in stacking_models}
        val_labels = []

        for images, masks, labels, birads, _ in val_loader:
            images, labels = images.to(device), labels.numpy()
            val_labels.extend(labels)
            for name in stacking_models:
                entry = models[name]
                cls_logits = extract_cls_logits(entry['model'], images, entry['cfg']['arch'])
                cls_logits_f = extract_cls_logits(entry['model'], torch.flip(images, dims=[3]), entry['cfg']['arch'])
                avg_logits = (cls_logits + cls_logits_f) / 2.0
                val_all_logits[name].extend(avg_logits.cpu().numpy())

        X_val = np.column_stack([val_all_logits[name] for name in stacking_models])
        y_val = np.array(val_labels)
        lr = LogisticRegression(C=100.0, max_iter=1000)
        lr.fit(X_val, y_val)
        stacking_coefs = lr.coef_[0]
        stacking_intercept = lr.intercept_[0]
        print(f"  Stacking weights: {dict(zip(stacking_models, stacking_coefs))}")
        print(f"  Stacking intercept: {stacking_intercept:.4f}")

    # --- Test inference ---
    y_true, all_probs = [], []
    per_model_logits = {name: [] for name in enabled}

    for images, masks, labels, birads, filenames in test_loader:
        images = images.to(device)
        B = images.size(0)

        for name in enabled:
            entry = models[name]
            model_logit_sum = torch.zeros(B, device=device)

            for scale in scales:
                if scale != 1.0:
                    h, w = images.shape[2], images.shape[3]
                    new_h = round(h * scale / 32) * 32
                    new_w = round(w * scale / 32) * 32
                    scaled_images = torch.nn.functional.interpolate(
                        images, size=(new_h, new_w), mode='bilinear', align_corners=False)
                else:
                    scaled_images = images

                cls_orig = extract_cls_logits(entry['model'], scaled_images, entry['cfg']['arch'])
                cls_flip = extract_cls_logits(entry['model'], torch.flip(scaled_images, dims=[3]), entry['cfg']['arch'])
                model_logit_sum += (cls_orig + cls_flip) / 2.0

            model_logit = model_logit_sum / len(scales)
            per_model_logits[name].extend(model_logit.cpu().numpy())

        y_true.extend(labels.numpy())

    y_true_np = np.array(y_true)
    logits_matrix = np.column_stack([per_model_logits[name] for name in enabled])

    if stacking_coefs is not None:
        ensemble_logits = logits_matrix @ stacking_coefs + stacking_intercept
        all_probs_np = 1.0 / (1.0 + np.exp(-ensemble_logits))
    else:
        prob_matrix = 1.0 / (1.0 + np.exp(-logits_matrix))
        if fusion == 'prob_mean':
            if weights_arr is None:
                all_probs_np = prob_matrix.mean(axis=1)
            else:
                all_probs_np = prob_matrix @ weights_arr
        elif fusion == 'logit_mean':
            if weights_arr is None:
                fused_logits = logits_matrix.mean(axis=1)
            else:
                fused_logits = logits_matrix @ weights_arr
            all_probs_np = 1.0 / (1.0 + np.exp(-fused_logits))
        elif fusion == 'legacy_bug':
            if weights_arr is not None:
                raise ValueError("--weights is not supported with --fusion legacy_bug")
            all_probs_np = 1.0 / (1.0 + np.exp(-logits_matrix)).mean(axis=1)
        else:
            raise ValueError(f"Unknown fusion mode: {fusion}")

    all_probs = all_probs_np.tolist()
    y_true = y_true_np

    per_model_probs = {}
    for name in enabled:
        logits_arr = np.array(per_model_logits[name])
        per_model_probs[name] = (1.0 / (1.0 + np.exp(-logits_arr))).tolist()

    y_true, all_probs = np.array(y_true), np.array(all_probs)

    # Print results
    print(f"\n=== {' + '.join(enabled)} Results ===")
    if use_tta_scales:
        print(f"(Multi-scale TTA: {scales})")
    if stacking_coefs is not None:
        print(f"(Stacking: LR weights)")
    auc = roc_auc_score(y_true, all_probs)
    best_f1 = {'threshold': None, 'f1': -1.0, 'rec': 0.0, 'spec': 0.0}
    for thresh in np.linspace(0.05, 0.95, 181):
        preds = (all_probs > thresh).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1['f1']:
            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
            best_f1 = {
                'threshold': float(thresh),
                'f1': float(f1),
                'rec': float(recall_score(y_true, preds)),
                'spec': float(tn / (tn + fp)) if tn + fp > 0 else 0.0,
            }
    print(
        f"  best_f1: thresh={best_f1['threshold']:.3f} "
        f"F1={best_f1['f1']:.4f} Rec={best_f1['rec']:.4f} Spec={best_f1['spec']:.4f}"
    )

    for thresh in [0.40, 0.45, 0.50, 0.55]:
        preds = (all_probs > thresh).astype(int)
        rec = recall_score(y_true, preds)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        spec = tn / (tn + fp) if tn + fp > 0 else 0
        prec = precision_score(y_true, preds)
        f1 = f1_score(y_true, preds)
        acc = accuracy_score(y_true, preds)
        print(f"  thresh={thresh:.2f}: AUC={auc:.4f} Acc={acc:.4f} Rec={rec:.4f} Spec={spec:.4f} Prec={prec:.4f} F1={f1:.4f}")

    print(f"\n{'='*60}")
    print("Per-Model (TTA flip only, no multi-scale):")
    for name in enabled:
        yp = np.array(per_model_probs[name])
        auc_i = roc_auc_score(y_true, yp)
        for t in [0.45, 0.50]:
            pr = (yp > t).astype(int)
            rec_i = recall_score(y_true, pr)
            cm = confusion_matrix(y_true, pr).ravel()
            spec_i = cm[0] / (cm[0] + cm[1]) if len(cm) >= 2 else 0
            f1_i = f1_score(y_true, pr)
            print(f"  {name} @{t:.2f}: AUC={auc_i:.4f} Rec={rec_i:.4f} Spec={spec_i:.4f} F1={f1_i:.4f}")

    return y_true, all_probs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, default='V2.3,V5.1,V5.2',
                        help='Comma-separated model names')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--fusion', type=str, default='prob_mean',
                        choices=['prob_mean', 'logit_mean', 'legacy_bug'])
    parser.add_argument('--weights', type=str, default=None,
                        help='Comma-separated fusion weights aligned with --models')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--tta_scales', action='store_true',
                        help='Enable multi-scale TTA (0.8x/1.0x/1.2x)')
    parser.add_argument('--stacking', action='store_true',
                        help='Learn stacking weights via Logistic Regression')
    args = parser.parse_args()

    enabled = args.models.split(',')
    weights = None
    if args.weights:
        weights = [float(x.strip()) for x in args.weights.split(',') if x.strip()]
    ensemble_test(enabled, use_tta_scales=args.tta_scales,
                  stacking_models=enabled if args.stacking else None,
                  device_arg=args.device, fusion=args.fusion,
                  batch_size=args.batch_size,
                  fusion_weights=weights)
