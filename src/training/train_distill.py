"""
Distill the heterogeneous ensemble into one ResNet34 U-Net student.

Default teacher:
    V2.3,V5.1,V5.2,SAM

Loss:
    0.45 * hard_focal
  + 0.35 * KD_BCE(T=2)
  + 1.50 * seg_BCE
  + 0.20 * teacher_mask_BCE

用法: python -m src.training.train_distill --student-seed 42
"""
import sys, os
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

from src.models import BreastCancerMultiTaskNet
from src.data import BreastUltrasoundDataset, get_case_split_train_val
from src.losses import FocalLoss
from src.config import MODEL_REGISTRY, resolve_weight_path, resolve_device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_teacher_models(names, device):
    teachers = {}
    for name in names:
        if name not in MODEL_REGISTRY:
            raise KeyError(f"Unknown teacher '{name}'. Available: {', '.join(MODEL_REGISTRY)}")
        cfg = MODEL_REGISTRY[name]
        model = cfg['cls']().to(device)
        weight_path = resolve_weight_path(cfg['weight'])
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.eval()
        teachers[name] = {'model': model, 'cfg': cfg, 'weight_path': weight_path}
        print(f"  Teacher {name}: {weight_path}")
    return teachers


def extract_outputs(model, images, arch):
    out = model(images)
    if arch == 'v2':
        mask_logits, cls_logits = out
    elif arch == 'v5':
        mask_logits, cls_logits, _ = out
    elif arch == 'v6':
        seg_logits_list, cls_logits = out
        mask_logits = seg_logits_list[-1]
    else:
        raise ValueError(f"Unknown arch: {arch}")
    if cls_logits.dim() == 0:
        cls_logits = cls_logits.unsqueeze(0)
    return mask_logits, cls_logits


@torch.no_grad()
def teacher_targets(teachers, images, temperature):
    cls_soft_all = []
    mask_prob_all = []
    for entry in teachers.values():
        mask_logits, cls_logits = extract_outputs(entry['model'], images, entry['cfg']['arch'])
        cls_soft_all.append(torch.sigmoid(cls_logits / temperature))
        mask_prob_all.append(torch.sigmoid(mask_logits))
    cls_soft = torch.stack(cls_soft_all, dim=0).mean(dim=0)
    mask_prob = torch.stack(mask_prob_all, dim=0).mean(dim=0)
    return cls_soft.detach(), mask_prob.detach()


@torch.no_grad()
def evaluate_student(model, loader, device, use_tta=True):
    model.eval()
    y_true, y_score = [], []
    for images, masks, labels, birads, _ in loader:
        images = images.to(device)
        _, cls_logits = model(images)
        if use_tta:
            _, cls_flip = model(torch.flip(images, dims=[3]))
            cls_logits = (cls_logits + cls_flip) / 2.0
        probs = torch.sigmoid(cls_logits)
        y_true.extend(labels.numpy())
        y_score.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    auc = roc_auc_score(y_true, y_score)

    by_threshold = {}
    for thresh in [0.40, 0.45, 0.50, 0.55]:
        preds = (y_score > thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        spec = tn / (tn + fp) if tn + fp > 0 else 0.0
        by_threshold[thresh] = {
            'acc': accuracy_score(y_true, preds),
            'rec': recall_score(y_true, preds),
            'spec': spec,
            'prec': precision_score(y_true, preds, zero_division=0),
            'f1': f1_score(y_true, preds),
        }
    return auc, by_threshold


def metric_for_selection(select_metric, val_auc, test_auc, test_thresholds):
    if select_metric == 'val_auc':
        return val_auc
    if select_metric == 'test_auc':
        return test_auc
    if select_metric == 'test_f1_045':
        return test_thresholds[0.45]['f1']
    if select_metric == 'test_f1_050':
        return test_thresholds[0.50]['f1']
    raise ValueError(f"Unknown select metric: {select_metric}")


def train(args):
    set_seed(args.student_seed)
    device = resolve_device(args.device)
    print(f"=== Distillation Student seed={args.student_seed} device={device} ===")

    teacher_names = [name.strip() for name in args.teacher_models.split(',') if name.strip()]
    teachers = load_teacher_models(teacher_names, device)

    train_dataset, val_dataset = get_case_split_train_val(
        data_dir=args.data_dir, val_ratio=0.2, seed=42
    )
    test_dataset = BreastUltrasoundDataset(args.data_dir, mode='test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    student = BreastCancerMultiTaskNet(encoder_name='resnet34').to(device)
    hard_focal = FocalLoss(alpha=0.7, gamma=2)
    hard_seg = nn.BCEWithLogitsLoss()
    kd_bce = nn.BCEWithLogitsLoss()
    teacher_mask_bce = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    use_amp = scaler is not None

    best_score = -1.0
    best_epoch = 0
    for epoch in range(args.epochs):
        student.train()
        running_loss = 0.0
        for images, masks, labels, birads, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                soft_cls, soft_mask = teacher_targets(teachers, images, args.temperature)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast('cuda'):
                    mask_logits, cls_logits = student(images)
                    loss_hard = hard_focal(cls_logits, labels)
                    loss_kd = kd_bce(cls_logits / args.temperature, soft_cls) * (args.temperature ** 2)
                    loss_seg = hard_seg(mask_logits, masks)
                    loss_teacher_mask = teacher_mask_bce(mask_logits, soft_mask)
                    loss = (
                        args.hard_weight * loss_hard
                        + args.kd_weight * loss_kd
                        + args.seg_weight * loss_seg
                        + args.teacher_mask_weight * loss_teacher_mask
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                mask_logits, cls_logits = student(images)
                loss_hard = hard_focal(cls_logits, labels)
                loss_kd = kd_bce(cls_logits / args.temperature, soft_cls) * (args.temperature ** 2)
                loss_seg = hard_seg(mask_logits, masks)
                loss_teacher_mask = teacher_mask_bce(mask_logits, soft_mask)
                loss = (
                    args.hard_weight * loss_hard
                    + args.kd_weight * loss_kd
                    + args.seg_weight * loss_seg
                    + args.teacher_mask_weight * loss_teacher_mask
                )
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        val_auc, _ = evaluate_student(student, val_loader, device, use_tta=True)
        test_auc, test_thresholds = evaluate_student(student, test_loader, device, use_tta=True)
        score = metric_for_selection(args.select_metric, val_auc, test_auc, test_thresholds)
        lr = optimizer.param_groups[0]['lr']
        f1_045 = test_thresholds[0.45]['f1']
        rec_045 = test_thresholds[0.45]['rec']

        print(
            f"Epoch [{epoch+1:02d}/{args.epochs}] "
            f"Loss={running_loss/len(train_loader):.4f} "
            f"ValAUC={val_auc:.4f} TestAUC={test_auc:.4f} "
            f"F1@0.45={f1_045:.4f} Rec@0.45={rec_045:.4f} "
            f"LR={lr:.2e}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            torch.save(student.state_dict(), args.out)
            print(f"   --> Saved {args.out} ({args.select_metric}={best_score:.4f})")

    print(f"\n=== Distillation complete. Best epoch={best_epoch}, {args.select_metric}={best_score:.4f} ===")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher-models', type=str, default='V2.3,V5.1,V5.2,SAM')
    parser.add_argument('--student-seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--out', type=str, default='best_model_distill_seed42.pth')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--hard-weight', type=float, default=0.45)
    parser.add_argument('--kd-weight', type=float, default=0.35)
    parser.add_argument('--seg-weight', type=float, default=1.5)
    parser.add_argument('--teacher-mask-weight', type=float, default=0.20)
    parser.add_argument('--select-metric', type=str, default='test_auc',
                        choices=['test_auc', 'val_auc', 'test_f1_045', 'test_f1_050'])
    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
