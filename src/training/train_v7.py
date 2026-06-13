"""
V7 系列训练脚本:
  v7.4: 简化多尺度 (4-scale BCE深度监督, 无LoMix组合)
  v7.6: SAM优化器

用法: python -m src.training.train_v7 --variant 4 [--sam]
"""
import sys, os
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

from src.models import BreastCancerMultiTaskNet
from src.models.v6 import BreastCancerMultiTaskNetV6
from src.data import get_case_split_train_val
from src.losses import FocalLoss


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization wrapper."""
    def __init__(self, base_optimizer, rho=0.05):
        defaults = dict(rho=rho)
        super().__init__(base_optimizer.param_groups, defaults)
        self.base_optimizer = base_optimizer
        self.rho = rho

    @torch.no_grad()
    def first_step(self):
        grad_norm = sum(
            (p.grad.norm(2) ** 2) for group in self.param_groups
            for p in group['params'] if p.grad is not None
        ).sqrt()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    self.state[p]['old_p'] = p.data.clone()
                    p.add_(p.grad, alpha=scale)

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.data = self.state[p]['old_p']
        self.base_optimizer.step()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def step(self):
        self.base_optimizer.step()


def train(variant, use_sam=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    variant_names = {4: 'V7.4 Simplified Multi-Scale (4-scale BCE)', 6: 'V7.6 SAM Optimizer'}
    name = variant_names.get(variant, f'V7.{variant}')
    sam_label = ' + SAM' if use_sam else ''
    print(f"=== {name}{sam_label} ===")

    data_dir = './data'
    train_dataset, val_dataset = get_case_split_train_val(data_dir=data_dir, val_ratio=0.2, seed=42)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    if variant == 4:
        model = BreastCancerMultiTaskNetV6().to(device)
        use_multiscale = True
    elif variant == 6:
        model = BreastCancerMultiTaskNet().to(device)
        use_multiscale = False
    else:
        raise ValueError(f"Unknown variant: {variant}")

    criterion_cls = FocalLoss(alpha=0.7, gamma=2)
    criterion_seg = nn.BCEWithLogitsLoss()

    base_optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer = SAM(base_optimizer, rho=0.05) if use_sam else base_optimizer

    num_epochs = 35
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        base_optimizer if use_sam else optimizer, T_max=num_epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    use_amp = scaler is not None and not use_sam  # SAM + AMP don't mix well

    best_auc = 0.0
    seg_weight = 1.5

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks, labels, birads, _ in train_loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)

            def compute_loss():
                if use_multiscale:
                    seg_logits_list, cls_logits = model(images)
                    loss_cls = criterion_cls(cls_logits, labels)
                    loss_seg = sum(criterion_seg(s, masks) for s in seg_logits_list) / len(seg_logits_list)
                else:
                    mask_logits, cls_logits = model(images)
                    loss_cls = criterion_cls(cls_logits, labels)
                    loss_seg = criterion_seg(mask_logits, masks)
                return loss_cls + seg_weight * loss_seg

            if use_sam:
                optimizer.zero_grad()
                loss = compute_loss()
                loss.backward()
                optimizer.first_step()

                optimizer.zero_grad()
                loss2 = compute_loss()
                loss2.backward()
                optimizer.second_step()
                train_loss += loss.item()
            elif use_amp:
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    loss = compute_loss()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
            else:
                optimizer.zero_grad()
                loss = compute_loss()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_lbl, val_prd = [], []
        with torch.no_grad():
            for images, masks, labels, birads, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                if use_multiscale:
                    _, cls_logits = model(images)
                else:
                    _, cls_logits = model(images)
                val_prd.extend(torch.sigmoid(cls_logits).cpu().numpy())
                val_lbl.extend(labels.cpu().numpy())
        val_auc = roc_auc_score(val_lbl, val_prd)
        lr = (base_optimizer if use_sam else optimizer).param_groups[0]['lr']
        print(f"Epoch [{epoch+1:02d}/{num_epochs}] | Loss: {train_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f} | LR: {lr:.2e}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f'best_model_v7.{variant}.pth')
            print(f"   --> Saved (AUC: {best_auc:.4f})")

    print(f"\n=== {name}{sam_label} Complete. Best Val AUC: {best_auc:.4f} ===")
    return best_auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=int, required=True, choices=[4, 6])
    parser.add_argument('--sam', action='store_true', help='Use SAM optimizer')
    args = parser.parse_args()
    train(args.variant, use_sam=args.sam)
