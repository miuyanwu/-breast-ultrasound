"""
V5 统一训练脚本 — SWA + 异质架构
  v5.1: V2.3 + BI-RADS辅助头 + SWA
  v5.2: V2.3 + U-Net++ + SWA

用法: python -m src.training.train_v5 --variant 1
"""
import sys, os
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse, copy
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.models.v5 import V5Model, V5ModelUNetPP
from src.data import get_case_split_train_val
from src.losses import FocalLoss


def create_model(variant):
    if variant == 1:
        return V5Model(), {'name': 'V5.1', 'desc': 'ResNet34 + U-Net + BI-RADS + SWA', 'birads_head': True}
    else:
        return V5ModelUNetPP(), {'name': 'V5.2', 'desc': 'ResNet34 + U-Net++ + SWA', 'birads_head': False}


# ============================================================
# SWA utilities
# ============================================================
@torch.no_grad()
def update_swa(swa_model, model, n_averaged):
    """Running average: swa = (swa * n + model) / (n + 1)"""
    for swa_p, p in zip(swa_model.parameters(), model.parameters()):
        swa_p.data.copy_((swa_p.data * n_averaged + p.data) / (n_averaged + 1))
    for swa_b, b in zip(swa_model.buffers(), model.buffers()):
        swa_b.data.copy_((swa_b.data * n_averaged + b.data) / (n_averaged + 1))


def update_bn(swa_model, model, train_loader, device):
    """Update SWA model BN statistics with a forward pass on training data."""
    swa_model.train()
    with torch.no_grad():
        for images, masks, labels, birads, _ in train_loader:
            images = images.to(device)
            _ = swa_model(images)


# ============================================================
# Training
# ============================================================
def train(variant):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = create_model(variant)
    model = model.to(device)

    swa_model = copy.deepcopy(model)
    swa_model.to(device)
    n_swa = 0

    print(f"=== {cfg['name']}: {cfg['desc']} ===")
    print(f"SWA start epoch: 28/35")

    data_dir = './data'
    train_dataset, val_dataset = get_case_split_train_val(data_dir=data_dir, val_ratio=0.2, seed=42)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    criterion_cls = FocalLoss(alpha=0.7, gamma=2)
    criterion_seg = nn.BCEWithLogitsLoss()
    criterion_birads = nn.CrossEntropyLoss() if cfg['birads_head'] else None

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    num_epochs = 35
    swa_start = 28
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    use_amp = scaler is not None
    best_auc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks, labels, birads, _ in train_loader:
            images, masks, labels, birads = images.to(device), masks.to(device), labels.to(device), birads.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    mask_logits, cls_logits, birads_logits = model(images)
                    loss = criterion_cls(cls_logits, labels) + 1.5 * criterion_seg(mask_logits, masks)
                    if birads_logits is not None:
                        loss = loss + 0.3 * criterion_birads(birads_logits, birads)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                mask_logits, cls_logits, birads_logits = model(images)
                loss = criterion_cls(cls_logits, labels) + 1.5 * criterion_seg(mask_logits, masks)
                if birads_logits is not None:
                    loss = loss + 0.3 * criterion_birads(birads_logits, birads)
                loss.backward()
                optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        if epoch + 1 >= swa_start:
            n_swa += 1
            update_swa(swa_model, model, n_swa)

        # Validation
        model.eval()
        val_lbl, val_prd = [], []
        with torch.no_grad():
            for images, masks, labels, birads, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                mask_logits, cls_logits, _ = model(images)
                val_prd.extend(torch.sigmoid(cls_logits).cpu().numpy())
                val_lbl.extend(labels.cpu().numpy())
        val_auc = roc_auc_score(val_lbl, val_prd)
        lr = optimizer.param_groups[0]['lr']
        swa_mark = ' [SWA]' if epoch + 1 >= swa_start else ''
        print(f"Epoch [{epoch+1:02d}/{num_epochs}] | Loss: {train_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f} | LR: {lr:.2e}{swa_mark}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f'best_model_v5.{variant}.pth')
            print(f"   --> Saved best (AUC: {best_auc:.4f})")

    # Update SWA BN and save
    print(f"\nUpdating SWA BatchNorm statistics...")
    update_bn(swa_model, model, train_loader, device)
    torch.save(swa_model.state_dict(), f'swa_model_v5.{variant}.pth')
    print(f"SWA model saved (averaged over last {n_swa} epochs)")

    # Evaluate SWA model
    swa_model.eval()
    val_lbl, val_prd = [], []
    with torch.no_grad():
        for images, masks, labels, birads, _ in val_loader:
            images, labels = images.to(device), labels.to(device)
            _, cls_logits, _ = swa_model(images)
            val_prd.extend(torch.sigmoid(cls_logits).cpu().numpy())
            val_lbl.extend(labels.cpu().numpy())
    swa_auc = roc_auc_score(val_lbl, val_prd)
    print(f"SWA Val AUC: {swa_auc:.4f}  |  Best Val AUC: {best_auc:.4f}")

    print(f"\n=== {cfg['name']} Complete. Best: {best_auc:.4f} | SWA: {swa_auc:.4f} ===")
    return best_auc, swa_auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=int, required=True, choices=[1, 2])
    args = parser.parse_args()
    train(args.variant)
