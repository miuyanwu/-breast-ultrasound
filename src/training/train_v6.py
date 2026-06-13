"""
V6 LoMix: 多尺度分割损失 + FocalLoss 分类训练
  - LoMix CombinatorialMutationsLossModule 自学习多尺度融合权重
  - 优化器同时包含 model params 和 loss_module params

用法: python -m src.training.train_v6
"""
import sys, os
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.models import BreastCancerMultiTaskNet
from src.data import BreastUltrasoundDataset, get_case_split_train_val
from src.losses import FocalLoss, CombinatorialMutationsLossModule, BinaryDiceLoss


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 50)
    print("V6 LoMix: Multi-scale Logits Fusion Training")
    print("=" * 50)

    data_dir = './data'
    train_dataset, val_dataset = get_case_split_train_val(
        data_dir=data_dir, val_ratio=0.2, seed=42
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = BreastCancerMultiTaskNet(encoder_name="resnet34").to(device)

    criterion_cls = FocalLoss(alpha=0.7, gamma=2)
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = BinaryDiceLoss()

    loss_module = CombinatorialMutationsLossModule(
        original_num_maps=4,
        num_classes=1,
        selecetd_num_maps=4,
        operations=['add', 'mul', 'concat', 'weighted_fusion'],
        use_learnable_weights=True,
        supervision='mutation',
        lc1=0.3,
        lc2=0.7
    ).to(device)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(loss_module.parameters()),
        lr=1e-4, weight_decay=1e-4
    )

    num_epochs = 35
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    seg_weight = 1.5
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    use_amp = scaler is not None

    best_auc = 0.0

    for epoch in range(num_epochs):
        model.train()
        loss_module.train()
        train_loss = 0.0

        for images, masks, labels, birads, filenames in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            birads = birads.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    seg_logits_list, cls_logits = model(images)
                    loss_cls = criterion_cls(cls_logits, labels)
                    loss_seg, deep_loss, mut_loss = loss_module(
                        seg_logits_list, masks, bce_loss, dice_loss
                    )
                    loss = loss_cls + seg_weight * loss_seg

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                seg_logits_list, cls_logits = model(images)
                loss_cls = criterion_cls(cls_logits, labels)
                loss_seg, deep_loss, mut_loss = loss_module(
                    seg_logits_list, masks, bce_loss, dice_loss
                )
                loss = loss_cls + seg_weight * loss_seg
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_labels_all = []
        val_preds_all = []

        with torch.no_grad():
            for images, masks, labels, birads, filenames in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                seg_logits_list, cls_logits = model(images)
                probs = torch.sigmoid(cls_logits)
                val_labels_all.extend(labels.cpu().numpy())
                val_preds_all.extend(probs.cpu().numpy())

        val_auc = roc_auc_score(val_labels_all, val_preds_all)
        avg_train_loss = train_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch + 1:02d}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | "
              f"LR: {current_lr:.2e}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'best_model_v6.pth')
            print(f"   --> Saved best (AUC: {best_auc:.4f})")

        if (epoch + 1) % 5 == 0:
            print("  [LoMix Weights]")
            loss_module.print_weights()

    print("\n=== Final LoMix Weights ===")
    loss_module.print_weights()
    loss_module.save_weights('lomix_weights_final.pth')

    print(f"\n{'='*50}")
    print(f"V6 LoMix Complete. Best Val AUC: {best_auc:.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    train()
