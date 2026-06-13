"""
单模型测试脚本 — TTA 水平翻转 + 多阈值评估

用法: python -m src.evaluation.test_single
"""
import sys, os
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

from src.data import BreastUltrasoundDataset
from src.models import BreastCancerMultiTaskNet


def evaluate_test_set():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = './data'

    output_mask_dir = './outputs/测试集预测Masks_v6'
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(os.path.join(output_mask_dir, 'benign'), exist_ok=True)
    os.makedirs(os.path.join(output_mask_dir, 'malignant'), exist_ok=True)

    print("Loading test dataset...")
    test_dataset = BreastUltrasoundDataset(data_dir=data_dir, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    print("Loading V6 LoMix best model weights...")
    model = BreastCancerMultiTaskNet(encoder_name="resnet34").to(device)
    model.load_state_dict(torch.load('best_model_v6.pth', map_location=device, weights_only=True))
    model.eval()

    y_true = []
    y_scores = []

    print("Running inference with TTA (Horizontal Flip)...")
    with torch.no_grad():
        for images, masks, labels, birads, filenames in test_loader:
            images = images.to(device)

            seg_logits_list, cls_logits = model(images)
            probs_orig = torch.sigmoid(cls_logits)
            pred_masks_orig = torch.sigmoid(seg_logits_list[-1])

            images_flipped = torch.flip(images, dims=[3])
            seg_list_flip, cls_logits_flip = model(images_flipped)

            probs_flip = torch.sigmoid(cls_logits_flip)
            pred_masks_flip_raw = torch.sigmoid(seg_list_flip[-1])
            pred_masks_flip_inv = torch.flip(pred_masks_flip_raw, dims=[3])

            final_probs = (probs_orig + probs_flip) / 2.0
            final_pred_masks = (pred_masks_orig + pred_masks_flip_inv) / 2.0

            y_true.extend(labels.numpy())
            y_scores.extend(final_probs.cpu().numpy())

            final_pred_masks_bin = (final_pred_masks > 0.5).float().cpu().numpy()
            for i in range(len(filenames)):
                file_name = filenames[i]
                label_val = labels[i].item()
                class_folder = 'malignant' if label_val == 1.0 else 'benign'

                single_mask = final_pred_masks_bin[i][0]
                mask_img_array = (single_mask * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_img_array, mode='L')

                save_name = file_name.replace('.png', '_pred.png')
                save_path = os.path.join(output_mask_dir, class_folder, save_name)
                mask_pil.save(save_path)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    print(f"\n{'='*60}")
    print("        赛题技术指标评测报告         ")
    print(f"{'='*60}")
    for thresh in [0.40, 0.45, 0.50, 0.55]:
        preds = (y_scores > thresh).astype(int)
        auc = roc_auc_score(y_true, y_scores)
        acc = accuracy_score(y_true, preds)
        rec = recall_score(y_true, preds)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = precision_score(y_true, preds)
        f1 = f1_score(y_true, preds)
        print(f"  thresh={thresh:.2f}: AUC={auc:.4f} Acc={acc:.4f} "
              f"Rec={rec:.4f} Spec={spec:.4f} Prec={prec:.4f} F1={f1:.4f}")

    print(f"{'='*60}")
    print(f"[*] 测试集分割掩膜已保存至: {output_mask_dir}")


if __name__ == '__main__':
    evaluate_test_set()
