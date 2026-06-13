"""
LoMix Loss Module — adapted from NeurIPS 2025 LoMix paper.
Extracted from lomix/trainer.py, adapted for binary segmentation:
  - CrossEntropyLoss → BCEWithLogitsLoss
  - Multi-class DiceLoss → Binary DiceLoss

Core idea: combinatorial multi-scale logits fusion with learnable weights.
Zero inference overhead — only the finest scale is used at test time.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations


# ============================================================
# Binary Dice Loss
# ============================================================
class BinaryDiceLoss(nn.Module):
    """Dice loss for binary segmentation (single-channel sigmoid output)."""
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersect = (probs * targets).sum(dim=1)
        union = (probs * probs).sum(dim=1) + (targets * targets).sum(dim=1)
        dice = (2.0 * intersect + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()


# ============================================================
# Weighted Fusion — per-pixel attention-weighted fusion
# ============================================================
class WeightedFusion(nn.Module):
    def __init__(self, num_stages, num_classes):
        super().__init__()
        self.weight_convs = nn.ModuleList(
            [nn.Conv2d(num_classes, 1, kernel_size=1) for _ in range(num_stages)]
        )

    def forward(self, predictions):
        weight_maps = [conv(pred) for pred, conv in zip(predictions, self.weight_convs)]
        weights = torch.stack(weight_maps, dim=1)
        weights = F.softmax(weights, dim=1)
        preds = torch.stack(predictions, dim=1)
        fused = torch.sum(weights * preds, dim=1)
        return fused


# ============================================================
# Combinatorial Mutations Loss Module
# ============================================================
class CombinatorialMutationsLossModule(nn.Module):
    """
    Free-floating weight approach:
      - Each raw param → positive weight via Softplus (0, ∞).
      - No sum constraint — weights are independent, avoiding coupling.
      - Useless combinations are automatically pushed to zero.

    Adapted for binary segmentation: BCEWithLogitsLoss + BinaryDiceLoss.
    """
    def __init__(
        self,
        original_num_maps,
        num_classes,
        selecetd_num_maps,
        operations=None,
        use_learnable_weights=True,
        supervision='mutation',
        lc1=0.3,
        lc2=0.7
    ):
        super().__init__()
        if operations is None:
            operations = ['add', 'mul', 'concat', 'weighted_fusion']

        self.num_maps = selecetd_num_maps
        self.num_classes = num_classes
        self.operations = operations
        self.use_learnable_weights = use_learnable_weights
        self.supervision = supervision
        self.lc1 = lc1
        self.lc2 = lc2

        if 'concat' in self.operations:
            self.concat_convs = nn.ModuleDict({
                str(k): nn.Conv2d(k * num_classes, num_classes, kernel_size=1)
                for k in range(2, selecetd_num_maps + 1)
            })

        if use_learnable_weights:
            self.original_weights = nn.ParameterList(
                [nn.Parameter(torch.zeros(1)) for _ in range(original_num_maps)]
            )

        self.combination_indices = {}
        if use_learnable_weights:
            self.synthesized_weights = nn.ModuleDict()

        for op in self.operations:
            comb_list = []
            for k in range(2, selecetd_num_maps + 1):
                comb_list.extend(list(combinations(range(selecetd_num_maps), k)))
            self.combination_indices[op] = comb_list

            if use_learnable_weights:
                self.synthesized_weights[op] = nn.ParameterList(
                    [nn.Parameter(torch.zeros(1)) for _ in range(len(comb_list))]
                )

        self.weighted_fusion_modules = nn.ModuleDict({
            str(k): WeightedFusion(k, num_classes)
            for k in range(2, selecetd_num_maps + 1)
        })

    def _compute_all_weights(self):
        if not self.use_learnable_weights:
            return None, None
        orig_vals = [F.softplus(w) for w in self.original_weights]
        synth_vals = {}
        for op, plist in self.synthesized_weights.items():
            synth_vals[op] = [F.softplus(p) for p in plist]
        return orig_vals, synth_vals

    def forward(self, output_maps, label_batch=None, bce_loss=None, dice_loss=None):
        device = output_maps[0].device
        deep_supervision_loss = 0.0
        mutation_loss = 0.0
        fused_logits = []

        if not self.use_learnable_weights:
            orig_weights = [torch.ones(1, device=device) for _ in output_maps]
            syn_weights = {}
            for op, combos in self.combination_indices.items():
                syn_weights[op] = [torch.ones(1, device=device) for _ in combos]
        else:
            orig_vals, synth_vals = self._compute_all_weights()
            orig_weights = orig_vals
            syn_weights = synth_vals

        for i, fmap in enumerate(output_maps):
            w_i = orig_weights[i]
            if label_batch is None and bce_loss is None and dice_loss is None:
                continue
            loss_bce = bce_loss(fmap, label_batch)
            loss_d = dice_loss(fmap, label_batch)
            combined_loss = self.lc1 * loss_bce + self.lc2 * loss_d
            deep_supervision_loss += w_i * combined_loss

        for op in self.operations:
            combos = self.combination_indices[op]
            w_list = (syn_weights[op] if self.use_learnable_weights
                      else [torch.ones(1, device=device)] * len(combos))
            for idx, comb in enumerate(combos):
                if op == 'add':
                    mutated = sum(output_maps[c] for c in comb)
                elif op == 'avg':
                    mutated = sum(output_maps[c] for c in comb) / len(comb)
                elif op == 'sub':
                    mutated = output_maps[comb[0]] - sum(output_maps[c] for c in comb[1:])
                elif op == 'mul':
                    mutated = output_maps[comb[0]]
                    for c in comb[1:]:
                        mutated = mutated * output_maps[c]
                elif op == 'concat':
                    cat = torch.cat([output_maps[c] for c in comb], dim=1)
                    mutated = self.concat_convs[str(len(comb))](cat)
                elif op in ['weighted_fusion', 'wf']:
                    mod = self.weighted_fusion_modules[str(len(comb))]
                    mutated = mod([output_maps[c] for c in comb])
                elif op == 'max':
                    mutated = torch.stack([output_maps[c] for c in comb], dim=0).max(dim=0).values
                else:
                    raise ValueError(f"Unsupported op: {op}")

                fused_logits.append(mutated)
                if label_batch is None and bce_loss is None and dice_loss is None:
                    continue
                loss_bce = bce_loss(mutated, label_batch)
                loss_d = dice_loss(mutated, label_batch)
                combined_loss = self.lc1 * loss_bce + self.lc2 * loss_d
                mutation_loss += w_list[idx] * combined_loss

        if label_batch is None and bce_loss is None and dice_loss is None:
            return fused_logits

        if self.supervision in ['mutation', 'lomix']:
            final_loss = deep_supervision_loss + mutation_loss
        else:
            final_loss = deep_supervision_loss
        return final_loss, deep_supervision_loss, mutation_loss

    def print_weights(self):
        if not self.use_learnable_weights:
            print("No learnable weights. Using uniform weighting.")
            return
        orig_vals, synth_vals = self._compute_all_weights()
        raw_orig = [p.item() for p in self.original_weights]
        print(f"Original Weights (raw): {' '.join(f'{x:.4f}' for x in raw_orig)}")
        print(f"Original Weights (softplus): {' '.join(f'{v.item():.4f}' for v in orig_vals)}")
        print(f"   => sum(original) = {float(torch.stack(orig_vals).sum()):.4f}")
        for op in self.operations:
            if op not in self.synthesized_weights:
                continue
            rvals = [p.item() for p in self.synthesized_weights[op]]
            svals = synth_vals[op]
            print(f"Synthesized '{op}' (raw): {' '.join(f'{rv:.4f}' for rv in rvals)}")
            print(f"Synthesized '{op}' (softplus): {' '.join(f'{sv.item():.4f}' for sv in svals)}")
            print(f"   => sum({op}) = {float(torch.stack(svals).sum()):.4f}")

    def save_weights(self, save_path):
        torch.save(self.state_dict(), save_path)
        print(f"Saved LoMix weights to {save_path}")

    def load_weights(self, load_path):
        self.load_state_dict(torch.load(load_path))
        print(f"Loaded LoMix weights from {load_path}")
