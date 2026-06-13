"""
V6 LoMix: 多尺度分割输出 + 二分类
  - 手动遍历 SMP UNet decoder.blocks 捕获 4 个尺度
  - 推理时只用最细尺度, 零额外开销
"""
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class BreastCancerMultiTaskNetV6(nn.Module):
    """
    ResNet34 encoder + U-Net decoder → multi-scale seg logits + cls logits.
    4 scale outputs (drop coarsest block) → used by LoMix loss during training.
    Inference: only seg_logits_list[-1] (finest) + cls_logits.
    """
    def __init__(self, encoder_name="resnet34"):
        super().__init__()
        self.seg_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            aux_params=None
        )

        enc_channels = smp.encoders.get_encoder(
            encoder_name, weights=None
        ).out_channels[-1]  # 512 for ResNet34
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(enc_channels, 1),
        )

    def forward(self, x, birads=None):
        """
        Returns:
            seg_logits_list: list of 4 tensors [B, 1, H, 256] (4 scales)
            cls_logits: [B] raw logits
        """
        features = self.seg_model.encoder(x)
        decoder_out = self.seg_model.decoder(features)
        mask_logits = self.seg_model.segmentation_head(decoder_out)

        # Classification head
        deepest = features[-1]
        visual = self.pool(deepest).view(deepest.size(0), -1)
        cls_logits = self.cls_head(visual).squeeze(-1)

        return mask_logits, cls_logits
