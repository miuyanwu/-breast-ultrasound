"""
V5 异质架构模型定义
  V5Model:      U-Net + BI-RADS 辅助预测头 (V5.1)
  V5ModelUNetPP: U-Net++ 无 BI-RADS (V5.2)
"""
import torch.nn as nn
import segmentation_models_pytorch as smp


class V5Model(nn.Module):
    """V5.1: U-Net + BI-RADS 辅助预测头"""
    def __init__(self, encoder_name="resnet34", num_birads=4):
        super().__init__()
        self.seg_model = smp.Unet(
            encoder_name=encoder_name, encoder_weights="imagenet",
            in_channels=3, classes=1, aux_params=None
        )
        enc_channels = smp.encoders.get_encoder(
            encoder_name, weights=None
        ).out_channels[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(enc_channels, 1)
        )
        self.birads_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(enc_channels, num_birads)
        )

    def forward(self, x):
        features = self.seg_model.encoder(x)
        decoder_out = self.seg_model.decoder(features)
        mask_logits = self.seg_model.segmentation_head(decoder_out)
        deepest = features[-1]
        visual = self.pool(deepest).view(deepest.size(0), -1)
        cls_logits = self.cls_head(visual).squeeze(-1)
        birads_logits = self.birads_head(visual)
        return mask_logits, cls_logits, birads_logits


class V5ModelUNetPP(nn.Module):
    """V5.2: U-Net++ 无 BI-RADS"""
    def __init__(self, encoder_name="resnet34"):
        super().__init__()
        self.seg_model = smp.UnetPlusPlus(
            encoder_name=encoder_name, encoder_weights="imagenet",
            in_channels=3, classes=1, aux_params=None
        )
        enc_channels = smp.encoders.get_encoder(
            encoder_name, weights=None
        ).out_channels[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(enc_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        features = self.seg_model.encoder(x)
        decoder_out = self.seg_model.decoder(features)
        mask_logits = self.seg_model.segmentation_head(decoder_out)
        deepest = features[-1]
        visual = self.pool(deepest).view(deepest.size(0), -1)
        cls_logits = self.cls_head(visual).squeeze(-1)
        return mask_logits, cls_logits, None
