"""
V2.3 冠军版模型: ResNet34 + U-Net + SMP 内置分类头
"""
import torch.nn as nn
import segmentation_models_pytorch as smp


class BreastCancerMultiTaskNet(nn.Module):
    def __init__(self, encoder_name="resnet34"):
        super().__init__()
        aux_params = dict(
            pooling='avg',
            dropout=0.4,
            activation=None,
            classes=1,
        )
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            aux_params=aux_params
        )

    def forward(self, x, birads=None):
        mask_logits, class_logits = self.model(x)
        return mask_logits, class_logits.squeeze(-1)
