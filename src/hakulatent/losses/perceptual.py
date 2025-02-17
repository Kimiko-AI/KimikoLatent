import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from torchvision import models
from convnext_perceptual_loss import ConvNextPerceptualLoss, ConvNextType


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_index=23):
        super(VGGFeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg16.features.children())[:layer_index])
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)


class PerceptualLoss(nn.Module):
    def __init__(self, layer_index=23, loss_type="mse"):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = VGGFeatureExtractor(layer_index=layer_index)
        self.loss_type = loss_type

    def forward(self, x_real, x_recon):
        feat_real = self.feature_extractor(x_real)
        feat_recon = self.feature_extractor(x_recon)

        if self.loss_type == "l1":
            loss = F.l1_loss(feat_recon, feat_real)
        else:
            loss = F.mse_loss(feat_recon, feat_real)

        return loss


class LPIPSLoss(nn.Module):
    def __init__(self, net="alex"):
        super(LPIPSLoss, self).__init__()
        self.lpips_model = lpips.LPIPS(net=net)
        self.lpips_model.eval().requires_grad_(False)

    def forward(self, x_real, x_recon):
        loss = self.lpips_model(x_real, x_recon)
        return loss.mean()


class ConvNeXtPerceptualLoss(nn.Module):
    def __init__(
        self,
        model_type=ConvNextType.BASE,
        feature_layers=[0, 2, 4, 6, 8, 10, 12, 14],
        feature_weights=None,
        use_gram=False,
        input_range=(0, 1),
        layer_weight_decay=0.99,
    ):
        super(ConvNeXtPerceptualLoss, self).__init__()
        self.model = ConvNextPerceptualLoss(
            model_type=model_type,
            feature_layers=feature_layers,
            feature_weights=feature_weights,
            use_gram=use_gram,
            input_range=input_range,
            layer_weight_decay=layer_weight_decay,
        )

    def forward(self, x_real, x_recon):
        loss = self.model(x_real, x_recon)
        return loss
