import torch
import torch.nn as nn
from convnext_perceptual_loss import ConvNextType

from .adversarial import AdvLoss
from .perceptual import PerceptualLoss, LPIPSLoss, ConvNeXtPerceptualLoss
from .vq_loss import KeplerQuantizerRegLoss


loss_table = {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "huber": nn.HuberLoss,
    "gnll": nn.GaussianNLLLoss,
}


class ReconLoss(nn.Module):
    def __init__(
        self,
        loss_type="mse",
        lpips_net="alex",
        convnext_type=None,
        convnext_kwargs={},
        loss_weights={},
    ):
        super(ReconLoss, self).__init__()
        self.loss = loss_table[loss_type]()
        self.loss_weight = loss_weights.get(loss_type, 1.0)
        if lpips_net is not None:
            self.lpips_loss = LPIPSLoss(lpips_net)
            self.lpips_weight = loss_weights.get("lpips", 1.0)
        else:
            self.lpips_loss = None
        if convnext_type is not None:
            self.convn_loss = ConvNeXtPerceptualLoss(
                model_type=convnext_type, **convnext_kwargs
            )
            self.convn_weight = loss_weights.get("convnext", 1.0)
        else:
            self.convn_loss = None

    def forward(self, x_real, x_recon):
        if isinstance(self.loss, nn.GaussianNLLLoss):
            x_recon, var = torch.split(
                x_recon, (x_real.size(1), x_recon.size(1) - x_real.size(1)), dim=1
            )
            # var = var.expand(-1, x_real.size(1), -1, -1)
            base = self.loss(x_recon, x_real, torch.abs(var) + 1) * self.loss_weight
        else:
            base = self.loss(x_recon, x_real) * self.loss_weight
        if self.lpips_loss is not None:
            lpips = self.lpips_loss(x_recon, x_real)
            base += lpips * self.lpips_weight
        if self.convn_loss is not None:
            convn = self.convn_loss(x_recon, x_real)
            base += convn * self.convn_weight
        return base
