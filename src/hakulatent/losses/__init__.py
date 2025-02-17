import torch.nn as nn
from convnext_perceptual_loss import ConvNextType

from .adversarial import AdvLoss
from .perceptual import PerceptualLoss, LPIPSLoss, ConvNeXtPerceptualLoss


loss_table = {"mse": nn.MSELoss, "l1": nn.L1Loss, "huber": nn.HuberLoss}


class ReconLoss(nn.Module):
    def __init__(
        self, loss_type="mse", lpips_net="alex", convnext_type=None, convnext_kwargs={}
    ):
        super(ReconLoss, self).__init__()
        self.loss = loss_table[loss_type]()
        if lpips_net is not None:
            self.lpips_loss = LPIPSLoss(lpips_net)
        else:
            self.lpips_loss = None
        if convnext_type is not None:
            self.convn_loss = ConvNeXtPerceptualLoss(
                model_type=convnext_type, **convnext_kwargs
            )
        else:
            self.convn_loss = None

    def forward(self, x_real, x_recon):
        base = self.loss(x_real, x_recon)
        if self.lpips_loss is not None:
            lpips = self.lpips_loss(x_real, x_recon)
            base += lpips
        if self.convn_loss is not None:
            convn = self.convn_loss(x_real, x_recon)
            base += convn
        return base
