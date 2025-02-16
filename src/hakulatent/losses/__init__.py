import torch.nn as nn

from .adversarial import AdvLoss
from .perceptual import PerceptualLoss, LPIPSLoss


loss_table = {"mse": nn.MSELoss, "l1": nn.L1Loss, "huber": nn.HuberLoss}


class ReconLoss(nn.Module):
    def __init__(self, loss_type="mse", lpips_net="alex"):
        super(ReconLoss, self).__init__()
        self.loss = loss_table[loss_type]()
        self.lpips_loss = LPIPSLoss(lpips_net)

    def forward(self, x_real, x_recon):
        return self.loss(x_real, x_recon) + self.lpips_loss(x_real, x_recon)
