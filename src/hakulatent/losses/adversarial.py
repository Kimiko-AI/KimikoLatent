import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.discriminator import R3GANDiscriminator


def hinge_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(F.softplus(-logits_real)) + torch.mean(F.softplus(logits_fake))
    )
    return d_loss


class AdvLoss(nn.Module):
    def __init__(self, start_iter: int, disc_loss: str = "hinge", **kwargs):
        super().__init__()
        self.start_iter = start_iter
        if "n_layers" not in kwargs:
            kwargs["n_layers"] = 5
        self.discriminator = R3GANDiscriminator(**kwargs)
        self.d_loss = hinge_loss if disc_loss == "hinge" else vanilla_loss

    def calc_adaptive_weight(self, rec_loss, g_loss, last_layer: nn.Module):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                rec_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def forward(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        rec_loss: torch.Tensor | None,
        opt_idx: int,
        last_layer: nn.Module,
    ):
        if rec_loss is None:
            rec_loss = torch.abs(real.contiguous() - fake.contiguous())

        if opt_idx == 0:
            logits_fake = self.discriminator(fake)
            g_loss = -torch.mean(logits_fake)
            d_weight = self.calc_adaptive_weight(rec_loss, g_loss, last_layer)
            final_g_loss = d_weight * g_loss
            return final_g_loss
        else:
            logits_real = self.discriminator(real)
            logits_fake = self.discriminator(fake)
            d_loss = self.d_loss(logits_real, logits_fake)
            return d_loss
