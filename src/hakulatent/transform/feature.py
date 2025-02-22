import random
from math import sin, cos, radians

from numpy import isin
import torch
import torch.nn.functional as F

from .base import LatentTransformBase


class BlendingTransform(LatentTransformBase):
    def __init__(self, alpha: float = 0.5, method: str = "random"):
        self.alpha = alpha
        self.method = method
        self.offset = 0

    def __call__(
        self, x: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.method == "random":
            order = list(range(x.size(0)))
            random.shuffle(order)
        elif self.method == "roundrobin":
            order = list(range(x.size(0)))
            offset = self.offset % x.size(0)
            order = order[offset:] + order[:offset]
            self.offset = (self.offset + 1) % x.size(0)

        if isinstance(self.alpha, list):
            alpha = random.random() * (self.alpha[1] - self.alpha[0]) + self.alpha[0]
        else:
            alpha = self.alpha

        x = x * (1 - alpha) + x[order] * alpha
        latent = latent * (1 - alpha) + latent[order] * alpha
        return x, latent


if __name__ == "__main__":
    from .base import LatentTransformCompose

    x = torch.randn(1, 3, 512, 512)
    latent = torch.randn(1, 4, 64, 64)
    transform = BlendingTransform
    for _ in range(10):
        x_trns, latent_trns = transform(x, latent)
        print(x_trns.shape, latent_trns.shape)
