import random
import torch


class LatentTransformBase:
    def __call__(
        self, x: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class LatentTransformCompose(LatentTransformBase):
    def __init__(self, *transforms: LatentTransformBase):
        self.transforms = transforms

    def __call__(
        self, x: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for transform in self.transforms:
            x, latent = transform(x, latent)
        return x, latent


class LatentTransformSwitch(LatentTransformBase):
    def __init__(self, *transforms, method="random"):
        self.transforms = list(transforms)
        self.method = method

    def __call__(
        self, x: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.method == "random":
            transform = random.choice(self.transforms)
        elif self.method == "roundrobin":
            transform = self.transforms.pop(0)
            self.transforms.append(transform)
        return transform(x, latent)
