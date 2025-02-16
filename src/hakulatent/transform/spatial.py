import torch
import torch.nn.functional as F

from .base import LatentTransformBase


def rotate(x: torch.Tensor, angle: int) -> torch.Tensor:
    # x: (B, C, H, W)
    # return: (B, C, H, W) or (B, C, W, H)
    if angle == 0:
        return x
    elif angle == 1:
        return x.flip(2).transpose(2, 3)
    elif angle == 2:
        return x.flip(2).flip(3)
    elif angle == 3:
        return x.transpose(2, 3).flip(2)


def crop(x: torch.Tensor, size: int | tuple[int, int], position: str) -> torch.Tensor:
    # x: (B, C, H, W)
    _, _, h, w = x.shape
    if isinstance(size, int):
        size = (size, size)
    size_h, size_w = size
    if position == "ul":
        return x[:, :, :size_h, :size_w]
    elif position == "ur":
        return x[:, :, :size_h, w - size_w :]
    elif position == "dl":
        return x[:, :, h - size_h :, :size_w]
    elif position == "dr":
        return x[:, :, h - size_h :, w - size_w :]
    elif position == "center":
        return x[
            :,
            :,
            (h - size_h) // 2 : (h + size_h) // 2,
            (w - size_w) // 2 : (w + size_w) // 2,
        ]


class RotationTransform(LatentTransformBase):
    def __init__(self, method: str = "random"):
        # method: random, roundrobin
        # 0: 0 degrees, 1: 90 degrees, 2: 180 degrees, 3: 270 degrees
        self.method = method
        self.angles = [0, 1, 2, 3]

    def __call__(
        self, x: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_angle = self.angles[0]
        if self.method == "random":
            current_angle = torch.randint(0, 4, (1,)).item()
        elif self.method == "roundrobin":
            self.angles = self.angles[1:] + self.angles[:1]
            current_angle = self.angles[0]

        x = rotate(x, current_angle)
        latent = rotate(latent, current_angle)
        return x, latent


class ScaleDownTransform(LatentTransformBase):
    def __init__(
        self, scale_factors=[0.25, 0.5, 0.75], method: str = "random", mode="bilinear"
    ):
        self.scale_factors = scale_factors
        self.method = method
        self.mode = mode

    def __call__(
        self, x: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_scale = self.scale_factors[0]
        if self.method == "random":
            current_scale = self.scale_factors[
                torch.randint(0, len(self.scale_factors), (1,)).item()
            ]
        elif self.method == "roundrobin":
            self.scale_factors = self.scale_factors[1:] + self.scale_factors[:1]
            current_scale = self.scale_factors[0]

        x = F.interpolate(x, scale_factor=current_scale, mode=self.mode)
        latent = F.interpolate(latent, scale_factor=current_scale, mode=self.mode)
        return x, latent


class CropTransform(LatentTransformBase):
    def __init__(self, scale_factors=[0.25, 0.5, 0.75], method="random"):
        self.scale_factors = scale_factors
        self.positions = ["ul", "ur", "dl", "dr", "center"]
        self.method = method

    def __call__(
        self, x: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.method == "random":
            scale_factors = self.scale_factors[
                torch.randint(0, len(self.scale_factors), (1,)).item()
            ]
            position = self.positions[
                torch.randint(0, len(self.positions), (1,)).item()
            ]
        elif self.method == "roundrobin":
            self.scale_factors = self.scale_factors[1:] + self.scale_factors[:1]
            self.position = self.positions[1:] + self.positions[:1]
            scale_factors = self.scale_factors[0]
            position = self.positions[0]
        size = (int(x.shape[2] * scale_factors), int(x.shape[3] * scale_factors))
        latent_size = (int(latent.shape[2] * scale_factors), int(latent.shape[3] * scale_factors))
        x = crop(x, size, position)
        latent = crop(latent, latent_size, position)
        return x, latent


class ScaleUpCropTransform(LatentTransformBase):
    def __init__(
        self, scale_factors=[1.25, 1.5, 2], method: str = "random", mode="bicubic"
    ):
        self.scale_factors = scale_factors
        self.method = method
        self.mode = mode
        self.crop_position = ["ul", "ur", "dl", "dr", "center"]

    def __call__(
        self, x: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_scale = self.scale_factors[0]
        current_crop = self.crop_position[0]
        if self.method == "random":
            current_scale = self.scale_factors[
                torch.randint(0, len(self.scale_factors), (1,)).item()
            ]
            current_crop = self.crop_position[
                torch.randint(0, len(self.crop_position), (1,)).item()
            ]
        elif self.method == "roundrobin":
            self.scale_factors = self.scale_factors[1:] + self.scale_factors[:1]
            current_scale = self.scale_factors[0]
            self.crop_position = self.crop_position[1:] + self.crop_position[:1]
            current_crop = self.crop_position[0]

        org_x_shape = x.shape[2:]
        org_latent_shape = latent.shape[2:]
        x = F.interpolate(x, scale_factor=current_scale, mode=self.mode)
        latent = F.interpolate(latent, scale_factor=current_scale, mode=self.mode)

        x = crop(x, org_x_shape, current_crop)
        latent = crop(latent, org_latent_shape, current_crop)
        return x, latent


if __name__ == "__main__":
    from .base import LatentTransformCompose

    x = torch.randn(1, 3, 256, 256)
    latent = torch.randn(1, 4, 32, 32)
    transform = LatentTransformCompose(
        RotationTransform(method="random"),
        ScaleDownTransform(method="random"),
        ScaleUpCropTransform(method="random"),
    )
    for _ in range(10):
        x_trns, latent_trns = transform(x, latent)
        print(x_trns.shape, latent_trns.shape)
