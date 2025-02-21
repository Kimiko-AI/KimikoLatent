import random
from math import sin, cos, radians

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


def crop(
    x: torch.Tensor,
    size: int | tuple[int, int],
    position: str,
) -> torch.Tensor:
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


def affine_transform(
    x: torch.Tensor,
    matrix: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "reflection",
):
    """
    Applies a 2x3 affine transform to x using grid_sample with reflection padding.
    x: (B, C, H, W)
    matrix: (B, 2, 3) or (2, 3)  # the affine transform
    """
    B, C, H, W = x.shape

    # If matrix is (2,3), expand to batch dimension
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0).expand(B, -1, -1)  # (B, 2, 3)

    # Create an affine grid
    # size = (B, C, H, W)
    grid = F.affine_grid(matrix, size=x.size(), align_corners=False).to(x.device)

    # Apply the transform
    x_out = F.grid_sample(
        x, grid, mode=mode, padding_mode=padding_mode, align_corners=False
    )
    return x_out


def build_affine_matrix(
    angle_deg: float = 0.0,
    translate: tuple[float, float] = (0.0, 0.0),
    scale: tuple[float, float] = (1.0, 1.0),
    shear_deg: tuple[float, float] = (0.0, 0.0),
    # The image/latent height & width to interpret translation in absolute or normalized coords
    height: int = 1,
    width: int = 1,
    translate_in_pixels: bool = False,
) -> torch.Tensor:
    """
    Builds a 2x3 affine transformation matrix combining rotation, scaling, shear, translation.
    - angle_deg: rotation angle in degrees
    - translate: (tx, ty) either in normalized range [-1..1] if not in pixels
    - scale: (sx, sy)
    - shear_deg: (shear_x, shear_y) in degrees
    - if translate_in_pixels == False, translate is in fraction of the image dimension
    """
    # Convert to radians
    angle = radians(angle_deg)
    shear_x = radians(shear_deg[0])
    shear_y = radians(shear_deg[1])

    # Rotation matrix R
    # [ cosθ, -sinθ ]
    # [ sinθ,  cosθ ]
    R = torch.tensor(
        [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]], dtype=torch.float32
    )

    # Scaling matrix S
    # [ sx,  0 ]
    # [ 0,  sy ]
    S = torch.tensor([[scale[0], 0.0], [0.0, scale[1]]], dtype=torch.float32)

    # Shear matrix SH (for x-then-y shear)
    # for shear_x (along x-axis), shear_y (along y-axis):
    # [ 1,  shear_x ]
    # [ shear_y, 1 ]
    SH = torch.tensor([[1.0, shear_x], [shear_y, 1.0]], dtype=torch.float32)

    # Compose these transformations: M = R * SH * S
    # (You could do a different order if desired.)
    # 2x2
    M_2x2 = R @ SH @ S

    # Translation
    # if not in pixels, interpret as fraction of width/height
    if not translate_in_pixels:
        tx = 2.0 * translate[0]  # scale to [-1..1] in normalized coords
        ty = 2.0 * translate[1]
    else:
        # from absolute pixels to normalized coords in [-1..1]
        tx = 2.0 * translate[0] / max(width - 1, 1)
        ty = 2.0 * translate[1] / max(height - 1, 1)

    # 2x3
    affine_mat = torch.tensor(
        [[M_2x2[0, 0], M_2x2[0, 1], tx], [M_2x2[1, 0], M_2x2[1, 1], ty]],
        dtype=torch.float32,
    )

    return affine_mat


def create_base_grid(B, H, W, device, dtype):
    """
    Creates a base sampling grid in normalized coords [-1..1].
    Returned shape: (B, H, W, 2).
    """
    ys = torch.linspace(-1, 1, steps=H, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, steps=W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
    base_grid = torch.empty(B, H, W, 2, device=device, dtype=dtype)
    base_grid[..., 0] = grid_x  # X
    base_grid[..., 1] = grid_y  # Y
    return base_grid


def gaussian_blur_2d(tensor, kernel_size=15, sigma=3.0):
    """
    Very naive Gaussian blur on (B, 2, H, W) or similar shape.
    You could use kornia.filters.gaussian_blur2d if installed.

    TODO: Implement this

    For demonstration, a simplified approach:
    """
    # One-liner if kornia is available:
    # import kornia
    # return kornia.filters.gaussian_blur2d(tensor, (kernel_size, kernel_size), (sigma, sigma))
    #
    # Otherwise, implement your own or skip smoothing:
    return tensor  # for demonstration, skip smoothing. Real code: implement your own or use Kornia.


def get_perspective_transform_4point(src_pts, dst_pts):
    """
    Minimal function that mimics torchvision's get_perspective_transform,
    or you can directly import:
    from torchvision.transforms.functional import _get_perspective_coeffs
    For clarity, we provide an example here.
    src_pts, dst_pts: (4,2) each in normalized [-1..1] coords
    returns 3x3 homography (torch.Tensor)
    """
    # We can solve a standard linear system.
    # For brevity, let's assume you can adapt from the TorchVision or Kornia sources:
    #    https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py
    # We'll do a quick stand-in:
    # We'll rely on the actual built-in if possible:
    import torchvision.transforms.functional as F_tv

    coeffs = F_tv._get_perspective_coeffs(src_pts.tolist(), dst_pts.tolist())
    # coeffs are [a, b, c, d, e, f, g, h]
    # Construct 3x3
    a, b, c, d, e, f, g, h = coeffs
    M = torch.tensor(
        [[a, b, c], [d, e, f], [g, h, 1]], device=src_pts.device, dtype=src_pts.dtype
    )
    return M


def warp_perspective_reflect(x, M):
    """
    Warps a batch of images x with perspective matrix M using reflection padding.
    x: (B, C, H, W)
    M: (B, 3, 3)
    """
    B, C, H, W = x.shape
    # create normalized meshgrid for each pixel
    # we can use kornia if installed (kornia.geometry.transform.warp_perspective)
    # or do it manually:
    device = x.device
    # build base grid
    # (H, W, 2)
    base_grid = torch.empty(B, H, W, 3, device=device, dtype=x.dtype)
    # fill with pixel coords in normalized range [-1,1]
    ys = torch.linspace(-1, 1, steps=H, device=device)
    xs = torch.linspace(-1, 1, steps=W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    # shape (H, W)
    base_grid[:, :, :, 0] = grid_x  # X
    base_grid[:, :, :, 1] = grid_y  # Y
    base_grid[:, :, :, 2] = 1  # homogeneous

    # now multiply by M^T for each batch item
    # out_grid = base_grid @ M^T
    # but we do it per-batch. We'll reshape
    base_grid_reshaped = base_grid.view(B, H * W, 3)  # (B, HW, 3)
    M_t = M.transpose(1, 2)  # (B, 3, 3)

    out_coords = base_grid_reshaped.bmm(M_t)  # (B, HW, 3)
    # Convert to 2D normalized
    # out_x = out_coords[:,:,0] / out_coords[:,:,2]
    # out_y = out_coords[:,:,1] / out_coords[:,:,2]
    denom = out_coords[..., 2:3] + 1e-8  # avoid div by zero
    out_xy = out_coords[..., 0:2] / denom

    # reshape to (B, H, W, 2)
    out_xy = out_xy.view(B, H, W, 2)

    # Now we can sample with grid_sample
    x_out = F.grid_sample(
        x, out_xy, mode="bilinear", padding_mode="reflection", align_corners=False
    )
    return x_out


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
        latent_size = (
            int(latent.shape[2] * scale_factors),
            int(latent.shape[3] * scale_factors),
        )
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


class RandomAffineTransform(LatentTransformBase):
    """
    Applies a random affine transform (rotation, scale, shear, translation)
    with reflection padding to avoid black borders.
    """

    def __init__(
        self,
        rotate_range=(-30, 30),  # in degrees
        scale_range=(0.8, 1.2),  # uniform scale factor range
        shear_range=(-10, 10),  # in degrees for x-shear & y-shear
        translate_range=(0.0, 0.0),  # fraction of image dimension
        method="random",
    ):
        """
        method: "random" (choose random each time) or "roundrobin" (cycle systematically).
        For simplicity, we'll illustrate "random".
        """
        self.rotate_range = rotate_range
        self.scale_range = scale_range
        if isinstance(shear_range[0], tuple):
            # if user gives ((-10,10),(-5,5)) for x-shear, y-shear
            self.shear_range_x = shear_range[0]
            self.shear_range_y = shear_range[1]
        else:
            # same range for both x-shear, y-shear
            self.shear_range_x = shear_range
            self.shear_range_y = shear_range

        self.translate_range = translate_range
        self.method = method

    def __call__(self, x: torch.Tensor, latent: torch.Tensor):
        B, C, H, W = x.shape

        if self.method == "random":
            angle_deg = random.uniform(self.rotate_range[0], self.rotate_range[1])
            scale_val = random.uniform(self.scale_range[0], self.scale_range[1])
            sx = random.uniform(self.shear_range_x[0], self.shear_range_x[1])
            sy = random.uniform(self.shear_range_y[0], self.shear_range_y[1])
            tx = random.uniform(-self.translate_range[0], self.translate_range[0])
            ty = random.uniform(-self.translate_range[1], self.translate_range[1])
        else:
            # you could implement "roundrobin" or a deterministic scheme here
            angle_deg, scale_val, sx, sy, tx, ty = 0, 1.0, 0, 0, 0, 0

        # Build 2x3 matrix for input image
        mat_x = build_affine_matrix(
            angle_deg=angle_deg,
            translate=(tx, ty),
            scale=(scale_val, scale_val),
            shear_deg=(sx, sy),
            height=H,
            width=W,
            translate_in_pixels=False,  # interpreting translate as fraction
        )
        # Build 2x3 matrix for latent (which might have different spatial dims)
        BH, BW = latent.shape[2], latent.shape[3]
        mat_latent = build_affine_matrix(
            angle_deg=angle_deg,
            translate=(tx, ty),
            scale=(scale_val, scale_val),
            shear_deg=(sx, sy),
            height=BH,
            width=BW,
            translate_in_pixels=False,
        )

        # Apply transforms with reflection
        x_out = affine_transform(x, mat_x, mode="bilinear", padding_mode="reflection")
        latent_out = affine_transform(
            latent, mat_latent, mode="bilinear", padding_mode="reflection"
        )

        return x_out, latent_out


class RandomPerspectiveTransform(LatentTransformBase):
    """
    Applies a random perspective (projective) transform, again using reflection
    to avoid black corners. Here we use torch's built-in 'grid_sample' after
    generating a random perspective transform in the form of a 3x3 homography.
    """

    def __init__(self, distortion_scale=0.5):
        """
        distortion_scale (0..1): how extreme the perspective change can be.
        """
        self.distortion_scale = distortion_scale

    def __call__(self, x: torch.Tensor, latent: torch.Tensor):
        # We create random displacement of each corner by up to distortion_scale of image dims
        B, C, H, W = x.shape
        device = x.device

        # For demonstration, let's assume a single transform for the whole batch
        # If you want per-sample transforms, you'd do it B times.
        # Randomly shift corners:
        half_h, half_w = int(H * self.distortion_scale), int(W * self.distortion_scale)

        # corners of the input in normalized coords [-1..1]
        # top-left, top-right, bottom-left, bottom-right
        base_coords = torch.tensor(
            [[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=torch.float32, device=device
        )

        # random offsets
        offsets = torch.empty_like(base_coords).uniform_(
            -self.distortion_scale, self.distortion_scale
        )
        # new corners
        new_coords = base_coords + offsets

        # Now we have original 4 corners -> new 4 corners in normalized coords
        # We can use kornia or we can solve for a 3x3 perspective matrix. For brevity,
        # let's do a quick function that solves for perspective using TorchVision's get_perspective_transform:
        M = get_perspective_transform_4point(base_coords, new_coords).unsqueeze(
            0
        )  # (1,3,3)
        # We'll do the same for latent (notice latent H/W might differ)
        BH, BW = latent.shape[2], latent.shape[3]
        base_coords_latent = torch.tensor(
            [[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=torch.float32, device=device
        )
        # We can reuse the same "offsets" to keep them consistent
        new_coords_latent = base_coords_latent + offsets
        M_latent = get_perspective_transform_4point(
            base_coords_latent, new_coords_latent
        ).unsqueeze(0)

        # random perspective matrix of shape (1,3,3)
        M = M.expand(B, -1, -1)  # (B,3,3)
        M_latent = M_latent.expand(B, -1, -1)  # (B,3,3)

        # but for perspective, we might want F.grid_sample with a 3x3 matrix -> we can use
        # a custom approach with "torch.nn.functional.grid_sample + custom perspective" or kornia's warp_perspective.
        # For demonstration, let's do the safe route with kornia if available.
        # If not, we'll do a simpler approximate approach:

        x_out = warp_perspective_reflect(
            x, M
        )  # We'll define warp_perspective_reflect below
        latent_out = warp_perspective_reflect(latent, M_latent)

        return x_out, latent_out


class RandomElasticDeformation(LatentTransformBase):
    """
    Example of an elastic (non-rigid) transform: we create a random displacement field,
    smooth it with a Gaussian, and warp the image via reflection padding.
    """

    def __init__(self, alpha=30.0, sigma=4.0):
        """
        alpha: scaling factor for displacement
        sigma: Gaussian kernel size for smoothing the displacement
        """
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, x: torch.Tensor, latent: torch.Tensor):
        B, C, H, W = x.shape

        # Make random displacement fields for x & y
        # shape (B, 2, H, W)
        disp = torch.randn(B, 2, H, W, device=x.device, dtype=x.dtype)

        # Optionally, blur (smooth) the displacement with a Gaussian
        # TODO fix this function
        # disp = gaussian_blur_2d(disp, kernel_size=int(4*self.sigma), sigma=self.sigma)

        disp = disp * self.alpha  # scale up displacements

        # We'll build a sampling grid from the base grid plus disp in normalized coords
        base_grid = create_base_grid(
            B, H, W, device=x.device, dtype=x.dtype
        )  # shape (B,H,W,2)
        # disp is in pixel coords => convert to normalized
        disp_norm_x = 2.0 * disp[:, 0] / max(W - 1, 1)
        disp_norm_y = 2.0 * disp[:, 1] / max(H - 1, 1)
        # shape (B,H,W)

        # Add to base grid
        out_grid = torch.empty_like(base_grid)
        out_grid[..., 0] = base_grid[..., 0] + disp_norm_x
        out_grid[..., 1] = base_grid[..., 1] + disp_norm_y

        # Warp x
        x_out = F.grid_sample(
            x, out_grid, mode="bilinear", padding_mode="reflection", align_corners=False
        )

        # We want the *same* displacement for latent (to remain equivariant)
        BH, BW = latent.shape[2], latent.shape[3]
        # If latent has different H, W, we re-generate or we rescale appropriately?
        # For simplicity, let's do a separate approach: re-generate disp for latent shape
        # But let's ensure the *same randomness* for consistent transform. We can fix a seed or pass it in.
        # Alternatively, you might want to scale the displacements to latent shape.
        # Here, let's do the same style. This requires the same random state or seed:

        disp_latent = torch.randn(
            B, 2, BH, BW, device=latent.device, dtype=latent.dtype
        )
        disp_latent = gaussian_blur_2d(
            disp_latent, kernel_size=int(4 * self.sigma), sigma=self.sigma
        )
        disp_latent = disp_latent * self.alpha

        base_grid_latent = create_base_grid(
            B, BH, BW, device=latent.device, dtype=latent.dtype
        )
        disp_norm_x_latent = 2.0 * disp_latent[:, 0] / max(BW - 1, 1)
        disp_norm_y_latent = 2.0 * disp_latent[:, 1] / max(BH - 1, 1)

        out_grid_latent = torch.empty_like(base_grid_latent)
        out_grid_latent[..., 0] = base_grid_latent[..., 0] + disp_norm_x_latent
        out_grid_latent[..., 1] = base_grid_latent[..., 1] + disp_norm_y_latent

        latent_out = F.grid_sample(
            latent,
            out_grid_latent,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=False,
        )

        return x_out, latent_out


if __name__ == "__main__":
    from .base import LatentTransformCompose

    x = torch.randn(1, 3, 512, 512)
    latent = torch.randn(1, 4, 64, 64)
    transform = LatentTransformCompose(
        RotationTransform(method="random"),
        ScaleDownTransform(method="random"),
        ScaleUpCropTransform(method="random"),
        RandomAffineTransform(method="random"),
        RandomPerspectiveTransform(),
        RandomElasticDeformation(),
    )
    for _ in range(10):
        x_trns, latent_trns = transform(x, latent)
        print(x_trns.shape, latent_trns.shape)
