import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class ProjResNet(nn.Module):
    def __init__(self, in_ch, dim, depth=4):
        super().__init__()
        layers = []
        # input projection to dim
        layers.append(nn.Conv2d(in_ch, dim, kernel_size=1, bias=False))
        # stack residual blocks
        for _ in range(depth):
            layers.append(ResBlock(dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VFLoss(nn.Module):
    def __init__(self, df=1024, m1=0, m2=0, eps=1e-6,
                 dinov2_name='vit_small_patch14_dinov2.lvd142m'):
        """
        df : foundation model feature dimension
        m1 : margin for marginal cosine similarity
        m2 : margin for marginal distance matrix similarity
        eps : numerical stability
        dinov2_name : model name from timm
        """
        super().__init__()
        self.df = df
        self.m1 = m1
        self.m2 = m2
        self.eps = eps
        self.proj = None  # lazy init projection (1x1 conv)

        # Load DINOv2 backbone
        self.dinov2 = model = torch.hub.load(
            repo_or_dir='dinov3',
            model='dinov3_vitl16',
            source="local",
            weights='/content/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
        )
        self.dinov2.eval()
        for p in self.dinov2.parameters():
            p.requires_grad_(False)
        self.proj = nn.Sequential(
            nn.Conv2d(32, 1024, kernel_size=1, bias=False),
            *[ResBlock(self.df) for _ in range(4)]
        )

    @torch.no_grad()
    def get_dinov2_features(self, x):
        """
        Extract final spatial patch features from DINOv2.
        Returns [B, df, H_f, W_f].
        """
        feats = self.dinov2.forward_features(x)
        feats, cls = feats["x_norm_patchtokens"], feats["x_norm_clstoken"]
        feats = feats.reshape(feats.size(0), 16, 16, 1024).permute(0, 3, 1, 2)
        return feats, cls

    def forward(self, z, img):
        """
        z : (B, C_z, H, W) -> VAE latents
        img : (B, 3, H_img, W_img) -> raw images for DINOv2 feature extraction
        """
        B, C_z, H, W = z.shape

        # --- 1. Get Ground Truth Features from DINOv2 ---
        # These are our target features, extracted from the real image.
        f_patches, f_cls = self.get_dinov2_features(img)

        # --- 2. Project VAE Latents into DINOv2 Feature Space ---
        # This is the "student" network's output that we want to align.
        z_proj_patches = self.proj(z)

        # --- 3. Calculate Projection Loss for Patch Tokens ---

        # Flatten spatial dimensions to [B, df, N] where N = H*W
        f_patches_flat = f_patches.flatten(2)
        z_proj_patches_flat = z_proj_patches.flatten(2)

        # Normalize along the feature dimension (dim=1)
        f_patches_norm = F.normalize(f_patches_flat, p=2, dim=1, eps=self.eps)
        z_proj_patches_norm = F.normalize(z_proj_patches_flat, p=2, dim=1, eps=self.eps)

        # Calculate negative cosine similarity.
        # The dot product of two unit vectors is their cosine similarity.
        # We want to maximize similarity, which means minimizing its negative.
        # The sum over dim=1 is the dot product. .mean() averages over batch and spatial locations.
        patch_projection_loss = 1 - (f_patches_norm * z_proj_patches_norm).sum(dim=1).mean()

        # --- 4. Calculate Projection Loss for CLS Token ---

        # To get a comparable vector from our projected latents, we use global average pooling.
        z_proj_pooled = torch.mean(z_proj_patches, dim=[2, 3])  # [B, df]

        # Normalize both the ground truth CLS token and our pooled vector
        f_cls_norm = F.normalize(f_cls, p=2, dim=1, eps=self.eps)
        z_proj_pooled_norm = F.normalize(z_proj_pooled, p=2, dim=1, eps=self.eps)

        # Calculate negative cosine similarity for the CLS token representation.
        cls_projection_loss = 1 - (f_cls_norm * z_proj_pooled_norm).sum(dim=1).mean()

        # --- 5. Combine Losses ---
        # A simple sum is a standard way to combine them. You could also weight them.
        total_loss = patch_projection_loss + cls_projection_loss

        return total_loss
