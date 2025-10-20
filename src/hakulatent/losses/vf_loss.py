import torch
import torch.nn as nn
import torch.nn.functional as F
import timm





class VFLoss(nn.Module):
    def __init__(self, df=512, m1=0, m2=0, eps=1e-6,
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
        self.dinov2 = timm.create_model(
    'vit_large_patch16_dinov3_qkvb.lvd1689m',
        pretrained=True,
        num_classes=0,
)
        self.dinov2.eval()
        for p in self.dinov2.parameters():
            p.requires_grad_(False)

        self.proj =nn.Conv2d(32, 1024, kernel_size=7, bias=True, padding=3)


    @torch.no_grad()
    def get_dinov2_features(self, x):
        """
        Extract final spatial patch features from DINOv2.
        Returns [B, df, H_f, W_f].
        """
        feats = self.dinov2.forward_features(x)
        feats, cls = feats[0], feats[5:]
        feats = feats.reshape(feats.size(0), 16, 16, 1024).permute(0, 3, 1, 2)
        return feats, cls

    def forward(self, z, img):
        """
        z : (B, C_z, H, W) -> VAE latents
        img : (B, 3, H_img, W_img) -> raw images for DINOv2 feature extraction
        """
        B, C_z, H, W = z.shape

        f_patches, f_cls = self.get_dinov2_features(img)

        z_proj_patches = self.proj(z)

        f_patches_flat = f_patches.flatten(2)
        z_proj_patches_flat = z_proj_patches.flatten(2)

        f_patches_norm = F.normalize(f_patches_flat, p=2, dim=1, eps=self.eps)
        z_proj_patches_norm = F.normalize(z_proj_patches_flat, p=2, dim=1, eps=self.eps)

        patch_projection_loss = 1 - (f_patches_norm * z_proj_patches_norm).sum(dim=1).mean()


        z_proj_pooled = torch.mean(z_proj_patches, dim=[2, 3])  # [B, df]

        # Normalize both the ground truth CLS token and our pooled vector
        f_cls_norm = F.normalize(f_cls, p=2, dim=1, eps=self.eps)
        z_proj_pooled_norm = F.normalize(z_proj_pooled, p=2, dim=1, eps=self.eps)

        cls_projection_loss = 1 - (f_cls_norm * z_proj_pooled_norm).sum(dim=1).mean()

        # --- 5. Combine Losses ---
        # A simple sum is a standard way to combine them. You could also weight them.
        total_loss = patch_projection_loss + cls_projection_loss

        return total_loss