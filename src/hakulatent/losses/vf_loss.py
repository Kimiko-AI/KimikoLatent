import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class VFLoss(nn.Module):
    def __init__(self, df=384, m1=0.5, m2=0.25, eps=1e-6,
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
            dinov2_name, pretrained=True, num_classes=0, global_pool=''
        )
        self.dinov2.eval()
        for p in self.dinov2.parameters():
            p.requires_grad_(False)

    def _init_proj(self, dz):
        self.proj = nn.Conv2d(dz, self.df, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity='linear')

    @torch.no_grad()
    def get_dinov2_features(self, x):
        """
        Extract final spatial patch features from DINOv2.
        Returns [B, df, H_f, W_f].
        """
        feats = self.dinov2.get_intermediate_layers(x, n=1, reshape=True)[0]
        return feats

    def forward(self, z, img):
        """
        z : (B, C_z, H, W) -> VAE latents
        img : (B, 3, H_img, W_img) -> raw images for DINOv2 feature extraction
        """
        B, C_z, H, W = z.shape

        # Get foundation features
        f = self.get_dinov2_features(img)  # [B, df, Hf, Wf]
        _, C_f, Hf, Wf = f.shape

        # Create projection if needed
        if self.proj is None or self.proj.in_channels != C_z:
            self._init_proj(C_z)

        # Project z to df channels
        z_proj = self.proj(z)  # (B, df, H, W)

        # Match spatial resolution
        if (Hf, Wf) != (H, W):
            f = F.interpolate(f, size=(H, W), mode='bilinear', align_corners=False)

        assert f.shape[1] == self.df, "Foundation feature dim must match df"

        # Flatten spatial dims
        N = H * W
        z_flat = z_proj.view(B, self.df, N)
        f_flat = f.view(B, self.df, N)

        # Normalize
        z_norm = F.normalize(z_flat, dim=1, eps=self.eps)
        f_norm = F.normalize(f_flat, dim=1, eps=self.eps)

        # --- Lmcos ---
        cos_sim = (z_norm * f_norm).sum(dim=1)  # (B, N)
        Lmcos = F.relu(1.0 - self.m1 - cos_sim).mean()

        # --- Lmdms ---
        z_pos = z_norm.permute(0, 2, 1)  # (B, N, df)
        f_pos = f_norm.permute(0, 2, 1)  # (B, N, df)

        sim_z = torch.bmm(z_pos, z_pos.transpose(1, 2))  # (B, N, N)
        sim_f = torch.bmm(f_pos, f_pos.transpose(1, 2))  # (B, N, N)

        Lmdms = F.relu(torch.abs(sim_z - sim_f) - self.m2).mean()

        return (Lmcos + Lmdms)
