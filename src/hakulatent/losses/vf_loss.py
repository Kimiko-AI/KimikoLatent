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
    def __init__(self, df=1024, m1= 0, m2=0, eps=1e-6,
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
        source="local" ,
        weights = '/content/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
)
        self.dinov2.eval()
        for p in self.dinov2.parameters():
            p.requires_grad_(False)
        self.proj = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, bias=False),
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
        feats = feats.view(feats.size(0), 16, 16, 1024)
        return feats, cls

    def forward(self, z, img):
        """
        z : (B, C_z, H, W) -> VAE latents
        img : (B, 3, H_img, W_img) -> raw images for DINOv2 feature extraction
        """
        B, C_z, H, W = z.shape

        # Get foundation features
        f, cls = self.get_dinov2_features(img)  # [B, df, Hf, Wf]
        _, C_f, Hf, Wf = f.shape

        # Project z to df channels
        z_proj = self.proj(z)  # (B, df, H, W)

        # Flatten spatial dims
        N = H * W
        z_flat = z_proj.view(B, self.df, N)
        f_flat = f.view(B, self.df, N)

        # Normalize
        z_norm = F.normalize(z_flat, dim=1, eps=self.eps)
        f_norm = F.normalize(f_flat, dim=1, eps=self.eps)
        z_norm_pooled = z_norm.mean(dim=2)

        cos_sim_cls = (z_norm_pooled * cls).sum(dim=1)  # (B, N)
        Lmcos_cls = F.relu(1.0- cos_sim_cls).mean()
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
