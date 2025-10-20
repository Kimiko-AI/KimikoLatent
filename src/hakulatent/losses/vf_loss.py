import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------------
# SAME-PADDING CONVOLUTION
# -------------------------------------------------------------------------
class Conv2dSame(nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(ih, self.kernel_size[0], self.stride[0], self.dilation[0])
        pad_w = self.calc_same_pad(iw, self.kernel_size[1], self.stride[1], self.dilation[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2,
                          pad_h // 2, pad_h - pad_h // 2])
        return super().forward(x)


# -------------------------------------------------------------------------
# RESNET BLOCK
# -------------------------------------------------------------------------
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout_prob=0.0):
        super().__init__()
        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.conv1 = Conv2dSame(in_channels, out_channels, kernel_size=3, bias=False)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.conv2 = Conv2dSame(out_channels, out_channels, kernel_size=3, bias=False)

        self.has_shortcut = in_channels != out_channels
        if self.has_shortcut:
            self.nin_shortcut = Conv2dSame(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.conv2(x)

        if self.has_shortcut:
            residual = self.nin_shortcut(residual)

        return x + residual


# -------------------------------------------------------------------------
# VECTOR QUANTIZER (No Entropy Regularization)
# -------------------------------------------------------------------------
class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck from VQ-VAE / MaskGIT-VQGAN.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, hidden_states, return_loss=True):
        # hidden_states: (B, C, H, W)
        x = hidden_states.permute(0, 2, 3, 1).contiguous()  # (B,H,W,C)
        flat = x.view(-1, self.embedding_dim)

        # compute distances
        distances = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )

        # find nearest embeddings
        indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(indices, self.num_embeddings).type(flat.dtype)

        # quantize
        quantized = encodings @ self.embedding.weight
        quantized = quantized.view_as(x)

        if return_loss:
            e_loss = F.mse_loss(quantized.detach(), x)
            q_loss = F.mse_loss(quantized, x.detach())
            vq_loss = q_loss + self.commitment_cost * e_loss
        else:
            vq_loss = None

        # straight-through estimator
        quantized = x + (quantized - x).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        indices = indices.view(hidden_states.shape[0], -1)
        return quantized, indices, vq_loss


# -------------------------------------------------------------------------
# ENCODER
# -------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=128, z_channels=128, dropout=0.0):
        super().__init__()

        self.conv_in = Conv2dSame(in_channels, hidden_channels // 2, kernel_size=3, bias=False)

        self.block1 = nn.Sequential(
            ResnetBlock(hidden_channels // 2, hidden_channels // 2, dropout),
            ResnetBlock(hidden_channels // 2, hidden_channels // 2, dropout),
        )
        self.down1 = nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=4, stride=2, padding=1)

        self.block2 = nn.Sequential(
            ResnetBlock(hidden_channels, hidden_channels, dropout),
            ResnetBlock(hidden_channels, hidden_channels, dropout),
        )
        self.norm_out = nn.GroupNorm(32, hidden_channels, eps=1e-6, affine=True)
        self.conv_out = Conv2dSame(hidden_channels, z_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


# -------------------------------------------------------------------------
# DECODER
# -------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, out_channels=4, hidden_channels=128, z_channels=128, dropout=0.0):
        super().__init__()

        self.conv_in = Conv2dSame(z_channels, hidden_channels, kernel_size=3)

        self.block1 = nn.Sequential(
            ResnetBlock(hidden_channels, hidden_channels, dropout),
            ResnetBlock(hidden_channels, hidden_channels, dropout),
        )

        self.up1 = nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1)

        self.block2 = nn.Sequential(
            ResnetBlock(hidden_channels // 2, hidden_channels // 2, dropout),
            ResnetBlock(hidden_channels // 2, hidden_channels // 2, dropout),
        )

        self.norm_out = nn.GroupNorm(32, hidden_channels // 2, eps=1e-6, affine=True)
        self.conv_out = Conv2dSame(hidden_channels // 2, out_channels, kernel_size=3)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.block1(x)
        x = self.up1(x)
        x = self.block2(x)
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


# -------------------------------------------------------------------------
# FULL MODEL
# -------------------------------------------------------------------------
class VQVAE(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=128, z_channels=128,
                 num_embeddings=4096, commitment_cost=0.25):
        super().__init__()

        self.encoder = Encoder(in_channels, hidden_channels, z_channels)
        self.quantizer = VectorQuantizer(num_embeddings, z_channels, commitment_cost)
        self.decoder = Decoder(in_channels, hidden_channels, z_channels)

    def forward(self, x):
        z = self.encoder(x)
        z_q, _, vq_loss = self.quantizer(z, return_loss=True)
        x_recon = self.decoder(z_q)
        recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + 0.05 * vq_loss
        return x_recon, total_loss, recon_loss, vq_loss
