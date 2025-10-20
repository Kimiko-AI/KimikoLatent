import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        if not self.same_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity
        return self.relu(out)

# --------------------------------
# Vector Quantizer
# --------------------------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=4096, embedding_dim=128, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, inputs):
        # inputs: (B, C, H, W)
        flatten = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = flatten.view(-1, self.embedding_dim)

        # Distances to embeddings
        distances = (
            flat_input.pow(2).sum(1, keepdim=True)
            - 2 * flat_input @ self.embeddings.weight.t()
            + self.embeddings.weight.pow(2).sum(1)
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = encodings @ self.embeddings.weight
        quantized = quantized.view(flatten.shape)

        # VQ losses
        e_latent_loss = F.mse_loss(quantized.detach(), flatten)
        q_latent_loss = F.mse_loss(quantized, flatten.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = flatten + (quantized - flatten).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, loss

# --------------------------------
# Encoder: Down → Res → Down → Res
# --------------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=128):
        super().__init__()
        self.stage1_down = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 4, 2, 1),  # /2
            nn.ReLU(),
        )
        self.stage1_res = nn.Sequential(
            ResidualBlock(hidden_dim // 2, hidden_dim // 2),
            ResidualBlock(hidden_dim // 2, hidden_dim // 2)
        )
        self.stage2_down = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim, 4, 2, 1),  # /4 total
            nn.ReLU(),
        )
        self.stage2_res = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        x = self.stage1_down(x)
        x = self.stage1_res(x)
        x = self.stage2_down(x)
        x = self.stage2_res(x)
        return x

# --------------------------------
# Decoder: Up → Res → Up → Res
# --------------------------------
class Decoder(nn.Module):
    def __init__(self, out_channels=4, hidden_dim=128):
        super().__init__()
        self.stage1_up = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1),  # *2
            nn.ReLU(),
        )
        self.stage1_res = nn.Sequential(
            ResidualBlock(hidden_dim // 2, hidden_dim // 2),
            ResidualBlock(hidden_dim // 2, hidden_dim // 2)
        )
        self.stage2_up = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim // 2, out_channels, 4, 2, 1),  # *4 total
        )
        # final residual block on 4ch output not needed, since we map to image

    def forward(self, x):
        x = self.stage1_up(x)
        x = self.stage1_res(x)
        x = self.stage2_up(x)
        return x

# --------------------------------
# VQ-VAE (Full Model)
# --------------------------------
class VQVAE(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=128, num_embeddings=4096):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dim)
        self.quantizer = VectorQuantizer(num_embeddings, hidden_dim)
        self.decoder = Decoder(in_channels, hidden_dim)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss = self.quantizer(z)
        x_recon = self.decoder(z_q)
        recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + vq_loss * 0.05
        return x_recon, total_loss, recon_loss, vq_loss