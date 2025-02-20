import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentApproxDecoder(nn.Module):
    def __init__(self, latent_dim=4, out_channels=3, shuffle=2):
        super().__init__()
        self.conv_in = nn.Conv2d(
            latent_dim, out_channels * shuffle**2, 5, stride=1, padding=2
        )
        self.conv_out = nn.Conv2d(out_channels, out_channels, 5, stride=1, padding=2)
        self.shuffle = shuffle
        nn.init.zeros_(self.conv_out.bias)
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, x):
        x = self.conv_in(x)
        x = F.pixel_shuffle(x, self.shuffle)
        x = self.conv_out(x)
        return x
