import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentApproxDecoder(nn.Module):
    def __init__(
        self, latent_dim=4, out_channels=3, shuffle=2, post_conv=True, logvar=False
    ):
        super().__init__()
        self.shuffle = shuffle
        self.post_conv = post_conv
        out_channels = out_channels + logvar
        self.conv_in = nn.Conv2d(
            latent_dim, out_channels * shuffle**2, 5, stride=1, padding=2
        )
        if post_conv:
            self.conv_out = nn.Conv2d(
                out_channels, out_channels, 5, stride=1, padding=2
            )
            nn.init.zeros_(self.conv_out.bias)
            nn.init.zeros_(self.conv_out.weight)
        else:
            nn.init.zeros_(self.conv_in.bias)
            nn.init.zeros_(self.conv_in.weight)

    def last_layer(self):
        return self.conv_out if self.post_conv else self.conv_in

    def forward(self, x):
        x = self.conv_in(x)
        x = F.pixel_shuffle(x, self.shuffle)
        if self.post_conv:
            x = self.conv_out(x)
        return x
