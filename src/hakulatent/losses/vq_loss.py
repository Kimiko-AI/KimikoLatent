"""
We use VectorQuantizer but only utilize their codebook loss as regularizations.
"""

import torch
import torch.nn as nn

from ..vq.kepler import KeplerQuantizer, KeplerLoss


class KeplerQuantizerRegLoss(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_embed=1024,
        partitions=4,
        scale=1,
        beta=0.25,
        use_kepler_loss=False,
        kl_weight=1e-8,
    ):
        super(KeplerQuantizerRegLoss, self).__init__()
        self.quantizer = KeplerQuantizer(
            embed_dim=embed_dim,
            scale=scale,
            partitions=partitions,
            n_embed=num_embed,
            beta=beta,
            kepler_loss=KeplerLoss(
                use=use_kepler_loss, kl_weight=kl_weight, n_e=int(num_embed * scale)
            ),
            legacy=True,
        )

    def forward(self, z):
        quantized, loss = self.quantizer(z)
        print(loss)
        return loss.mean()


if __name__ == "__main__":
    # Example configuration (as provided by the official team):
    embed_dim = 4
    scale = 1  # k = n_embed when scale == 1
    partitions = 1  # partition number d in the paper
    n_embed = 1024  # K in the paper
    beta = 0.25  # Example commitment loss weight

    loss_fn = KeplerQuantizerRegLoss(embed_dim, n_embed, partitions, scale, beta)

    test_latent = torch.randn(1, embed_dim, 32, 32)
    loss = loss_fn(test_latent)
    print(loss)
