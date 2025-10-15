import torch
import torch.nn as nn
import torch.nn.functional as F

class R3GANDiscBlock(nn.Module):
    """
    A residual discriminator block based on R3GAN principles, now with variable depth.
    Uses AvgPool for downsampling, GroupNorm, and SiLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        groups (int): Number of groups for GroupNorm.
        num_conv_blocks (int): The number of conv->norm->act blocks in the main path.
    """

    def __init__(self, in_channels, out_channels, groups=8, num_conv_blocks=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # The number of conv blocks must be at least 1
        if num_conv_blocks < 1:
            raise ValueError("num_conv_blocks must be at least 1.")

        # Use a 1x1 conv for the skip connection if channels change
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

        # Dynamically build the main path
        layers = []
        # First convolution block handles the transition from in_channels to out_channels
        layers.extend([
            nn.GroupNorm(num_groups=min(groups, in_channels), num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        ])

        # Add subsequent convolution blocks, which all operate on out_channels
        for _ in range(num_conv_blocks - 1):
            layers.extend([
                nn.GroupNorm(num_groups=groups, num_channels=out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            ])

        self.main = nn.Sequential(*layers)

        # R3GAN principle: Separate downsampling from convolution
        self.downsample = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        # Main path
        h = self.main(x)
        # Skip connection path
        x_skip = self.skip(x)

        # Add residual and then downsample.
        # The residual connection skips over all internal conv blocks.
        out = self.downsample(x_skip + h)
        return out


class R3GANDiscriminator(nn.Module):
    """
    A PatchGAN-style discriminator built with modern R3GAN blocks.
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=5, groups=8):
        """
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the base conv layer
            n_layers (int)  -- the number of downsampling blocks
            groups (int)    -- the number of groups for GroupNorm
        """
        super().__init__()

        # Initial convolution to map input image to feature space
        self.initial_conv = nn.Conv2d(input_nc, ndf, kernel_size=3, padding=1)

        blocks = []
        in_feat = ndf
        for i in range(n_layers):
            # Double the features at each layer, up to a max of 8*ndf
            out_feat = min(ndf * (2 ** (i + 1)), ndf * 8)
            blocks.append(R3GANDiscBlock(in_feat, out_feat, groups=groups))
            in_feat = out_feat

        # Add a final processing block without downsampling
        self.blocks = nn.Sequential(*blocks)
        self.final_block = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=in_feat),
            nn.SiLU(),
            nn.Conv2d(in_feat, in_feat, kernel_size=3, padding=1),
        )

        # Final 1x1 convolution to produce a 1-channel prediction map
        self.final_conv = nn.Conv2d(in_feat, 1, kernel_size=1)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.blocks(x)
        x = self.final_block(x)
        x = self.final_conv(x)
        return x

# Example Usage
if __name__ == '__main__':
    # Create a dummy input image tensor
    # Batch size 1, 3 channels (RGB), 256x256 pixels
    dummy_image = torch.randn(1, 3, 256, 256)

    # Initialize the discriminator
    discriminator = R3GANDiscriminator(input_nc=3, ndf=64, n_layers=4)

    # Get the output (a patch-based prediction map)
    output_map = discriminator(dummy_image)

    # The output is a grid where each value judges a patch of the input
    print("R3GAN Discriminator")
    print(f"Input shape: {dummy_image.shape}")
    print(f"Output shape: {output_map.shape}") # Should be [1, 1, 16, 16] for 4 layers

    # Calculate number of parameters
    num_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params / 1e6:.2f} M")