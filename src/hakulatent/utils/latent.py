import torch


def pca_to_rgb(latents: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert a latent tensor of shape [B, C, H, W] to an RGB tensor [B, 3, H, W]
    using PCA projection on the channel dimension.

    Args:
        latents (torch.Tensor): Tensor of shape [B, C, H, W], where C is between 4 and 16.

    Returns:
        torch.Tensor: Tensor of shape [B, 3, H, W] obtained by projecting the channels onto the top 3 principal components.
    """
    B, C, H, W = latents.shape

    # Reshape to [N, C] where N = B * H * W.
    X = latents.permute(0, 2, 3, 1).reshape(-1, C)

    # Center the data by subtracting the mean (computed per channel).
    X_mean = X.mean(dim=0, keepdim=True)
    X_centered = X - X_mean

    # Compute the covariance matrix (C x C).
    cov = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)

    # Since the covariance matrix is symmetric, use torch.linalg.eigh.
    # eigenvectors are sorted in ascending order, so the last 3 correspond to the top 3 components.
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    top3_eigenvectors = eigenvectors[:, -3:]  # Shape: [C, 3]

    # Project the centered data onto the top 3 principal components.
    X_projected = X_centered @ top3_eigenvectors  # Shape: [B * H * W, 3]

    # Reshape back to [B, H, W, 3] and then permute to [B, 3, H, W].
    rgb = X_projected.reshape(B, H, W, 3).permute(0, 3, 1, 2)

    # Normalize each image to [0, 1] using min-max scaling.
    # Here, we compute the min and max across all channels and spatial dimensions per image.
    rgb_min = rgb.reshape(B, -1).min(dim=1)[0].reshape(B, 1, 1, 1)
    rgb_max = rgb.reshape(B, -1).max(dim=1)[0].reshape(B, 1, 1, 1)
    rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min + eps)

    return rgb_norm
