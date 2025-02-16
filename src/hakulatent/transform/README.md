# EQ-VAE reproduction

[EQ-VAE: Equivariance Regularized Latent Space forImproved Generative Image Modeling](https://arxiv.org/pdf/2502.09509)

In EQ-VAE, we implement transformation which "intuitively" should provide same effect on both latent and original space.

For example: spatial rotation and scaling should provide same effect in both 2D image latent and 2D image.

Therefore, training with those transformation can ensure(enforece, force) the latent space to share similar property of original space, which make the latent space to be more friendly than naive latent.

In this folder we implement a transformation framework which designed to work on both latent and input image so we can reproduce EQ-VAE.

The basic pseudo code will be:

```python
def training_step(batch)
    x = batch
    mu, log_var = encoder(x)
    latent = sample(mu, log_var)
    x, latent = transform(x, latent)
    x_rec = decoder(latent)
    loss = l_recon(x, x_rec) + l_disc(x_rec) + kl_div(mu, log_var)
    return loss
```
