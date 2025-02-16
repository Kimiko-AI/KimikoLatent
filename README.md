# [WIP] HakuLatent

A comprehensive codebase for training and finetuning Image <> Latent models.

## TODOs

- [X] Trainer for VAE or VQ-VAE or direct AE
  - [X] Basic Trainer
  - [ ] Decoder-only finetuning
  - [ ] PEFT
- [X] Equivariance Regularization [EQ-VAE](https://arxiv.org/abs/2502.09509)
- [X] Adversarial Loss
  - [ ] Investigate better discriminator setup
- [ ] Models
  - [ ] MAE for latent
  - [ ] windowed/natten attention for commonly used VAE setup
