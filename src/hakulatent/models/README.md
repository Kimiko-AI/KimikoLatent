# Model API

In our implementation, we allow arbitrary model implementation to be used

But they should follow specific API definition

## VAE

For VAE, we decide to directly use the definition from diffuser with minor modifications

* encode(x: torch.Tensor) -> DiagonalGaussianDistribution
* decode(z: torch.Tensor) -> torch.Tensor
* forward(forward): decode(encode().sample) or decode(encode().mode)
* get_last_layer() -> nn.Parameters
  * Return the weight of output layer of decoder so Adversarial loss can calculate the adaptive weight on its gradient.

## Losses

* recon loss:
  * forward(x_real, x_fake) -> recon loss
* disc loss:
  * forward(x_real, x_fake, rec_loss, opt_idx, last_layer) -> adv loss or disc loss
