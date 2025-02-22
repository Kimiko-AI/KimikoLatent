from hmac import new
import torch
import torch.nn as nn
from diffusers import AutoencoderKL


def expand_latent_dim(vae: AutoencoderKL, new_latent_dim: int):
    vae.config.latent_channels = new_latent_dim
    encoder_out = vae.encoder.conv_out
    enc_out_weight = encoder_out.weight.data
    enc_out_weight_mu = enc_out_weight[: vae.config.latent_channels]
    enc_out_weight_lv = enc_out_weight[vae.config.latent_channels :]
    enc_out_bias = encoder_out.bias.data
    enc_out_bias_mu = enc_out_bias[: vae.config.latent_channels]
    enc_out_bias_lv = enc_out_bias[vae.config.latent_channels :]

    new_encoder_out = nn.Conv2d(
        encoder_out.in_channels,
        new_latent_dim * 2,
        encoder_out.kernel_size,
        encoder_out.stride,
        encoder_out.padding,
    )
    new_encoder_out.weight.data = torch.cat(
        [
            enc_out_weight_mu,
            torch.zeros_like(enc_out_weight_mu),
            enc_out_weight_lv,
            torch.zeros_like(enc_out_weight_lv),
        ],
        dim=0,
    )
    new_encoder_out.bias.data = torch.cat(
        [
            enc_out_bias_mu,
            torch.zeros_like(enc_out_bias_mu),
            enc_out_bias_lv,
            torch.zeros_like(enc_out_bias_lv),
        ],
        dim=0,
    )
    vae.encoder.conv_out = new_encoder_out

    decoder_in = vae.decoder.conv_in
    new_decoder_in = nn.Conv2d(
        new_latent_dim,
        decoder_in.out_channels,
        decoder_in.kernel_size,
        decoder_in.stride,
        decoder_in.padding,
    )
    new_decoder_in.weight.data = torch.cat(
        [
            decoder_in.weight.data,
            torch.randn_like(decoder_in.weight.data) * decoder_in.weight.data.std(),
        ],
        dim=1,
    )
    new_decoder_in.bias.data = torch.cat(
        [
            decoder_in.bias.data,
            torch.randn_like(decoder_in.bias.data) * decoder_in.bias.data.std(),
        ]
    )
    vae.decoder.conv_in = new_decoder_in

    if vae.quant_conv is not None:
        quant_conv = vae.quant_conv
        qc_weight_mu = quant_conv.weight.data[: vae.config.latent_channels]
        qc_weight_lv = quant_conv.weight.data[vae.config.latent_channels :]
        qc_bias_mu = quant_conv.bias.data[: vae.config.latent_channels]
        qc_bias_lv = quant_conv.bias.data[vae.config.latent_channels :]
        new_quant_conv = nn.Conv2d(
            2 * new_latent_dim,
            2 * new_latent_dim,
            quant_conv.stride,
            quant_conv.padding,
        )
        nn.init.zeros_(new_quant_conv.weight)
        nn.init.zeros_(new_quant_conv.bias)
        new_quant_conv.weight.data[: qc_weight_mu.size(0), : qc_weight_mu.size(1)] = (
            qc_weight_mu
        )
        new_quant_conv.weight.data[
            new_latent_dim : new_latent_dim + qc_weight_lv.size(0),
            : qc_weight_mu.size(1),
        ] = (
            torch.zeros_like(qc_weight_mu) * qc_weight_mu.std()
        )
        new_quant_conv.weight.data = torch.cat(
            [
                qc_weight_mu,
                torch.zeros_like(qc_weight_mu) * qc_weight_mu.std(),
                qc_weight_lv,
                torch.zeros_like(qc_weight_lv) * qc_weight_lv.std(),
            ],
            dim=0,
        )
        new_quant_conv.bias.data = torch.cat(
            [
                qc_bias_mu,
                torch.zeros_like(qc_bias_mu),
                qc_bias_lv,
                torch.zeros_like(qc_bias_lv),
            ],
            dim=0,
        )
        vae.quant_conv = new_quant_conv

    if vae.post_quant_conv is not None:
        post_quant_conv = vae.post_quant_conv
        new_post_quant_conv = nn.Conv2d(
            new_latent_dim,
            post_quant_conv.out_channels,
            post_quant_conv.kernel_size,
            post_quant_conv.stride,
            post_quant_conv.padding,
        )
        nn.init.zeros_(new_post_quant_conv.weight)
        nn.init.zeros_(new_post_quant_conv.bias)
        vae.post_quant_conv = new_post_quant_conv

    return vae
