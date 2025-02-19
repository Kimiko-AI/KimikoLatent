import warnings

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import lpips
from PIL import Image
from diffusers import AutoencoderKL
from convnext_perceptual_loss import ConvNextType, ConvNextPerceptualLoss

from hakulatent.trainer import LatentTrainer
from hakulatent.utils.latent import pca_to_rgb
from hakulatent.logging import logger


DEVICE = "cuda"
DTYPE = torch.float16
SHORT_AXIS_SIZE = 1024

BASE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
# BASE_MODEL = "black-forest-labs/FLUX.1-schnell"
# BASE_MODEL = "stabilityai/sd-vae-ft-mse"
SUB_FOLDER = None
# SUB_FOLDER = "vae"

BASE_MODEL2 = BASE_MODEL
SUB_FOLDER2 = SUB_FOLDER
# BASE_MODEL2 = "stabilityai/stable-diffusion-3.5-large"
# SUB_FOLDER2 = "vae"

# CKPT_PATH = "epoch=1-step=44000.ckpt"
CKPT_PATH = "./HakuLatent/ubeak8fi/checkpoints/epoch=0-step=1000.ckpt"
CKPT_PATH = "Y:/epoch=2-step=48000.ckpt"
# CKPT_PATH = "Y:/epoch=1-step=16000.ckpt"


warnings.filterwarnings(
    "ignore",
    ".*Found keys that are not in the model state dict but in the checkpoint.*",
)


def process(x):
    return x * 2 - 1


def deprocess(x):
    return x * 0.5 + 0.5


lpips_loss = lpips.LPIPS(net="vgg").eval().to(DEVICE).requires_grad_(False)
convn_loss = (
    ConvNextPerceptualLoss(
        device=DEVICE,
        model_type=ConvNextType.TINY,
        feature_layers=[10, 12, 14],
        input_range=(0, 1),
        use_gram=False,
    )
    .eval()
    .requires_grad_(False)
)


def metrics(inp, recon):
    inp = inp.to(DEVICE).float()
    recon = recon.to(DEVICE).float()
    mse = F.mse_loss(inp, recon)
    psnr = 10 * torch.log10(1 / mse)
    return (
        mse.cpu(),
        psnr.cpu(),
        lpips_loss(inp, recon, normalize=True).mean().cpu(),
        convn_loss(inp, recon).mean().cpu(),
    )


if __name__ == "__main__":
    test_img = Image.open("test.jpg").convert("RGB")
    test_img = VF.to_tensor(test_img)
    test_inp = process(VF.resize(test_img, SHORT_AXIS_SIZE)[None].to(DEVICE).to(DTYPE))
    test_img = VF.resize(test_img, SHORT_AXIS_SIZE * 2)[None]

    logger.info("Loading models...")
    vae_ref: AutoencoderKL = AutoencoderKL.from_pretrained(
        BASE_MODEL, subfolder=SUB_FOLDER
    )
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        BASE_MODEL2, subfolder=SUB_FOLDER2
    )
    if BASE_MODEL2 == BASE_MODEL and SUB_FOLDER2 == SUB_FOLDER:
        trainer_module = LatentTrainer.load_from_checkpoint(
            CKPT_PATH,
            vae=vae,
            map_location="cpu",
            strict=False,
        )

    vae = vae.to(DTYPE).eval().requires_grad_(False).to(DEVICE)
    vae_ref = vae_ref.to(DTYPE).eval().requires_grad_(False).to(DEVICE)

    logger.info("Running Encoding...")

    original_latent = vae_ref.encode(test_inp).latent_dist.mode()
    new_latent = vae.encode(test_inp).latent_dist.mode()

    logger.info("Running Decoding...")

    original_recon = deprocess(vae_ref.decode(original_latent).sample.cpu().float())
    new_recon = deprocess(vae.decode(new_latent).sample.cpu().float())

    logger.info("Done, calculating results...")
    orig_latent_rgb = F.interpolate(
        pca_to_rgb(original_latent.cpu().float()[:, :4]),
        original_recon.shape[-2:],
        mode="nearest",
    )
    new_latent_rgb = F.interpolate(
        pca_to_rgb(new_latent.cpu().float()[:, :4]),
        new_recon.shape[-2:],
        mode="nearest",
    )
    test_inp = deprocess(F.interpolate(test_inp, new_recon.shape[-2:], mode="bilinear"))
    test_img = F.interpolate(
        test_img, [i * 2 for i in new_recon.shape[-2:]], mode="bilinear"
    )

    ref_mse, ref_psnr, ref_lpips, ref_convn = metrics(test_inp, original_recon)
    new_mse, new_psnr, new_lpips, new_convn = metrics(test_inp, new_recon)

    logger.info(
        f"  - Orig: MSE: {ref_mse:.3e}, PSNR: {ref_psnr:.4f}, "
        f"LPIPS: {ref_lpips:.4f}, ConvNeXt: {ref_convn:.3e}"
    )
    logger.info(
        f"  - New : MSE: {new_mse:.3e}, PSNR: {new_psnr:.4f}, "
        f"LPIPS: {new_lpips:.4f}, ConvNeXt: {new_convn:.3e}"
    )

    logger.info("Saving results...")
    result_grid = torch.cat(
        [
            test_img,
            torch.cat(
                [
                    torch.cat([orig_latent_rgb, new_latent_rgb], dim=-2),
                    torch.cat([original_recon, new_recon], dim=-2),
                ],
                dim=-1,
            ),
        ],
        dim=-1,
    )
    result_grid = result_grid.clamp(0, 1)
    result_grid = result_grid.permute(0, 2, 3, 1).cpu()
    result_grid = result_grid[0]
    result_grid = (result_grid * 255).to(torch.uint8).numpy()
    result_grid = Image.fromarray(result_grid)
    result_grid.save("result.jpg", quality=95)
    logger.info("All done!")
