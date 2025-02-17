import test
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import lpips
from PIL import Image
from diffusers import AutoencoderKL

from hakulatent.trainer import LatentTrainer
from hakulatent.utils.latent import pca_to_rgb
from hakulatent.logging import logger


lpips_loss = lpips.LPIPS(net="vgg").eval().to("mps").requires_grad_(False)


def metrics(inp, recon):
    mse = F.mse_loss(inp.cpu().float(), recon.cpu().float())
    psnr = 10 * torch.log10(1 / mse)
    return (
        mse,
        psnr,
        lpips_loss(inp.to("mps").float(), recon.to("mps").float()).mean().cpu(),
    )


if __name__ == "__main__":
    test_img = Image.open("test3.png")
    test_img = VF.to_tensor(test_img)
    test_inp = VF.resize(test_img, 1024).unsqueeze(0).to("mps").half() * 2 - 1
    test_img = VF.resize(test_img, 2048).unsqueeze(0)

    logger.info("Loading models...")
    vae_ref: AutoencoderKL = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix"
    )
    vae: AutoencoderKL = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
    trainer_module = LatentTrainer.load_from_checkpoint(
        "./epoch=0-step=23000.ckpt", vae=vae, map_location="cpu", strict=False
    )

    vae = vae.half().eval().requires_grad_(False).to("mps")
    vae_ref = vae_ref.half().eval().requires_grad_(False).to("mps")

    logger.info("Running Encoding...")

    original_latent = vae_ref.encode(test_inp).latent_dist.mode()
    new_latent = vae.encode(test_inp).latent_dist.mode()

    logger.info("Running Decoding...")

    original_recon = vae_ref.decode(original_latent).sample.cpu().float() * 0.5 + 0.5
    new_recon = vae.decode(new_latent).sample.cpu().float() * 0.5 + 0.5

    logger.info("Done, calculating results...")
    orig_latent_rgb = F.interpolate(
        pca_to_rgb(original_latent.cpu().float()),
        original_recon.shape[-2:],
        mode="nearest",
    )
    new_latent_rgb = F.interpolate(
        pca_to_rgb(new_latent.cpu().float()), new_recon.shape[-2:], mode="nearest"
    )
    test_inp = (
        F.interpolate(test_inp, new_recon.shape[-2:], mode="bilinear") * 0.5 + 0.5
    )
    test_img = F.interpolate(
        test_img, [i * 2 for i in new_recon.shape[-2:]], mode="bilinear"
    )

    ref_mse, ref_psnr, ref_lpips = metrics(test_inp, original_recon)
    new_mse, new_psnr, new_lpips = metrics(test_inp, new_recon)

    logger.info(
        f"  - Orig: MSE: {ref_mse:.4f}, PSNR: {ref_psnr:.2f}, LPIPS: {ref_lpips:.4f}"
    )
    logger.info(
        f"  - New : MSE: {new_mse:.4f}, PSNR: {new_psnr:.2f}, LPIPS: {new_lpips:.4f}"
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
    result_grid.save("result.webp", quality=95)
