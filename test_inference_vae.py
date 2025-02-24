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
from hakulatent.models.approx import LatentApproxDecoder


DEVICE = "cuda"
DTYPE = torch.float16
SHORT_AXIS_SIZE = 1024

NAMES = [
    "SDXL           ",
    "EQ-SDXL-VAE    ",
    "EQ-SDXL-VAE-advON",
]
BASE_MODELS = [
    "madebyollin/sdxl-vae-fp16-fix",
    "KBlueLeaf/EQ-SDXL-VAE",
    "KBlueLeaf/EQ-SDXL-VAE",
    # "./models/EQ-SDXL-VAE-ch8",
]
SUB_FOLDERS = [None, None, None]
CKPT_PATHS = [
    None, 
    None, 
    "Y:/EQ-SDXL-VAE-advft-ckpt/epoch=0-step=2000.ckpt",
]
USE_APPROXS = [False, False, False]

warnings.filterwarnings(
    "ignore",
    ".*Found keys that are not in the model state dict but in the checkpoint.*",
)


def process(x):
    return x * 2 - 1


def deprocess(x):
    return x * 0.5 + 0.5


lpips_loss = lpips.LPIPS(net="vgg").eval().to(DEVICE).requires_grad_(False).float()
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
    .float()
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


def model_distance(model1, model2):
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    distance = 0
    total = 0

    for key in state_dict1.keys():
        if key in state_dict2 and state_dict1[key].shape == state_dict2[key].shape:
            distance += torch.dist(state_dict1[key], state_dict2[key])
            total += 1

    return distance/total


if __name__ == "__main__":
    test_img = Image.open("test4.png").convert("RGB")
    test_img = VF.to_tensor(test_img)
    test_inp = process(VF.resize(test_img, SHORT_AXIS_SIZE)[None].to(DEVICE).to(DTYPE))
    test_img = VF.resize(test_img, SHORT_AXIS_SIZE * 2)[None]

    test_img2 = test_img
    test_inp2 = test_inp

    # test_img2 = Image.open("test5.png").convert("RGB")
    # test_img2 = VF.to_tensor(test_img2)
    # test_inp2 = process(
    #     VF.resize(test_img2, list(test_inp.shape[-2:]))[None].to(DEVICE).to(DTYPE)
    # )
    # test_img2 = VF.resize(test_img2, list(test_img.shape[-2:]))[None]

    logger.info("Loading models...")
    vaes = []
    for base_model, sub_folder, ckpt_path, use_approx in zip(
        BASE_MODELS, SUB_FOLDERS, CKPT_PATHS, USE_APPROXS
    ):
        vae = AutoencoderKL.from_pretrained(base_model, subfolder=sub_folder)
        if use_approx:
            vae.decoder = LatentApproxDecoder(
                latent_dim=vae.config.latent_channels,
                out_channels=3,
                shuffle=2,
                # post_conv=False,
                # logvar=True,
            )
            vae.decode = lambda x: vae.decoder(x)
            vae.get_last_layer = lambda: vae.decoder.conv_out.weight
        if ckpt_path:
            LatentTrainer.load_from_checkpoint(
                ckpt_path, vae=vae, map_location="cpu", strict=False
            )
        vae = vae.to(DTYPE).eval().requires_grad_(False).to(DEVICE)
        vaes.append(vae)

    logger.info("Calculating distances...")
    for i in range(len(vaes)):
        for j in range(i + 1, len(vaes)):
            distance = model_distance(vaes[i], vaes[j])
            logger.info(f"Distance between {i} and {j}: {distance}")

    logger.info("Running Encoding...")

    latent1s = []
    for vae in vaes:
        latent1s.append(vae.encode(test_inp).latent_dist.mode().float())

    latent2s = []
    for vae in vaes:
        latent2s.append(vae.encode(test_inp2).latent_dist.mode().float())

    latents = [
        latent1 * 0.5 + latent2 * 0.5 for latent1, latent2 in zip(latent1s, latent2s)
    ]
    logger.info("Running Decoding...")

    recons = []
    for vae, latent in zip(vaes, latents):
        recon = vae.decode(latent.to(DTYPE))
        if hasattr(recon, "sample"):
            recon = recon.sample
        recon = deprocess(recon.cpu().float())[:, :3]
        recons.append(recon)

    logger.info("Done, calculating results...")
    latent_rgbs = []
    for latent in latents:
        latent_rgbs.append(
            F.interpolate(
                pca_to_rgb(latent.cpu().float()), recons[-1].shape[-2:], mode="nearest"
            )
        )

    test_inp = deprocess(
        F.interpolate(test_inp, recons[-1].shape[-2:], mode="bilinear")
    )
    test_inp2 = deprocess(
        F.interpolate(test_inp2, recons[-1].shape[-2:], mode="bilinear")
    )
    test_inp = 0.5 * test_inp + 0.5 * test_inp2
    test_img = F.interpolate(
        test_img, [i * 2 for i in recons[-1].shape[-2:]], mode="bilinear"
    )
    test_img2 = F.interpolate(
        test_img2, [i * 2 for i in recons[-1].shape[-2:]], mode="bilinear"
    )
    test_img = 0.5 * test_img + 0.5 * test_img2

    mses, psnrs, lpipses, convnes = [], [], [], []
    for name, recon in zip(NAMES, recons):
        mse, psnr, lpips_, convn = metrics(test_inp.float(), recon.float())
        logger.info(
            f"  - {name}: MSE: {mse:.3e}, PSNR: {psnr:.4f}, "
            f"LPIPS: {lpips_:.4f}, ConvNeXt: {convn:.3e}"
        )

    logger.info("Saving results...")
    result_grid = torch.cat(
        [
            test_img,
            torch.cat(
                [
                    torch.cat([recon, latent_rgb], dim=-2)
                    for recon, latent_rgb in zip(recons, latent_rgbs)
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
