if __name__ == "__main__":
    import warnings

    warnings.filterwarnings(
        "ignore",
        ".*Found keys that are not in the model state dict but in the checkpoint.*",
    )

    import torch
    import torch.nn.functional as F
    import torch.utils.data as data
    import lpips
    from tqdm import tqdm
    from torchvision.transforms import (
        Compose,
        Resize,
        ToTensor,
        CenterCrop,
    )
    from diffusers import AutoencoderKL
    from convnext_perceptual_loss import ConvNextType, ConvNextPerceptualLoss

    from src.hl_dataset.imagenet import ImageNetDataset
    from src.hakulatent.trainer import LatentTrainer
    from src.hakulatent.logging import logger
    from src.hakulatent.models.approx import LatentApproxDecoder

    DEVICE = "cuda"
    DTYPE = torch.bfloat16
    SHORT_AXIS_SIZE = 512


NAMES = [
    "EQ-SDXL-VAE      ",
]
BASE_MODELS = [
    "diffusers/FLUX.1-vae",
]
SUB_FOLDERS = [None]
CKPT_PATHS = [
    "The-Final-VAE/yxfnmgen/checkpoints/epoch=2-step=28134.ckpt"  ,
]
USE_APPROXS = [False, ]


def process(x):
    return x * 2 - 1


def deprocess(x):
    return x * 0.5 + 0.5


if __name__ == "__main__":
    lpips_loss = torch.compile(
        lpips.LPIPS(net="vgg").eval().to(DEVICE).requires_grad_(False)
    )
    convn_loss = torch.compile(
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

    @torch.compile
    def metrics(inp, recon):
        mse = F.mse_loss(inp, recon)
        psnr = 10 * torch.log10(1 / mse)
        return (
            mse.cpu(),
            psnr.cpu(),
            lpips_loss(inp, recon, normalize=True).mean().cpu(),
            convn_loss(inp, recon).mean().cpu(),
        )

    transform = Compose(
        [
            Resize(SHORT_AXIS_SIZE),
            CenterCrop(SHORT_AXIS_SIZE),
            ToTensor(),
        ]
    )
    valid_dataset = ImageNetDataset('validation', tran=transform)
    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        pin_memory_device=DEVICE,
    )
    next(iter(valid_loader))

    logger.info("Loading models...")
    vaes = []
    for base_model, sub_folder, ckpt_path, use_approx in zip(
        BASE_MODELS, SUB_FOLDERS, CKPT_PATHS, USE_APPROXS
    ):
        vae = AutoencoderKL.from_pretrained(base_model, subfolder=sub_folder, ignore_mismatched_sizes=True)
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
        vae.encoder = torch.compile(vae.encoder)
        vae.decoder = torch.compile(vae.decoder)
        vaes.append(torch.compile(vae))

    logger.info("Running Validation")
    total = 0
    all_latents = [[] for _ in range(len(vaes))]
    all_mse = [[] for _ in range(len(vaes))]
    all_psnr = [[] for _ in range(len(vaes))]
    all_lpips = [[] for _ in range(len(vaes))]
    all_convn = [[] for _ in range(len(vaes))]
    for idx, batch in enumerate(tqdm(valid_loader)):
        image = batch[0].to(DEVICE)
        test_inp = process(image).to(DTYPE)
        batch_size = test_inp.size(0)

        for i, vae in enumerate(vaes):
            latent = vae.encode(test_inp).latent_dist.mode()
            recon = deprocess(vae.decode(latent).sample.float())
            all_latents[i].append(latent.cpu().float())
            mse, psnr, lpips_, convn = metrics(image, recon)
            all_mse[i].append(mse.cpu() * batch_size)
            all_psnr[i].append(psnr.cpu() * batch_size)
            all_lpips[i].append(lpips_.cpu() * batch_size)
            all_convn[i].append(convn.cpu() * batch_size)

        total += batch_size

    for i in range(len(vaes)):
        all_latents[i] = torch.cat(all_latents[i], dim=0)
        all_mse[i] = torch.stack(all_mse[i]).sum() / total
        all_psnr[i] = torch.stack(all_psnr[i]).sum() / total
        all_lpips[i] = torch.stack(all_lpips[i]).sum() / total
        all_convn[i] = torch.stack(all_convn[i]).sum() / total

        logger.info(
            f"  - {NAMES[i]}: MSE: {all_mse[i]:.3e}, PSNR: {all_psnr[i]:.4f}, "
            f"LPIPS: {all_lpips[i]:.4f}, ConvNeXt: {all_convn[i]:.3e}"
        )

    logger.info("Saving results...")
    for name, latents in zip(NAMES, all_latents):
        torch.save(latents, f"./output/{name.strip()}-latent.pt")
