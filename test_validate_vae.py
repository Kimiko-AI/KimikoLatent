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

    from hl_dataset.imagenet import ImageNetDataset
    from hakulatent.trainer import LatentTrainer
    from hakulatent.logging import logger

    DEVICE = "cuda"
    DTYPE = torch.float16
    SHORT_AXIS_SIZE = 256

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


def process(x):
    return x * 2 - 1


def deprocess(x):
    return x * 0.5 + 0.5


if __name__ == "__main__":
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

    valid_dataset = ImageNetDataset(
        "val",
        transform=Compose(
            [
                Resize(SHORT_AXIS_SIZE),
                CenterCrop(SHORT_AXIS_SIZE),
                ToTensor(),
            ]
        ),
    )
    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        pin_memory_device=DEVICE,
    )
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

    logger.info("Running Validation")
    all_ref_mse, all_ref_psnr, all_ref_lpips, all_ref_convn = [], [], [], []
    all_new_mse, all_new_psnr, all_new_lpips, all_new_convn = [], [], [], []
    for idx, batch in enumerate(tqdm(valid_loader)):
        test_inp = process(batch[0].to(DTYPE).to(DEVICE))

        original_latent = vae_ref.encode(test_inp).latent_dist.mode()
        new_latent = vae.encode(test_inp).latent_dist.mode()
        original_recon = deprocess(vae_ref.decode(original_latent).sample.cpu().float())
        new_recon = deprocess(vae.decode(new_latent).sample.cpu().float())

        test_inp = deprocess(test_inp)

        ref_mse, ref_psnr, ref_lpips, ref_convn = metrics(test_inp, original_recon)
        new_mse, new_psnr, new_lpips, new_convn = metrics(test_inp, new_recon)

        all_ref_mse.append(ref_mse * test_inp.shape[0])
        all_ref_psnr.append(ref_psnr * test_inp.shape[0])
        all_ref_lpips.append(ref_lpips * test_inp.shape[0])
        all_ref_convn.append(ref_convn * test_inp.shape[0])

        all_new_mse.append(new_mse * test_inp.shape[0])
        all_new_psnr.append(new_psnr * test_inp.shape[0])
        all_new_lpips.append(new_lpips * test_inp.shape[0])
        all_new_convn.append(new_convn * test_inp.shape[0])

    ref_mse = torch.stack(all_ref_mse).float().sum() / len(valid_dataset)
    ref_psnr = torch.stack(all_ref_psnr).float().sum() / len(valid_dataset)
    ref_lpips = torch.stack(all_ref_lpips).float().sum() / len(valid_dataset)
    ref_convn = torch.stack(all_ref_convn).float().sum() / len(valid_dataset)
    new_mse = torch.stack(all_new_mse).float().sum() / len(valid_dataset)
    new_psnr = torch.stack(all_new_psnr).float().sum() / len(valid_dataset)
    new_lpips = torch.stack(all_new_lpips).float().sum() / len(valid_dataset)
    new_convn = torch.stack(all_new_convn).float().sum() / len(valid_dataset)

    logger.info(
        f"  - Orig: MSE: {ref_mse:.3e}, PSNR: {ref_psnr:.4f}, "
        f"LPIPS: {ref_lpips:.4f}, ConvNeXt: {ref_convn:.3e}"
    )
    logger.info(
        f"  - New : MSE: {new_mse:.3e}, PSNR: {new_psnr:.4f}, "
        f"LPIPS: {new_lpips:.4f}, ConvNeXt: {new_convn:.3e}"
    )

    logger.info("Saving results...")
