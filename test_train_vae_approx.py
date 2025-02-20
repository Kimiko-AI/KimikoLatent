"""
A demo script to finetune a pretrained VAE on ImageNet with the EQ-VAE setup.
"""

if __name__ == "__main__":
    import torch
    import torch.utils.data as data
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from torchvision.transforms import (
        Compose,
        Resize,
        RandomCrop,
        RandomHorizontalFlip,
        RandomVerticalFlip,
        ToTensor,
        Lambda,
    )
    from diffusers import AutoencoderKL
    from convnext_perceptual_loss import ConvNextType

    from hl_dataset.imagenet import ImageNetDataset
    from hakulatent.trainer import LatentTrainer
    from hakulatent.losses import AdvLoss, ReconLoss
    from hakulatent.models.approx import LatentApproxDecoder
else:
    # This if-else can speedup multi-worker dataloader in windows
    print("Subprocess Starting:", __name__)


BASE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
SUB_FOLDER = None
EPOCHS = 1
BATCH_SIZE = 32
GRAD_ACC = 1
GRAD_CKPT = False

LOSS_TYPE = "l1"
LPIPS_NET = "vgg"
USE_CONVNEXT = True
ADV_START_ITER = 1000

NUM_WORKERS = 12
SIZE = 512
LR = 1e-3
DLR = 5e-3


def process(x):
    return x * 2 - 1


def deprocess(x):
    return x * 0.5 + 0.5


if __name__ == "__main__":
    split = "train"
    transform = Compose(
        [
            Resize(SIZE),
            RandomCrop((SIZE, SIZE)),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor(),
            Lambda(process),
        ]
    )
    dataset = ImageNetDataset(split, transform)
    loader = data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        persistent_workers=bool(NUM_WORKERS),
    )
    next(iter(loader))

    vae: AutoencoderKL = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder=SUB_FOLDER)
    if GRAD_CKPT:
        vae.enable_gradient_checkpointing()
    vae = vae.eval().requires_grad_(False)
    vae.encoder = torch.compile(vae.encoder)
    vae.decoder = None
    vae.post_quant_conv = None

    approx = LatentApproxDecoder(
        latent_dim=vae.config.latent_channels, out_channels=3, shuffle=2
    )
    vae.decoder = approx
    vae.decode = lambda x: approx(x)
    vae.get_last_layer = lambda: approx.conv_out.weight

    trainer_module = LatentTrainer(
        vae=vae,
        recon_loss=ReconLoss(
            loss_type=LOSS_TYPE,
            lpips_net=LPIPS_NET,
            convnext_type=ConvNextType.TINY if USE_CONVNEXT else None,
            convnext_kwargs={
                "feature_layers": [10, 12, 14],
                "use_gram": False,
                "input_range": (-1, 1),
                "device": "cuda",
            },
            loss_weights={
                LOSS_TYPE: 5.0,
                "lpips": 0.5,
                "convnext": 2.0,
            },
        ),
        adv_loss=None,
        img_deprocess=deprocess,
        log_interval=100,
        loss_weights={
            "recon": 1.0,
            "adv": 0,
            "kl": 0,
        },
        name="SDXL-VAE-approx",
        lr=LR,
        lr_disc=DLR,
        optimizer="torch.optim.AdamW",
        opt_configs={"betas": (0.9, 0.98)},
        lr_sch_configs={
            "lr": {
                "end": len(loader) * EPOCHS // GRAD_ACC,
                "value": 1.0,
                "min_value": 0.1,
                "mode": "cosine",
                "warmup": 0,
            }
        },
        grad_acc=GRAD_ACC,
    )

    logger = WandbLogger(
        project="HakuLatent",
        name="SDXL-VAE-approx",
        # offline=True,
    )
    trainer = pl.Trainer(
        logger=logger,
        devices=1,
        max_epochs=EPOCHS,
        precision="16-mixed",
        callbacks=[
            ModelCheckpoint(every_n_train_steps=1000),
            LearningRateMonitor(logging_interval="step"),
        ],
        log_every_n_steps=1,
    )
    trainer.fit(trainer_module, loader)
