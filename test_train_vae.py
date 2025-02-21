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
    from hakulatent.transform import (
        LatentTransformCompose,
        LatentTransformSwitch,
        RotationTransform,
        ScaleDownTransform,
        ScaleUpCropTransform,
        CropTransform,
        RandomAffineTransform,
    )
    from hakulatent.trainer import LatentTrainer
    from hakulatent.losses import AdvLoss, ReconLoss
else:
    # This if-else can speedup multi-worker dataloader in windows
    print("Subprocess Starting:", __name__)


BASE_MODEL = "KBlueLeaf/EQ-SDXL-VAE"
SUB_FOLDER = None
EPOCHS = 1
BATCH_SIZE = 8
GRAD_ACC = 4
GRAD_CKPT = False

LOSS_TYPE = "mse"
LPIPS_NET = "vgg"
USE_CONVNEXT = True
ADV_START_ITER = 1000

NUM_WORKERS = 8
SIZE = 256
LR = 5e-5
DLR = 1e-3


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
    # loader warmup
    next(iter(loader))

    vae: AutoencoderKL = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder=SUB_FOLDER)
    if GRAD_CKPT:
        vae.enable_gradient_checkpointing()
    vae.encoder = torch.compile(vae.encoder)
    vae.get_last_layer = lambda: vae.decoder.conv_out.weight

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
        ),
        adv_loss=AdvLoss(start_iter=ADV_START_ITER, n_layers=5),
        img_deprocess=deprocess,
        log_interval=100,
        transform_prob=0.5,
        latent_transform=RandomAffineTransform(
            rotate_range=(-180, 180),
            scale_range=(0.8, 1.2),
            shear_range=((-10, 10), (-5, 5)),
            translate_range=(0.1, 0.1),
            method="random",
        ),  # Thanks AmericanPresidentJimmyCarter
        loss_weights={
            "recon": 1.0,
            "adv": 0.25,
            "kl": 5e-8,
        },
        name="EQ-SDXL-VAE-random-affine",
        lr=LR,
        lr_disc=DLR,
        optimizer="torch.optim.AdamW",
        opt_configs={"betas": (0.9, 0.98)},
        lr_sch_configs={
            "lr": {
                "end": len(loader) * EPOCHS // GRAD_ACC,
                "value": 1.0,
                "min_value": 0.2,
                "mode": "cosine",
                "warmup": 0,
            }
        },
        grad_acc=GRAD_ACC,
    )

    logger = WandbLogger(
        project="HakuLatent",
        name="EQ-SDXL-VAE-random-affine",
        offline=True
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
