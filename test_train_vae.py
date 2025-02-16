"""
A demo script to finetune a pretrained VAE on ImageNet with the EQ-VAE setup.
"""

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
)
from diffusers import AutoencoderKL

from hl_dataset.imagenet import ImageNetDataset
from hakulatent.transform import (
    LatentTransformCompose,
    LatentTransformSwitch,
    RotationTransform,
    ScaleDownTransform,
    ScaleUpCropTransform,
    CropTransform,
)
from hakulatent.trainer import LatentTrainer
from hakulatent.losses import AdvLoss, ReconLoss


EPOCHS = 1
BATCH_SIZE = 8
GRAD_ACC = 8

ADV_START_ITER = 100

NUM_WORKERS = 8
SIZE = 256
LR = 2e-5
DLR = 1e-4


if __name__ == "__main__":
    split = "train"
    transform = Compose(
        [
            Resize(SIZE),
            RandomCrop((SIZE, SIZE)),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor(),
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
    vae: AutoencoderKL = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
    vae.get_last_layer = lambda: vae.decoder.conv_out.weight

    trainer_module = LatentTrainer(
        vae=vae,
        recon_loss=torch.compile(ReconLoss(loss_type="l1", lpips_net="vgg")),
        adv_loss=torch.compile(AdvLoss(start_iter=2 * GRAD_ACC * ADV_START_ITER)),
        latent_transform=LatentTransformCompose(
            RotationTransform(method="random"),
            LatentTransformSwitch(
                ScaleDownTransform(
                    method="random", scale_factors=[i / 32 for i in range(16, 32, 2)]
                ),
                ScaleUpCropTransform(
                    method="random", scale_factors=[i / 32 for i in range(32, 48, 2)]
                ),
                CropTransform(
                    method="random", scale_factors=[i / 32 for i in range(16, 32, 2)]
                ),
            ),
        ),
        loss_weights={
            "recon": 1.0,
            "adv": 1.0,
            "kl": 2e-7,
        },
        name="EQ-VAE-sdxl-imgnet",
        lr=LR,
        lr_disc=DLR,
        optimizer="torch.optim.AdamW",
        opt_configs={"betas": (0.9, 0.98)},
        lr_sch_configs={
            "lr": {
                "end": len(loader) * EPOCHS,
                "value": 1.0,
                "min_value": 0.25,
                "mode": "cosine",
                "warmup": 100,
            }
        },
        grad_acc=GRAD_ACC,
    )

    logger = WandbLogger(
        project="HakuLatent",
        name="EQ-VAE-sdxl-imgnet-AdvON",
    )
    trainer = pl.Trainer(
        logger=logger,
        devices=1,
        max_epochs=EPOCHS,
        precision="16-mixed",
        callbacks=[
            ModelCheckpoint(
                every_n_train_steps=5000,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
    )
    trainer.fit(trainer_module, loader)
else:
    print("Subprocess Running:", __name__)
