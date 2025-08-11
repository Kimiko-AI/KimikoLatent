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

    from src.hl_dataset.imagenet import ImageNetDataset
    from src.hakulatent.transform import (
        LatentTransformCompose,
        LatentTransformSwitch,
        RotationTransform,
        ScaleDownTransform,
        ScaleUpCropTransform,
        CropTransform,
        RandomAffineTransform,
        BlendingTransform,
    )
    from src.hakulatent.trainer import LatentTrainer
    from src.hakulatent.losses import AdvLoss, ReconLoss, KeplerQuantizerRegLoss
    from src.hakulatent.extune.linear import ScaleLinear, ScaleConv2d
else:
    # This if-else can speedup multi-worker dataloader in windows
    print("Subprocess Starting:", __name__)
torch.set_float32_matmul_precision('medium' )
from torchvision.transforms import InterpolationMode

BASE_MODEL = "diffusers/FLUX.1-vae"
SUB_FOLDER = None
EPOCHS = 2
BATCH_SIZE = 4
GRAD_ACC = 4
GRAD_CKPT = True
TRAIN_DEC_ONLY = False

LOSS_TYPE = "huber"
LPIPS_NET = "vgg"
USE_CONVNEXT = True
ADV_START_ITER = 0

NUM_WORKERS = 8
SIZE =384
LR = 1e-4
DLR = 1e-4

NEW_LATENT_DIM = None
PRETRAIN = False


def process(x):
    return x * 2 - 1


def deprocess(x):
    return x * 0.5 + 0.5


if __name__ == "__main__":
    split = "train"
    transform = Compose(
        [
            Resize(512,  interpolation=InterpolationMode.BICUBIC),
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

    if NEW_LATENT_DIM:
        vae.config.latent_channels = NEW_LATENT_DIM
        vae.encoder.conv_out = ScaleConv2d(
            "",
            vae.encoder.conv_out,
            vae.encoder.conv_out.in_channels,
            NEW_LATENT_DIM * 2,
        ).generate_module()
        vae.decoder.conv_in = ScaleConv2d(
            "", vae.decoder.conv_in, NEW_LATENT_DIM, vae.decoder.conv_in.out_channels
        ).generate_module()
        vae.quant_conv = ScaleConv2d(
            "", vae.quant_conv, NEW_LATENT_DIM * 2, NEW_LATENT_DIM * 2, inputs_groups=[]
        ).generate_module()
        vae.post_quant_conv = ScaleConv2d(
            "", vae.post_quant_conv, NEW_LATENT_DIM, NEW_LATENT_DIM
        ).generate_module()

    if PRETRAIN:
        vae = AutoencoderKL(
            down_block_types=["DownEncoderBlock2D"] * 4,
            block_out_channels=[128, 256, 512, 512],
            latent_channels=NEW_LATENT_DIM,
            up_block_types=["UpDecoderBlock2D"] * 4,
            layers_per_block=2,
        )
        vae.save_pretrained("./models/Kohaku-VAE")
        vae.compile()

    if TRAIN_DEC_ONLY:
        vae.requires_grad_(False)
        vae.decoder.requires_grad_(True)

    vae.get_last_layer = lambda: vae.decoder.conv_out.weight
    vae.compile()
    trainer_module = LatentTrainer(
        vae=vae,
        recon_loss=ReconLoss(
            loss_type=LOSS_TYPE,
            lpips_net=LPIPS_NET,
            convnext_type=ConvNextType.TINY if USE_CONVNEXT else None,
            convnext_kwargs={
                "feature_layers": [2, 6, 10, 14],
                "use_gram": False,
                "input_range": (-1, 1),
                "device": "cuda",
            },
            loss_weights={
                LOSS_TYPE: 0.25,
                "lpips": 0.3,
                "convnext": 0.45,
            },
        ),
        latent_loss=KeplerQuantizerRegLoss(
            embed_dim=vae.config.latent_channels,
            scale=1,
            partitions=1,
            num_embed=1024,
            beta=0.25,
            use_kepler_loss=False,
        ),
        adv_loss=AdvLoss(start_iter=ADV_START_ITER, disc_loss="vanilla", n_layers=4),
        img_deprocess=deprocess,
        log_interval=100,
        loss_weights={
            "recon": 1,
            "adv": 0.25,
            "kl": 0.00001,
            "reg": 0,
            "cycle": 0.25,

        },
        name="EQ-SDXL-VAE-random-affine",
        lr=LR,
        lr_disc=DLR,
        optimizer="torch.optim.AdamW",
        opt_configs={"betas": (0.9, 0.98), "weight_decay": 1e-2},
        lr_sch_configs={
            "lr": {
                "end": EPOCHS * (len(loader) + 1) // GRAD_ACC,
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
        name="EQ-VAE-ch16-randomaffine",
        # offline=True
    )
    trainer = pl.Trainer(
        logger=logger,
        devices=1,
        max_epochs=EPOCHS,
        precision="32",
        callbacks=[
            ModelCheckpoint(every_n_train_steps=500),
            ModelCheckpoint(every_n_epochs = 1, save_on_train_epoch_end = True),
            LearningRateMonitor(logging_interval="step"),
        ],
        log_every_n_steps=1,
    )
    trainer.fit(trainer_module, loader)
