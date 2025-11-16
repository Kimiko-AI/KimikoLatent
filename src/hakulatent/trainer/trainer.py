import os
import random
from typing import Any, Iterator, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sch
import lightning.pytorch as pl
import wandb
from ..losses.vf_loss import VFLoss
from diffusers import (
    AutoencoderKL,
)
import lejepa
import random
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from anyschedule import AnySchedule
from ..utils import instantiate
from ..utils.latent import pca_to_rgb
from ..transform import LatentTransformBase
from ..losses.adversarial import AdvLoss
from ..losses.wavelet_loss import SWTLoss
from ..losses.vf_loss import VFLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvNeXtFeatureLoss:
    def __init__(self, model_name="convnext_large.dinov3_lvd1689m", device="cpu"):
        self.device = device

        self.model = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True
        ).to(device).eval()
        for p in self.model.parameters():
          p.requires_grad = False

        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.imagenet_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def normalize_from_minus1_1(self, x):
        x = (x + 1) / 2                     # back to [0,1]
        return (x - self.imagenet_mean) / self.imagenet_std

    def extract_all_features(self, x):
        return self.model(x)

    def feature_loss(self, f1, f2):
        a = F.normalize(f1, dim=1)
        b = F.normalize(f2, dim=1)
        return F.mse_loss(a, b)

    def loss_from_tensors(self, x1, x2):
        x1 = self.normalize_from_minus1_1(x1.to(self.device))
        x2 = self.normalize_from_minus1_1(x2.to(self.device))

        feats1 = self.extract_all_features(x1)
        feats2 = self.extract_all_features(x2)

        total = 0.0
        for a, b in zip(feats1, feats2):
            total = total + self.feature_loss(a, b)

        return total


class BaseTrainer(pl.LightningModule):
    def __init__(
            self,
            *args,
            name: str = "",
            lr: float = 1e-5,
            optimizer: type[optim.Optimizer] = optim.AdamW,
            opt_configs: dict[str, Any] | list[dict[str, Any]] = {
                "weight_decay": 0.01,
                "betas": (0.9, 0.999),
            },
            lr_sch_configs: dict[str, dict[str, Any]] | list[dict[str, Any]] = {
                "lr": {
                    "end": 10000,
                    "value": 1.0,
                    "min_value": 0.01,
                    "mode": "cosine",
                    "warmup": 1000,
                }
            },
            multiple_optimizers: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.name = name
        self.train_params: Iterator[nn.Parameter] = None
        self.optimizer = instantiate(optimizer)
        self.opt_configs = opt_configs
        self.lr = lr
        self.lr_sch_configs = lr_sch_configs
        self.multiple_optimizers = multiple_optimizers

    def log(self, *args, **kwargs):
        if self._trainer is not None:
            super(BaseTrainer, self).log(*args, **kwargs)

    def configure_optimizers(self):
        assert self.train_params is not None

        if not self.multiple_optimizers:
            self.train_params = [self.train_params]
        if not isinstance(self.lr, list):
            self.lr = [self.lr] * len(self.train_params)
        if not isinstance(self.optimizer, list):
            self.optimizer = [self.optimizer] * len(self.train_params)
        if not isinstance(self.opt_configs, list):
            self.opt_configs = [self.opt_configs] * len(self.train_params)
        if not isinstance(self.lr_sch_configs, list):
            self.lr_sch_configs = [self.lr_sch_configs] * len(self.train_params)

        results = []
        for train_params, lr, optimizer, opt_configs, lr_sch_configs in zip(
                self.train_params,
                self.lr,
                self.optimizer,
                self.opt_configs,
                self.lr_sch_configs,
        ):
            optimizer = optimizer(train_params, lr=lr, **opt_configs)

            lr_scheduler = None
            if lr_sch_configs:
                lr_scheduler = AnySchedule(optimizer, config=lr_sch_configs)

            if lr_scheduler is None:
                results.append(optimizer)
            else:
                results.append(
                    {
                        "optimizer": optimizer,
                        "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
                    }
                )
        return results


class LatentTrainer(BaseTrainer):
    automatic_optimization = False

    def __init__(
            self,
            vae,
            vae_compile: bool = False,
            lycoris_model: nn.Module | None = None,
            recon_loss: nn.Module = nn.MSELoss(),
            latent_loss: nn.Module | None = None,
            adv_loss: AdvLoss | None = None,
            img_deprocess: Callable | None = None,
            loss_weights: tuple[int] = (1.0, 0.5, 1e-6),
            latent_transform: LatentTransformBase | None = None,
            transform_prob: float = 0.5,
            log_interval: int = 500,
            *args,
            name: str = "",
            lr: float = 1e-5,
            lr_disc: float = None,
            grad_acc: int | dict[int, int] = 1,
            optimizer: type[optim.Optimizer] = optim.AdamW,
            opt_configs: dict[str, Any] = {
                "weight_decay": 0.01,
                "betas": (0.9, 0.999),
            },
            lr_sch_configs: dict[str, dict[str, Any]] = {
                "lr": {
                    "end": 10000,
                    "value": 1.0,
                    "min_value": 0.01,
                    "mode": "cosine",
                    "warmup": 1000,
                }
            },
            full_config: dict[str, Any] = {},
            **kwargs,
    ):
        super().__init__(
            *args,
            name=name,
            lr=[lr, lr_disc] if lr_disc is not None else lr,
            optimizer=optimizer,
            opt_configs=opt_configs,
            lr_sch_configs=lr_sch_configs,
            **kwargs,
        )
        self.save_hyperparameters(
            ignore=[
                "vae",
                "lycoris_model",
                "recon_loss",
                "adv_loss",
                "img_deprocess",
                "latent_transform",
                "args",
                "kwargs",
            ]
        )
        if vae_compile and vae is not None:
            vae = torch.compile(vae)
        if lycoris_model is not None:
            vae.requires_grad_(False).eval()
        self.vae = vae
        self.lycoris_model = lycoris_model
        self.vf_loss = VFLoss()
        self.img_deprocess = img_deprocess or (lambda x: x)
        self.transform = latent_transform
        self.transform_prob = transform_prob
        self.log_interval = log_interval
        self.convnext_criterion = ConvNeXtFeatureLoss()

        self.latent_loss = latent_loss
        self.recon_loss = recon_loss
        self.swt = SWTLoss(loss_weight_ll=0.05, loss_weight_lh=0.025, loss_weight_hl=0.025, loss_weight_hh=0.02)
        self.vf_loss = VFLoss()
        self.vf_loss.proj.reset_parameters()
        self.vf_loss.proj.requires_grad_(True)
        self.adv_loss = adv_loss
        if isinstance(loss_weights, dict):
            self.recon_loss_weight = loss_weights.get("recon", 1.0)
            self.adv_loss_weight = loss_weights.get("adv", 0.5)
            self.kl_loss_weight = loss_weights.get("kl", 1e-6)
            self.reg_loss_weight = loss_weights.get("reg", 1.0)
            self.cycle_loss_weight = loss_weights.get("cycle", 0)
            self.swt_loss_weight = loss_weights.get("swt", 1)

        else:
            (
                self.recon_loss_weight,
                self.adv_loss_weight,
                self.kl_loss_weight,
                self.reg_loss_weight,
                self.cycle_loss_weight,
                self.swt_loss_weight,
            ) = loss_weights

        self.grad_acc = grad_acc or 1
        self.current_grad_acc = 1
        self.epoch = 0
        self.opt_step = 0
        self.ema_loss = 0
        self.ema_decay = 0.999

        if lycoris_model is not None:
            self.lycoris_model.train()
            self.train_params = list(self.lycoris_model.parameters())
        else:
            self.train_params = [i for i in self.vae.parameters() if i.requires_grad] # + [i for i in self.vf_loss.proj.parameters() if i.requires_grad]

        if self.latent_loss is not None:
            self.train_params = self.train_params + list(self.latent_loss.parameters())

        if self.adv_loss is not None and self.adv_loss_weight > 0:
            self.multiple_optimizers = True
            self.train_params = [self.train_params, self.adv_loss.parameters()]
            self.ema_d_loss = 0
            self.start_iter = self.adv_loss.start_iter - 1
        univariate_test = lejepa.univariate.EppsPulley(n_points=17)
        self.lejepa_loss = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=1024
        )
    def on_train_epoch_start(self):
        if isinstance(self.grad_acc, dict):
            self.current_grad_acc = self.grad_acc.get(
                self.current_epoch, self.current_grad_acc
            )
        else:
            self.current_grad_acc = self.grad_acc

    def on_train_epoch_end(self) -> None:
        self.epoch += 1
        if self.lycoris_model is not None:
            dir = "./lycoris_weight"
            epoch = self.epoch
            if self._trainer is not None:
                trainer = self._trainer
                epoch = trainer.current_epoch
                if len(trainer.loggers) > 0:
                    if trainer.loggers[0].save_dir is not None:
                        save_dir = trainer.loggers[0].save_dir
                    else:
                        save_dir = trainer.default_root_dir
                    name = trainer.loggers[0].name
                    version = trainer.loggers[0].version
                    version = (
                        version if isinstance(version, str) else f"version_{version}"
                    )
                    dir = os.path.join(save_dir, str(name), version, "lycoris_weight")
                else:
                    # if no loggers, use default_root_dir
                    dir = os.path.join(trainer.default_root_dir, "lycoris_weight")
            os.makedirs(dir, exist_ok=True)
            model_weight = {
                k: v for k, v in self.unet.named_parameters() if v.requires_grad
            }
            lycoris_weight = self.lycoris_model.state_dict() | model_weight
            torch.save(lycoris_weight, os.path.join(dir, f"epoch={epoch}.pt"))

    def basic_step(self, x):
        dist = self.vae.encode(x)
        if hasattr(dist, "latent_dist"):
            dist = dist.latent_dist
        dist.deterministic = False

        latent = dist.latent
        origin = latent.clone()

        if self.transform is not None and random.random() < self.transform_prob:
            x, latent = self.transform(x, latent)

        num_channels = latent.shape[1]
        batch_size = latent.shape[0]

        mask = torch.ones_like(latent)
        start_channels = []  # store only start indices

        for i in range(batch_size):
            # Randomly choose start channel
            #start_ch = random.randint(1, num_channels - 1)
            start_ch = 32
            # Mask from start_ch to end
            mask[i, start_ch:, :, :] = 0
            # Record the start index
            start_channels.append(start_ch)

        #latent = latent * mask

        x_rec = self.vae.decode(latent)
        if hasattr(x_rec, "sample"):
            x_rec = x_rec.sample
        if x.shape[2:] != x_rec.shape[2:]:
            x = F.interpolate(x, size=x_rec.shape[2:], mode="bicubic")

        # Return the start indices instead of lists or tensor mask
        return origin, x, x_rec, latent, dist, start_channels

    def recon_step(self, x, x_rec, latent, dist, g_opt, g_sch, batch_idx, grad_acc, imags):
        recon_loss = self.recon_loss(x, x_rec)
        vf_loss = self.vf_loss(latent, imags)
        #vf_loss = torch.tensor(0.0, device=x.device)
        # --- Cycle loss ---
        cycle_loss = torch.tensor(0.0, device=x.device)
        if self.cycle_loss_weight > 0:
            with torch.no_grad():  # detach to avoid gradient loops on the VAE
                latent_cycle = self.vae.encode(x).latent_dist.sample()
                latent_cycle2 = self.vae.encode(x_rec).latent_dist.sample()

            cycle_loss = F.mse_loss(latent_cycle2, latent_cycle)

        #kl_loss = torch.sum(dist.kl()) / x_rec.numel()
        #jepa_loss = self.lejepa_loss(dist.latent.reshape(x.shape[0], -1))
        kl_loss = torch.tensor(0.0, device=x.device)
        reg_loss = self.convnext_criterion.loss_from_tensors(x, x_rec)
        swt = self.swt(x_rec, x)
        if self.latent_loss is not None:
            reg_loss += self.latent_loss(latent)
        loss = (
                recon_loss * self.recon_loss_weight
                + kl_loss * self.kl_loss_weight
                #+ jepa_loss * self.kl_loss_weight
                + reg_loss
                + cycle_loss * self.cycle_loss_weight
                + swt * 0.1 + vf_loss )
        adv_loss = torch.tensor(0.0, device=x.device)
        if (
                self.adv_loss is not None
                and self.adv_loss_weight > 0
        ):
            adv_loss = self.adv_loss(x, x_rec, recon_loss, 0, self.vae.get_last_layer())
        loss += adv_loss * self.adv_loss_weight
        self.manual_backward(loss / grad_acc)

        if (batch_idx + 1) % grad_acc == 0:
            self.clip_gradients(g_opt, 0.1)
            g_opt.step()
            g_opt.zero_grad()
            if g_sch is not None:
                g_sch.step()

        ema_decay = min(self.opt_step / (10 + self.opt_step), self.ema_decay)
        self.ema_loss = ema_decay * self.ema_loss + (1 - ema_decay) * loss.item()
        self.opt_step += 1
        self.log("train/loss", loss.item(), on_step=True, logger=True)
        self.log(
            "train/recon_loss",
            recon_loss.item(),
            on_step=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train/kl_loss", kl_loss.item(), on_step=True, prog_bar=True, logger=True
        )
        self.log("train/cycle_loss", cycle_loss.item(), on_step=True, prog_bar=True, logger=True)
        self.log("train/swt_loss", swt.item(), on_step=True, prog_bar=True, logger=True)

        self.log(
            "train/vf_loss", vf_loss.item(), on_step=True, prog_bar=True, logger=True
        )
        self.log("train/adv_loss", adv_loss.item(), prog_bar=True, logger=True)
        self.log(
            "train/ema_loss",
            self.ema_loss,
            on_step=True,
            logger=True,
            prog_bar=True,
        )

    def on_load_checkpoint(self, checkpoint):
        # Keep only the state_dict
        keys_to_keep = ["state_dict"]
        for k in list(checkpoint.keys()):
            if k not in keys_to_keep:
                checkpoint.pop(k, None)

    def adv_step(self, x, x_rec, d_opt, d_sch, batch_idx, grad_acc):
        d_opt = d_opt[0]
        d_sch = d_sch[0]
        x_rec = x_rec.detach()
        adv_loss = self.adv_loss(x, x_rec, None, 1, None)
        loss = adv_loss
        self.manual_backward(loss / grad_acc)

        if (batch_idx + 1) % grad_acc == 0:
            self.clip_gradients(d_opt, 0.1)
            d_opt.step()
            d_opt.zero_grad()
            if d_sch is not None:
                d_sch.step()

        ema_decay = min(self.opt_step / (10 + self.opt_step), self.ema_decay)
        self.ema_d_loss = ema_decay * self.ema_d_loss + (1 - ema_decay) * loss.item()
        self.opt_step += 1
        self.log("train/disc_loss", loss.item(), on_step=True, logger=True)
        self.log(
            "train/ema_d_loss",
            self.ema_d_loss,
            on_step=True,
            logger=True,
            prog_bar=True,
        )

    @torch.no_grad()
    @torch.autocast("cuda", enabled=False)
    def log_images(self, org_x, x, x_rec, latent, mask_channels):
        from PIL import Image

        org_x = org_x.float()[:8]
        x = x.float()[:8]
        x_rec = x_rec.float()[:8, : x.size(1)]
        latent = latent.float()[:8]
        try:
            latent_rgb = pca_to_rgb(latent)
        except Exception as e:
            latent_rgb =  x
            print(e)
        x = F.interpolate(x, size=org_x.shape[2:], mode="bicubic")
        x_rec = F.interpolate(x_rec, size=org_x.shape[2:], mode="bicubic")
        latent_rgb = F.interpolate(latent_rgb, size=org_x.shape[2:], mode="nearest")

        concat_images = torch.cat(
            [org_x, x, x_rec, latent_rgb], dim=-2
        )  # concat on height
        # take first 8 sample and concat them on width
        concat_images = torch.cat(list(concat_images), dim=-1).cpu()
        concat_images = (
            (concat_images.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
        )
        concat_images = Image.fromarray(concat_images)
        if mask_channels is not None:
            caption = f"Masked channels: {mask_channels}"
        if hasattr(self.logger, "log_image"):
            self.logger.log_image("train/samples", [concat_images], caption=[caption])
        else:
            concat_images.save("train_samples.png")

    def training_step(self, batch, idx):
        x: torch.Tensor
        dist: DiagonalGaussianDistribution
        x, dino = batch
        org_x = x.clone()
        origin_latent, x, x_rec, latent, dist, mask_channels = self.basic_step(x)
        try:
            g_opt, *d_opt = self.optimizers()
            g_sch, *d_sch = self.lr_schedulers()
        except:
            g_opt = self.optimizers()
            g_sch = self.lr_schedulers()
            d_opt = []
            d_sch = []
        grad_acc = self.grad_acc

        if idx % self.log_interval == 0:
            self.log_images(
                self.img_deprocess(org_x),
                self.img_deprocess(x),
                self.img_deprocess(x_rec),
                latent,
                mask_channels
            )

        # VAE Loss
        self.recon_step(x, x_rec, latent, dist, g_opt, g_sch, idx, grad_acc, dino)

        ## Discriminator Loss
        d_opt = list(d_opt)
        d_sch = list(d_sch)
        if d_opt != [] and self.global_step >= self.start_iter:
            self.adv_step(x, x_rec, d_opt, d_sch, idx, grad_acc)
