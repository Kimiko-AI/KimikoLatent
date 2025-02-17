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

from diffusers import (
    AutoencoderKL,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from anyschedule import AnySchedule
from ..utils import instantiate
from ..utils.latent import pca_to_rgb
from ..transform import LatentTransformBase
from ..losses.adversarial import AdvLoss


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
        vae: AutoencoderKL | None = None,
        vae_compile: bool = False,
        lycoris_model: nn.Module | None = None,
        recon_loss: nn.Module = nn.MSELoss(),
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

        self.img_deprocess = img_deprocess or (lambda x: x)
        self.transform = latent_transform
        self.transform_prob = transform_prob
        self.log_interval = log_interval

        self.recon_loss = recon_loss
        self.adv_loss = adv_loss
        if isinstance(loss_weights, dict):
            self.recon_loss_weight = loss_weights.get("recon", 1.0)
            self.adv_loss_weight = loss_weights.get("adv", 0.5)
            self.kl_loss_weight = loss_weights.get("kl", 1e-6)
        else:
            (
                self.recon_loss_weight,
                self.adv_loss_weight,
                self.kl_loss_weight,
            ) = loss_weights

        self.grad_acc = grad_acc or 1
        self.current_grad_acc = 1
        self.epoch = 0
        self.opt_step = 0
        self.ema_loss = 0
        self.ema_decay = 0.999

        if lycoris_model is not None:
            self.lycoris_model.train()
            self.train_params = self.lycoris_model.parameters()
        else:
            self.train_params = [i for i in self.vae.parameters() if i.requires_grad]

        if self.adv_loss is not None and self.adv_loss_weight > 0:
            self.multiple_optimizers = True
            self.train_params = [self.train_params, self.adv_loss.parameters()]
            self.ema_d_loss = 0
            self.start_iter = self.adv_loss.start_iter - 1

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

        latent = dist.sample()
        if self.transform is not None and random.random() < self.transform_prob:
            x, latent = self.transform(x, latent)

        x_rec = self.vae.decode(latent)
        if hasattr(x_rec, "sample"):
            x_rec = x_rec.sample
        return x, x_rec, latent, dist

    def recon_step(self, x, x_rec, dist, g_opt, g_sch, batch_idx, grad_acc):
        recon_loss = self.recon_loss(x, x_rec)
        kl_loss = torch.sum(dist.kl()) / x_rec.numel()
        loss = recon_loss * self.recon_loss_weight + kl_loss * self.kl_loss_weight
        adv_loss = torch.tensor(0.0, device=x.device)
        if (
            self.adv_loss is not None
            and self.adv_loss_weight > 0
            and self.global_step >= self.start_iter
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
        self.log("train/adv_loss", adv_loss.item(), prog_bar=True, logger=True)
        self.log(
            "train/ema_loss",
            self.ema_loss,
            on_step=True,
            logger=True,
            prog_bar=True,
        )

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
    def log_images(self, org_x, x, x_rec, latent):
        from PIL import Image

        org_x = org_x.float()
        x = x.float()
        x_rec = x_rec.float()
        latent = latent.float()
        latent_rgb = pca_to_rgb(latent)

        x = F.interpolate(x, size=org_x.shape[2:], mode="bicubic")
        x_rec = F.interpolate(x_rec, size=org_x.shape[2:], mode="bicubic")
        latent_rgb = F.interpolate(latent_rgb, size=org_x.shape[2:], mode="nearest")

        concat_images = torch.cat(
            [org_x, x, x_rec, latent_rgb], dim=-2
        )  # concat on height
        # take first 8 sample and concat them on width
        concat_images = torch.cat(list(concat_images[:8]), dim=-1).cpu()
        concat_images = (
            (concat_images.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
        )
        concat_images = Image.fromarray(concat_images)
        if hasattr(self.logger, "log_image"):
            self.logger.log_image("train/samples", [concat_images])
        else:
            concat_images.save("train_samples.png")

    def training_step(self, batch, idx):
        x: torch.Tensor
        dist: DiagonalGaussianDistribution
        x, *_ = batch
        org_x = x.clone()
        x, x_rec, latent, dist = self.basic_step(x)
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
            )

        # VAE Loss
        self.recon_step(x, x_rec, dist, g_opt, g_sch, idx, grad_acc)

        ## Discriminator Loss
        d_opt = list(d_opt)
        d_sch = list(d_sch)
        if d_opt != [] and self.global_step >= self.start_iter:
            self.adv_step(x, x_rec, d_opt, d_sch, idx, grad_acc)
