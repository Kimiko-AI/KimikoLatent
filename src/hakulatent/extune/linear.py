from typing import *
from itertools import chain

from weakref import ReferenceType

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ScaleLayer


class ScaleLinear(ScaleLayer):
    def __init__(
        self,
        name: str,
        orig_module: nn.Linear,
        in_dim: int,
        out_dim: int,
        outputs_groups=[],
        inputs_groups=[],
        same_mean=False,
        full_zero=False,
        not_zero=False,
    ):
        super().__init__(name, orig_module)
        assert not (full_zero and not_zero)
        self.orig_module: ReferenceType[nn.Linear]
        use_bias = orig_module.bias is not None

        if in_dim is None:
            in_dim = orig_module.in_features
        if out_dim is None:
            out_dim = orig_module.out_features

        self.in_feature = in_dim
        self.out_feature = out_dim
        self.orig_in = orig_module.in_features
        self.orig_out = orig_module.out_features

        self.addon_in = in_dim - self.orig_in
        self.addon_out = out_dim - self.orig_out
        self.same_mean = same_mean

        assert self.addon_in >= 0 and self.addon_out >= 0
        if outputs_groups:
            self.outputs_groups = True
            self.orig_out_split = tuple(i[0] for i in outputs_groups)
            self.addon_out_split = tuple(i[1] for i in outputs_groups)
            assert sum(self.orig_out_split) == self.orig_out
            assert sum(self.addon_out_split) == self.addon_out
        else:
            self.outputs_groups = False

        if inputs_groups:
            self.inputs_groups = True
            self.in_group_split = tuple(i[0] + i[1] for i in inputs_groups)
            self.orig_in_split = tuple(i[0] for i in inputs_groups)
            self.addon_in_split = tuple(i[1] for i in inputs_groups)
            assert sum(self.orig_in_split) == self.orig_in
            assert sum(self.addon_in_split) == self.addon_in
        else:
            self.inputs_groups = False

        if self.addon_in:
            self.addon_inout = nn.Linear(self.addon_in, out_dim, bias=use_bias)
            self.in_split = (self.orig_in, self.addon_in)
            if self.same_mean:
                nn.init.constant_(self.addon_inout.weight, 0)
                if use_bias:
                    nn.init.constant_(self.addon_inout.bias, 0)
            else:
                nn.init.constant_(
                    self.addon_inout.weight[: self.orig_out, : self.orig_in], 0
                )
                if use_bias:
                    nn.init.constant_(self.addon_inout.bias[: self.orig_out], 0)
        else:
            self.in_split = None

        if self.addon_out:
            self.addon_origout = nn.Linear(self.orig_in, self.addon_out, bias=use_bias)
            if not not_zero:
                nn.init.constant_(self.addon_origout.weight, 0)
                if use_bias:
                    nn.init.constant_(self.addon_origout.bias, 0)
            if self.same_mean:
                self.addon_origout.weight.data.data.copy_(
                    orig_module.weight.data.mean(dim=0)
                )
                if use_bias:
                    self.addon_origout.bias.data.data.copy_(
                        orig_module.bias.data.mean()
                    )
        else:
            self.addon_origout = None

        if full_zero:
            for p in self.parameters():
                torch.nn.init.constant_(p, 0)

    def generate_module(self):
        orig_weight = self.orig_module().weight.detach().cpu()
        orig_bias = self.orig_module().bias.detach().cpu()

        if self.addon_origout is not None:
            addon_weight = self.addon_origout.weight.detach().cpu()
            addon_bias = self.addon_origout.bias.detach().cpu()
            orig_weight = torch.cat([orig_weight, addon_weight], dim=0)
            orig_bias = torch.cat([orig_bias, addon_bias])

        if self.addon_inout is not None:
            addon_weight = self.addon_inout.weight.detach().cpu()
            addon_bias = self.addon_inout.bias.detach().cpu()
            orig_weight = torch.cat([orig_weight, addon_weight], dim=1)
            orig_bias = orig_bias + addon_bias

        new_layer = nn.Linear(self.in_feature, self.out_feature)
        new_layer.weight.data.copy_(orig_weight)
        new_layer.bias.data.copy_(orig_bias)
        return new_layer

    def forward(self, x):
        if self.in_split is not None:
            if self.inputs_groups:
                in_groups = torch.split(x, self.in_group_split, dim=-1)
                in_groups = [
                    torch.split(i, spliter, dim=-1)
                    for i, spliter in zip(
                        in_groups, zip(self.orig_in_split, self.addon_in_split)
                    )
                ]
                x = torch.cat([i[0] for i in in_groups], dim=-1)
                addon_x = torch.cat([i[1] for i in in_groups], dim=-1)
            else:
                x, addon_x = torch.split(x, self.in_split, dim=-1)
            # print(getattr(self, 'name', ''), torch.sum(torch.abs(addon_x)), 'use groups' if self.inputs_groups else 'no groups')
            addon_out = self.addon_inout(addon_x)
        else:
            addon_out = torch.tensor(0, dtype=x.dtype, device=x.device)
        orig_out = self.orig_forward(x)

        if self.addon_origout is not None:
            orig_addon_out = self.addon_origout(x)
            if self.same_mean:
                import math

                # print(
                #     'same mean with addon out',
                #     torch.mean(orig_out),
                #     torch.mean(orig_addon_out),
                #     math.sqrt((orig_out.shape[-1]+orig_addon_out.shape[-1])/orig_out.shape[-1]),
                #     torch.std(orig_out)/torch.std(torch.cat([orig_out, orig_addon_out], dim=-1))
                # )
            orig_out = torch.cat([orig_out, orig_addon_out], dim=-1)

        out = orig_out + addon_out

        if self.outputs_groups:
            orig, addon = torch.split(out, [self.orig_out, self.addon_out], dim=-1)
            orig_groups = torch.split(orig, self.orig_out_split, dim=-1)
            addon_groups = torch.split(addon, self.addon_out_split, dim=-1)
            out = torch.cat(list(chain(*zip(orig_groups, addon_groups))), dim=-1)

        # print(addon_out.shape, x.shape, out.shape)
        # assert torch.min(torch.abs(out)) > 0 or not 'attentions' in self.name, f'{self.name} {torch.min(torch.abs(out))}'
        return out


class ScaleConv2d(ScaleLayer):
    def __init__(
        self,
        name,
        orig_module: nn.Conv2d,
        in_ch,
        out_ch,
        outputs_groups=[],
        inputs_groups=[],
        same_mean=False,
    ):
        super().__init__(name, orig_module)
        if in_ch is None:
            in_ch = orig_module.in_channels
        if out_ch is None:
            out_ch = orig_module.out_channels

        self.orig_module: ReferenceType[nn.Conv2d]
        use_bias = orig_module.bias is not None

        self.in_channels = in_ch
        self.out_channels = out_ch
        self.kernel_size = orig_module.kernel_size
        self.stride = orig_module.stride
        self.padding = orig_module.padding

        self.orig_in = orig_module.in_channels
        self.orig_out = orig_module.out_channels

        self.addon_in = in_ch - self.orig_in
        self.addon_out = out_ch - self.orig_out
        self.same_mean = same_mean

        assert self.addon_in >= 0 and self.addon_out >= 0
        if outputs_groups:
            self.outputs_groups = True
            self.orig_out_split = tuple(i[0] for i in outputs_groups)
            self.addon_out_split = tuple(i[1] for i in outputs_groups)
            assert sum(self.orig_out_split) == self.orig_out
            assert sum(self.addon_out_split) == self.addon_out
        else:
            self.outputs_groups = False

        if inputs_groups:
            self.inputs_groups = True
            self.in_group_split = tuple(i[0] + i[1] for i in inputs_groups)
            self.orig_in_split = tuple(i[0] for i in inputs_groups)
            self.addon_in_split = tuple(i[1] for i in inputs_groups)
            assert sum(self.orig_in_split) == self.orig_in
            assert sum(self.addon_in_split) == self.addon_in
        else:
            self.inputs_groups = False

        if self.addon_in:
            self.addon_inout = nn.Conv2d(
                self.addon_in,
                out_ch,
                self.kernel_size,
                self.stride,
                self.padding,
                bias=use_bias,
            )
            self.in_split = (self.orig_in, self.addon_in)
            if self.same_mean:
                nn.init.constant_(self.addon_inout.weight, 0)
                if use_bias:
                    nn.init.constant_(self.addon_inout.bias, 0)
            else:
                nn.init.constant_(
                    self.addon_inout.weight[: self.orig_out, : self.orig_in], 0
                )
                if use_bias:
                    nn.init.constant_(self.addon_inout.bias[: self.orig_out], 0)
        else:
            self.addon_inout = None
            self.in_split = None

        if self.addon_out:
            self.addon_origout = nn.Conv2d(
                self.orig_in,
                self.addon_out,
                self.kernel_size,
                self.stride,
                self.padding,
                bias=use_bias,
            )
            nn.init.constant_(self.addon_origout.weight, 0)
            if use_bias:
                nn.init.constant_(self.addon_origout.bias, 0)
            if self.same_mean:
                self.addon_origout.weight.data.data.copy_(
                    orig_module.weight.data.mean(dim=0)
                )
                if use_bias:
                    self.addon_origout.bias.data.data.copy_(
                        orig_module.bias.data.mean()
                    )
        else:
            self.addon_origout = None

    def generate_module(self):
        orig_weight = self.orig_module().weight.detach().cpu()
        orig_bias = self.orig_module().bias.detach().cpu()

        if self.addon_origout is not None:
            addon_weight = self.addon_origout.weight.detach().cpu()
            addon_bias = self.addon_origout.bias.detach().cpu()
            orig_weight = torch.cat([orig_weight, addon_weight], dim=0)
            orig_bias = torch.cat([orig_bias, addon_bias])

        if self.addon_inout is not None:
            addon_weight = self.addon_inout.weight.detach().cpu()
            addon_bias = self.addon_inout.bias.detach().cpu()
            orig_weight = torch.cat([orig_weight, addon_weight], dim=1)
            orig_bias = orig_bias + addon_bias

        new_layer = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
        )
        new_layer.weight.data.copy_(orig_weight)
        new_layer.bias.data.copy_(orig_bias)
        return new_layer

    def forward(self, x):
        if self.in_split is not None:
            if self.inputs_groups:
                in_groups = torch.split(x, self.in_group_split, dim=1)
                in_groups = [
                    torch.split(i, spliter, dim=1)
                    for i, spliter in zip(
                        in_groups, zip(self.orig_in_split, self.addon_in_split)
                    )
                ]
                x = torch.cat([i[0] for i in in_groups], dim=1)
                addon_x = torch.cat([i[1] for i in in_groups], dim=1)
                # print('in_groups', self.in_group_split)
            else:
                x, addon_x = torch.split(x, self.in_split, dim=1)  # [..., c, h, w]
            # print(
            #     getattr(self, 'name', ''),
            #     torch.mean(torch.abs(addon_x)),
            #     'use groups' if self.inputs_groups else 'no groups',
            #     list(zip(self.orig_in_split, self.addon_in_split)) if self.inputs_groups else self.in_split
            # )
            addon_out = self.addon_inout(addon_x)
        else:
            addon_out = torch.tensor(0, dtype=x.dtype, device=x.device)

        orig_out = self.orig_forward(x)

        if self.addon_origout is not None:
            orig_addon_out = self.addon_origout(x)
            # print(getattr(self, 'name', ''), 'addon_out', torch.mean(torch.abs(orig_addon_out)))
            # if self.same_mean:
            #     import math
            #     print(
            #         'same mean with addon out',
            #         torch.mean(orig_out),
            #         torch.mean(orig_addon_out),
            #         math.sqrt((orig_out.shape[1]+orig_addon_out.shape[1])/orig_out.shape[1]),
            #         torch.mean(torch.std(orig_out, dim=1)/torch.std(torch.cat([orig_out, orig_addon_out], dim=1), dim=1))
            #     )
            orig_out = torch.cat([orig_out, orig_addon_out], dim=1)

        out = orig_out + addon_out

        if self.outputs_groups:
            orig, addon = torch.split(out, [self.orig_out, self.addon_out], dim=1)
            orig_groups = torch.split(orig, self.orig_out_split, dim=1)
            addon_groups = torch.split(addon, self.addon_out_split, dim=1)
            out = torch.cat(list(chain(*zip(orig_groups, addon_groups))), dim=1)

        # print(addon_out.shape, x.shape, out.shape)
        return out
