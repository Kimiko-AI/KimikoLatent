from typing import *
from itertools import chain

from weakref import proxy, ref, ReferenceType

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleLayer(nn.Module):
    def __init__(self, name: str, orig_module: nn.Module):
        super().__init__()
        self.name = name
        self.orig_module: ReferenceType[nn.Module] = ref(orig_module)
        self.orig_forward = orig_module.forward
        self.multiplier = 1

    def apply_to(self):
        self.orig_module().forward = self.forward
