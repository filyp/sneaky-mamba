# %%
from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
# %%
delta = torch.arange(12).reshape(3, 4)
A = torch.arange(10, 30).reshape(4, 5)
# %%
einsum(delta, A, 'three four, four five -> three four five')
# %%
