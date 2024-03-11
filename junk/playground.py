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
model.parameters
for name, param in model.named_parameters():
    print(name)

# %%
model.layers[0].mixer.shrink
# trainer.train_dataset[0]

# %%
task_lenghts = list(range(10))
eval_dataset = DirectTasksDataset(tokenizer, task_lenghts)
ex = eval_dataset[-2]
ex
out = model(input_ids=ex["input_ids"].reshape(1, -1).to("cuda"))
ans = out.argmax(axis=2)
# %%
print(tokenizer.decode(ex["input_ids"]))
print(tokenizer.decode(ex["labels"]))
print(tokenizer.decode(ans.flatten()))
# %%
x = model.embedding(inputs)
m = model.layers[0]
x = m.norm(x)
x.shape

# %%
from einops import einsum, rearrange, repeat

self = m.mixer
(b, l, d) = x.shape

x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
(x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

x = rearrange(x, 'b l d_in -> b d_in l')
x = self.conv1d(x)[:, :, :l]
x = rearrange(x, 'b d_in l -> b l d_in')

# %%
x = F.silu(x)
x

# %%
b, l, d = x.shape

x_small = einsum(x, self.shrink, "b l d, d d_sh -> b l d_sh")

_h = torch.zeros((b, self.d_h), device=x.device)
_hs = []
# %%
i = 0
_x = x_small[:, i, :]
_x
# %%

_h = einsum(_h, self.T, _x, "b h_in, h_in d_sh h_out, b d_sh -> b h_out")
_h

# %%

# ReLU
_h = F.relu(_h)
_hs.append(_h)
# %%
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer.zero_grad()   # Reset gradients
# outputs = model(inputs) # Forward pass
# outputs

# loss = criterion(outputs, targets) # Compute loss
# loss.backward()         # Backpropagation
# optimizer.step()        # Update parameters

