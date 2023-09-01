from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlockEnc(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, patch_size, input_res, attn_mask: torch.Tensor = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_res // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.width = width
        self.layers = layers
        self.enc_layers = nn.Sequential(*[ResidualAttentionBlockEnc(width, heads, attn_mask) for _ in range(layers)])

        # self.ln_post = LayerNorm(width)
        # self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        return self.enc_layers(x) #LND



class VisionTransformer(nn.Module):
    def __init__(self, input_res: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        ##TODO: we can train different vit for two images, or use one to encode both; or freeze vit
        self.ori_vit1 = Encoder(width, layers, heads, patch_size, input_res)
        self.ori_vit2 = Encoder(width, layers, heads, patch_size, input_res)
        
        self.attn1 = nn.MultiheadAttention(width, heads)
        self.attn2 = nn.MultiheadAttention(width, heads)
        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(width, width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(width * 4, width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(width, width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(width * 4, width))
        ]))
        self.ln_1 = LayerNorm(width)
        self.ln_2 = LayerNorm(width)
        self.ln_final = LayerNorm(width*2)
        self.linear1 = nn.Linear(width*2, width)
        self.linear2 = nn.Linear(width, output_dim)


    def forward(self, p1: torch.Tensor, p2): # x is a pair of (p1, p2)
        # p1, p2 = x
        enc1_output = self.ori_vit1(p1)  #LND
        enc2_output = self.ori_vit2(p2)  #LND
        out1 = self.attn1(enc2_output, enc1_output, enc1_output, need_weights=False, attn_mask=None)[0]
        out1 = self.mlp1(self.ln_1(out1))
        out2 = self.attn2(enc1_output, enc2_output, enc2_output, need_weights=False, attn_mask=None)[0]
        out2 = self.mlp2(self.ln_2(out2))
        out = self.ln_final(torch.cat((out1, out2), dim=1))
        out = self.linear1(out)
        out = self.linear2(out)
        return out


test = VisionTransformer(224, 16, 768, 6, 8, 1)
# from torchsummary import summary
# summary(test, [(3, 224, 224), (3, 224, 224)])
out = test(torch.randn(1,3,224,224), torch.randn(1,3,224,224))
