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

class ResidualAttentionBlockDec(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn1 = nn.MultiheadAttention(d_model, n_head)
        self.attn2 = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.ln_3 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def self_attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn1(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def cross_attention(self, x: torch.Tensor, enc_output: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn2(x, enc_output, enc_output, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor):
        x = x + self.self_attention(self.ln_1(x))
        x = x + self.cross_attention(self.ln_2(x), enc_output)
        x = x + self.mlp(self.ln_3(x))
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
        return self.enc_layers(x)


class Decoder(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, patch_size: int, input_res: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_res // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.width = width
        self.layers = layers
        self.dec_layers = nn.ModuleList([ResidualAttentionBlockDec(width, heads, attn_mask) for _ in range(layers)])

        # self.ln_post = LayerNorm(width)
        # self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        for layer in self.dec_layers:
            x = layer(x, enc_output)
        x = x.permute(1, 0, 2)  # LND -> NLD (batch_size, num_patches, embed_dim)

        # x = self.ln_post(x[:, 0, :])

        # if self.proj is not None:
        #     x = x @ self.proj

        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_res: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.encoder = Encoder(width, layers, heads, patch_size, input_res)
        self.decoder = Decoder(width, layers, heads, patch_size, input_res)

        ###TODO:
        self.ln_post = LayerNorm(width)
        self.final_layer = nn.Linear(width, output_dim)

    def forward(self, p1: torch.Tensor, p2): # x is a pair of (p1, p2)
        # p1, p2 = x
        enc_output = self.encoder(p1)  #LND
        dec_output = self.decoder(p2, enc_output)  #NLD

        ###TODO:
        out = self.ln_post(dec_output[:, 0, :])
        out = self.final_layer(out)
        return out
        # final_output = None
test = VisionTransformer(224, 16, 768, 6, 8, 1)
# from torchsummary import summary
# summary(test, [(3, 224, 224), (3, 224, 224)])
out = test(torch.randn(1,3,224,224), torch.randn(1,3,224,224))
