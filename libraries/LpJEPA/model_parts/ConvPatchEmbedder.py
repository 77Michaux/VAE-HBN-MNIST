import numpy as np
import torch
import torch.nn as nn


class ConvPatchEmbedder(nn.Module):
    def __init__(
        self, image_size: int = 512, patch_size: int = 32, in_channels: int = 1, d_model: int = 768
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.d_model = d_model
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=d_model, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.conv(x)
        return x.flatten(2).transpose(1, 2)  # [B, D, P_w, P_h] -> [B, P(w*h), D]


def get_1d_sin_cos_pos_embed_from_grid(pos: np.ndarray, d_model: int = 768):
    assert d_model % 2 == 0
    omega = np.arange(d_model // 2, dtype=float)
    omega /= d_model / 2.0
    omega = 1.0 / 1e4**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)  # [P, D/2]
    emb_cos = np.cos(out)  # [P, D/2]
    return np.concatenate([emb_sin, emb_cos], axis=1)  # [P, D]


def get_1d_sin_cos_pos_embed(grid_size, d_model, cls_token=True):
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sin_cos_pos_embed_from_grid(grid, d_model)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, d_model]), pos_embed], axis=0)  # [1+P, D]
    return pos_embed


def get_2d_sin_cos_pos_embed_from_grid(grid, d_model):
    assert d_model % 2 == 0
    emb_height = get_1d_sin_cos_pos_embed_from_grid(grid[0], d_model // 2)
    emb_width = get_1d_sin_cos_pos_embed_from_grid(grid[1], d_model // 2)
    return np.concatenate([emb_height, emb_width], axis=1)  # [H*W, D]


def get_2d_sin_cos_pos_embed(grid_size, d_model, cls_token: bool = True):
    grid_height = np.arange(grid_size, dtype=float)
    grid_width = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_width, grid_height)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sin_cos_pos_embed_from_grid(grid, d_model)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, d_model]), pos_embed], axis=0)
    return torch.from_numpy(pos_embed)
