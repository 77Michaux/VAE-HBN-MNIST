import logging
import random

import numpy as np
import torch

logger = logging.getLogger(f'tiny_jepa.{__name__}')


def generate_masks(
    input_size: tuple[int, int] | int = (512, 512),
    patch_size: int = 8,
    num_target_blocks: int = 4,
    context_scale_range: tuple[float, float] = (0.85, 1.0),
    context_aspect_ratio_range: tuple[float, float] = (0.85, 1),
    target_scale_range: tuple[float, float] = (0.15, 0.2),
    target_aspect_ratio_range: tuple[float, float] = (0.75, 1.5),
):

    # 1 ----------- Grid dimensions -----------

    # An image is split into a grid of patches, which become tokens for the vision transformer.
    # So 512 x 512 image with patch_size of 8 makes:
    # height = 512 / 8 = 64 patches high
    # width = 512 / 8 = 64 patches wide
    # num_patches = height * width = 4096 patches
    if not isinstance(input_size, tuple):
        logger.warning(f'input_size is an int. Making it a tuple ({input_size}, {input_size})')
        input_size = (input_size,) * 2
    image_h, image_w = input_size
    grid_h, grid_w = image_h // patch_size, image_w // patch_size
    num_patches = grid_h * grid_w

    # Context and targets cover patches, so we work with num_patches.
    # They are 1D arrays that will cover the "flattened" vector of patches.
    # Masks are initialized as False at every position.
    # To follow the paper, we start by generating target masks that can go anywhere.
    # Then we create a big context block, and we remove the overlap from the context block.
    target_mask = np.zeros(num_patches, dtype=bool)

    # 2 ----------- Generate target blocks -----------

    for _ in range(num_target_blocks):
        # Calculate target block size
        target_h, target_w, target_top, target_left = generate_block(
            num_patches, target_scale_range, target_aspect_ratio_range, grid_h, grid_w
        )

        # Add this block to overall target mask
        for i in range(target_top, target_top + target_h):
            for j in range(target_left, target_left + target_w):
                target_mask[i * grid_w + j] = True

    # 3 ----------- Generate context block -----------

    candidate_context_mask = np.zeros(num_patches, dtype=bool)
    context_h, context_w, context_top, context_left = generate_block(
        num_patches, context_scale_range, context_aspect_ratio_range, grid_h, grid_w
    )

    # For everything in the range, make it the context mask
    for i in range(context_top, context_top + context_h):
        for j in range(context_left, context_left + context_w):
            candidate_context_mask[i * grid_w + j] = True

    # 4 ----------- Remove overlap -----------

    context_mask = candidate_context_mask & ~target_mask

    return torch.from_numpy(context_mask), torch.from_numpy(target_mask)


def generate_block(
    num_patches: int,
    scale_range: tuple[float, float],
    aspect_ratio_range: tuple[float, float],
    grid_h: int,
    grid_w: int,
):

    # Calculate block size
    scale = random.uniform(*scale_range)
    aspect_ratio = random.uniform(*aspect_ratio_range)
    num_patches = int(num_patches * scale)
    height = int(np.sqrt(num_patches / aspect_ratio))
    width = int(height * aspect_ratio)
    height = max(1, min(height, grid_h))
    width = max(1, min(width, grid_w))

    # Select a top-left corner
    top = random.randint(0, grid_h - height)
    left = random.randint(0, grid_w - width)

    return height, width, top, left
