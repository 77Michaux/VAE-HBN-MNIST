import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2


class GPUAugmentation(nn.Module):
    def __init__(self, crop_size=512, p_degrade=0.8, patch_size: int = 32):
        super().__init__()
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.degradations = v2.Compose(
            [
                v2.RandomApply(
                    [
                        v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                        v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                    ],
                    p=p_degrade,
                ),
                v2.Normalize(mean=[0.5], std=[0.225]),
            ]
        )

    def forward(self, x):
        # x: [B * V, 1, H, W]
        BV, C, H, W = x.shape

        cropped_images = []
        cropped_masks = []
        all_coords = []

        # 1. Générer le masque global de l'anatomie AVANT le crop
        # On considère comme "fond" ce qui est très proche du minimum (ex: < 0.05 après normalisation)
        # Ajuste le seuil (0.05) selon la façon dont tes pixels noirs sont codés.
        global_masks = (x > 0.05).float()
        if self.training:
            for i in range(BV):
                img = x[i : i + 1]
                mask = global_masks[i : i + 1]

                top, left, height, width = v2.RandomResizedCrop.get_params(
                    img, scale=(0.1, 1.0), ratio=(0.75, 1.33)
                )

                # On applique EXACTEMENT le même crop et resize sur l'image et son masque
                img_crop = v2.functional.resized_crop(
                    img,
                    top,
                    left,
                    height,
                    width,
                    size=(self.crop_size, self.crop_size),
                    antialias=True,
                )
                mask_crop = v2.functional.resized_crop(
                    mask,
                    top,
                    left,
                    height,
                    width,
                    size=(self.crop_size, self.crop_size),
                    interpolation=v2.InterpolationMode.NEAREST,
                )

                # Flips identiques
                if torch.rand(1).item() > 0.5:
                    img_crop = v2.functional.horizontal_flip(img_crop)
                    mask_crop = v2.functional.horizontal_flip(mask_crop)
                if torch.rand(1).item() > 0.5:
                    img_crop = v2.functional.vertical_flip(img_crop)
                    mask_crop = v2.functional.vertical_flip(mask_crop)

                cropped_images.append(img_crop)
                cropped_masks.append(mask_crop)
                all_coords.append([top, top + height, left, left + width])

            x = torch.cat(cropped_images, dim=0)
            x = self.degradations(x)

            masks = torch.cat(cropped_masks, dim=0)  # [B * V, 1, crop_size, crop_size]

            # 2. Downsample du masque à la taille de la grille des patchs (ex: 16x16)
            # On utilise un pool2d de la taille de ton patch_size (ex: 32)
            # Si un seul pixel du patch touche à l'anatomie, le patch entier est conservé (valeur 1)
            patch_size = self.patch_size  # À synchroniser avec ton hyperparamètre
            attn_mask = F.avg_pool2d(
                masks, kernel_size=patch_size, stride=patch_size
            )  # [B * V, 1, target_h, target_w]
            attn_mask = (attn_mask > 0.2).float()  # On garde le patch si > 20% d'anatomie
            # Flatten pour correspondre à la séquence N du Transformer -> [B * V, N]
            attn_mask = attn_mask.view(BV, -1)
            coords = torch.tensor(all_coords, dtype=torch.float32, device=x.device)
            return x, coords, attn_mask
        else:
            normalizer = v2.Normalize(mean=[0.5], std=[0.225])
            x = normalizer(x)
            right, bottom = x.shape[2:]
            global_mask = (x > 0.05).float()
            coords = torch.tensor([0, right, 0, bottom]).unsqueeze(0).repeat(BV, 1).to(x.device)
            attn_mask = F.avg_pool2d(
                global_mask, kernel_size=self.patch_size, stride=self.patch_size
            )  # [B * V, 1, target_h, target_w]
            attn_mask = (attn_mask > 0.05).float()
            attn_mask = attn_mask.view(BV, -1)
            return x, coords, attn_mask
