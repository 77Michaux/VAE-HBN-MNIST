import torch
import torch.nn as nn
import torch.nn.functional as F

from rt_jepa.downstream.linear_probe import Perceptron
from rt_jepa.model.ConvPatchEmbedder import ConvPatchEmbedder
from rt_jepa.model.model_parts import Encoder, Projector, RectifiedLpLoss, VISRegLoss
from rt_jepa.model.transformers import TransformerBlock
from rt_jepa.utils.transforms import GPUAugmentation


def _init_weights(m):
    """
    Initialize the weights of the model.

    Parameters
    ----------
    m: nn.Module

    Returns
    -------
    """
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Embedding):
        nn.init.trunc_normal_(m.weight, std=0.02)


class LpJEPA(nn.Module):
    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 32,
        in_channels: int = 1,
        d_model: int = 256,
        num_views: int = 5,
        proj_dim: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        mlp_ratio: int = 4,
        mlp_drop: float = 0.0,
        drop_path_rate: float = 0.3,
        lambda_reg: float = 0.3,
        mu: float = -0.5,
        sigma: float = 1.0,
        p: int = 1,
        K: int = 256,
        rdm: bool = False,
        num_projs: int = 256,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.mlp_ratio = mlp_ratio
        self.mlp_drop = mlp_drop
        self.drop_path_rate = drop_path_rate
        self.lambda_reg = lambda_reg
        self.mu = mu
        self.num_views = num_views
        self.gpu_augmentation = GPUAugmentation(
            crop_size=self.image_size, patch_size=self.patch_size
        )
        self.rdm = rdm

        self.patch_embedder = ConvPatchEmbedder(
            image_size=self.image_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            d_model=self.d_model,
        )
        if drop_path_rate > 0:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_encoder_layers)]
        else:
            dpr = [0.0] * num_encoder_layers
        encoder_blocks: nn.ModuleList = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    mlp_drop=self.mlp_drop,
                    drop_path=dpr[i],
                )
                for i in range(num_encoder_layers)
            ]
        )
        self.encoder: Encoder = Encoder(
            encoder_blocks,
            d_model=self.d_model,
            num_heads=self.num_heads,
            img_size=self.image_size,
            patch_size=self.patch_size,
        )
        self.projector = Projector(
            input_dim=self.d_model,
            hidden_dim=int(self.d_model * mlp_ratio),
            proj_dim=self.proj_dim,
            rdm=self.rdm,
        )

        if self.rdm:
            self.dist_cost = RectifiedLpLoss(
                latent_dim=self.proj_dim, num_projs=num_projs, mu=self.mu, sigma=sigma, p=p
            )
        else:
            self.dist_cost = VISRegLoss(D_proj=self.proj_dim, K=K)
        self.class_predictor = Perceptron(self.proj_dim, 1)
        self.apply(_init_weights)

    @staticmethod
    def _prepare_class(c: torch.Tensor, V: int):
        return c.repeat(V)

    def _prepare_tokens(self, x: torch.Tensor):
        if self.training:
            V: int = self.num_views
        else:
            V = 1
        # x: [B, C, W, H]
        x = x.repeat(V, 1, 1, 1)  # [B*V, C, W, H]
        x, crop_coords, attn_mask = self.gpu_augmentation(x)  # Récupère le masque ici
        tokens = self.patch_embedder(x)  # [B*V, P, D]
        return tokens, crop_coords, attn_mask

    def embed_tokens(self, x):
        B = x.shape[0]
        if self.training:
            B = x.shape[0]
            V = self.num_views
            tokens, crop_coords, attn_mask = self._prepare_tokens(x)
            encoded_seq = self.encoder(
                tokens, crop_coords=crop_coords, attn_mask=attn_mask
            )  # [B*V, P, D]
            P, D = encoded_seq.shape[1:]
            projs = self.projector(encoded_seq.flatten(0, 1))  # [B*V*P, D]
            return projs.view(B, V, P, D)  # [B, V, P, D]
        else:
            V = 1
            tokens, crop_coords, attn_mask = self._prepare_tokens(x)
            encoded_seq = self.encoder(tokens, crop_coords=crop_coords, attn_mask=attn_mask)
            P, D = encoded_seq.shape[1:]
            z_proj = self.projector(encoded_seq.flatten(0, 1))
            z = z_proj.view(B, V, P, D)
            # z = z.squeeze(1)
            return z

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None):
        B = x.shape[0]
        V = self.num_views if self.training else 1
        proj = self.embed_tokens(x)  # [B, V, P, D]
        proj_mean = proj.mean(dim=1, keepdim=True)  # [B, 1, P, D]
        P, D = proj.shape[2:]
        inv_loss = F.mse_loss(proj, proj_mean.expand_as(proj))
        dist_loss = self.dist_cost(proj.reshape(B * V * P, -1), reduction='mean')
        with torch.no_grad():
            sparsity = torch.tensor((proj == 0).float().mean().item())
        if c is None:
            loss = inv_loss + self.lambda_reg * dist_loss
            return loss, inv_loss, dist_loss, sparsity
        else:
            c = self._prepare_class(c, V)
            class_pred = self.class_predictor(proj.reshape(B * V, -1))
            class_pred_loss = F.binary_cross_entropy(class_pred, c)
            loss = (0.8 * inv_loss + 0.2 * class_pred_loss) + self.lambda_reg * dist_loss
            return loss, inv_loss, dist_loss, class_pred_loss, sparsity
