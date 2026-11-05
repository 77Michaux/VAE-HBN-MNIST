import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class SIGReg(torch.nn.Module):
    """
    Class that calculates Sketch-Isotropic Gaussian distribution fitness using the Cramer-World
    theorem. It checks normality on random projections on segments.
    """

    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(start=0, end=3, steps=knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer(name='t', tensor=t)
        self.register_buffer(name='phi', tensor=window)
        self.register_buffer(name='weights', tensor=weights * window)

    def forward(self, proj, reduction: str = 'mean'):
        # This squashes (B, L, D) or (B, H, W, D) into (N, D)
        # where N is the total number of vectors in the batch.
        if proj.ndim > 2:
            proj = proj.reshape(-1, proj.size(-1))
        A = torch.randn(proj.size(-1), 256, device=proj.device, requires_grad=False)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t

        if reduction == 'mean':
            err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
            statistic = (err @ self.weights).mean() * proj.size(0)
            return statistic

        elif reduction == 'sum':
            err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
            statistic = (err @ self.weights).sum() * proj.size(0)
            return statistic

        elif reduction is None:
            pointwise_err = (x_t.cos() - self.phi).square() + x_t.sin().square()
            pointwise_stat = pointwise_err.mean(1) @ self.weights
            return pointwise_stat

        return None


class VISRegLoss(nn.Module):
    def __init__(
        self,
        D_proj: int,
        K: int = 64,
        w_scale: float = 1.0,
        w_shape: float = 1.0,
        w_center: float = 1.0,
    ):
        """
        Régularisation VISReg unifiée pour l'entraînement et l'évaluation pointwise (IG / Visu).

        Parameters
        ----------
        D_proj : int
            La dimension de l'espace projeté (ex: 128, 256).
        K : int
            Nombre de projections aléatoires.
        """
        super().__init__()
        self.K = K
        self.D_proj = D_proj
        self.w_scale = w_scale
        self.w_shape = w_shape
        self.w_center = w_center
        self.normal_dist = Normal(0.0, 1.0)

        # On enregistre une matrice de projection fixe pour l'inférence (Visu / IG)
        W_fixed = torch.randn(D_proj, K)
        W_fixed = W_fixed / W_fixed.norm(p=2, dim=0, keepdim=True)
        self.register_buffer('W_fixed', W_fixed)

    def forward(self, z: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        if reduction is None:
            reduction = 'none'

        M, D = z.shape
        if M <= 1:
            if reduction == 'none':
                return torch.zeros(M, device=z.device, dtype=z.dtype, requires_grad=True)
            return torch.tensor(0.0, device=z.device, dtype=z.dtype, requires_grad=True)

        # 1. Center loss
        mu = z.mean(dim=0)
        L_center_scalar = mu.pow(2).mean()

        # 2. Scale loss
        z_cent = z - mu
        std = z_cent.std(dim=0, unbiased=False) + 1e-7
        L_scale_scalar = (1.0 - std).pow(2).mean()

        # 3. Shape loss: Sliced Wasserstein Distance
        z_norm = z_cent / std.detach()

        # --- SÉLECTION DES PROJECTIONS ---
        if self.training:
            # En entraînement, on veut de la stochasticité pour régulariser
            W = torch.randn(D, self.K, device=z.device, dtype=z.dtype)
            W = W / W.norm(p=2, dim=0, keepdim=True)
        else:
            # En eval / IG, on utilise les mêmes projections pour stabiliser le gradient
            W = self.W_fixed.to(device=z.device, dtype=z.dtype)

        p = z_norm @ W
        p_sorted, indices = torch.sort(p, dim=0)

        u = torch.arange(1, M + 1, device=z.device, dtype=z.dtype) / (M + 1)
        target = self.normal_dist.icdf(u).unsqueeze(1)  # [M, 1]
        error_sorted = (p_sorted - target).pow(2)  # [M, K]

        if reduction == 'none':
            # Inversion du tri via scatter pour réassigner la perte aux bons éléments
            pointwise_shape_error = torch.zeros_like(error_sorted)
            pointwise_shape_error.scatter_(0, indices, error_sorted)

            L_shape = pointwise_shape_error.mean(dim=1)  # [M]

            L_scale = torch.full((M,), L_scale_scalar, device=z.device, dtype=z.dtype)
            L_center = torch.full((M,), L_center_scalar, device=z.device, dtype=z.dtype)

            total_loss = (
                (self.w_scale * L_scale) + (self.w_shape * L_shape) + (self.w_center * L_center)
            )
        else:
            L_shape = error_sorted.mean()
            total_loss = (
                (self.w_scale * L_scale_scalar)
                + (self.w_shape * L_shape)
                + (self.w_center * L_center_scalar)
            )

            if reduction == 'sum':
                total_loss = total_loss * M

        return total_loss


class RectifiedLpLoss(nn.Module):
    def __init__(self, latent_dim, mu=0.0, sigma=1.0, p=1.0, num_projs=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.mu = mu
        self.sigma = sigma
        self.p = p
        self.num_projs = num_projs

        # 1. Projections fixes (Stable)
        # On utilise register_buffer pour que ça soit sauvegardé dans le state_dict
        # sans être un paramètre optimisable.
        projs = torch.randn(num_projs, latent_dim)
        projs = projs / (projs.norm(p=self.p, dim=-1, keepdim=True) + 1e-9)
        self.register_buffer('projections', projs)

    def _sample_ggd(self, batch_size, device):
        """
        Sampler efficace pour la distribution généralisée gaussienne (sans mpmath).
        """
        # On échantillonne la forme cible GGD
        # X = sign * (p * Gamma(1/p))^(1/p)
        shape = (batch_size, self.latent_dim)

        sign = torch.rand(shape, device=device).sign_()

        # Gamma dist: concentration=1/p, rate=1.0
        gamma_dist = torch.distributions.Gamma(1.0 / self.p, 1.0)
        g = gamma_dist.sample(shape).to(device)

        # Scaling
        x = sign * (self.p * g).pow(1.0 / self.p) * self.sigma + self.mu
        return torch.relu(x)

    def forward(self, z, reduction='mean'):
        """
        z: [B, D]
        reduction: 'mean' (SWD pour l'entraînement) ou None (Pointwise pour Audit/IG)
        """
        B, D = z.shape
        device = z.device

        # 1. Projections
        p_z = torch.matmul(z, self.projections.T)  # [B, num_projs]

        # Mode Entraînement : SWD global
        if reduction == 'mean':
            target = self._sample_ggd(B, device)
            p_t = torch.matmul(target, self.projections.T)  # [B, num_projs]

            p_z_sorted, _ = torch.sort(p_z, dim=0)
            p_t_sorted, _ = torch.sort(p_t, dim=0)

            if self.p == 1:
                return F.l1_loss(p_z_sorted, p_t_sorted)
            return F.mse_loss(p_z_sorted, p_t_sorted)

        # Mode Audit / IG : Écart Pointwise
        # On ne trie pas, on mesure l'écart de chaque projection par rapport
        # à la moyenne projetée de la distribution cible (mu projeté).
        else:
            # La moyenne de la RGG projetée est approximée par mu * somme(projections)
            # C'est une valeur scalaire par projection
            mu_proj = self.mu * self.projections.sum(dim=-1)

            # On mesure la distance de chaque échantillon à cette cible
            if self.p == 1:
                # L1 distance
                dist = F.l1_loss(p_z, mu_proj.expand_as(p_z), reduction='none')
            else:
                # L2 distance
                dist = F.mse_loss(p_z, mu_proj.expand_as(p_z), reduction='none')

            # On retourne la moyenne par échantillon (pour avoir un scalaire par item)
            return dist.mean(dim=1)


def precompute_freqs_cis_2d_batch(
    head_dim: int, crop_coords: torch.Tensor, img_size: int, patch_size: int, theta: float = 10000.0
):
    """
    Génère RoPE 2D pour chaque élément du batch global B*V et l'ajuste à la forme finale du ViT.
    crop_coords: [B * V, 4] -> [top, bottom, left, right] en pixels
    """
    BV = crop_coords.shape[0]
    dim = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    freqs = freqs.to(crop_coords.device)

    all_cos, all_sin = [], []

    # Division par patch_size pour passer dans l'espace des tokens
    coords_patches = crop_coords / patch_size

    for i in range(BV):
        # On garde les floats pour éviter les sauts brutaux avant l'interpolation
        y_start, y_end, x_start, x_end = coords_patches[i].tolist()

        # Détermination de la taille du crop d'origine en unités "patch"
        orig_h = max(int(round(y_end - y_start)), 1)
        orig_w = max(int(round(x_end - x_start)), 1)

        t_y = torch.linspace(y_start, y_end, steps=orig_h, device=crop_coords.device)
        t_x = torch.linspace(x_start, x_end, steps=orig_w, device=crop_coords.device)

        freqs_y = torch.outer(t_y, freqs)  # [orig_h, dim // 2]
        freqs_x = torch.outer(t_x, freqs)  # [orig_w, dim // 2]

        freqs_y = freqs_y.unsqueeze(1).repeat(1, orig_w, 1)  # [orig_h, orig_w, dim//2]
        freqs_x = freqs_x.unsqueeze(0).repeat(orig_h, 1, 1)  # [orig_h, orig_w, dim//2]

        # Grille brute du crop d'origine : [orig_h, orig_w, dim]
        freqs_2d = torch.cat([freqs_y, freqs_x], dim=-1)

        # --- L'ÉTAPÉ CLÉ : Interpolation vers la taille fixe du Transformer ---
        # On assume que ton patch embedder produit une grille carrée (ex: 16x16)
        # On va calculer cette taille cible fixe dynamiquement (N = H_target * W_target)
        # Ici, comme l'image augmentée est carrée (512x512) -> H_target = W_target = 512 // 32 = 16
        # On durcit la target à 16x16 ou on la déduit si nécessaire.
        # Pour être générique, on force une grille finale standard
        # (ex: 16x16 si target=512 et patch=32).
        target_h, target_w = (
            img_size // patch_size,
            img_size // patch_size,
        )  # Ajuste ici si ta taille d'image/patch change dynamiquement

        # Permute pour interpolate: [1, dim, orig_h, orig_w]
        freqs_2d = freqs_2d.permute(2, 0, 1).unsqueeze(0)
        freqs_2d = F.interpolate(
            freqs_2d, size=(target_h, target_w), mode='bilinear', align_corners=True
        )

        # Remettre au format standard : [target_h * target_w, dim]
        freqs_2d = freqs_2d.squeeze(0).permute(1, 2, 0).view(-1, dim)
        # ---------------------------------------------------------------------

        # Duplication pour rotate_half
        freqs_2d = torch.cat([freqs_2d, freqs_2d], dim=-1)

        all_cos.append(torch.cos(freqs_2d))
        all_sin.append(torch.sin(freqs_2d))

    # Maintenant, chaque tenseur fait strictement [256, head_dim]. Le stack va adorer.
    return torch.stack(all_cos, dim=0), torch.stack(all_sin, dim=0)


class Perceptron(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.GELU(),
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


class Encoder(torch.nn.Module):
    def __init__(
        self,
        blocks: nn.ModuleList,
        d_model: int,
        num_heads: int = 8,
        img_size: int = 512,
        patch_size: int = 16,
    ):
        super().__init__()
        self.blocks = blocks
        self.norm = nn.LayerNorm(d_model)
        self.head_dim = d_model // num_heads
        self.img_size = img_size
        self.patch_size = patch_size

    def forward(
        self, x: torch.Tensor, crop_coords: torch.Tensor | None, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        if crop_coords is None:
            crop_coords = (
                torch.tensor([0, self.img_size, 0, self.img_size])
                .repeat(x.shape[0], 1)
                .to(x.device)
            )
        cos, sin = precompute_freqs_cis_2d_batch(
            self.head_dim, crop_coords, self.img_size, self.patch_size
        )

        for block in self.blocks:
            x = block(x, freqs_cos=cos, freqs_sin=sin, attn_mask=attn_mask)
        return self.norm(x)


class Projector(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, proj_dim, rdm):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.rdm = rdm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.projection(x)) if self.rdm else self.projection(x)
