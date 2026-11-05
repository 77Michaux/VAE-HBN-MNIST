import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    """Sépare et tourne la moitié des dimensions."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    # x: [B*V, num_heads, N, head_dim]
    # cos, sin: [B*V, N, head_dim] -> unsqueeze(1) pour diffuser sur les têtes
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer=nn.GELU,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Dropout(p=drop_prob),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(p=drop_prob),
        )

    def forward(self, x):
        return self.mlp(x)


class Attention(nn.Module):
    def __init__(
        self, d_model, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = qk_scale
        self.dropout_p = attn_drop

        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cos=None, freqs_sin=None, attn_mask=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if freqs_cos is not None and freqs_sin is not None:
            q = apply_rotary_pos_emb(q, freqs_cos, freqs_sin)
            k = apply_rotary_pos_emb(k, freqs_cos, freqs_sin)

        # --- Gestion du masque d'attention pour SDPA ---
        # SDPA attend un masque de shape [B, num_heads, N_queries, N_keys]
        # ou un masque 2D [B, N_keys] qui sera broadcasté.
        if attn_mask is not None:  # noqa: SIM108
            # attn_mask est [B*V, N]. On doit le transformer en booléen pour SDPA.
            # On crée une matrice de connectivité [B*V, 1, N, N] : un patch valide parle aux autres
            # patchs valides.
            # Version simple et efficace acceptée par SDPA : shape [B*V, 1, 1, N]
            boolean_mask = attn_mask.unsqueeze(1).unsqueeze(2).bool()
        else:
            boolean_mask = None
        # -----------------------------------------------

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=boolean_mask,  # Injecté ici !
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio, mlp_drop, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model=d_model, num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(
            in_features=d_model,
            hidden_features=d_model * mlp_ratio,
            out_features=d_model,
            act_layer=nn.GELU,
            drop_prob=mlp_drop,
        )

    def forward(self, x, freqs_cos=None, freqs_sin=None, attn_mask=None):
        x = x + self.drop_path(
            self.attn(self.norm1(x), freqs_cos=freqs_cos, freqs_sin=freqs_sin, attn_mask=attn_mask)
        )
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
