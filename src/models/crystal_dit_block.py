import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .recip_rope_y import ReciprocalRoPE

class AdaLNModulation(nn.Module):
    """
    Generate 6 modulation vectors from the global condition C_t.

    Output vectors (each of shape (B, D)):
        gamma_1, beta_1, alpha_1   — for the attention sub-block
        gamma_2, beta_2, alpha_2   — for the FFN sub-block

    AdaLN-Zero: alpha vectors are initialized to zero so that each
    block acts as an identity at the start of training (DiT convention).

    Parameters
    ----------
    d_cond  : int  Condition vector dimension.
    d_model : int  Transformer hidden dimension.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        # Zero-init the linear layer so alpha starts at 0
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(self, C_t: Tensor) -> tuple[Tensor, ...]:
        """
        Parameters
        ----------
        C_t : (B, d_model)

        Returns
        -------
        gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2
            each (B, d_model)  — unsqueeze to (B, 1, D) when applied
        """
        out = self.proj(C_t)                               # (B, 6*D)
        return out.chunk(6, dim=-1)                        # 6 × (B, D)


# ════════════════════════════════════════════════════════════════
#  Multi-Head Self-Attention with ReciprocalRoPE
# ════════════════════════════════════════════════════════════════

class RoPEAttention(nn.Module):
    """
    Multi-head self-attention with ReciprocalRoPE applied to Q and K.

    Cannot use nn.MultiheadAttention because RoPE must be injected
    between the Q/K projection and the dot-product computation.

    RoPE is applied on the full (B, N, D) Q/K before reshaping into
    heads. This is equivalent to per-head RoPE because the rotation
    operates on consecutive pairs [0,1], [2,3], ... which map cleanly
    to head boundaries (head 0 = dims 0..d_head-1, etc.).

    Parameters
    ----------
    d_model : int   Hidden dimension (must be divisible by n_heads).
    n_heads : int   Number of attention heads.
    dropout : float Attention dropout rate.
    rope    : ReciprocalRoPE   Shared RoPE instance (created externally).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        rope:    ReciprocalRoPE = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5

        # joint Q/K/V projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        # output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.rope = rope

    def forward(
        self,
        x:           Tensor,   # (B, N_total, D)  full sequence [prefix + atoms]
        frac_coords: Tensor,   # (B, N_atoms, 3)  atom coords only (no prefix)
        k_t:         Tensor,   # (B, 6)           noisy k-parameters
        attn_mask:   Tensor,   # (B, N_total)     True = masked (padding)
        n_prefix:    int = 0,  # number of non-spatial prefix tokens (lattice, etc.)
    ) -> Tensor:
        """
        Returns
        -------
        out : (B, N_total, D)
        """
        B, N_total, D = x.shape

        # ── Q, K, V projection ──
        qkv = self.qkv_proj(x)                            # (B, N_total, 3D)
        Q, K, V = qkv.chunk(3, dim=-1)                    # each (B, N_total, D)

        # ── Apply RecipRoPE only to atom tokens (skip prefix) ──
        if self.rope is not None and n_prefix < N_total:
            Q_prefix, Q_atoms = Q[:, :n_prefix], Q[:, n_prefix:]
            K_prefix, K_atoms = K[:, :n_prefix], K[:, n_prefix:]

            Q_atoms = self.rope(Q_atoms, frac_coords, k_t)
            K_atoms = self.rope(K_atoms, frac_coords, k_t)

            Q = torch.cat([Q_prefix, Q_atoms], dim=1) if n_prefix > 0 else Q_atoms
            K = torch.cat([K_prefix, K_atoms], dim=1) if n_prefix > 0 else K_atoms
        Q = Q.view(B, N_total, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, N_total, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, N_total, self.n_heads, self.d_head).transpose(1, 2)
        # ── Scaled dot-product attention ──
        # attn_mask: (B, N_total) bool → (B, 1, 1, N_total) for broadcasting
        mask_4d = attn_mask.unsqueeze(1).unsqueeze(2)      # (B, 1, 1, N_total)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.masked_fill(mask_4d, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)                        # (B, H, N, d_head)
        # ── Merge heads and project ──
        out = out.transpose(1, 2).reshape(B, N_total, D)   # (B, N_total, D)
        return self.out_proj(out)                           # (B, N_total, D)


# ════════════════════════════════════════════════════════════════
#  SiLU-Gated Feed-Forward Network
# ════════════════════════════════════════════════════════════════

class FFN(nn.Module):
    """
    Simple two-layer FFN with SiLU activation.

    x → Linear(D, d_ffn) → SiLU → Dropout → Linear(d_ffn, D) → out

    Parameters
    ----------
    d_model : int   Input/output dimension.
    d_ffn   : int   Intermediate dimension.
    dropout : float Dropout rate.
    """

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)                                  # (B, N, D)


# ════════════════════════════════════════════════════════════════
#  CrystalDiT Block
# ════════════════════════════════════════════════════════════════

class CrystalDiTBlock(nn.Module):
    """
    Single CrystalDiT block: AdaLN + MHSA(RecipRoPE) + FFN.

    Parameters
    ----------
    d_model : int               Transformer hidden dim.
    n_heads : int               Number of attention heads.
    d_ffn   : int               FFN intermediate dim.
    dropout : float             Dropout rate.
    rope    : ReciprocalRoPE    Shared RoPE instance (all blocks share one).

    Shape
    -----
    Input:
        h           (B, N, D)        token sequence
        C_t         (B, d_model)      global condition vector
        frac_coords (B, N, 3)        fractional coordinates (noisy)
        k_t         (B, 6)           lattice k-parameters (noisy)
        attn_mask   (B, N)           True = masked (padding)
    Output:
        h_out       (B, N, D)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn:   int = 1024,
        dropout: float = 0.1,
        rope:    ReciprocalRoPE = None,
    ):
        super().__init__()

        self.adaln = AdaLNModulation(d_model)

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn  = RoPEAttention(d_model, n_heads, dropout, rope)

        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn   = FFN(d_model, d_ffn, dropout)

    def forward(
        self,
        h:           Tensor,   # (B, N_total, D)
        C_t:         Tensor,   # (B, d_model)
        frac_coords: Tensor,   # (B, N_atoms, 3)  atom coords only
        k_t:         Tensor,   # (B, 6)
        attn_mask:   Tensor,   # (B, N_total)
        n_prefix:    int = 0,  # non-spatial prefix tokens
    ) -> Tensor:

        # ── Generate AdaLN modulation params ──
        gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2 = self.adaln(C_t)
        # each: (B, D) → unsqueeze to (B, 1, D) for broadcasting over N
        gamma_1 = gamma_1.unsqueeze(1)
        beta_1  = beta_1.unsqueeze(1)
        alpha_1 = alpha_1.unsqueeze(1)
        gamma_2 = gamma_2.unsqueeze(1)
        beta_2  = beta_2.unsqueeze(1)
        alpha_2 = alpha_2.unsqueeze(1)

        # ── Attention sub-block ──
        h_norm = self.norm1(h)                              # (B, N_total, D)
        h_mod  = (1 + gamma_1) * h_norm + beta_1           # AdaLN modulate
        h_attn = self.attn(h_mod, frac_coords, k_t, attn_mask, n_prefix=n_prefix)
        h      = h + alpha_1 * h_attn                      # gated residual

        # ── FFN sub-block ──
        h_norm = self.norm2(h)
        h_mod  = (1 + gamma_2) * h_norm + beta_2
        h_ffn  = self.ffn(h_mod)
        h      = h + alpha_2 * h_ffn                       # gated residual

        return h
