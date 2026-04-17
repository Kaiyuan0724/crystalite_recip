from __future__ import annotations

import torch
from torch import nn

from src.models.embeddings import LatticeEmbedder, TimeEmbedder
from src.models.heads import CrystalHeads
from src.models.crystal_dit_block import CrystalDiTBlock
from src.models.recip_rope_y import ReciprocalRoPE


def mod1(x: torch.Tensor) -> torch.Tensor:
    """Wrap values into [0, 1)."""
    return x - torch.floor(x)


class RecipModel(nn.Module):

    def __init__(
            self,
            vz: int,
            d_model: int = 256,
            d_cond:  int = 256,
            d_ffn:   int = 1024,
            n_layers:   int = 6,
            n_heads: int = 8,
            type_dim: int | None = None,
            lattice_embed_mode: str = "rff",
            lattice_rff_dim:    int = 256,
            lattice_rff_sigma:  float = 5.0,
            dropout:            float = 0.0,
            lattice_repr:       str = "y1",
            coord_head_mode:    str = "direct",
            sigma_init:         float = 0.5,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        # Bug 2 fixed: handle type_dim=None (same logic as CrystaliteModel)
        self.type_dim = (vz + 1) if type_dim is None else int(type_dim)
        self.type_proj = nn.Sequential(
            nn.Linear(self.type_dim, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )
        self.lattice_embed = LatticeEmbedder(
            d_model=d_model,
            mode=lattice_embed_mode,
            rff_dim=lattice_rff_dim,
            rff_sigma=lattice_rff_sigma,
        )
        self.segment_embed = nn.Embedding(2, d_model)
        self.time = TimeEmbedder(d_model=d_model)
        self.rope = ReciprocalRoPE(
            model_dim=d_model,
            n_heads=n_heads,
            sigma_init=sigma_init,
        )
        self.trunk = nn.ModuleList([
            CrystalDiTBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_cond=d_cond,
                d_ffn=d_ffn,
                dropout=dropout,
                rope=self.rope,
            )
            for _ in range(n_layers)
        ])
        self.heads = CrystalHeads(
            d_model=d_model,
            vz=vz,
            type_out_dim=self.type_dim,
            coord_head_mode=coord_head_mode,
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        type_feats: torch.Tensor,
        frac_coords: torch.Tensor,
        lattice_feats: torch.Tensor,
        pad_mask: torch.Tensor,
        t_sigma: torch.Tensor,
        # Bug 8 fixed: required by denoise_edm in edm_utils.py
        lattice_bias_feats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            type_feats:         (B, N, type_dim) relaxed type features with padding zeroed.
            frac_coords:        (B, N, 3) fractional coords (not necessarily wrapped).
            lattice_feats:      (B, 6)
            pad_mask:           (B, N) bool where True denotes padding.
            t_sigma:            (B,) scalar noise embedding c_noise = 0.25 * log(sigma).
            lattice_bias_feats: optional (B, 6), unused here but required by EDM interface.
        """
        frac_mod = mod1(frac_coords)
        # Bug 3 fixed: removed duplicate '+' operator
        h_type = self.type_proj(type_feats) + self.segment_embed.weight[0]
        h_lat = self.lattice_embed(lattice_feats) + self.segment_embed.weight[1]
        h_lat = h_lat[:, None, :]

        # Bug 7 fixed: lattice token placed FIRST so RoPEAttention skips it via
        # n_prefix=1, avoiding shape mismatch between sequence (B, N+1, D) and
        # frac_coords (B, N, 3).
        # Sequence layout: [lattice_token, atom_0, ..., atom_{N-1}]
        x = torch.cat([h_lat, h_type], dim=1)  # (B, N+1, D)

        # Lattice token is never padding; its False goes at the front.
        pad_seq = torch.cat(
            [
                torch.zeros((pad_mask.shape[0], 1), device=pad_mask.device, dtype=torch.bool),
                pad_mask.bool(),
            ],
            dim=1,
        )  # (B, N+1)

        t_emb = self.time(t_sigma, t_sigma)

        # Bug 4 fixed: feed h (not the original x) into each successive block
        # Bug 5 fixed: all positional args, no SyntaxError
        h = x
        for block in self.trunk:
            h = block(h, t_emb, frac_mod, lattice_feats, pad_seq, n_prefix=1)

        # Bug 6 fixed: apply norm_out before the output heads
        h = self.norm_out(h)

        # Unpack lattice-first sequence: position 0 = lattice, 1: = atoms
        h_lat_out = h[:, 0, :]   # (B, D)
        h_atoms   = h[:, 1:, :]  # (B, N, D)

        # Call head submodules directly (CrystalHeads.forward expects lattice-last,
        # so we slice manually and avoid that assumption here)
        type_logits = self.heads.type_head(h_atoms)
        lattice_vel = self.heads.lattice_head(h_lat_out)
        if self.heads.coord_head_mode == "direct":
            coord_vel = self.heads.coord_head(h_atoms)
        else:
            coord_vel = self.heads._coord_from_relative(h_atoms, frac_mod, pad_mask.bool())

        return {
            "type_logits": type_logits,
            "coord_vel":   coord_vel,
            "lattice_vel": lattice_vel,
        }
