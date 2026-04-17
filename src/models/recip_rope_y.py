"""
recip_rope.py — ReciprocalRoPE with y-parameterized lattice
============================================================
Lattice is represented as a 6-dim latent y, reconstructed into a
lower-triangular lattice matrix L(y):
    L = [[exp(y1),  0,        0      ],
         [y2,       exp(y3),  0      ],
         [y4,       y5,       exp(y6)]]
 
This replaces the k-coefficient + matrix_exp parameterization for
improved numerical stability and speed.
"""
 
import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple
 
 
# ════════════════════════════════════════════════════════════════
# Part 1  O_h group matrices (48 signed permutations)
# ════════════════════════════════════════════════════════════════
 
def get_Oh_matrices() -> List[np.ndarray]:
    matrices = []
    for signs in [(s0, s1, s2) for s0 in [1, -1] for s1 in [1, -1] for s2 in [1, -1]]:
        for perm in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
            M = np.zeros((3, 3), dtype=int)
            M[0, perm[0]] = signs[0]
            M[1, perm[1]] = signs[1]
            M[2, perm[2]] = signs[2]
            matrices.append(M)
    return matrices
 
 
# ════════════════════════════════════════════════════════════════
# Part 2  Orbit computation
# ════════════════════════════════════════════════════════════════
 
def _compute_orbit(seed: Tuple, group_matrices: List[np.ndarray]) -> List[Tuple]:
    h0 = np.array(seed, dtype=int)
    orbit = {tuple((M.T @ h0).tolist()) for M in group_matrices}
    return sorted(orbit)
 
 
def compute_master_orbits(group_matrices: List[np.ndarray], max_channels: int) -> List[List[Tuple]]:
    r_max = max(10, int(math.ceil(math.sqrt(max_channels * 3))))
    candidates = sorted(
        [(h, k, l)
         for h in range(-r_max, r_max + 1)
         for k in range(-r_max, r_max + 1)
         for l in range(-r_max, r_max + 1)
         if not (h == 0 and k == 0 and l == 0)],
        key=lambda v: (v[0]**2 + v[1]**2 + v[2]**2, v),
    )
 
    visited = set()
    orbits = []
    total = 0
    for seed in candidates:
        if seed in visited:
            continue
        orbit = _compute_orbit(seed, group_matrices)
        visited.update(orbit)
        if total + len(orbit) <= max_channels:
            orbits.append(orbit)
            total += len(orbit)
        if total >= max_channels:
            break
    return orbits
 
 
# ════════════════════════════════════════════════════════════════
# Part 3  y → B_recip (lower-triangular Cholesky-style construction)
# ════════════════════════════════════════════════════════════════
 
def _y_to_L(y: Tensor) -> Tensor:
    """
    Reconstruct lower-triangular lattice matrix L from 6-dim y.
 
        L = [[exp(y1),  0,        0      ],
             [y2,       exp(y3),  0      ],
             [y4,       y5,       exp(y6)]]
 
    Any y in R^6 maps to a valid lattice (positive diagonal guaranteed).
    """
    shape = y.shape[:-1]
    L = y.new_zeros(*shape, 3, 3)
    L[..., 0, 0] = y[..., 0].exp()
    L[..., 1, 0] = y[..., 1]
    L[..., 1, 1] = y[..., 2].exp()
    L[..., 2, 0] = y[..., 3]
    L[..., 2, 1] = y[..., 4]
    L[..., 2, 2] = y[..., 5].exp()
    return L
 
 
def _y_to_B_recip(y: Tensor, detach_geometry: bool = True) -> Tensor:
    """
    Convert y to reciprocal lattice matrix B* = inv(L)^T.
    """
    squeeze = y.dim() == 1
    if squeeze:
        y = y.unsqueeze(0)
 
    L = _y_to_L(y)                                  # (B, 3, 3)
    if detach_geometry:
        L = L.detach()
    B_recip = torch.linalg.inv(L).transpose(-2, -1)  # (B, 3, 3)
 
    if squeeze:
        B_recip = B_recip.squeeze(0)
    return B_recip
 
 
# ════════════════════════════════════════════════════════════════
# Part 4  ReciprocalRoPE
# ════════════════════════════════════════════════════════════════
 
class ReciprocalRoPE(nn.Module):
    """
    Reciprocal-space RoPE with per-head learnable sigma.
 
    Rotates Q/K dimensions by:
        phase_n(u) = 2π h_n · u          (fractional-coordinate phase)
        decay_n    = exp(-σ_h/2 |B* h_n|²) (Gaussian envelope, per head)
    """
 
    def __init__(
        self,
        model_dim:  int,
        n_heads:    int = 8,
        sigma_init: float = 0.5,
    ):
        super().__init__()
        assert model_dim % 2 == 0
        assert model_dim % n_heads == 0
 
        self.n_heads = n_heads
        self.d_head = model_dim // n_heads
 
        group = get_Oh_matrices()
        orbits = compute_master_orbits(group, max_channels=model_dim // 2)
        miller_list = [idx for orbit in orbits for idx in orbit]
        self.N_freq = len(miller_list)
        self.freq_per_head = self.d_head // 2
 
        self.register_buffer(
            "miller_indices",
            torch.tensor(miller_list, dtype=torch.float32),
        )
 
        if isinstance(sigma_init, (list, tuple)):
            assert len(sigma_init) == n_heads
            init_vals = torch.tensor([math.log(s) for s in sigma_init])
        else:
            log_s = math.log(sigma_init)
            spread = torch.linspace(-0.7, 0.7, n_heads)
            init_vals = log_s + spread
 
        self.log_sigma = nn.Parameter(init_vals.float())
 
        sigmas = self.log_sigma.exp().detach().tolist()
        sigma_str = ", ".join(f"{s:.3f}" for s in sigmas)
        print(
            f"[ReciprocalRoPE] O_h, n_heads={n_heads}, "
            f"N_freq={self.N_freq}/{model_dim // 2}, "
            f"freq_per_head={self.freq_per_head}, "
            f"sigma_init=[{sigma_str}]"
        )
 
    @property
    def sigma(self) -> Tensor:
        return self.log_sigma.exp()
 
    def _phases(self, frac_coords: Tensor) -> Tensor:
        return 2 * math.pi * torch.einsum(
            "nj,bsj->bsn", self.miller_indices, frac_coords
        )
 
    def _decay(self, B_recip: Tensor) -> Tensor:
        G = torch.einsum("bik,kn->bin", B_recip, self.miller_indices.T)
        G_sq = (G ** 2).sum(dim=1)
 
        sigma = self.sigma
        N_used = min(self.n_heads * self.freq_per_head, self.N_freq)
        sigma_per_freq = sigma.repeat_interleave(self.freq_per_head)[:N_used]
 
        decay = torch.zeros_like(G_sq)
        decay[:, :N_used] = torch.exp(
            -0.5 * sigma_per_freq.unsqueeze(0) * G_sq[:, :N_used]
        )
        if N_used < self.N_freq:
            decay[:, N_used:] = torch.exp(
                -0.5 * sigma.mean() * G_sq[:, N_used:]
            )
        return decay
 
    def _rotate(self, x: Tensor, phases: Tensor, decay: Tensor) -> Tensor:
        B, S, D = x.shape
        x_rope = x[..., : 2 * self.N_freq].reshape(B, S, self.N_freq, 2)
        x_pass = x[..., 2 * self.N_freq:]
 
        x_e, x_o = x_rope[..., 0], x_rope[..., 1]
        cos_p, sin_p = phases.cos(), phases.sin()
 
        x_e_rot = (x_e * cos_p - x_o * sin_p) * decay
        x_o_rot = (x_e * sin_p + x_o * cos_p) * decay
 
        x_rot = torch.stack([x_e_rot, x_o_rot], dim=-1).reshape(B, S, 2 * self.N_freq)
        return torch.cat([x_rot, x_pass], dim=-1)
 
    def forward(
        self,
        x:               Tensor,   # (B, N, D) Q or K
        frac_coords:     Tensor,   # (B, N, 3)
        y:               Tensor,   # (B, 6) or (6,)
        detach_geometry: bool = True,
    ) -> Tensor:
        if y.dim() == 1:
            y = y.unsqueeze(0).expand(x.shape[0], -1)
 
        B_recip = _y_to_B_recip(y, detach_geometry=detach_geometry)
        phases = self._phases(frac_coords)
        decay  = self._decay(B_recip).unsqueeze(1)
 
        return self._rotate(x, phases, decay)
 