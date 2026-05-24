"""Global quantum-state encoder.

对应 frame_RC_V1_1.md §4.5。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GlobalQuantumEncoder(nn.Module):
    """离散量子标签 → 高维潜变量 h_cond。

    Inputs (batch dict)
    -------------------
    Z              [B]                    long
    nele           [B]                    long
    charge         [B]                    long
    J              [B]                    long (= 2J，整数)
    parity         [B]                    long ∈ {0, 1}
    config_shells  [B, max_seq, 4]        long (n, l, 2j, occ)
    shell_mask     [B, max_seq]           bool

    Output
    ------
    h_cond         [B, d_cond]            float
    """

    def __init__(
        self,
        d_embed_z: int = 64,
        d_embed_shell: int = 32,
        d_gru_hidden: int = 128,
        d_cond: int = 256,
        max_z: int = 110,
        max_n: int = 10,
        max_l: int = 7,
        max_seq: int = 16,
    ) -> None:
        super().__init__()
        self.max_seq = max_seq
        self.d_cond = d_cond
        self.max_z = max_z
        self.max_n = max_n
        self.max_l = max_l
        self._j_vocab = 2 * max_l + 3
        self._occ_vocab = 2 * (2 * max_l + 1) + 1
        self._J_vocab = 2 * max_z + 1

        self.z_embed = nn.Embedding(max_z + 1, d_embed_z, padding_idx=0)
        self.nele_embed = nn.Embedding(max_z + 1, d_embed_z, padding_idx=0)
        self.charge_embed = nn.Embedding(max_z + 1, d_embed_z, padding_idx=0)
        self.J_embed = nn.Embedding(2 * max_z + 1, d_embed_z, padding_idx=0)
        self.parity_embed = nn.Embedding(2, d_embed_z)

        # 壳层：四个离散字段各自 embed 后拼接
        self.n_embed = nn.Embedding(max_n + 1, d_embed_shell, padding_idx=0)
        self.l_embed = nn.Embedding(max_l + 1, d_embed_shell)
        self.j_embed = nn.Embedding(2 * max_l + 3, d_embed_shell)
        self.occ_embed = nn.Embedding(2 * (2 * max_l + 1) + 1, d_embed_shell, padding_idx=0)

        self.shell_proj = nn.Linear(4 * d_embed_shell, d_gru_hidden)
        self.shell_gru = nn.GRU(
            input_size=d_gru_hidden,
            hidden_size=d_gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        fused_dim = 5 * d_embed_z + 2 * d_gru_hidden
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )
        self.film_parity = nn.Linear(d_embed_z, 2 * d_cond)
        self.ln = nn.LayerNorm(d_cond)

    def forward(self, batch: dict) -> Tensor:
        Z = batch["Z"].long().clamp(0, self.max_z)
        nele = batch["nele"].long().clamp(0, self.max_z)
        charge = batch["charge"].long().clamp(0, self.max_z)
        J = batch["J"].long().clamp(0, self._J_vocab - 1)
        parity = batch["parity"].long().clamp(0, 1)
        config = batch["config_shells"].long()        # [B, S, 4]
        shell_mask = batch["shell_mask"].bool()         # [B, S]

        z_emb = self.z_embed(Z)                          # [B, D_z]
        nele_emb = self.nele_embed(nele)
        charge_emb = self.charge_embed(charge)
        J_emb = self.J_embed(J)
        parity_emb = self.parity_embed(parity)

        # NIST Rydberg 态 n 可达 79；超出 max_n 时必须 clamp，否则 CUDA embedding 越界
        n_e = self.n_embed(config[..., 0].clamp(0, self.max_n))
        l_e = self.l_embed(config[..., 1].clamp(0, self.max_l))
        j_e = self.j_embed(config[..., 2].clamp(0, self._j_vocab - 1))
        o_e = self.occ_embed(config[..., 3].clamp(0, self._occ_vocab - 1))
        shell_emb = torch.cat([n_e, l_e, j_e, o_e], dim=-1)        # [B, S, 4*D_s]
        shell_emb = self.shell_proj(shell_emb)                     # [B, S, D_g]

        # mask shell pad positions
        shell_emb = shell_emb * shell_mask.unsqueeze(-1).float()

        gru_out, _ = self.shell_gru(shell_emb)                     # [B, S, 2*D_g]
        # masked mean pooling
        denom = shell_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        gru_pooled = (gru_out * shell_mask.unsqueeze(-1).float()).sum(dim=1) / denom

        fused = torch.cat([z_emb, nele_emb, charge_emb, J_emb, parity_emb, gru_pooled], dim=-1)
        h = self.fuse(fused)

        gamma_beta = self.film_parity(parity_emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        h = self.ln(h * (1.0 + gamma) + beta)
        return h
