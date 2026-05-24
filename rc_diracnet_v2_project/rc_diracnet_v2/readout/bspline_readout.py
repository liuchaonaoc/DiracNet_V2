"""B-spline factor readout with non-relativistic kinetic balance."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..constants import C_LIGHT


class BSplineReadout(nn.Module):
    def __init__(self, c_speed: float = C_LIGHT, force_c0_zero: bool = True, q_residual_scale: float = 0.0) -> None:
        super().__init__()
        self.inv_2c = 1.0 / (2.0 * float(c_speed))
        self.force_c0_zero = bool(force_c0_zero)
        self.q_residual_scale = float(q_residual_scale)

    def forward(
        self,
        c_P: Tensor,
        env: Tensor,
        denv: Tensor,
        d2env: Tensor,
        B_basis: Tensor,
        dB_basis: Tensor,
        d2B_basis: Tensor,
        kappa: Tensor,
        r_grid: Tensor,
        orb_mask: Tensor,
        c_Q: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if self.force_c0_zero and c_P.shape[-1] > 0:
            c_P = c_P.clone()
            c_P[..., 0] = 0.0
        f = 1.0 + torch.einsum("bok,kn->bon", c_P, B_basis.to(c_P.device).to(c_P.dtype))
        df = torch.einsum("bok,kn->bon", c_P, dB_basis.to(c_P.device).to(c_P.dtype))
        d2f = torch.einsum("bok,kn->bon", c_P, d2B_basis.to(c_P.device).to(c_P.dtype))
        P = env * f
        dPdr = denv * f + env * df
        d2Pdr = d2env * f + 2.0 * denv * df + env * d2f
        r_b = r_grid.to(P.device).to(P.dtype).view(1, 1, -1)
        kap_b = kappa.to(P.device).to(P.dtype).unsqueeze(-1)
        Q = self.inv_2c * (dPdr + kap_b * P / r_b)
        dQdr = self.inv_2c * (d2Pdr + kap_b * (dPdr / r_b - P / (r_b * r_b)))
        if self.q_residual_scale > 0.0 and c_Q is not None:
            if self.force_c0_zero and c_Q.shape[-1] > 0:
                c_Q = c_Q.clone(); c_Q[..., 0] = 0.0
            fQ = torch.einsum("bok,kn->bon", c_Q, B_basis.to(c_Q.device).to(c_Q.dtype))
            dfQ = torch.einsum("bok,kn->bon", c_Q, dB_basis.to(c_Q.device).to(c_Q.dtype))
            Q = Q + self.q_residual_scale * env * fQ
            dQdr = dQdr + self.q_residual_scale * (denv * fQ + env * dfQ)
        mask = orb_mask.to(P.device).unsqueeze(-1).to(P.dtype)
        return {
            "P": P * mask,
            "Q": Q * mask,
            "dPdr": dPdr * mask,
            "dQdr": dQdr * mask,
            "d2Pdr": d2Pdr * mask,
            "f": f * mask,
            "df": df * mask,
            "d2f": d2f * mask,
        }
