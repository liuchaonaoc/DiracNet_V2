"""Hydrogenic analytic references for gates and tests."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor
from scipy.special import genlaguerre, roots_genlaguerre


def hydrogenic_energy(Z: int | float, n: int | float) -> float:
    return -float(Z) ** 2 / (2.0 * float(n) ** 2)


def hydrogenic_P_analytic(r: Tensor, Z: int | float, n: int, l: int = 0) -> Tensor:
    if n <= l:
        raise ValueError("require n > l")
    rho = 2.0 * float(Z) * r / float(n)
    fact = math.factorial(n - l - 1)
    fact_n = math.factorial(n + l)
    norm = math.sqrt((2.0 * float(Z) / float(n)) ** 3 * fact / (2.0 * n * fact_n))
    rho_np = rho.detach().cpu().numpy()
    L = genlaguerre(n - l - 1, 2 * l + 1)(rho_np)
    R = norm * (rho_np ** l) * np.exp(-rho_np / 2.0) * L
    P = (rho_np / (2.0 * float(Z) / float(n))) * R
    out = torch.tensor(P, dtype=r.dtype, device=r.device)
    # Normalise in simple trapezoid sense if a grid object is unavailable.
    return out


def hydrogenic_radial_nodes(r: Tensor, Z: int | float, n: int, l: int = 0) -> Tensor:
    """Return analytic radial node locations for hydrogenic P(r).

    Nodes are zeros of ``L_{n-l-1}^{2l+1}(rho)`` with
    ``rho = 2 Z r / n``. The origin is not counted as a radial node.
    """
    degree = int(n) - int(l) - 1
    if degree <= 0:
        return torch.empty(0, dtype=r.dtype, device=r.device)
    rho_roots, _ = roots_genlaguerre(degree, 2 * int(l) + 1)
    r_nodes = rho_roots * float(n) / (2.0 * float(Z))
    return torch.tensor(r_nodes, dtype=r.dtype, device=r.device)


def hydrogenic_lobe_partition(r: Tensor, Z: int | float, n: int, l: int = 0) -> Tensor:
    """Boolean mask splitting ``r`` at the analytic radial nodes.

    The hydrogenic radial function ``P(r)`` for principal quantum number ``n`` and
    orbital quantum number ``l`` has ``K = n - l - 1`` radial nodes. These nodes
    partition the positive real line into ``K + 1`` "lobes". This routine returns
    a boolean tensor of shape ``[K + 1, N_grid]`` such that ``mask[k, i]`` is
    True iff grid point ``r[i]`` belongs to lobe ``k`` (counted from the
    innermost lobe).

    For nodeless orbitals (``K == 0``, e.g. 1s, 2p, 3d) a single all-True row is
    returned, so callers can still iterate over the segments uniformly.

    Parameters
    ----------
    r : Tensor[N_grid]
        Sorted positive radial grid.
    Z, n, l : quantum numbers.

    Returns
    -------
    mask : BoolTensor[K + 1, N_grid]
    """
    nodes = hydrogenic_radial_nodes(r, Z, n, l).to(r.device).to(r.dtype)
    n_grid = r.shape[-1]
    if nodes.numel() == 0:
        return torch.ones((1, n_grid), dtype=torch.bool, device=r.device)
    sorted_nodes, _ = torch.sort(nodes)
    n_lobes = int(sorted_nodes.numel()) + 1
    edges = torch.empty(n_lobes + 1, dtype=r.dtype, device=r.device)
    edges[0] = r.min() - 1.0  # ensures the innermost lobe captures r[0].
    edges[1:-1] = sorted_nodes
    edges[-1] = r.max() + 1.0  # ensures the outermost lobe captures r[-1].
    r_b = r.unsqueeze(0)  # [1, N_grid]
    lo = edges[:-1].unsqueeze(-1)  # [n_lobes, 1]
    hi = edges[1:].unsqueeze(-1)
    return (r_b >= lo) & (r_b < hi)


def cosine_signed(P_model: Tensor, P_ref: Tensor, grid) -> Tensor:
    num = grid.integrate(P_model * P_ref, dim=-1)
    den = grid.integrate(P_model * P_model, dim=-1).sqrt() * grid.integrate(P_ref * P_ref, dim=-1).sqrt()
    return num / den.clamp_min(1e-12)
