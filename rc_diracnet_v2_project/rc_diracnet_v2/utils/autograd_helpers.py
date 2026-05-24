"""Autograd-based radial derivatives.

对应 frame_RC_V1_1.md §4.2 autograd_helpers.py。

设计说明：
    RC-DiracNet 主路径使用解析导数（``reservoir.basis_generator`` 与
    ``reservoir.envelope`` 都直接给出 ``dphi/dr``、``denv/dr``），
    因此这里的 autograd 工具主要服务于：
      1) ``continuum.PhaseAmplitudeNet`` 等通用 INR；
      2) 单元测试中比对解析与自动微分。
"""

from __future__ import annotations

import torch
from torch import Tensor


def grad_wrt_r(
    y: Tensor,
    r: Tensor,
    create_graph: bool = True,
) -> Tensor:
    """∂y/∂r。

    参数
    ----
    y : Tensor [..., N_grid]
        必须由 ``r`` 派生且为 pointwise 依赖（即 ``y[..., i]`` 只依赖 ``r[i]``）。
    r : Tensor [N_grid]
        必须 ``requires_grad=True``。
    create_graph : bool
        是否保留二阶导计算图。

    返回
    ----
    Tensor [..., N_grid]
    """
    if not r.requires_grad:
        raise RuntimeError(
            "grad_wrt_r: r must have requires_grad=True. "
            "Call `r = r.detach().requires_grad_()` before forward."
        )

    flat_y = y.reshape(-1, y.shape[-1])  # [B', N_grid]
    b_prime = flat_y.shape[0]

    grads = []
    for k in range(b_prime):
        (g,) = torch.autograd.grad(
            outputs=flat_y[k].sum(),
            inputs=r,
            create_graph=create_graph,
            retain_graph=True,
        )
        grads.append(g)
    stacked = torch.stack(grads, dim=0)  # [B', N_grid]
    return stacked.reshape(y.shape)


def second_grad_wrt_r(y: Tensor, r: Tensor) -> Tensor:
    """∂²y/∂r²。要求 r.requires_grad=True 且 y 由 r 派生。"""
    dy = grad_wrt_r(y, r, create_graph=True)
    d2y = grad_wrt_r(dy, r, create_graph=True)
    return d2y


def make_r_with_grad(r_grid: Tensor) -> Tensor:
    """复制并启用梯度，避免污染 ``RadialGrid`` 内部 buffer。"""
    return r_grid.detach().clone().requires_grad_(True)
