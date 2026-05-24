import torch
from rc_diracnet_v2.basis import BSplineBasis


def test_bspline_partition_of_unity_interior():
    r = torch.linspace(1.0, 30.0, 200)
    b = BSplineBasis(n_basis=16, order=3, r_min=1e-4, r_max=50.0); b.precompute(r)
    B, _, _ = b(); assert torch.allclose(B.sum(0), torch.ones_like(r), atol=1e-3)


def test_bspline_endpoint_zero():
    r = torch.tensor([1e-4, 50.0])
    b = BSplineBasis(n_basis=16, order=3, r_min=1e-4, r_max=50.0); b.precompute(r)
    B, _, _ = b(); assert B[1:,0].abs().max() < 1e-6 and B[:-1,-1].abs().max() < 1e-6
