import torch
from rc_diracnet_v2.kan import KANCoeffNet, MLPCoeffNet, PyKANCoeffNet, SplineKANLayer


def test_mlp_coeff_zero_init():
    net=MLPCoeffNet(d_cond=16,n_basis=8); h=torch.randn(2,16); f=torch.ones(2,3,8); m=torch.ones(2,3,dtype=torch.bool)
    lam,c=net(h,f,m); assert lam.abs().max()<1e-6 and c.abs().max()<1e-6


def test_spline_kan_layer_has_edge_spline_parameters():
    layer = SplineKANLayer(in_features=4, out_features=3, grid_size=7)
    x = torch.randn(5, 4).clamp(-1, 1)
    y = layer(x)
    assert y.shape == (5, 3)
    assert layer.spline_weight.shape == (3, 4, 7)


def test_kan_coeff_zero_init_and_not_mlp_wrapper():
    net = KANCoeffNet(d_cond=16, n_basis=8, hidden=12, grid_size=7)
    h = torch.randn(2, 16)
    f = torch.ones(2, 3, 8)
    m = torch.ones(2, 3, dtype=torch.bool)
    lam, c = net(h, f, m)
    assert hasattr(net.net.layer1, "spline_weight")
    assert hasattr(net.net.layer2, "spline_weight")
    assert lam.abs().max() < 1e-6
    assert c.abs().max() < 1e-6


def test_pykan_backend_is_optional_or_zero_init():
    try:
        net = PyKANCoeffNet(d_cond=16, n_basis=8, hidden=12, grid_size=5)
    except ImportError as exc:
        assert "kan.kind='pykan'" in str(exc)
        return
    h = torch.randn(2, 16)
    f = torch.ones(2, 3, 8)
    m = torch.ones(2, 3, dtype=torch.bool)
    lam, c = net(h, f, m)
    assert lam.abs().max() < 1e-6
    assert c.abs().max() < 1e-6
