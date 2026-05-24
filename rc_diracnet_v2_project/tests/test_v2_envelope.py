import torch
from rc_diracnet_v2.readout import AnalyticEnvelope


def test_envelope_h1s_analytic():
    r = torch.linspace(0.01, 20.0, 200)
    env, _, _ = AnalyticEnvelope()(r, torch.tensor([[1.0]]), torch.tensor([[-1]]), torch.tensor([1]))
    assert torch.allclose(env[0,0], r * torch.exp(-r), atol=1e-4)
