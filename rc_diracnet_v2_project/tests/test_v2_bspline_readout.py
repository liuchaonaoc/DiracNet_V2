import torch
from rc_diracnet_v2.readout import BSplineReadout


def test_readout_init_equals_env():
    Bsz,Norb,K,N = 2,3,8,50
    c = torch.zeros(Bsz,Norb,K); env = torch.randn(Bsz,Norb,N); denv=torch.randn_like(env); d2env=torch.randn_like(env)
    basis = torch.randn(K,N); r=torch.linspace(0.1,5,N); mask=torch.ones(Bsz,Norb,dtype=torch.bool); kappa=torch.ones(Bsz,Norb,dtype=torch.long)*-1
    out = BSplineReadout()(c,env,denv,d2env,basis,basis,basis,kappa,r,mask)
    assert torch.allclose(out['P'], env)
