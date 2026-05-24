import torch
from rc_diracnet_v2.losses import AsymptoticTailLoss


def test_asymptotic_zero_on_decaying():
    r=torch.linspace(0.1,50,500); P=(r*torch.exp(-r)).view(1,1,-1); L=AsymptoticTailLoss()(P,r,torch.tensor([[True]])); assert L < 1e-3
