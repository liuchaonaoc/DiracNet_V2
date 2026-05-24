import torch
from rc_diracnet_v2.losses import BSplineSmoothLoss


def test_smooth_zero_on_constant():
    c=torch.ones(2,3,8); L=BSplineSmoothLoss()(c,torch.ones(2,3,dtype=torch.bool)); assert L < 1e-10
