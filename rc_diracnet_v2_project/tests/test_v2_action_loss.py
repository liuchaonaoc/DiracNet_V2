import torch
from rc_diracnet_v2.losses import BohrSommerfeldActionLoss
from rc_diracnet_v2.utils.grid import RadialGrid


def test_action_loss_grad_finite():
    grid=RadialGrid(r_min=1e-4,r_max=50,n_grid=256)
    E=torch.tensor([[-0.5]], requires_grad=True); V=-1.0/grid.r.view(1,-1)
    L=BohrSommerfeldActionLoss()(E,V,torch.tensor([[1]]),torch.tensor([[0]]),torch.tensor([[True]]),grid)
    L.backward(); assert torch.isfinite(E.grad).all()
