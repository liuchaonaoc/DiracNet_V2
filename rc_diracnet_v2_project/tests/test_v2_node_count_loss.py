import torch
from rc_diracnet_v2.losses import NodeCountLoss


def test_node_count_zero_nodes():
    r=torch.linspace(0.1,10,200); P=torch.exp(-r).view(1,1,-1); L=NodeCountLoss()(P,torch.tensor([[0]]),torch.tensor([[True]])); assert L < 0.1
