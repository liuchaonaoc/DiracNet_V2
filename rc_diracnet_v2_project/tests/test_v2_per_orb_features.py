import torch
from rc_diracnet_v2.data.batch_builder import V2BatchBuilder


def test_per_orb_features_h1s():
    batch={"Z":torch.tensor([1]),"config_shells":torch.tensor([[[1,0,1,1]]]).long(),"orb_mask":torch.tensor([[True]])}
    f=V2BatchBuilder(1)._build_per_orb_features(batch); assert f.shape==(1,1,8); assert f[0,0,5] == -1
