from .action_loss import BohrSommerfeldActionLoss
from .anti_collapse_loss import AntiCollapseLoss, anti_collapse_per_orbital
from .asymptotic_loss import AsymptoticTailLoss
from .bspline_smooth_loss import BSplineSmoothLoss
from .decay_consistency_loss import DecayConsistencyLoss
from .nist_scalar_loss import NISTScalarHuberLoss
from .node_count_loss import NodeCountLoss
from .orthonormality_loss import OrthonormalityLoss
from .pde_loss import DiracPDELoss
from .virial_loss import VirialLoss

try:
    from .loss_balancer import LossBalancer
except Exception:
    LossBalancer = None

__all__ = [
    "BohrSommerfeldActionLoss", "AntiCollapseLoss", "anti_collapse_per_orbital",
    "AsymptoticTailLoss", "BSplineSmoothLoss",
    "DecayConsistencyLoss", "NISTScalarHuberLoss", "NodeCountLoss",
    "OrthonormalityLoss", "DiracPDELoss", "VirialLoss", "LossBalancer",
]
