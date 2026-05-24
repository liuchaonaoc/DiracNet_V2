"""Data layer: NIST loader, dataset, batch builder.

注意：``BatchBuilder`` 不在此处自动 re-export，需要的调用方请显式
``from rc_diracnet_v2.data.batch_builder import BatchBuilder``。
这是为了避免 ``data → priors → data`` 的循环导入：
``priors.coupling_tensors`` 依赖 ``data.config_parser.Shell``，
而 ``batch_builder`` 反过来依赖 ``priors.coupling_tensors``。
"""

from __future__ import annotations

from .config_parser import Shell, encode_config_to_tensor, parse_config_string
from .dataset import AtomicSpectraDataset, LevelRowDataset, collate_atoms, collate_levelwise
from .samplers import GroupByIonSampler
from .level_encoder import LevelFeatureEncoder, TermVocabulary
from .nist_loader import NISTLevelLoader

__all__ = [
    "Shell",
    "encode_config_to_tensor",
    "parse_config_string",
    "AtomicSpectraDataset",
    "LevelRowDataset",
    "collate_atoms",
    "collate_levelwise",
    "GroupByIonSampler",
    "TermVocabulary",
    "LevelFeatureEncoder",
    "NISTLevelLoader",
]
