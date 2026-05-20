# 08 — Data Pipeline

## 1. Manifest 文件

V2 沿用 V1 的 parquet manifest 格式，**所有列均不变**：

```
Z              int64
ion_charge     int64
parent_config  str          e.g. "1s1"
level_config   str          e.g. "1s1"
J              float64      half-integer
parity         int (0|1)
level_eV       float64      绝对能 (Hartree×27.2114) 或激发能
uncertainty_eV float64
term           str          NIST term symbol
```

V2 直接复用 V1 的 `data/nist_loader.py` 与 `data/dataset.LevelRowDataset`。

## 2. Phase 1 — Hydrogenic 沙盒

### 2.1 `manifest_hydrogenic_v2.parquet`

包含 18 行：

| Z | ion_charge | n | level_config | level_eV (absolute, eV) |
|---|------------|---|--------------|--------------------------|
| 1 | 0 | 1 | "1s1" | -13.6057 |
| 1 | 0 | 2 | "2s1" | -3.4014 |
| 1 | 0 | 3 | "3s1" | -1.5118 |
| 1 | 0 | 4 | "4s1" | -0.8504 |
| 1 | 0 | 5 | "5s1" | -0.5442 |
| 1 | 0 | 6 | "6s1" | -0.3780 |
| 2 | 1 | 1 | "1s1" | -54.4228 |
| 2 | 1 | 2 | "2s1" | -13.6057 |
| ... | ... | ... | ... | ... |

构造脚本：直接抄 V1 `scripts/prepare_hydrogenic_phase1_v2.py`，把 `n-levels: 5 → 6` 并写到 `data_cache/manifest_hydrogenic_v2.parquet`。

### 2.2 `manifest_hydrogenic_v2_extended.parquet`（推荐）

按 V1 Recommendation 2 增加 Z=4..10 类氢离子破除几何退化：

```python
cases = [(Z, Z-1) for Z in range(1, 11)]   # H..Ne
for Z, ion_q in cases:
    for n in range(1, 7):
        ...
```

共 60 行（10 ions × 6 n）。这是 V2 Phase 1 默认数据集。

### 2.3 p-shell 数据（可选 Sprint 3）

把 `2p1, 3p1, 4p1` 加入，验证 V2 对 l > 0 的 Laguerre 形状能力。

```python
for Z, q in cases:
    for n, l in [(2, 1), (3, 1), (4, 1)]:
        cfg = f"{n}p1"
        E_au = -(Z**2) / (2 * n**2)         # NR 退化忽略 fine structure
        ...
```

## 3. Phase 2 — NIST 全量

### 3.1 来源

V1 已有 `data_cache/manifest_nist_full.parquet`，**直接复用，不重新爬取**。
若 V2 需要扩展，复用 V1 `scripts/prepare_nist_dataset.py`。

### 3.2 字段对齐

`align_in_loader: false`（绝对能）是 V2 默认；
对齐留给 `losses/nist_scalar_loss.py` 的 `align_ground` flag 在训练时处理。

### 3.3 拆分策略

| 数据集 | 训练用 | 用途 |
|--------|--------|------|
| `manifest_hydrogenic_v2.parquet` (18 行) | Stage 1 (PDE only) | 门禁基准 |
| `manifest_hydrogenic_v2_extended.parquet` (60 行) | Stage 1 主训练 | 破除几何退化 |
| `manifest_nist_full.parquet` (~36k 行) | Stage 2 (NIST residual) | 残差 fine-tune |

注意：Stage 1 故意 **不用 NIST 大数据集**，只用类氢；这是为了让 PDE 在解析对照
能精确验证，避免在 NIST 噪声里看不清"是否真正学到 Laguerre"。

## 4. Dataset / DataLoader

直接复用 V1：

```python
from rc_diracnet.data.dataset import LevelRowDataset, collate_levelwise
from rc_diracnet.data.samplers import GroupByIonSampler
from rc_diracnet.data.batch_builder import BatchBuilder
```

`BatchBuilder` 在 V2 中 **可以不再调用** `build_coeff_tensor_per_batch`（那是 V1 的方案 C
所需）；可以简化为：

```python
class V2BatchBuilder(BatchBuilder):
    def build(self, batch):
        # 不构造 coeff_tensor；仅添加 V2 需要的 per-orbital phys features
        batch["per_orb_features"] = self._build_per_orb_features(batch)
        return batch
    
    def _build_per_orb_features(self, batch):
        """[B, N_orb, D_phys] with (Z, Z_eff, n, n_star, l, kappa, occ, is_outer)."""
        ...
```

## 5. `per_orb_features` 详细计算

```python
def build_per_orbital_features(batch, n_orb_max=16):
    Z = batch["Z"].float()                     # [B]
    config = batch["config_shells"].long()      # [B, max_seq, 4]
    orb_mask = batch["orb_mask"]                # [B, N_orb]
    B = Z.shape[0]
    
    # truncate to n_orb_max
    n_idx     = config[:, :n_orb_max, 0].float()  # [B, N_orb]
    l_idx     = config[:, :n_orb_max, 1].float()
    j2_idx    = config[:, :n_orb_max, 2].float()
    occ       = config[:, :n_orb_max, 3].float()
    
    # kappa
    kappa = torch.where(j2_idx == 2 * l_idx - 1, l_idx,
                        -(l_idx + 1)).long()             # [B, N_orb]
    
    # Slater Z_eff (placeholder; can be improved)
    Z_eff = Z.unsqueeze(-1) - 0.85 * (n_idx - 1)        # very rough
    Z_eff = Z_eff.clamp_min(0.1)
    
    # effective quantum number n* (Slater)
    n_star = torch.where(
        n_idx <= 3, n_idx,
        torch.where(n_idx == 4, torch.full_like(n_idx, 3.7),
                   torch.where(n_idx == 5, torch.full_like(n_idx, 4.0),
                              torch.full_like(n_idx, 4.2)))
    )
    
    # is_outer
    is_outer = torch.zeros(B, n_orb_max, dtype=Z.dtype, device=Z.device)
    for b in range(B):
        outer_idx = orb_mask[b].sum() - 1
        if outer_idx >= 0:
            is_outer[b, outer_idx] = 1.0
    
    features = torch.stack([
        Z.unsqueeze(-1).expand_as(n_idx),  # [B, N_orb]
        Z_eff, n_idx, n_star, l_idx, kappa.float(), occ, is_outer
    ], dim=-1)  # [B, N_orb, 8]
    return features
```

## 6. 数据准备脚本

```python
# scripts/v2_prepare_hydrogenic.py
"""Generate manifest_hydrogenic_v2.parquet for Stage 1 gate."""
# 直接基于 V1 prepare_hydrogenic_phase1_v2.py 复制；改 cases 范围

# scripts/v2_prepare_hydrogenic_extended.py
"""Generate manifest_hydrogenic_v2_extended.parquet with Z=1..10."""
```

## 7. 数据预校验

进入训练前，必做：

```python
# tests/test_v2_data_pipeline.py
def test_hydrogenic_v2_manifest():
    ds = LevelRowDataset("data_cache/manifest_hydrogenic_v2.parquet")
    assert len(ds) == 18
    for i in range(len(ds)):
        sample = ds[i]
        Z, n = int(sample["Z"]), int(sample["config_shells"][0, 0])
        E_exp = -(Z**2) / (2 * n**2) * HARTREE_eV
        assert abs(sample["E_target"] * HARTREE_eV - E_exp) < 1e-3

def test_per_orb_features_shape():
    ...
```

## 8. 与 V1 完全复用的模块

| 模块 | V2 状态 |
|------|---------|
| `data/nist_loader.py` | **完全复用** |
| `data/dataset.py` | **完全复用** (`LevelRowDataset`) |
| `data/level_encoder.py` | **完全复用** |
| `data/samplers.py` | **完全复用** |
| `data/config_parser.py` | **完全复用** |
| `data/batch_builder.py` | 复用 + 加 `_build_per_orb_features` |
