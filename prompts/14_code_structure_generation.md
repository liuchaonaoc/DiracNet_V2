# 14 — Code Structure Generation Prompt

> 本文件不是算法解释，而是给代码生成 Agent 的 **工程骨架生成提示词**。
> 执行者应先按这里搭出 `rc_diracnet_v2_project/` 的可运行代码结构，再逐个实现
> `02`-`13` 中的细节。

## 0. Agent 角色

你是一个负责实现 DiracNet V2 的代码生成 Agent。你的任务不是重新设计理论，
而是 **严格按照现有文档生成项目骨架、模块边界、文件职责、配置、脚本和测试结构**。

必须优先阅读并遵守：

1. `docs/design_rationale.md`
2. `prompts/00_overview.md`
3. `prompts/01_architecture.md`
4. `prompts/02_bspline_basis.md`
5. `prompts/03_kan_hypernet.md`
6. `prompts/04_envelope_kinetic.md`
7. `prompts/05_physics_losses.md`
8. `prompts/06_two_stage_training.md`
9. `prompts/07_evaluation.md`
10. `prompts/08_data_pipeline.md`
11. `prompts/09_project_layout.md`
12. `prompts/10_sprint_plan.md`
13. `prompts/11_test_plan.md`
14. `prompts/12_port_from_v1.md`
15. `prompts/13_action_and_residual_safety.md`

若文档之间出现冲突，以更新后的安全约束为准：

```text
prompts/13_action_and_residual_safety.md
prompts/06_two_stage_training.md
prompts/07_evaluation.md
```

特别是：`E_orb-only` 是主结果；`Δ_residual` 只是 optional calibration。

## 1. 不可违反的硬约束

### 1.1 训练纪律

- Stage 1 绝对不能让 NIST loss 进入 backward 图。
- Stage 1 loss 只允许：

```text
L_PDE + L_ortho + L_node + L_action_BS + L_asym + L_smooth
```

- `L_action_BS` 是 Bohr-Sommerfeld 作用量量子化约束，不是“作用量本身”。
- Stage 2 默认关闭；如果启用，只能作为 calibration。
- `v2_evaluate.py` 必须同时输出：

```text
E_orb-only metrics      ← 主结果
E_orb + Δ_res metrics   ← 次结果，若启用 residual
residual safety metrics
physics gate metrics
```

### 1.2 反 cheating 纪律

禁止：

- `zn_bias_table`
- `(Z, n)` 直接 lookup bias
- 每个 manifest row 一个 learnable bias
- Stage 1 使用 NIST 梯度
- 只报告 calibrated 指标
- Stage 2 默认解冻 KAN 主路径

必须：

- 主报告先给 `E_orb-only`
- residual safety verdict 不通过时，不得输出 `PASS`
- Stage 2 后复查 `|cos|`, `λ drift`, `L_PDE`, `L_action_BS`
- LOO / OOD 至少覆盖 leave-one-n、leave-one-Z、leave-one-ion

### 1.3 V1 复用纪律

从 `DiracNet_V1/rc_diracnet_project/rc_diracnet/` 复制可复用模块到
`rc_diracnet_v2_project/rc_diracnet_v2/`，不要在 V2 中直接 import `rc_diracnet`。

可复制清单以 `prompts/12_port_from_v1.md` 为准。

## 2. 目标目录树

生成以下目录：

```text
DiracNet_V2/
└── rc_diracnet_v2_project/
    ├── README.md
    ├── pyproject.toml
    ├── requirements.txt
    ├── configs/
    │   ├── default.yaml
    │   ├── v2_smoke.yaml
    │   ├── v2_phase1_stage1_pde_only.yaml
    │   ├── v2_phase1_stage1_pde_only_extended.yaml
    │   ├── v2_phase1_stage2_nist.yaml
    │   ├── v2_phase1_full.yaml
    │   ├── v2_phase1_loo.yaml
    │   └── v2_phase2_nist_full.yaml
    ├── data_cache/
    ├── data_raw/
    ├── checkpoints/
    ├── logs/
    ├── results/
    ├── docs/
    ├── rc_diracnet_v2/
    │   ├── __init__.py
    │   ├── constants.py
    │   ├── data/
    │   ├── encoders/
    │   ├── physics/
    │   ├── basis/
    │   ├── kan/
    │   ├── readout/
    │   ├── losses/
    │   ├── models/
    │   ├── training/
    │   ├── utils/
    │   └── observables/
    ├── scripts/
    └── tests/
```

`data_cache/`, `data_raw/`, `checkpoints/`, `logs/`, `results/` 应写入 `.gitignore`。

## 3. Python package 文件清单

### 3.1 Root

```text
rc_diracnet_v2/
├── __init__.py
└── constants.py
```

`__init__.py`：

```python
__version__ = "2.0.0"
```

`constants.py` 从 V1 复制，包含 `ALPHA`, `C_LIGHT`, `HARTREE_eV` 等。

### 3.2 `utils/`

复制 V1：

```text
utils/
├── __init__.py
├── autograd_helpers.py
├── checkpoint.py
├── config.py
├── grid.py
├── hungarian.py
├── logging.py
├── numeric.py
└── tensor_ops.py
```

要求：

- `RadialGrid` 保持 V1 行为。
- 所有 imports 改为 `rc_diracnet_v2.*`。

### 3.3 `data/`

```text
data/
├── __init__.py
├── batch_builder.py
├── config_parser.py
├── dataset.py
├── level_encoder.py
├── nist_loader.py
└── samplers.py
```

复制 V1 后改造 `batch_builder.py`：

```python
class V2BatchBuilder:
    def __init__(self, n_orb_max: int = 16) -> None: ...
    def build(self, batch: dict[str, Any]) -> dict[str, Any]: ...
    def _build_per_orb_features(self, batch: dict[str, Any]) -> Tensor: ...
```

`per_orb_features` shape:

```text
[B, N_orb, 8] = [Z, Z_eff, n, n_star, l, kappa, occ, is_outer]
```

### 3.4 `encoders/`

复制 V1：

```text
encoders/
├── __init__.py
└── quantum_encoder.py
```

### 3.5 `physics/`

```text
physics/
├── __init__.py
├── dirac_operator.py
├── slater_lda.py
├── hydrogenic_spinor.py
└── hydrogenic_analytic.py
```

前三个从 V1 复制。

新增 `hydrogenic_analytic.py`：

- `hydrogenic_energy(Z, n) -> float`
- `hydrogenic_P_analytic(r, Z, n, l=0) -> Tensor`
- `cosine_signed(P_model, P_ref, grid) -> Tensor`

### 3.6 `basis/`

```text
basis/
├── __init__.py
└── bspline_basis.py
```

`BSplineBasis` 必须实现：

```python
class BSplineBasis(nn.Module):
    def __init__(
        self,
        n_basis: int = 32,
        order: int = 3,
        r_min: float = 1e-4,
        r_max: float = 50.0,
        knot_scheme: str = "log",
        boundary_clamp: bool = True,
    ) -> None: ...

    @torch.no_grad()
    def precompute(self, r_grid: Tensor) -> None: ...

    def forward(self) -> tuple[Tensor, Tensor, Tensor]: ...
```

Buffers:

```text
knots [K + p + 1]
B     [K, N_grid]
dB    [K, N_grid]
d2B   [K, N_grid]
```

### 3.7 `kan/`

```text
kan/
├── __init__.py
├── kan_coeff_net.py
├── mlp_coeff_net.py
└── visualization.py
```

必须先实现 `MLPCoeffNet` fallback，再接 `KANCoeffNet`。

共同接口：

```python
def forward(
    self,
    h_cond: Tensor,           # [B, D_cond]
    per_orb_features: Tensor, # [B, N_orb, 8]
    orb_mask: Tensor,         # [B, N_orb]
) -> tuple[Tensor, Tensor]:
    """Return lam_log_res [B,N_orb], c [B,N_orb,K]."""
```

硬要求：

- 输出层 zero-init。
- 初始 `lam_log_res = 0`, `c = 0`。
- 输入归一化到 `[-1, 1]`。
- `c[..., 0]` 后续在 readout 强制为 0。

### 3.8 `readout/`

```text
readout/
├── __init__.py
├── envelope.py
├── bspline_readout.py
└── orthogonalizer.py
```

`orthogonalizer.py` 从 V1 复制。

`envelope.py`：

```python
class AnalyticEnvelope(nn.Module):
    def forward(
        self,
        r_grid: Tensor,
        lam: Tensor,
        kappa: Tensor,
        Z: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]: ...
```

`bspline_readout.py`：

```python
class BSplineReadout(nn.Module):
    def forward(
        self,
        c_P: Tensor,
        env: Tensor,
        denv: Tensor,
        d2env: Tensor,
        B_basis: Tensor,
        dB_basis: Tensor,
        d2B_basis: Tensor,
        kappa: Tensor,
        r_grid: Tensor,
        orb_mask: Tensor,
        c_Q: Tensor | None = None,
    ) -> dict[str, Tensor]: ...
```

必须输出：

```text
P, Q, dPdr, dQdr, d2Pdr, f, df, d2f
```

Q 默认由 kinetic balance 构造，Stage 1 禁止 free-Q。

### 3.9 `losses/`

```text
losses/
├── __init__.py
├── pde_loss.py
├── orthonormality_loss.py
├── nist_scalar_loss.py
├── loss_balancer.py
├── node_count_loss.py
├── action_loss.py
├── asymptotic_loss.py
└── bspline_smooth_loss.py
```

从 V1 复制：

- `pde_loss.py`
- `orthonormality_loss.py`
- `nist_scalar_loss.py`
- `loss_balancer.py`

新增：

- `NodeCountLoss`
- `BohrSommerfeldActionLoss`
- `AsymptoticTailLoss`
- `BSplineSmoothLoss`

`BohrSommerfeldActionLoss` 的接口与公式必须完全遵守 `prompts/13_action_and_residual_safety.md`。

### 3.10 `models/`

```text
models/
├── __init__.py
├── dirac_net_v2.py
└── level_residual_head.py
```

`DiracNetV2` 负责端到端 forward：

```text
batch
→ QuantumEncoder + LevelFeatureEncoder
→ V2BatchBuilder/per_orb_features
→ KAN/MLP coeff net
→ λ = (Z_eff/n) * exp(lam_log_res)
→ AnalyticEnvelope
→ BSplineBasis
→ BSplineReadout
→ lowdin_orthonormalize
→ DiracRadialOperator
→ E_orb
→ E_orb_sum
→ optional Δ_residual
```

Forward 输出必须包含：

```python
{
    "h_cond": h_cond,
    "per_orb_features": per_orb_features,
    "lam": lam,
    "lam_log_res": lam_log_res,
    "c_raw": c,
    "wavefunctions": {"P": P, "Q": Q, "dPdr": dPdr, "dQdr": dQdr},
    "v_eff": V_eff,
    "E_orb": E_orb,
    "macro": {
        "E_orb_sum": E_orb_sum,
        "E_pred_main": E_orb_sum,
        "delta_residual": delta_residual_or_none,
        "E_pred_calibrated": E_orb_sum + delta_residual_or_E_orb_sum,
    },
}
```

`LevelResidualHead`：

- 默认不实例化或不启用。
- zero-init。
- `delta_max_meV` 默认 5。
- 禁止 lookup table。

### 3.11 `training/`

```text
training/
├── __init__.py
├── _helpers.py
├── optimizer_factory.py
├── scheduler.py
├── stage_gate.py
└── two_stage_trainer.py
```

`stage_gate.py`：

- `check_stage1_gate(model, loader, thresholds) -> GateReport`
- `residual_safety_verdict(metrics: dict[str, float]) -> str`
- `GateReport.to_markdown()`

`two_stage_trainer.py`：

- `run_stage1()`
- `run_stage2_calibration()`
- `fit()`

Stage 1 backward 不得包含 NIST。

## 4. Scripts

生成：

```text
scripts/
├── v2_prepare_hydrogenic.py
├── v2_prepare_hydrogenic_extended.py
├── v2_train.py
├── v2_train_stage1_only.py
├── v2_train_stage2_from.py
├── v2_evaluate.py
├── v2_gate_analytic.py
├── v2_visualize_kan.py
├── v2_plot_wavefunctions.py
├── v2_plot_training.py
└── v2_diagnose_physics_chain.py
```

`v2_train.py`：

- 默认只跑 Stage 1。
- 只有 `cfg.stage2.enabled: true` 时才跑 Stage 2 calibration。

`v2_evaluate.py`：

- 必须输出 `E_orb-only` 主指标。
- 若 residual 启用，额外输出 calibrated 指标。
- 必须输出 residual safety verdict。

## 5. Configs

生成 `configs/default.yaml`，至少包含：

```yaml
seed: 42
device: cuda

grid:
  r_min: 1.0e-4
  r_max: 50.0
  n_grid: 512
  scheme: loglinear

bspline:
  n_basis: 32
  order: 3
  knot_scheme: log
  force_c0_zero: true

kan:
  kind: mlp        # Sprint 1 fallback; later kan
  grid_size: 8
  spline_order: 3

stage1:
  n_epochs: 300
  weights:
    pde: 10.0
    ortho: 1.0
    node: 0.0
    action: 0.0
    asym: 0.01
    smooth: 1.0e-4
  warmup:
    node_start_epoch: 10
    action_start_epoch: 20
    action_pde_threshold: 1.0e-2
    node_weight_after_warmup: 0.1
    action_weight_after_warmup: 0.05
  gate:
    cos_threshold: 0.99
    e_orb_meV_threshold: 1.0
    lambda_rel_threshold: 0.05
    L_PDE_threshold: 1.0e-3
    L_action_BS_threshold: 1.0e-3

stage2:
  enabled: false
  delta_max_meV: 5.0
  w_nist_max: 0.3
  warmup_epochs: 20
  freeze_kan_main: true
  residual_ratio_threshold: 0.2
  require_ood_pass: true
```

## 6. Tests

生成以下测试文件：

```text
tests/
├── conftest.py
├── test_v1_modules_still_work.py
├── test_v2_bspline_basis.py
├── test_v2_envelope.py
├── test_v2_bspline_readout.py
├── test_v2_kinetic_balance.py
├── test_v2_kan_coeff_net.py
├── test_v2_node_count_loss.py
├── test_v2_action_loss.py
├── test_v2_asymptotic_loss.py
├── test_v2_smooth_loss.py
├── test_v2_pde_loss.py
├── test_v2_dirac_operator_h1s.py
├── test_v2_dataset.py
├── test_v2_per_orb_features.py
├── test_v2_gate_pass_on_analytic.py
├── test_v2_gate_fail_on_random.py
├── test_v2_residual_safety.py
├── test_v2_two_stage_trainer_smoke.py
└── test_v2_full_smoke.py
```

Minimum first milestone:

```bash
pytest tests/test_v1_modules_still_work.py \
       tests/test_v2_bspline_basis.py \
       tests/test_v2_envelope.py \
       tests/test_v2_bspline_readout.py \
       tests/test_v2_action_loss.py -v
```

## 7. Implementation Order

按以下顺序生成代码：

1. 项目文件：`pyproject.toml`, `requirements.txt`, `.gitignore`, `README.md`
2. 复制 V1 modules，修 imports
3. `basis/bspline_basis.py`
4. `readout/envelope.py`
5. `readout/bspline_readout.py`
6. `kan/mlp_coeff_net.py`
7. `data/batch_builder.py` V2 features
8. `models/dirac_net_v2.py`
9. `losses/node_count_loss.py`
10. `losses/action_loss.py`
11. `losses/asymptotic_loss.py`
12. `losses/bspline_smooth_loss.py`
13. `physics/hydrogenic_analytic.py`
14. `training/stage_gate.py`
15. `training/two_stage_trainer.py` Stage 1 only
16. scripts for prepare/train/evaluate/gate
17. tests
18. optional `kan/kan_coeff_net.py`
19. optional Stage 2 calibration

Do not implement Stage 2 before Stage 1 gate can run.

## 8. Acceptance Criteria

代码结构生成完成后，至少满足：

```bash
python -c "import rc_diracnet_v2; print(rc_diracnet_v2.__version__)"
pytest tests/test_v1_modules_still_work.py -v
```

Sprint 1 完成后：

```bash
pytest tests/test_v2_bspline_basis.py \
       tests/test_v2_envelope.py \
       tests/test_v2_bspline_readout.py \
       tests/test_v2_kan_coeff_net.py \
       tests/test_v2_action_loss.py \
       tests/test_v2_dirac_net_v2_forward.py -v
```

Sprint 2 完成后：

```bash
python scripts/v2_train_stage1_only.py --config configs/v2_phase1_stage1_pde_only.yaml
python scripts/v2_gate_analytic.py \
  --ckpt checkpoints/v2_phase1_stage1/stage1_passed.pt \
  --manifest data_cache/manifest_hydrogenic_v2.parquet
```

Gate 必须包含：

```text
cos_min
lambda_drift_max
E_orb_error_max
L_PDE
L_action_BS
VERDICT
```

## 9. Output Style for the Code Generation Agent

每完成一批文件，输出：

```text
Created/updated:
- path/to/file.py — one-line purpose

Validation:
- command run
- result

Next:
- next file group
```

不要在没有测试的情况下宣布 Sprint 完成。

## 10. Final Reminder

DiracNet V2 的目标不是“把 NIST 数字拟合小”，而是：

```text
先让 E_orb-only 成为可信的物理基准；
再把 Δ_residual 限制为可解释、可关闭、可审计的 calibration。
```

任何代码结构若让 residual、lookup、NIST 梯度绕过这个原则，都必须重写。
