# 09 — Project Layout

## 1. 顶层目录

```
DiracNet_V2/
├── README.md
├── prompts/                        # 本套提示词
│   ├── 00_overview.md
│   ├── 01_architecture.md
│   ├── 02_bspline_basis.md
│   ├── 03_kan_hypernet.md
│   ├── 04_envelope_kinetic.md
│   ├── 05_physics_losses.md
│   ├── 06_two_stage_training.md
│   ├── 07_evaluation.md
│   ├── 08_data_pipeline.md
│   ├── 09_project_layout.md         (本文件)
│   ├── 10_sprint_plan.md
│   ├── 11_test_plan.md
│   ├── 12_port_from_v1.md
│   ├── 13_action_and_residual_safety.md
│   └── 14_code_structure_generation.md
├── docs/
│   └── design_rationale.md
└── rc_diracnet_v2_project/          # 实际代码项目（按 prompts/10 顺序生成）
    ├── README.md
    ├── pyproject.toml
    ├── requirements.txt
    ├── configs/
    ├── data_cache/                   # gitignored
    ├── data_raw/                     # gitignored
    ├── checkpoints/                  # gitignored
    ├── logs/                         # gitignored
    ├── results/                      # gitignored
    ├── docs/
    ├── rc_diracnet_v2/               # Python package
    │   ├── __init__.py
    │   ├── constants.py              # copy from V1
    │   ├── data/                     # mostly copy from V1
    │   ├── encoders/                 # copy from V1
    │   ├── physics/                  # copy from V1
    │   ├── basis/                    # NEW: B-spline basis
    │   ├── kan/                      # NEW: KAN hypernet
    │   ├── readout/                  # rewrite (envelope, bspline readout, ortho)
    │   ├── losses/                   # mostly copy from V1; add node + asymptotic
    │   ├── models/                   # NEW: DiracNetV2 main
    │   ├── training/                 # NEW: two_stage_trainer
    │   ├── utils/                    # copy from V1
    │   └── observables/              # copy from V1 (后期 Phase 用)
    ├── scripts/
    └── tests/
```

## 2. `rc_diracnet_v2/` 内部目录

### 2.1 `constants.py` (copy from V1)

```
α, c, Hartree↔eV 单位换算
```

### 2.2 `data/` (mostly copy from V1)

```
data/
├── __init__.py
├── nist_loader.py            # copy
├── dataset.py                # copy
├── samplers.py               # copy
├── level_encoder.py          # copy
├── config_parser.py          # copy
└── batch_builder.py          # copy + add `_build_per_orb_features`
```

### 2.3 `encoders/` (copy from V1)

```
encoders/
├── __init__.py
└── quantum_encoder.py        # copy
```

### 2.4 `physics/` (copy from V1)

```
physics/
├── __init__.py
├── dirac_operator.py         # copy
├── slater_lda.py             # copy
├── hydrogenic_spinor.py      # copy
├── hydrogenic_analytic.py    # NEW: analytic Laguerre P_nl for gate
└── (其它如 continuum / eigen_solver / hamiltonian_assembler 暂不用)
```

### 2.5 `basis/` (**NEW**)

```
basis/
├── __init__.py
└── bspline_basis.py          # BSplineBasis class
```

### 2.6 `kan/` (**NEW**)

```
kan/
├── __init__.py
├── kan_coeff_net.py          # KANCoeffNet (主)
├── mlp_coeff_net.py          # MLPCoeffNet (备选)
└── visualization.py          # 边样条可视化
```

### 2.7 `readout/` (rewrite)

```
readout/
├── __init__.py
├── envelope.py               # NEW: AnalyticEnvelope (无 MLP)
├── bspline_readout.py        # NEW: BSplineReadout (无可学习参数)
└── orthogonalizer.py         # copy from V1
```

### 2.8 `losses/` (mostly copy)

```
losses/
├── __init__.py
├── pde_loss.py               # copy
├── orthonormality_loss.py    # copy
├── nist_scalar_loss.py       # copy
├── node_count_loss.py        # NEW
├── action_loss.py            # NEW: Bohr-Sommerfeld action quantisation
├── asymptotic_loss.py        # NEW
├── bspline_smooth_loss.py    # NEW
└── loss_balancer.py          # copy (for compatibility)
```

### 2.9 `models/` (**NEW**)

```
models/
├── __init__.py
├── dirac_net_v2.py           # DiracNetV2 main model
└── level_residual_head.py    # Bounded Δ_residual (Stage 2)
```

### 2.10 `training/` (**NEW**)

```
training/
├── __init__.py
├── two_stage_trainer.py      # 主 trainer (Stage 1 / Stage 2 状态机)
├── stage_gate.py             # check_stage1_gate / rollback
├── optimizer_factory.py      # copy + extend (per-module lr)
└── scheduler.py              # copy
```

### 2.11 `utils/` (copy from V1)

```
utils/
├── __init__.py
├── grid.py                   # copy (核心)
├── autograd_helpers.py       # copy
├── checkpoint.py             # copy
├── config.py                 # copy
├── hungarian.py              # copy (Phase 2 备用)
├── logging.py                # copy
├── numeric.py                # copy
└── tensor_ops.py             # copy
```

## 3. `configs/`

```
configs/
├── default.yaml                          # base config
├── v2_phase1_stage1_pde_only.yaml        # Stage 1: hydrogenic, PDE only
├── v2_phase1_stage1_pde_only_extended.yaml  # Stage 1: Z=1..10
├── v2_phase1_stage2_nist.yaml            # Stage 2: NIST residual fine-tune
├── v2_phase1_full.yaml                   # Stage 1 + Stage 2 combined
├── v2_phase1_loo.yaml                    # LOO test
├── v2_smoke.yaml                         # 短训练验证不报错
└── v2_phase2_nist_full.yaml              # NIST 全量 (Sprint 4)
```

每个 config 必须 `defaults: default.yaml` 然后只 override 必要字段。

### 3.1 `default.yaml` 示例

```yaml
seed: 42
device: cuda

grid:
  r_min: 1.0e-4
  r_max: 50.0
  n_grid: 512
  scheme: loglinear

encoder:
  d_embed_z: 64
  d_embed_shell: 32
  d_gru_hidden: 128
  d_cond: 256
  max_z: 110
  max_n: 30
  max_l: 7
  max_seq: 32
  term_vocab_size: 256

bspline:
  n_basis: 32
  order: 3
  knot_scheme: log
  r_min: 1.0e-4
  r_max: 50.0
  force_c0_zero: true

kan:
  kind: kan                    # or mlp
  layers_hidden: [264, 128, 64, 33]   # 264 = d_cond + 8; 33 = 1 + n_basis
  grid_size: 8
  spline_order: 3
  scale_spline: 0.1
  grid_range: [-1, 1]

envelope:
  use_relativistic_gamma: false  # γ = |κ| (low Z) vs sqrt(κ² - (Zα)²)
  lambda_min: 0.05
  lambda_max: 20.0
  lam_log_res_clamp: 2.0

readout:
  n_orb_max: 16
  q_residual_scale: 0.0          # Stage 1 keeps 0

stage1:
  n_epochs: 300
  weights:
    pde: 10.0
    ortho: 1.0
    node: 0.0                 # warmup 后开到 0.1
    action: 0.0               # L_PDE 稳定后开到 0.05
    asym: 0.01
    smooth: 1.0e-4
  gate:
    cos_threshold: 0.99
    e_orb_meV_threshold: 1.0
    lambda_rel_threshold: 0.05
    L_PDE_threshold: 1.0e-3
    L_action_BS_threshold: 1.0e-3
    test_manifest: data_cache/manifest_hydrogenic_v2.parquet
  
stage2:
  n_epochs: 50
  enabled: false              # 默认只做 V2.0-physics；calibration 显式开启
  delta_max_meV: 5.0          # Phase 1 默认 1-5 meV；light atoms 可设 20 meV
  warmup_epochs: 20
  w_nist_max: 0.3
  freeze_kan_main: true
  reduce_w_nist_on_rollback: 0.5
  residual_ratio_threshold: 0.2
  require_ood_pass: true

losses:
  nist_huber_delta: 1.0e-3       # δ in Hartree (~27 meV)
  align_ground: false
  balancer: manual

optimizer:
  name: adamw
  lr_kan: 3.0e-4
  lr_encoder: 1.0e-4
  lr_residual: 1.0e-4
  weight_decay: 1.0e-4
  grad_clip: 5.0

scheduler:
  name: warmup_cosine
  warmup_steps: 200
  total_steps: 6000

training:
  batch_size: 8
  val_every: 5
  ckpt_dir: ./checkpoints/v2_default
  log_dir: ./logs/v2_default

dataset:
  manifest: data_cache/manifest_hydrogenic_v2.parquet
  max_orb: 16
  max_seq: 32
  group_by_ion: true
  val_ion_holdout: true
  val_holdout_per_ion: 2
  align_in_loader: false
```

## 4. `scripts/`

```
scripts/
├── v2_prepare_hydrogenic.py              # 18 行 Z=1..3
├── v2_prepare_hydrogenic_extended.py     # 60 行 Z=1..10
├── v2_train.py                           # 主训练入口（stage 1 → stage 2 自动）
├── v2_train_stage1_only.py               # 仅 Stage 1（调试）
├── v2_train_stage2_from.py               # 从指定 Stage 1 ckpt 起 Stage 2
├── v2_evaluate.py                        # 全量评估
├── v2_gate_analytic.py                   # 解析对照门禁
├── v2_visualize_kan.py                   # KAN 边样条可视化
├── v2_plot_wavefunctions.py              # P_model vs P_analytic
├── v2_plot_training.py                   # loss 曲线
└── v2_diagnose_physics_chain.py          # 物理链路诊断（从 V1 copy + 改）
```

## 5. `tests/`

```
tests/
├── __init__.py
├── conftest.py                            # 公共 fixture (grid, sample batch)
├── test_v2_bspline_basis.py
├── test_v2_envelope.py
├── test_v2_bspline_readout.py
├── test_v2_kinetic_balance.py
├── test_v2_kan_coeff_net.py
├── test_v2_node_count_loss.py
├── test_v2_action_loss.py
├── test_v2_asymptotic_loss.py
├── test_v2_smooth_loss.py
├── test_v2_pde_loss.py                    # 复用 V1 test
├── test_v2_dirac_operator_h1s.py          # 复用 V1 test
├── test_v2_dataset.py
├── test_v2_per_orb_features.py
├── test_v2_gate_pass_on_analytic.py       # 解析输入下门禁直接 PASS
├── test_v2_gate_fail_on_random.py
├── test_v2_two_stage_trainer_smoke.py     # 3 epoch smoke
├── test_v2_residual_safety.py             # Δ_residual anti-cheating
└── test_v2_full_smoke.py                  # 端到端冒烟
```

## 6. `pyproject.toml` / `requirements.txt`

```
# requirements.txt
torch >= 2.0
numpy
scipy                            # for B-spline reference & Laguerre
pandas
pyarrow                          # parquet
matplotlib
pyyaml
efficient-kan                    # optional; fall back to MLP if not installed
tqdm
```

## 7. 与 V1 的目录对比

```
V1: rc_diracnet/reservoir/{basis_generator, envelope}
V2: rc_diracnet_v2/basis/{bspline_basis}
    rc_diracnet_v2/readout/{envelope, bspline_readout}

V1: rc_diracnet/readout/wavefunction_readout
V2: rc_diracnet_v2/readout/bspline_readout

V1: rc_diracnet/models/{rc_diracnet, rc_diracnet_levelwise, level_energy_head}
V2: rc_diracnet_v2/models/{dirac_net_v2, level_residual_head}

V1: rc_diracnet/training/{trainer, levelwise_trainer, scheduler, optimizer_factory}
V2: rc_diracnet_v2/training/{two_stage_trainer, stage_gate, scheduler, optimizer_factory}
```
