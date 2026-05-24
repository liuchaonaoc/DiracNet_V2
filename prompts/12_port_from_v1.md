# 12 — Port from V1 (Reuse Checklist)

> 把 `DiracNet_V1/rc_diracnet_project/` 中可直接复用的模块列出来。
> **原则：在 V2 项目里 `from rc_diracnet_v2.xxx import ...`**，
> 即把 V1 文件 **复制** 到 V2 包内（不是 `import rc_diracnet`），
> 这样 V1/V2 完全独立，删除 V1 不影响 V2。

## 1. 完全复用（原样复制）

| V1 路径 | V2 路径 | 备注 |
|---------|---------|------|
| `rc_diracnet/constants.py` | `rc_diracnet_v2/constants.py` | – |
| `rc_diracnet/utils/grid.py` | `rc_diracnet_v2/utils/grid.py` | 核心 |
| `rc_diracnet/utils/autograd_helpers.py` | `rc_diracnet_v2/utils/autograd_helpers.py` | – |
| `rc_diracnet/utils/checkpoint.py` | `rc_diracnet_v2/utils/checkpoint.py` | – |
| `rc_diracnet/utils/config.py` | `rc_diracnet_v2/utils/config.py` | – |
| `rc_diracnet/utils/hungarian.py` | `rc_diracnet_v2/utils/hungarian.py` | Phase 2 备 |
| `rc_diracnet/utils/logging.py` | `rc_diracnet_v2/utils/logging.py` | – |
| `rc_diracnet/utils/numeric.py` | `rc_diracnet_v2/utils/numeric.py` | – |
| `rc_diracnet/utils/tensor_ops.py` | `rc_diracnet_v2/utils/tensor_ops.py` | – |
| `rc_diracnet/data/nist_loader.py` | `rc_diracnet_v2/data/nist_loader.py` | – |
| `rc_diracnet/data/dataset.py` | `rc_diracnet_v2/data/dataset.py` | – |
| `rc_diracnet/data/samplers.py` | `rc_diracnet_v2/data/samplers.py` | – |
| `rc_diracnet/data/config_parser.py` | `rc_diracnet_v2/data/config_parser.py` | – |
| `rc_diracnet/data/level_encoder.py` | `rc_diracnet_v2/data/level_encoder.py` | – |
| `rc_diracnet/encoders/quantum_encoder.py` | `rc_diracnet_v2/encoders/quantum_encoder.py` | – |
| `rc_diracnet/physics/dirac_operator.py` | `rc_diracnet_v2/physics/dirac_operator.py` | 核心 |
| `rc_diracnet/physics/slater_lda.py` | `rc_diracnet_v2/physics/slater_lda.py` | density helper |
| `rc_diracnet/physics/hydrogenic_spinor.py` | `rc_diracnet_v2/physics/hydrogenic_spinor.py` | 测试 fixture |
| `rc_diracnet/readout/orthogonalizer.py` | `rc_diracnet_v2/readout/orthogonalizer.py` | Löwdin |
| `rc_diracnet/losses/pde_loss.py` | `rc_diracnet_v2/losses/pde_loss.py` | Dirac PDE 残差 |
| `rc_diracnet/losses/orthonormality_loss.py` | `rc_diracnet_v2/losses/orthonormality_loss.py` | – |
| `rc_diracnet/losses/nist_scalar_loss.py` | `rc_diracnet_v2/losses/nist_scalar_loss.py` | – |
| `rc_diracnet/losses/loss_balancer.py` | `rc_diracnet_v2/losses/loss_balancer.py` | – |

复制命令（Sprint 0 一次性）：

```bash
mkdir -p rc_diracnet_v2_project/rc_diracnet_v2
cd rc_diracnet_v2_project/rc_diracnet_v2

V1=/home/chaos/workspace2/DiracNet_V1/rc_diracnet_project/rc_diracnet

# constants
cp "$V1/constants.py" .

# utils
mkdir -p utils && cp "$V1/utils/"{grid,autograd_helpers,checkpoint,config,hungarian,logging,numeric,tensor_ops,__init__}.py utils/

# data
mkdir -p data && cp "$V1/data/"{nist_loader,dataset,samplers,config_parser,level_encoder,batch_builder,__init__}.py data/

# encoders
mkdir -p encoders && cp "$V1/encoders/"{quantum_encoder,__init__}.py encoders/

# physics
mkdir -p physics && cp "$V1/physics/"{dirac_operator,slater_lda,hydrogenic_spinor,__init__}.py physics/

# readout (only orthogonalizer)
mkdir -p readout && cp "$V1/readout/"{orthogonalizer,__init__}.py readout/

# losses (4 个)
mkdir -p losses && cp "$V1/losses/"{pde_loss,orthonormality_loss,nist_scalar_loss,loss_balancer,__init__}.py losses/

# (NEW 目录) 留空
mkdir -p basis kan models training observables
```

然后把所有 `from rc_diracnet.xxx import` 改为 `from rc_diracnet_v2.xxx import`，
可以用 `sed`：

```bash
find rc_diracnet_v2 -name "*.py" -exec \
    sed -i 's/from rc_diracnet\./from rc_diracnet_v2./g; s/import rc_diracnet\b/import rc_diracnet_v2/g' {} \;
```

或 `ruff`/`isort` 配套 codemod。

## 2. 复用 + 改动

| V1 路径 | V2 路径 | 改动 |
|---------|---------|------|
| `rc_diracnet/data/batch_builder.py` | `rc_diracnet_v2/data/batch_builder.py` | 加 `_build_per_orb_features` 方法；不再调 `build_coeff_tensor_per_batch`（V2 第一版不用） |
| `rc_diracnet/training/optimizer_factory.py` | `rc_diracnet_v2/training/optimizer_factory.py` | 加 `build_optimizer(stage=1|2)`，支持 per-module lr |
| `rc_diracnet/training/scheduler.py` | `rc_diracnet_v2/training/scheduler.py` | 直接复用 |
| `rc_diracnet/training/trainer.py` 中的 `_all_grads_finite`, `_fmt_losses`, `_to_device` 等 helper | `rc_diracnet_v2/training/_helpers.py` | 提取 helper 后复用 |

## 3. 重新设计（不复用）

| V1 路径 | V2 替换 | 原因 |
|---------|---------|------|
| `rc_diracnet/reservoir/basis_generator.py` (RC tanh) | `rc_diracnet_v2/basis/bspline_basis.py` | 基底类型完全替换 |
| `rc_diracnet/reservoir/envelope.py` (含 λ-MLP) | `rc_diracnet_v2/readout/envelope.py` (纯解析) | λ 改外部输入 |
| `rc_diracnet/readout/wavefunction_readout.py` (含 W-MLP) | `rc_diracnet_v2/readout/bspline_readout.py` (无参) | 系数改外部输入 |
| `rc_diracnet/models/rc_diracnet_levelwise.py` | `rc_diracnet_v2/models/dirac_net_v2.py` | 主架构改 |
| `rc_diracnet/models/level_energy_head.py` (含 Δ_term + zn_bias) | `rc_diracnet_v2/models/level_residual_head.py` (optional Δ_res calibration) | 默认不启用；Phase 1 cap 1-5 meV，light atoms cap 20 meV；禁止 lookup |
| `rc_diracnet/training/levelwise_trainer.py` | `rc_diracnet_v2/training/two_stage_trainer.py` | 协议改两阶段 |

## 4. **不复制** 的模块

| V1 路径 | 为什么不复制 |
|---------|--------------|
| `rc_diracnet/reservoir/*` | RC 基底已被 B-spline 替换 |
| `rc_diracnet/readout/polynomial_readout.py`（已在 V1 B9 回滚时删除） | – |
| `rc_diracnet/models/rc_diracnet.py`（旧 [B,M] 接口） | V1.3 已弃用 |
| `rc_diracnet/physics/hamiltonian_assembler.py`、`eigen_solver.py`、`continuum.py`、`radial_integrals.py`、`poisson_solver.py` | V2 第一版不做多电子 HF / Sturmian / 连续谱 |
| `rc_diracnet/priors/*`（angular_momentum, coupling_tensors, selection_rules） | V2 第一版不做 CSF / Racah |
| `rc_diracnet/observables/*`（collision, radiative） | 后期 Phase 用 |
| `rc_diracnet/losses/{poisson_loss, wkb_loss, nist_loss}` | 当前不接入 |

## 5. V1 脚本中可参考但需要重写的

| V1 脚本 | V2 重写 | 改动 |
|---------|---------|------|
| `scripts/prepare_hydrogenic_phase1_v2.py` | `scripts/v2_prepare_hydrogenic.py` + `v2_prepare_hydrogenic_extended.py` | 加 Z=4..10，加 p-shell 可选 |
| `scripts/train_levelwise.py` | `scripts/v2_train.py` | 两阶段；KAN 输入归一化等 |
| `scripts/evaluate_levelwise.py` | `scripts/v2_evaluate.py` | 加 Stage 1/2 区分、解析对照 |
| `scripts/diagnose_physics_chain.py` | `scripts/v2_diagnose_physics_chain.py` | 复用 logic，更换 model 类 |
| `scripts/diagnose_levelwise_ckpt.py` | `scripts/v2_diagnose_v2_ckpt.py` | KAN 边样条 dump |
| `scripts/plot_wavefunctions.py` | `scripts/v2_plot_wavefunctions.py` | 同时绘制 P_analytic |

## 6. V1 配置参考

V2 起步阅读这几个 V1 yaml 作为对照：

- `configs/phase1_levelwise.yaml`（基线）
- `configs/phase1_levelwise_absolute.yaml`（R9 baseline）
- `configs/phase1_levelwise_absolute_b9.yaml`（B9 多尺度 RC）
- `configs/phase1_levelwise_absolute_b9_loo.yaml`（B9 LOO）

V2 完全 **不要** 用 `defaults: phase1_levelwise.yaml` 等指向 V1 的文件；
V2 配置必须自包含（`defaults: default.yaml` 仅指 V2 自己的 default）。

## 7. V1 文档强烈推荐先读

- `DiracNet_V1/rc_diracnet_project/docs/V1_3_sprint1_retrospective.md` —
  **V2 全部设计的因果起点**；不读不许动手。
- `DiracNet_V1/prompt_RC_V1_3_architecture.md` — V1 提出 Scheme A/B/C 时的设计稿
- `DiracNet_V1/frame_RC_V1_3.md` — V1 实施记录

## 8. 复用清单的 Sprint 0 验收

```bash
cd rc_diracnet_v2_project
pip install -e .
python -c "
from rc_diracnet_v2.utils.grid import RadialGrid
from rc_diracnet_v2.physics.dirac_operator import DiracRadialOperator
from rc_diracnet_v2.encoders.quantum_encoder import GlobalQuantumEncoder
from rc_diracnet_v2.data.dataset import LevelRowDataset
from rc_diracnet_v2.losses.pde_loss import DiracPDELoss
from rc_diracnet_v2.readout.orthogonalizer import lowdin_orthonormalize
print('All V1 ports OK')
"
```

输出 `All V1 ports OK` 即 Sprint 0 完成。
