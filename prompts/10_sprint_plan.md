# 10 — Sprint Plan & Milestones

> 总目标：把 V2 从空目录推到 Stage 1 物理门禁通过，并把 `E_orb-only` 作为主基准；
> Stage 2 residual calibration 仅作为可选次指标，不能替代物理主结果。
> 约需 4 个 Sprint，3-4 周时间（单人全职估算）。

## Sprint 0 — 项目初始化（半天）

| 步骤 | 输出 |
|------|------|
| 创建 `rc_diracnet_v2_project/` 目录结构（见 `09_project_layout.md`） | 空骨架 |
| 复用 V1 模块：`grid`, `constants`, `data/*`, `encoders/*`, `physics/*`（除 hamiltonian_assembler），`utils/*`，`readout/orthogonalizer.py`，`losses/{pde, ortho, nist_scalar, balancer}` | 见 `12_port_from_v1.md` |
| 写 `default.yaml` | 见 `09_project_layout.md` |
| `pyproject.toml` + `requirements.txt` | 项目可 `pip install -e .` |
| 写 `tests/conftest.py`（grid fixture、sample batch） | – |

**Sprint 0 验收**：

```bash
python -c "import rc_diracnet_v2; print(rc_diracnet_v2.__version__)"
pytest tests/test_v1_modules_still_work.py -v   # 已复用模块的烟雾
```

---

## Sprint 1 — B-spline + KAN + Forward （2-3 天）

| 步骤 | 关联文件 | 验收测试 |
|------|---------|----------|
| 实现 `basis/bspline_basis.py` (scipy 路径) | `02_bspline_basis.md` | `test_v2_bspline_basis.py` |
| 实现 `readout/envelope.py`（去 MLP） | `04_envelope_kinetic.md` | `test_v2_envelope.py` |
| 实现 `readout/bspline_readout.py` | `04_envelope_kinetic.md` | `test_v2_bspline_readout.py` |
| 实现 `kan/mlp_coeff_net.py`（备选） | `03_kan_hypernet.md` | `test_v2_kan_coeff_net.py` (MLP 路径) |
| 实现 `models/dirac_net_v2.py` 主 forward | `01_architecture.md` | `test_v2_dirac_net_v2_forward.py` |
| 加 `_build_per_orb_features` 到 batch_builder | `08_data_pipeline.md` | `test_v2_per_orb_features.py` |

**Sprint 1 验收**：

```bash
pytest tests/test_v2_bspline_basis.py \
       tests/test_v2_envelope.py \
       tests/test_v2_bspline_readout.py \
       tests/test_v2_dirac_net_v2_forward.py -v

# 一次手工 forward：从 manifest_hydrogenic_v2 一行数据
python scripts/v2_diagnose_physics_chain.py \
       --config configs/v2_phase1_stage1_pde_only.yaml \
       --row 0
# 期望输出：P_model 形状合理，与 P_analytic |cos| > 0.5（未训练 baseline）
```

---

## Sprint 2 — 物理 Loss + Stage 1 Trainer（3-4 天）

| 步骤 | 关联文件 |
|------|---------|
| 实现 `losses/node_count_loss.py` | `05_physics_losses.md` |
| 实现 `losses/action_loss.py` | `05_physics_losses.md` + `13_action_and_residual_safety.md` |
| 实现 `losses/asymptotic_loss.py` | `05_physics_losses.md` |
| 实现 `losses/bspline_smooth_loss.py` | `05_physics_losses.md` |
| 实现 `training/stage_gate.py`（解析对照） | `06_two_stage_training.md` §2.4 |
| 实现 `training/two_stage_trainer.py` （仅 Stage 1 分支） | `06_two_stage_training.md` §2.3 |
| 实现 `physics/hydrogenic_analytic.py` | `07_evaluation.md` §2.2 |
| 实现 `scripts/v2_train_stage1_only.py` | – |
| 实现 `scripts/v2_gate_analytic.py` | `07_evaluation.md` §2.4 |

**Sprint 2 中期检查（Day 2）**：

```bash
# Stage 1 跑通；不必通过门禁
python scripts/v2_train_stage1_only.py --config configs/v2_smoke.yaml
# 期望: L_PDE 在 50 步内下降至少 30%
```

**Sprint 2 验收（end）**：

```bash
# 用 18 行 hydrogenic 完整跑 Stage 1
python scripts/v2_train_stage1_only.py \
       --config configs/v2_phase1_stage1_pde_only.yaml
# 期望: ≤ 300 epoch 内通过 stage1 gate
#       PASS: 所有 18 行 |cos| > 0.99, |λ-Z/n|/Z/n < 5%, |ΔE| < 1 meV,
#             L_action_BS < 1e-3

# 评估
python scripts/v2_evaluate.py \
       --ckpt checkpoints/v2_phase1_stage1/stage1_passed.pt \
       --manifest data_cache/manifest_hydrogenic_v2.parquet
# 期望: 未对齐 RMS < 100 meV（无 NIST 学习，但 PDE 已让 E_orb 与 -Z²/2n² 一致）
```

**若 Sprint 2 验收失败**：参考 `06_two_stage_training.md` §2.5 的诊断表。
**不要进入 Sprint 3 直到 Stage 1 gate 通过**。

---

## Sprint 3 — KAN 升级 + 可选 Residual Calibration + Phase 1 全量（3-4 天）

| 步骤 | 关联文件 |
|------|---------|
| 实现 `models/level_residual_head.py` | `06_two_stage_training.md` §3.3 |
| 扩展 `training/two_stage_trainer.py` 加入 Stage 2 calibration 分支 + rollback | `06_two_stage_training.md` §3.4 |
| 实现 `kan/kan_coeff_net.py`（替换 MLP） | `03_kan_hypernet.md` §3.1 |
| 实现 `kan/visualization.py` + `scripts/v2_visualize_kan.py` | `03_kan_hypernet.md` §7 |
| 实现 `scripts/v2_train.py` (Stage 1 → Stage 2 自动) | – |
| 实现 `scripts/v2_plot_wavefunctions.py` | – |
| 实现 `tests/test_v2_residual_safety.py` | `13_action_and_residual_safety.md` |

**Sprint 3 验收**：

```bash
# A. 用 60 行 extended hydrogenic 跑完整 Stage 1，Stage 2 默认关闭
python scripts/v2_train.py \
       --config configs/v2_phase1_full.yaml

# B. 评估
python scripts/v2_evaluate.py \
       --ckpt checkpoints/v2_phase1_full/stage1_passed.pt \
       --manifest data_cache/manifest_hydrogenic_v2_extended.parquet
# 期望: E_orb-only aligned RMS < 50 meV, max |error| < 200 meV,
#       physics gate PASS, L_action_BS PASS

# B2. 可选 calibration（不是主结果）
python scripts/v2_train_stage2_from.py \
       --ckpt checkpoints/v2_phase1_full/stage1_passed.pt \
       --config configs/v2_phase1_stage2_nist.yaml
python scripts/v2_evaluate.py --ckpt checkpoints/v2_phase1_full/stage2_best.pt ...
# 期望: 同时报告 E_orb-only 与 E_orb+Δ；max |Δ| < 5 meV；residual ratio < 0.2

# C. LOO
python scripts/v2_train.py \
       --config configs/v2_phase1_loo.yaml   # hold out n=5,6 per ion
python scripts/v2_evaluate.py --ckpt ... --loo-mode
# 期望: E_orb-only hold-out RMS < 200 meV；calibrated OOD 不得崩溃

# D. KAN 可视化健康度
python scripts/v2_visualize_kan.py \
       --ckpt checkpoints/v2_phase1_full/stage2_best.pt \
       --out figures/v2_kan_edges/
# 期望: λ(Z, n) 边样条接近 Z/n 形状；c_k 模长 ≤ 1
```

---

## Sprint 4 — Phase 2 NIST 全量 + 论文级评估（5-7 天）

| 步骤 | 关联文件 |
|------|---------|
| 准备 NIST manifest（复用 V1 数据） | `08_data_pipeline.md` §3 |
| 实现 `configs/v2_phase2_nist_full.yaml` | – |
| 训练 Stage 1（仍用 hydrogenic gate 监控）+ Stage 2（NIST 大数据） | – |
| 实现完整 evaluator with per-element breakdown | – |
| 实现 LOO / OOD 评估 | – |

**Sprint 4 验收**：

```bash
python scripts/v2_train.py \
       --config configs/v2_phase2_nist_full.yaml

python scripts/v2_evaluate.py \
       --ckpt checkpoints/v2_phase2/stage2_best.pt \
       --manifest data_cache/manifest_nist_full.parquet \
       --out results/v2_phase2/
# 期望:
#   E_orb-only aligned RMS < 500 meV (chemical accuracy ~1 eV)
#   median |error| < 100 meV
#   |cos| min > 0.95 across hydrogenic test subset
#   L_action_BS within gate on hydrogenic subset
#   calibrated 指标若启用，必须报告 |Δ_res| max, residual ratio, OOD/LOO
```

最终 deliverable：

- `results/v2_phase2/SUMMARY.md` 完整对比 V1 R9 / B16 / B9-LOO 与 V2 `E_orb-only` / calibrated
- `figures/` 完整可视化
- `docs/v2_lessons_learned.md` 记录 V2 sprint 中发现的问题

---

## 总体里程碑表

| 里程碑 | 完成时间（累计） | 通过条件 |
|--------|------------------|----------|
| M1: Sprint 0 done | day 0.5 | 项目可 import |
| M2: Sprint 1 done | day 3 | 前向 + 单测过 |
| M3: Sprint 2 done | day 7 | Stage 1 gate PASS on 18 行 |
| M4: Sprint 3 done | day 11 | `E_orb-only` RMS < 50 meV on 60 行；calibration 通过安全门禁 |
| M5: Sprint 4 done | day 18 | Phase 2 `E_orb-only` RMS < 500 meV |

## 风险节点 & 备选路径

| 节点 | 风险 | 备选 |
|------|------|------|
| Sprint 2: Stage 1 gate 不通过 | B-spline 节点不够 / 节点损失权重不对 | `K = 32 → 48`，加 `L_smooth`，把 `force_freeze_lambda = true` 验证下界 |
| Sprint 3: KAN 不稳 | `efficient-kan` 数值问题 | 用 MLPCoeffNet 跑通 Sprint 3，KAN 升级延后 |
| Sprint 2: action loss 不稳定 | turning point 梯度或 Maslov 修正不匹配 | 延后开启 `L_action_BS`；`w_action: 0.05 → 0.01`；先只做 metric |
| Sprint 3: Stage 2 rollback 频繁 | NIST 数据噪声 / residual 正在尝试作弊 | 维持 `E_orb-only` 主结果；降低 `w_nist_max`；不要默认放大 Δ_max |
| Sprint 4: NIST 全量训练时长 | 36k 行 × 50 epoch ≈ days | 先用 NIST 子集（10% 抽样）验证收敛趋势 |

## 每日 standup 模版

```
[V2 Day N]
状态：Sprint X (Y/Z 项)
今日完成：
  - ...
今日卡点：
  - ...
明日目标：
  - ...
门禁状态（如已进入训练）：
  - cos_min: X.XX
  - λ_drift_max: X.X%
  - L_PDE: X.Xe-Y
```
