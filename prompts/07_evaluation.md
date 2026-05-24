# 07 — Evaluation & Anti-Cheating Gate

> V2 的核心原则：**任何指标只在通过物理对照后才能被采信**。
> V1 R9 报出 170 meV MAE 但波函数余弦只有 0.45，就是反面教材。

## 1. 三层评估

| 层级 | 工具 | 频率 | 失败后果 |
|------|------|------|----------|
| **A. 解析对照门禁** | `scripts/v2_gate_analytic.py` | 每 epoch | 决定能否进入 Stage 2；回滚 Stage 2 |
| **B. 能量准确度** | `scripts/v2_evaluate.py` | Stage 末 | 同时报告 `E_orb-only` 与 calibrated；只以前者作为主结果 |
| **C. KAN 可视化 / 诊断** | `scripts/v2_visualize.py` | 选做 | 解释训练动态 |

## 2. A — 解析对照门禁

### 2.1 测试集

合成 `manifest_hydrogenic_v2.parquet`，覆盖：

```
ions: H I (Z=1, q=0), He II (Z=2, q=1), Li III (Z=3, q=2)
n   : 1..6
l   : 0 (s-shells only, Stage 1 简化)
共 18 行；每行 level_config = "{n}s1"
E_target = -Z²/(2n²) Hartree (analytic)
```

### 2.2 解析参考波函数

直接用类氢 (Z=Z_nuc) 的解析径向 `P_{nl}^{exact}` (NR 极限)：

```
P_{n,0}^{exact}(r) = N · r · exp(-Z r / n) · L_{n-1}^{1}(2 Z r / n)
```

其中 `L_{n-1}^{1}` 是 generalised Laguerre polynomial。

实现：

```python
# rc_diracnet_v2/physics/hydrogenic_analytic.py

from scipy.special import genlaguerre

def hydrogenic_P_analytic(r, Z, n, l=0):
    """Return P_nl(r) on a torch tensor grid (NR Schrödinger)."""
    rho = 2.0 * Z * r / n
    # normalisation
    import math
    fact = math.factorial(n - l - 1)
    fact_n = math.factorial(n + l)
    norm = math.sqrt((2.0 * Z / n) ** 3 * fact / (2.0 * n * fact_n))
    L = genlaguerre(n - l - 1, 2 * l + 1)
    # P(r) = r · R_nl(r); for hydrogenic R(r) = norm · rho^l · exp(-rho/2) · L(rho)
    rho_np = rho.detach().cpu().numpy()
    L_vals = L(rho_np)
    R = norm * (rho_np ** l) * np.exp(-rho_np / 2.0) * L_vals
    P = torch.tensor(rho_np / (2.0 * Z / n) * R, dtype=torch.float32, device=r.device)
    return P  # [N_grid]
```

注意：`P_nl(r) = r · R_nl(r)`，与 V1 的 `WavefunctionReadout` 输出尺度一致（积分用 `∫ P² dr`）。

### 2.3 余弦相似度

```python
def cosine_signed(P_model, P_ref, grid):
    a = grid.integrate(P_model * P_ref, dim=-1)
    b = grid.integrate(P_model * P_model, dim=-1).sqrt()
    c = grid.integrate(P_ref * P_ref, dim=-1).sqrt()
    return a / (b * c + 1e-12)
# 实际门禁用 abs(...)（波函数整体符号不可观测）
```

### 2.4 门禁脚本

```python
# scripts/v2_gate_analytic.py
"""
Stage 1 / Stage 2 物理对照门禁。
对 manifest_hydrogenic_v2 上每一行：
  - 计算模型的 P_model, E_orb, λ
  - 与 P_analytic, E_analytic = -Z²/(2n²), λ_analytic = Z/n 对比
报告:
  PASS / FAIL  per row & overall
要求:
  |cos| > 0.99  for all rows
  |ΔE| < 1 meV  for all rows
  |λ_pred − Z/n| / (Z/n) < 5%
  |∫p_rdr − π(n-l-1/2)| < action_threshold
"""
```

调用示例：

```bash
python scripts/v2_gate_analytic.py \
    --ckpt checkpoints/v2_stage1/latest.pt \
    --manifest data_cache/manifest_hydrogenic_v2.parquet \
    --out results/v2_stage1_gate.txt
```

Trainer 内部直接调用 `gate.check(model, ds)` 函数，避免重复 IO。

### 2.5 门禁输出格式

```
== Stage 1 Gate Report ==
ckpt: checkpoints/v2_stage1/epoch_120.pt
manifest: manifest_hydrogenic_v2.parquet (18 rows)

Per-row results:
| Z | n |    λ_pred |     λ_ref |  Δλ/λ_ref |       |cos| |       E_orb |        E_ref |  ΔE (meV) | PASS |
|---|---|-----------|-----------|-----------|-------------|-------------|--------------|-----------|------|
| 1 | 1 |   0.99987 |   1.00000 |   0.013% |     0.99998 |  -0.5000023 |  -0.5000000  |    0.062  |  ✓   |
| 1 | 2 |   0.50031 |   0.50000 |   0.062% |     0.99961 |  -0.1249988 |  -0.1250000  |    0.033  |  ✓   |
...
| 3 | 5 |   ...     |   ...     |   ...    |     ...     |    ...      |    ...       |    ...    |  ?   |

Aggregate:
  cos_min = 0.99961    (threshold 0.99)   ✓
  Δλ_max  = 0.41%      (threshold 5%)     ✓
  ΔE_max  = 0.42 meV   (threshold 1 meV)  ✓
  L_PDE   = 7.4e-4     (threshold 1e-3)   ✓
  L_action_BS = 3.1e-5 (threshold 1e-3)   ✓

VERDICT: PASS
```

## 3. B — 能量准确度评估

复用 V1 `scripts/evaluate_levelwise.py` 的结构，加 V2 的两阶段输出：

```bash
python scripts/v2_evaluate.py \
    --ckpt checkpoints/v2_stage2/best.pt \
    --manifest data_cache/manifest_h_levelwise_abs.parquet \
    --out results/v2_phase1/eval.txt
```

输出必须把 `E_orb-only` 放在 calibrated 之前：

```
=== Stage 2 Evaluation ===
checkpoint: ...
manifest:   ...
levels:     15

=== E_orb-only (MAIN RESULT; no Δ_residual) ===
without ground alignment:
  RMS  |error| = 18.0 meV
  MAE  |error| = 12.0 meV

with align_ground=True:
  RMS  |error| = 14.0 meV
  MAE  |error| = 9.0 meV
  median |error| = 6.0 meV
  max  |error| = 42.0 meV

=== E_orb + Δ_residual (CALIBRATION ONLY) ===
without ground alignment:
  RMS  |error| = 7.1 meV
  MAE  |error| = 4.5 meV

with align_ground=True:
  RMS  |error| = 5.2 meV
  MAE  |error| = 3.4 meV
  median |error| = 2.0 meV
  max  |error| = 18.0 meV

=== Δ_residual Safety ===
  max |Δ_residual| = 4.8 meV  (cap 5.0 meV)    ✓
  median |Δ_residual| = 1.1 meV
  median |Δ| / median |E_orb - E_target| = 0.12 (threshold 0.20) ✓
  leave-one-n RMS = 41.0 meV                  ✓
  leave-one-Z RMS = 55.0 meV                  ✓
  leave-one-ion RMS = 60.0 meV                ✓

per (Z, ion_charge) — MAE & RMS (meV):
  Z=1 ion=0: n=5 MAE=4.5  RMS=7.1
  Z=2 ion=1: n=5 MAE=3.2  RMS=5.0
  Z=3 ion=2: n=5 MAE=5.0  RMS=8.2

=== Stage 1 vs Stage 2 ===
                Stage 1 MAE | Stage 2 MAE | Δ
Z=1 ion=0:       18.0       |  4.5        | -13.5 meV (NIST head learned)
Z=2 ion=1:       20.0       |  3.2        | -16.8 meV
Z=3 ion=2:       30.0       |  5.0        | -25.0 meV

=== Physics Gate (post-Stage 2) ===
cos_min: 0.9988  (threshold 0.99)  ✓
Δλ_max:  1.2%   (threshold 5%)     ✓
ΔE_orb_max (without Δ_res): 1.5 meV (threshold 5 meV)  ✓
L_action_BS: 8.0e-4 (threshold 1e-3) ✓
Δ_res_max: 4.8 meV  (cap 5 meV)     ✓

VERDICT: ACCEPT
```

如果 `E_orb-only` 不达标，但 calibrated 达标，VERDICT 必须是 `CALIBRATION-ONLY / NOT PHYSICS PASS`。

## 4. C — KAN 可视化与诊断

### 4.1 边样条图

```bash
python scripts/v2_visualize_kan.py \
    --ckpt checkpoints/v2_stage1/passed.pt \
    --out figures/v2_kan_edges/
```

输出：
- `lambda_vs_Z.png` — 不同 n 下，λ 随 Z 的关系（期望近似 `Z/n`）
- `lambda_vs_n.png` — 不同 Z 下，λ 随 n 的关系（期望 `1/n` 形状）
- `c_k_vs_n.png` — B-spline 系数 c_k 的分布
- `kan_layer_l_edge_ij.png` — 单条边样条曲线

### 4.2 波函数对比图

```bash
python scripts/v2_plot_wavefunctions.py \
    --ckpt checkpoints/v2_stage1/passed.pt \
    --manifest data_cache/manifest_hydrogenic_v2.parquet \
    --out figures/v2_wavefn/
```

每个 (Z, n) 一张图：
- 实线：`P_model(r)`
- 虚线：`P_analytic(r)`
- 散点：节点位置（实际 vs 理论 n-1）

### 4.3 训练动态图

```bash
python scripts/v2_plot_training.py \
    --logdir logs/v2_phase1/
```

每个 run 自动生成：
- `loss_curve.png` — 各 loss 的 epoch 轨迹
- `gate_history.png` — 解析对照每 epoch 的通过状态
- `nist_residual.png` — `E_pred - E_target` 的箱型图

## 5. 反作弊的硬约束（再次强调）

任何论文 / 报告中提到 V2 性能数字时，**必须** 附带：

```
[V2 Phase 1]
- E_orb-only RMS = X meV (aligned), Y meV (raw)   ← 主结果
- E_orb+Δ RMS = X2 meV (aligned), Y2 meV (raw)    ← 次结果，若启用
- |cos| min = Z   (must be > 0.99 if claimed)
- λ drift max = W%   (must be < 5%)
- L_action_BS = A   (must be < threshold)
- Δ_residual max = D meV
- median |Δ| / median |E_orb - E_target| = R (must be < 0.2)
- LOO RMS = L meV (if measured)
- leave-one-n / leave-one-Z / leave-one-ion OOD metrics
```

少一项，结果作废。这是 V2 的承诺。

### 5.1 `Δ_residual` 判定为 cheating 的规则

下列任一发生，calibrated 结果不得作为 V2 成功：

- `E_orb-only` 高误差，`E_orb+Δ` 低误差；
- residual 幅度接近 cap；
- residual 占比超过 20%；
- LOO/OOD 崩溃；
- Stage 2 后 `|cos|`, `λ`, `L_PDE`, `L_action_BS` 退化；
- residual 输入或参数结构等价于 `(Z, n)` lookup。

## 6. 与 V1 evaluator 的对照

| V1 工具 | V2 复用 / 替换 |
|---------|----------------|
| `scripts/evaluate_levelwise.py` | 大部分复用，加入 Stage 1/2 区分 |
| `scripts/diagnose_levelwise_ckpt.py` | 复用，扩展 KAN 字段 |
| `scripts/diagnose_physics_chain.py` | 复用 |
| `scripts/diagnose_analytic_levels.py` | **核心复用**：升级为 `v2_gate_analytic.py` |
| `scripts/plot_wavefunctions.py` | 复用 |

## 7. 检查清单（Sprint 验收前必做）

- [ ] `test_hydrogenic_analytic_P` 通过：解析 1s 喂入 → `E_orb ≈ -0.5 Ha`
- [ ] `test_gate_passes_on_analytic_init` 通过：初始（λ=Z/n, c=0）下门禁直接 PASS
- [ ] `test_gate_fails_on_random_init` 通过：随机 c 下门禁 FAIL（验证门禁敏感性）
- [ ] `test_action_loss_hydrogenic_quantization` 通过：解析 Coulomb 下 `L_action_BS ≈ 0`
- [ ] `v2_gate_analytic.py` 在 stage1_passed.pt 上 PASS
- [ ] `v2_evaluate.py` 同时输出 `E_orb-only` 与 calibrated 指标，且主指标达标
- [ ] residual safety 指标通过：`max |Δ|`, residual ratio, LOO/OOD
- [ ] LOO 测试：hold out 5s/6s 不参与训练，eval RMS < 200 meV
