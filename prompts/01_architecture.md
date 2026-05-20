# 01 — End-to-End Architecture

> 目标：把一个 batch 从 dataloader 到 `E_pred / wavefunctions / E_orb / loss` 的全过程
> 完整定义清楚。所有形状用 `[...]` 标注，所有可微/不可微节点标注 `∇✓ / ∇✗`。

## 1. 张量形状约定（与 V1 保持一致）

```
B        = batch_size
N_orb    = readout.n_orb_max          (e.g. 16 — 与 max_orb 一致)
N_grid   = grid.n_grid                (e.g. 512)
K        = bspline.n_basis            (e.g. 32) ← B-spline 基底数
D_cond   = encoder.d_cond             (e.g. 256)
```

- 所有波函数与能量张量按 `float32` 计算；积分权重等 buffer 由 `float64` 预计算后下采到 `float32`。
- `kappa[B, N_orb]` 是相对论 κ；正 / 负值都允许。
- `orb_mask[B, N_orb]` bool：标记每个 (b, a) 是否为有效活跃轨道；padding 位强制波函数 = 0。

## 2. 前向数据流（自顶向下）

```
batch = {
    Z [B] long, charge [B] long, nele [B] long,
    config_shells [B, max_seq, 4] long  (n, l, 2j, occ),
    shell_mask [B, max_seq] bool,
    kappa [B, N_orb] long, orb_mask [B, N_orb] bool,
    J [B] long, parity [B] long, term_id [B] long,
    E_target [B] float32 (Hartree),
}

#────────────────────────────────────────────────────────
# (A) 量子编码  ∇✓
#────────────────────────────────────────────────────────
h_global = QuantumEncoder(batch)                       # [B, D_cond]
h_cond   = LevelFeatureEncoder(h_global, J, π, term)   # [B, D_cond]

#────────────────────────────────────────────────────────
# (B) 取每个活跃轨道的物理坐标 (Z, n, l, κ, occ, 是否价电子)  ∇✗
#────────────────────────────────────────────────────────
per_orb_features = build_per_orbital_features(batch)   # [B, N_orb, D_phys]
# D_phys ≈ 8: [Z, n, l, κ, occ, is_outer, screen_Z_eff, n*]

#────────────────────────────────────────────────────────
# (C) KAN HyperNet  ∇✓
#────────────────────────────────────────────────────────
lam_log_res, c_raw = KANCoeffNet(
    h_cond,               # [B, D_cond]
    per_orb_features,     # [B, N_orb, D_phys]
)
# lam_log_res : [B, N_orb]   (log-residual to log(Z_eff/n))
# c_raw       : [B, N_orb, K]
# 末层 zero-init → 初值 λ = Z_eff/n, c_k = 0 → P = env

#────────────────────────────────────────────────────────
# (D) Envelope (解析)   ∇✓ via lam_log_res
#────────────────────────────────────────────────────────
λ = clamp( (Z_eff / n) * exp(lam_log_res), λ_min, λ_max )     # [B, N_orb]
γ = sqrt( κ² - (Z α)² )         # ≈ |κ| for low Z
env(r), denv(r), d2env(r) = AnalyticEnvelope(r_grid, λ, γ)     # [B, N_orb, N_grid]

#────────────────────────────────────────────────────────
# (E) B-spline 基底 (预算 buffer)   ∇✗
#────────────────────────────────────────────────────────
B(r), B'(r), B''(r) = BSplineBasis.eval_on_grid()              # [K, N_grid]
# 1) 在 r_min..r_max 上设计 K 个节点（log-spaced）
# 2) 端点处节点重复 (order+1) 次使 B_k(r_min) = 0 for k>0 → 不破坏 r^γ
# 3) 一次性预算，registered as buffer

#────────────────────────────────────────────────────────
# (F) Wavefunction Readout  ∇✓
#────────────────────────────────────────────────────────
f(r)        = 1 + Σ_k c[b,a,k] · B_k(r)                       # [B, N_orb, N_grid]
df/dr(r)    =     Σ_k c[b,a,k] · B'_k(r)
d²f/dr²(r)  =     Σ_k c[b,a,k] · B''_k(r)

P(r)   = env(r) · f(r)
dP/dr  = denv · f + env · df/dr
d²P/dr²= d2env · f + 2 · denv · df/dr + env · d²f/dr²

Q(r)   = (1/(2c)) · ( dP/dr + κ · P / r )                     # kinetic balance
dQ/dr  = (1/(2c)) · ( d²P/dr² + κ · dP/dr / r − κ · P / r² )

P, Q, dPdr, dQdr ← mask(orb_mask)

#────────────────────────────────────────────────────────
# (G) 正交归一化  ∇✓ (Stage 1 默认 norm-only; Stage 2 可选 full Löwdin)
#────────────────────────────────────────────────────────
P, Q, dPdr, dQdr = lowdin_orthonormalize(
    P, Q, dPdr, dQdr, grid, orb_mask, use_full_lowdin=stage1_done
)

#────────────────────────────────────────────────────────
# (H) Dirac 算子 → E_orb  ∇✓
#────────────────────────────────────────────────────────
V_eff(r) = -Z / r + V_HF(r)        # Stage 1: V_eff = -Z/r; Stage 2 可加 mean-field
LP, LQ   = DiracRadialOperator(P, Q, dPdr, dQdr, V_eff, κ, r_grid)
E_orb    = orbital_energy_from_dirac(P, Q, LP, LQ, grid)       # Rayleigh quotient [B, N_orb]

#────────────────────────────────────────────────────────
# (I) 标量能量预测  ∇✓
#────────────────────────────────────────────────────────
# Stage 1:
E_pred_stage1 = Σ_a occ_a · E_orb_a                            # [B]
# Stage 2:
Δ_res = LevelResidualHead(h_cond, J, π, term)                  # bounded ±50 meV [B]
E_pred_stage2 = E_pred_stage1 + Δ_res

#────────────────────────────────────────────────────────
# (J) Loss
#────────────────────────────────────────────────────────
L_PDE    = DiracPDELoss(P, Q, dPdr, dQdr, E_orb, V_eff, κ, r_grid, grid, orb_mask)
L_ortho  = OrthonormalityLoss(P, Q, grid, orb_mask)
L_node   = NodeCountLoss(P, n_required = n - l - 1, grid)
L_asym   = AsymptoticTailLoss(P, λ, γ, r_grid)
L_smooth = BSplineSmoothLoss(c)   # 二阶差分

# Stage 1:
loss = w_pde · L_PDE + w_ortho · L_ortho + w_node · L_node + w_asym · L_asym + w_smooth · L_smooth

# Stage 2 (extra):
L_NIST = NISTScalarHuberLoss(E_pred_stage2, E_target, Z, charge)
loss += w_nist · L_NIST
```

## 3. 模块依赖图

```
┌────────────────┐
│ QuantumEncoder │──┐         ┌──────────────┐
│ (V1 复用)      │  │         │ BSplineBasis │
└────────────────┘  │         │  (NEW)       │
                    ▼         └──────┬───────┘
                ┌─────────────────┐  │
                │ KANCoeffNet     │  │
                │  (NEW)          │  │
                └────────┬────────┘  │
                         ▼           ▼
                ┌────────────────────────┐
                │ AnalyticEnvelope (NEW) │
                │ + BSplineReadout (NEW) │
                │ + KineticBalance       │
                └─────────┬──────────────┘
                          ▼
              ┌─────────────────────────┐
              │ lowdin_orthonormalize   │
              │ (V1 复用)               │
              └────────┬────────────────┘
                       ▼
              ┌─────────────────────────┐
              │ DiracRadialOperator     │
              │ orbital_energy_from_... │
              │ (V1 复用)               │
              └────────┬────────────────┘
                       ▼
              ┌──────────────────────┐
              │ LevelResidualHead    │   only Stage 2
              │  (NEW)               │
              └────────┬─────────────┘
                       ▼
              ┌──────────────────────┐
              │  Losses              │
              │  L_PDE / L_ortho /   │
              │  L_node / L_asym /   │
              │  L_NIST (Stage 2)    │
              └──────────────────────┘
```

## 4. 关键张量初值

| 张量 | 初值 | 训练后期望 |
|------|------|------------|
| `lam_log_res` | 0 (zero-init last layer) | `O(0.01)` 残差 |
| `c_raw` | 0 (zero-init last layer) | 任意符号；幅度 `≤ 1/K` 量级 |
| `λ` | `Z_eff / n` | ≈ `Z_eff / n` ± 5% |
| `f(r)` | 1 (constant) | Laguerre-like 多项式 |
| `P(r)` | env(r) = r^γ exp(-λr) | r^γ exp(-λr) · L_{n-l-1}^{2l+1}(2λr) |
| `Q(r)` | kinetic-balance(P) | 同左（≈相对论小分量） |
| `Δ_res` | 0 (Stage 2 起步) | ≤ 50 meV |
| `E_orb` | ≈ `-Z²/(2n²)` from analytic init | true eigenvalue |
| `L_PDE` (init) | ~1e0..1e1 | < 1e-3 (Stage 1 后) |
| `L_NIST` (Stage 2 init) | depends | < 1e-4 |

## 5. 与 V1 forward 的差异（清单）

| V1 step | V2 step | 差异 |
|---------|---------|------|
| `reservoir = RCBasisGenerator(d_res=512)` | `bspline = BSplineBasis(K=32)` | 基底类型完全替换 |
| `envelope(h_cond, kappa, n_idx, Z)` 内部预测 λ | `KANCoeffNet` 同时预测 (λ_residual, c) | (λ, c) 来源统一 |
| `WavefunctionReadout(h_cond)` 出 `W [B, N_orb, 2, D_res]` | `BSplineReadout(c)` 出 `P = env · (1 + Σ c_k B_k)` | 系数 KAN 给，不再 MLP→`W` |
| `LevelEnergyHead(h_cond, E_orb, mask, Z, n)` 含 `Δ_term + zn_bias` | Stage 1: 直接 `Σ occ_a · E_orb_a` | 无 Δ_term 在 Stage 1 |
| – | Stage 2: 额外 `LevelResidualHead` | 残差 head 仅在 Stage 2 接入 |
| `NIST` 损失全程参与 | Stage 1 不接 NIST 梯度 | 训练协议变化 |

## 6. 训练流程的最简骨架（伪代码）

```python
# Stage 1: PDE-only
optimizer_stage1 = AdamW(
    params=[*kan.params, *residual_lambda.params, *bspline_coef.params],
    lr=3e-4,
)
for epoch in range(N1):
    for batch in train_loader:
        out = model(batch, stage=1)
        L = w_pde * L_PDE + w_ortho * L_ortho + w_node * L_node + w_asym * L_asym
        L.backward()
        optimizer_stage1.step()
    # log NIST as monitoring metric (no grad)
    log_nist_residual(model, val_loader)
    
    if epoch % 10 == 0 and check_stage1_gate(model):
        save_checkpoint("stage1_passed.pt")
        break

# Stage 2: NIST-residual fine-tune
freeze(kan_main_params)
add(LevelResidualHead)  # bounded ±50 meV, zero-init
optimizer_stage2 = AdamW(
    params=[*residual_head.params],
    lr=1e-4,
)
for epoch in range(N2):
    for batch in train_loader:
        out = model(batch, stage=2)
        L = w_pde * L_PDE + w_nist * L_NIST_residual
        L.backward()
        optimizer_stage2.step()
    if violates_gate(model, threshold=0.99):
        rollback_to_last_good()
```

## 7. 实现顺序建议

1. **B-spline 基底**（`02_bspline_basis.md`）：本身可单测，与外界解耦；
2. **AnalyticEnvelope（V2 版）**（`04_envelope_kinetic.md`）：复用 V1，但 λ 接 KAN 而非内部 MLP；
3. **KANCoeffNet**（`03_kan_hypernet.md`）：可先用 `MLP+zero-init` 替代占位；
4. **BSplineReadout** + kinetic balance：把 (env, B-spline coef) → (P, Q, dPdr, dQdr) 串联；
5. **Losses**：复用 V1 PDE/Ortho，新增 Node + Asymptotic；
6. **Trainer**：Stage 1 / Stage 2 切换协议；
7. **Eval**：解析对照门禁（`07_evaluation.md`）。

每完成一步跑对应单元测试再继续；不要批量改完再测。
