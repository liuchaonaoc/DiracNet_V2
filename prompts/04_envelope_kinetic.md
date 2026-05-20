# 04 — Envelope + Kinetic Balance + Readout

## 1. Envelope (与 V1 几乎一致，但 λ 来源换成 KAN)

### 1.1 形式

```
env_a(r) = r^{γ_a} · exp(-λ_a · r)
γ_a = sqrt(κ_a² - (Z α)²)        # 严格 Dirac 1s
低 Z 极限: γ_a ≈ |κ_a|

denv_a/dr  = env · (γ_a / r − λ_a)
d²env_a/dr²= env · ( (γ_a / r − λ_a)² − γ_a / r² )
```

`γ_a` 取决于 `κ_a` 与 `Z`，但对低 Z 用 `|κ_a|`（V1 已经这么做）。

### 1.2 与 V1 的差异

V1 中 envelope 自己持有一个小 MLP `predict_lambda(h_cond, kappa, n)` 来出 `λ`。
**V2 中 envelope 不再持有 MLP**，它只接受外部传入的 `λ [B, N_orb]`：

```python
# rc_diracnet_v2/readout/envelope.py
class AnalyticEnvelope(nn.Module):
    def __init__(self, alpha: float = ALPHA) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(
        self,
        r_grid: Tensor,      # [N_grid]
        lam: Tensor,         # [B, N_orb]  ← 来自 KAN
        kappa: Tensor,       # [B, N_orb]  long (signed)
        Z: Tensor,           # [B]
    ) -> tuple[Tensor, Tensor, Tensor]:
        ...
```

`λ` 由 `models/dirac_net.py` 在主 forward 里调 KAN 后传入。

### 1.3 γ 的精确公式

对相对论严格的 Dirac 解，`γ_a = sqrt(κ² - (Zα)²)`。Z α 很小时（低 Z）退化为 `|κ|`。
要在低 Z 下精确，可以分支：

```python
def compute_gamma(kappa, Z, alpha=ALPHA):
    zalpha = Z.to(kappa.dtype) * alpha
    gamma_sq = (kappa.to(zalpha.dtype) ** 2) - zalpha.view(-1, 1) ** 2
    return gamma_sq.clamp_min(1e-6).sqrt()
```

`Z α > κ` 时（仅 Z > 137 才发生）需要外加 finite-nuclear-size 处理，本期不考虑。

## 2. Kinetic Balance — Q 由 P 解析推出

非相对论 kinetic balance（NKB，V1 已用）：

```
Q_a(r) = (1 / (2c)) · ( dP_a/dr + κ_a · P_a / r )
dQ_a/dr = (1 / (2c)) · ( d²P_a/dr² + κ_a · dP_a/dr / r − κ_a · P_a / r² )
```

这是 Dirac 方程小分量在 `ε / c² → 0` 极限下的解析关系。V2 **默认且强制使用 NKB**
（V1 中是 `kinetic_balance: true` 的可选项）。

不再保留 "free Q + residual MLP" 的备选路径，因为 V1 retrospective 说明这条路径
让模型有"补偿空间"。

### 2.1 Q 的小幅相对论修正

如果未来想引入 RKB / DKB（dual kinetic balance for heavy atoms），在 `BSplineReadout` 中
增加可选的 `q_residual_scale` 参数（zero-init），与 V1 接口一致。Phase 1 关闭，
Phase 2 (高 Z) 再开启。

```python
if cfg.readout.q_residual_scale > 0:
    Q_residual = scale * env * Σ c_q_k · B_k(r)        # 单独一组 c_q
    Q = Q_nkb + Q_residual
```

注意：这又会引入额外可学习量，**在 Stage 1 必须保持 `q_residual_scale = 0`**，
仅在 Stage 2 末期或后续 Phase 中尝试。

## 3. BSplineReadout — 把 (env, B-spline coef) → (P, Q, dPdr, dQdr)

```python
# rc_diracnet_v2/readout/bspline_readout.py

class BSplineReadout(nn.Module):
    """Compute P, Q, dPdr, dQdr from (c, env, B, dB, d2B, κ, r_grid).
    
    c is provided externally by the KAN; this module is purely tensor algebra
    and holds NO trainable parameters.
    """
    
    def __init__(
        self,
        c_speed: float = C_LIGHT,
        force_c0_zero: bool = True,
        q_residual_scale: float = 0.0,   # ≥ 0; only > 0 in Stage 2+ ablation
    ) -> None:
        super().__init__()
        self.inv_2c = 1.0 / (2.0 * c_speed)
        self.force_c0_zero = bool(force_c0_zero)
        self.q_residual_scale = float(q_residual_scale)

    def forward(
        self,
        c_P: Tensor,            # [B, N_orb, K]
        env: Tensor,            # [B, N_orb, N_grid]
        denv: Tensor,           # [B, N_orb, N_grid]
        d2env: Tensor,          # [B, N_orb, N_grid]
        B_basis: Tensor,        # [K, N_grid]
        dB_basis: Tensor,       # [K, N_grid]
        d2B_basis: Tensor,      # [K, N_grid]
        kappa: Tensor,          # [B, N_orb]
        r_grid: Tensor,         # [N_grid]
        orb_mask: Tensor,       # [B, N_orb]
        c_Q: Tensor | None = None,    # optional Stage 2+ Q residual coefficients
    ) -> dict[str, Tensor]:
        # 1) 屏蔽 c_0
        if self.force_c0_zero:
            c_P = c_P.clone()
            c_P[..., 0] = 0.0
        
        # 2) factor f(r) = 1 + Σ c_k B_k(r)
        f    = 1.0 + torch.einsum("bok,kn->bon", c_P, B_basis)
        df   =       torch.einsum("bok,kn->bon", c_P, dB_basis)
        d2f  =       torch.einsum("bok,kn->bon", c_P, d2B_basis)
        
        # 3) P, dP/dr, d²P/dr²
        P     = env * f
        dPdr  = denv * f + env * df
        d2Pdr = d2env * f + 2.0 * denv * df + env * d2f
        
        # 4) Q via NKB
        r_b   = r_grid.view(1, 1, -1).to(P.dtype)
        kap_b = kappa.to(P.dtype).unsqueeze(-1)
        Q     = self.inv_2c * (dPdr + kap_b * P / r_b)
        dQdr  = self.inv_2c * (d2Pdr + kap_b * (dPdr / r_b - P / (r_b * r_b)))
        
        # 5) optional Q residual (Stage 2 only)
        if self.q_residual_scale > 0.0 and c_Q is not None:
            if self.force_c0_zero:
                c_Q = c_Q.clone(); c_Q[..., 0] = 0.0
            fQ  = torch.einsum("bok,kn->bon", c_Q, B_basis)
            dfQ = torch.einsum("bok,kn->bon", c_Q, dB_basis)
            Q    = Q    + self.q_residual_scale * env * fQ
            dQdr = dQdr + self.q_residual_scale * (denv * fQ + env * dfQ)
        
        # 6) mask
        mask = orb_mask.unsqueeze(-1).to(P.dtype)
        P, Q, dPdr, dQdr = P * mask, Q * mask, dPdr * mask, dQdr * mask
        
        return {"P": P, "Q": Q, "dPdr": dPdr, "dQdr": dQdr,
                "d2Pdr": d2Pdr, "f": f, "df": df, "d2f": d2f}
```

注意：`d2Pdr` 也返回出来，供 PDE / smoothness loss 复用，避免重复计算。

## 4. 数值稳健性细节

### 4.1 `r_min` 处的奇异性

`κ · P / r` 在 `r → 0` 处趋于发散。env 的 `r^γ` 因子保证 `P → 0`，但分母仍要求
`r_min > 0`：

- 取 `grid.r_min = 1e-4`（与 V1 一致）；
- `BSplineBasis.r_min` 必须等于或略小于 `grid.r_min`；
- 不要在公式里加 `r.clamp_min(1e-12)`，那会引入额外的不平滑性。

### 4.2 `Q_NKB` 数值方差

`dP/dr` 在 P 的峰值附近显著大于 `P/r`，但在 `r ≫ peak` 处 `dP/dr → -λ · P`，
所以 `Q ≈ (1/(2c)) · (-λ + κ/r) · P`，与 V1 实验一致地 **总体小 10⁴ 倍**。
PDE 损失对小分量的贡献已经按 `2c²` 归一（见 V1 `pde_loss.py`），保持不变。

### 4.3 `Löwdin` 后导数失同步

V1 中 Löwdin 归一化时把 `(P, Q, dPdr, dQdr)` 同步缩放，但 **full Löwdin 后** `dP/dr`
其实不再是新 `P` 的精确导数（因为 `inv_sqrt(S)` 把不同轨道线性组合）。
**Stage 1 仅做 norm-only Löwdin**（V1 默认行为），导数同步缩放精确。
Stage 2 如果开启 full Löwdin，**必须 detach 导数并在 PDE loss 里禁用** 或
**重新解析求导**（推荐前者，更稳）。

## 5. 复用 V1 模块对照表

| V2 模块 | V1 来源 | 改动 |
|---------|---------|------|
| `AnalyticEnvelope` | `reservoir/envelope.py` | 删除内部 λ-MLP；λ 改外部传入 |
| `BSplineReadout`  | `readout/wavefunction_readout.py` | 拆掉 W 投影 MLP；c 改外部传入 |
| `KineticBalance` | 内嵌在 V1 `wavefunction_readout.py` 的 `kinetic_balance=True` 分支 | 提取为独立 helper （或保留在 readout 内） |

## 6. 单元测试要点（详见 `11_test_plan.md`）

- `test_envelope_consistency`：给定 `λ=1, κ=-1, Z=1`，env 应等于 `r · exp(-r)`；
  数值导数与解析导数误差 < 1e-4；
- `test_bspline_readout_anchor`：初始 `c=0` 时，`P == env`；
- `test_kinetic_balance_h1s`：用解析 1s 波函数喂入 → `Q_NKB ≈ (Zα/2) · P`（Pauli 极限）；
- `test_dirac_orbital_energy_h1s`：把上述 (P, Q) 喂入 V1 的 `orbital_energy_from_dirac`
  → `E_orb ≈ -0.5 Ha` (low Z)。

通过这 4 项测试后再开始训练。
