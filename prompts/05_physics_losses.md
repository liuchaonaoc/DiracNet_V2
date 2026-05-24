# 05 — Physics Losses

## 1. 总览

V2 在 Stage 1 共用 6 个 physics loss（NIST **不参与梯度**）：

| Loss | 数学 | 维度 | 默认权重 |
|------|------|------|----------|
| `L_PDE` | `‖ H_D ψ − E_orb ψ ‖²` (V1 复用) | scalar | **10.0** |
| `L_ortho` | `‖ S − I ‖_F²` (V1 复用) | scalar | 1.0 |
| `L_node` | 节点计数惩罚（n - l - 1） | scalar | 0.1 |
| `L_action_BS` | Bohr-Sommerfeld 作用量量子化 `∫p_rdr = π(n-l-1/2)` | scalar | 0.0 → 0.05 |
| `L_asym` | 远端衰减一致性 `P(r_max) ≈ 0` & log-slope ≈ -λ | scalar | 0.01 |
| `L_smooth` | B-spline 系数二阶差分 `Σ (c_{k+1} - 2c_k + c_{k-1})²` | scalar | 1e-4 |

Stage 2 增加：

| Loss | 数学 | 维度 | 默认权重 |
|------|------|------|----------|
| `L_NIST_res` | `Huber(E_orb_sum + Δ_res − E_target)` | scalar | 可选；默认 calibration-only |

**关键纪律**：Stage 1 训练中 NIST 仍要每个 batch 计算并写入 metrics log，
但 **`loss.backward()` 里不出现它**。这条线在 trainer 里硬编码（详见 `06_two_stage_training.md`）。

## 2. `L_PDE` — Dirac 残差

直接复用 V1 `losses/pde_loss.py`：

```python
from rc_diracnet.losses.pde_loss import DiracPDELoss
```

无需修改。`small_component_scale = 2c²` 已经合理；保留它。

## 3. `L_ortho` — 跨轨道正交

直接复用 V1 `losses/orthonormality_loss.py`：

```python
from rc_diracnet.losses.orthonormality_loss import OrthonormalityLoss
```

无需修改。

**Stage 1 何时启用 full Löwdin？**

- 仅 Stage 1 末期，且 `L_ortho < 1e-3`（norm-only Löwdin + 软约束已经够好）；
- 否则保持 `use_full_lowdin=False`（V1 默认）；
- 开了 full Löwdin 后 `L_ortho` 形式上 = 0，但要保留它作为数值健康度指标。

## 4. `L_node` — 节点计数惩罚（新）

### 4.1 物理动机

n-l-1 节点数是径向 Schrödinger 方程在固定 V_eff 下的本征解的根本特征：
- 1s (n=1, l=0): 0 nodes
- 2s (n=2, l=0): 1 node
- 3s (n=3, l=0): 2 nodes
- 2p (n=2, l=1): 0 nodes
- 3p (n=3, l=1): 1 node

PDE-only 训练在 Rayleigh quotient 下倾向于坍缩到基态（无节点解）。
即使 KAN 显式吃 `n`，模型仍可能学到"n=3 也输出 0 节点的 P"——这就是退化。
节点损失把"节点数 = n-l-1"硬性写入梯度。

### 4.2 实现

```python
# rc_diracnet_v2/losses/node_count_loss.py

def _signed_amplitude(P: Tensor) -> Tensor:
    """Smoothed sign of P on radial grid.
    
    Use tanh(P / σ) where σ is a learnable / adaptive scale (median |P| × 0.01).
    """
    sigma = P.abs().median(dim=-1, keepdim=True).values.clamp_min(1e-6) * 1e-2
    return torch.tanh(P / sigma)

def _approx_node_count(P: Tensor) -> Tensor:
    """Differentiable node count via |sign change| sum / 2.
    
    For P [B, N_orb, N_grid]:
      n_nodes ≈ 0.5 * Σ_i |s_i - s_{i-1}|
    where s = tanh(P/σ).
    """
    s = _signed_amplitude(P)
    diff = (s[..., 1:] - s[..., :-1]).abs()
    return 0.5 * diff.sum(dim=-1)

class NodeCountLoss(nn.Module):
    def forward(
        self,
        P: Tensor,           # [B, N_orb, N_grid]
        n_required: Tensor,  # [B, N_orb] long  (n - l - 1)
        orb_mask: Tensor,    # [B, N_orb]
    ) -> Tensor:
        n_pred = _approx_node_count(P)                  # [B, N_orb] float
        err = (n_pred - n_required.to(n_pred.dtype))    # both float
        # Smooth Huber on count error
        mask = orb_mask.to(err.dtype)
        l = F.huber_loss(err * mask, torch.zeros_like(err), delta=0.5, reduction="sum")
        denom = mask.sum().clamp_min(1.0)
        return l / denom
```

### 4.3 数值要点

- `sigma` 自适应：避免 `tanh(P/σ)` 在很小的 P 处也认为是大跳跃；
- 同一 `(B, orb)` 内 σ 共享；
- 不要把 `_approx_node_count` 用作度量；仅用作 loss。真实节点统计要在 eval 时
  用解析过零点（`scripts/diagnose_v2_levels.py`）。

### 4.4 `n_required` 的来源

```python
# 在 forward 里：
n_idx = config_shells[..., 0]                 # [B, max_seq]
l_idx = config_shells[..., 1]                 # [B, max_seq]
n_required = (n_idx - l_idx - 1).clamp_min(0)
# pad to [B, N_orb] with -1 sentinel; mask later
```

## 5. `L_asym` — 渐近衰减一致性（新）

### 5.1 物理动机

正确的束缚态满足 `P(r→∞) ~ exp(-λ r)`。如果模型某个轨道在外层失控（B-spline 节点
没覆盖远端），`P(r_max)` 不收敛 → 能量谱不对。

### 5.2 实现

```python
class AsymptoticTailLoss(nn.Module):
    """Penalise |P(r_max)| being non-negligible relative to peak."""
    
    def forward(
        self,
        P: Tensor,            # [B, N_orb, N_grid]
        r_grid: Tensor,       # [N_grid]
        orb_mask: Tensor,
        tail_window: int = 32,
    ) -> Tensor:
        P_tail = P[..., -tail_window:]                          # [B, N_orb, W]
        P_peak = P.abs().max(dim=-1, keepdim=True).values + 1e-12
        rel_tail = (P_tail.abs() / P_peak).max(dim=-1).values   # [B, N_orb]
        mask = orb_mask.to(rel_tail.dtype)
        return (rel_tail * mask).sum() / mask.sum().clamp_min(1.0)
```

可选增强：检查 log-slope。

```python
# d log|P|/dr ≈ -λ on tail
slope_pred = (torch.log(P[..., -1].abs() + 1e-12) - torch.log(P[..., -tail_window].abs() + 1e-12))
             / (r_grid[-1] - r_grid[-tail_window])
slope_err = (slope_pred + λ).pow(2)   # want slope_pred ≈ -λ
```

但这需要 λ 张量也传入；简单起见，**Sprint 1 先用 `|P(r_max)|` 单项**。

## 6. `L_smooth` — B-spline 二阶差分（新）

### 6.1 动机

避免 KAN 给出"震荡"的系数 c_k，让相邻节点差异过大。物理上 Laguerre 系数是平滑变化的。

### 6.2 实现

```python
class BSplineSmoothLoss(nn.Module):
    def forward(self, c: Tensor, orb_mask: Tensor) -> Tensor:
        # c [B, N_orb, K]
        d2 = c[..., 2:] - 2.0 * c[..., 1:-1] + c[..., :-2]    # [B, N_orb, K-2]
        sq = d2.pow(2).sum(dim=-1)                              # [B, N_orb]
        mask = orb_mask.to(sq.dtype)
        return (sq * mask).sum() / mask.sum().clamp_min(1.0)
```

权重 1e-4 即可；过大会扼杀表达力。

## 7. `L_action_BS` — 作用量量子化约束（新）

### 7.1 为什么不是“作用量本身”

MIT 2026 的 *On computing quantum waves exactly from classical action* 表明，在 Hamilton-Jacobi
作用量 `S` 与密度 `ρ` 的合适表述下，Schrödinger 波可以从 classical action 精确计算出来。
但 V2 当前求的是 **1D 径向束缚定态**，空间相位可取实，完整 Hamilton-Jacobi + density 方程会退化为
与 `L_PDE` 等价的局部 Schrödinger/Dirac 条件。

因此，直接加入“作用量极值”不会带来新信息；真正有增益的是定态半经典中的全局量子化条件：

```
∫_{r_-}^{r_+} p_r(r) dr = π · (n - l - 1/2)
```

这条约束携带的是 **第几个本征态** 的信息，可与 `L_node` 一起防止 PDE-only 阶段坍缩到基态。

### 7.2 物理形式

原子单位制下：

```
p_r²(r) = 2 · (E_orb - V_eff(r) - l(l+1)/(2r²))
S_r     = ∫ sqrt(max(p_r², 0)) dr
target  = π · (n - l - 1/2)
L_action_BS = mean_active_orbitals( (S_r - target)² )
```

对 Coulomb 势，带 Langer/Maslov 修正后这条 Bohr-Sommerfeld 量子化条件是精确的；
对一般 `V_eff`，它是半经典近似，但仍是有用的全局 regularizer。

### 7.3 实现

```python
# rc_diracnet_v2/losses/action_loss.py

class BohrSommerfeldActionLoss(nn.Module):
    """Global action quantisation loss for bound radial orbitals.

    L_BS = mean_active ( ∫ p_r dr - π(n - l - 1/2) )²
    """

    def __init__(self, eps: float = 1.0e-12) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        E_orb: Tensor,        # [B, N_orb], Hartree
        V_eff: Tensor,        # [B, N_grid], Hartree
        n_idx: Tensor,        # [B, N_orb], long
        l_idx: Tensor,        # [B, N_orb], long
        orb_mask: Tensor,     # [B, N_orb]
        grid,                 # RadialGrid
    ) -> Tensor:
        r = grid.r.to(E_orb.device).to(E_orb.dtype)
        r_b = r.view(1, 1, -1)

        l = l_idx.to(E_orb.dtype)
        centrifugal = (l * (l + 1.0)).unsqueeze(-1) / (2.0 * r_b * r_b)
        v_total = V_eff.unsqueeze(1).to(E_orb.dtype) + centrifugal

        p_sq = 2.0 * (E_orb.unsqueeze(-1) - v_total)
        # Smooth allowed-region projection. Clamp is acceptable for a first pass;
        # softplus can replace it if turning-point gradients become unstable.
        p_r = (p_sq.clamp_min(0.0) + self.eps).sqrt()
        S = grid.integrate(p_r, dim=-1)

        target = math.pi * (n_idx.to(E_orb.dtype) - l - 0.5).clamp_min(0.0)
        err = S - target
        mask = orb_mask.to(err.dtype)
        return (err.pow(2) * mask).sum() / mask.sum().clamp_min(1.0)
```

### 7.4 何时开启

不要在训练第一步就打开 `L_action_BS`。推荐 warmup：

```python
weights = {
    "pde": 10.0,
    "ortho": 1.0,
    "node": 0.0,
    "action_bs": 0.0,
    "asym": 0.01,
    "smooth": 1e-4,
}

if epoch >= 10:
    weights["node"] = 0.1
if epoch >= 20 and running_avg["L_PDE"] < 1e-2:
    weights["action_bs"] = 0.05
```

解释：

- `L_PDE` 先把局部方程拉到合理范围；
- `L_node` 再给整数节点约束；
- `L_action_BS` 最后给全局作用量约束，避免早期 `E_orb` 未稳定时把梯度打乱。

### 7.5 何时失败

| 现象 | 含义 |
|------|------|
| `L_PDE≈0` 但 `L_action_BS` 大 | 模型可能满足某个局部本征方程，但收敛到错误 n |
| `L_action_BS≈0` 但 `L_PDE` 大 | 作用量积分对了，但局部波函数仍错 |
| `L_node` 与 `L_action_BS` 给出冲突 | 节点估计不稳或 `V_eff` / Maslov 修正不匹配 |

二者同时小，才可认为 PDE-only 阶段确实锁定到了目标本征态。

## 8. NIST 损失（Stage 2 / calibration-only）

直接复用 V1 `losses/nist_scalar_loss.py`：

```python
from rc_diracnet.losses.nist_scalar_loss import NISTScalarHuberLoss
```

但在 V2 里这是 **作用在受限校准残差**上的损失。流程：

```
Stage 1 终点保存 E_orb_stage1_table [B, N_orb] (per row)
Stage 2:
   E_orb_a = orbital_energy_from_dirac(...)    # 仍按 Dirac 算
   E_orb_sum = Σ_a occ_a · E_orb_a             # Koopmans 求和
   Δ_res = LevelResidualHead(h_cond, J, π, term)   # bounded; Phase 1 默认 1-5 meV，轻原子 20 meV
   E_pred = E_orb_sum + Δ_res
   L_NIST_res = Huber(E_pred − E_target)
```

注意：

- V2 主报告必须先给 `E_orb_sum` 的指标，不能只报告 `E_orb_sum + Δ_res`；
- `Δ_residual` 默认不是主模型组件，而是 calibration experiment；
- 默认冻结 KAN / encoder / readout，只训练 `LevelResidualHead`；
- 禁止 `zn_bias_table` 或任何直接 `(Z, n)` lookup；
- 如果 `Δ_residual` 的改善主要来自训练集记忆，或者 OOD/LOO 崩溃，结果判为 cheating。

## 9. 损失权重与 ramp 协议（Stage 1 → Stage 2）

```python
# Stage 1
weights_stage1 = {
    "pde": 10.0,
    "ortho": 1.0,
    "node": 0.0,        # epoch >= 10 后开到 0.1
    "action_bs": 0.0,   # L_PDE < 1e-2 且 epoch >= 20 后开到 0.05
    "asym": 0.01,
    "smooth": 1e-4,
    # nist not in loss
}

# Stage 2
weights_stage2 = {
    "pde": 10.0,       # 保持，防 cheating
    "ortho": 1.0,
    "node": 0.1,
    "action_bs": 0.05,
    "asym": 0.01,
    "smooth": 1e-4,
    "nist": ramp(epoch, start=0.0, end=cfg.stage2.w_nist_max, n_warmup=20),
}
```

NIST 权重 ramp 是关键——直接给 1.0 会让 NIST 梯度瞬间压制 PDE，重蹈 V1 R9 覆辙。
"线性 ramp 20 个 epoch"是经验值，可调。

## 10. PDE-only 阶段的 **关键防退化机制**

> 为什么 Stage 1 PDE-only 不会坍缩到 n=1?

| 防退化点 | 实现位置 | 效果 |
|----------|----------|------|
| 跨轨道正交（`L_ortho`） | `losses/orthonormality_loss.py` | 不同 n 的 P 在内积下相互排斥 |
| KAN 显式 `n` 输入 | `kan_hypernet.KANCoeffNet` | 网络对不同 n 可以输出不同 c |
| `λ` 初始 `Z_eff/n` | `models/dirac_net.py` 在 KAN 输出后 clamp | 不同 n 的 env 衰减率不同 |
| 节点损失 `L_node` | `losses/node_count_loss.py` | 强制 n=k 必须有 k-1 节点 |
| 作用量量子化 `L_action_BS` | `losses/action_loss.py` | 用全局 `∫p_rdr` 约束目标 n |
| 渐近损失 `L_asym` | `losses/asymptotic_loss.py` | 远端衰减一致 |
| `c_0 = 0` 强制 | `readout/bspline_readout.py` | 起点完全由 env 控制 |

**这 7 点联手**，确保即使在 PDE-only + Rayleigh quotient 下，每个 n 都被锁定到其本征解附近，
不再坍缩到基态。这是 V2 相对 V1 PDE-only 方案的根本进步。

## 11. 数值监控（强烈推荐）

每个 batch 记录：

```python
metrics = {
    "L_PDE": float,
    "L_ortho": float,
    "L_node": float,
    "L_action_BS": float,
    "L_asym": float,
    "L_smooth": float,
    "L_NIST_monitor": float,             # 不参与梯度，但要看
    "E_orb_only_MAE": float,              # 主基准
    "E_calibrated_MAE": float,            # 若 Stage 2 启用 Δ_residual
    "delta_residual_abs_max": float,
    "delta_residual_ratio": float,
    "per_orb_node_pred": [...],           # 用于 sanity check
    "lambda_drift": (λ_pred / (Z_eff/n)).histogram,
    "c_norm": ||c||_2,
}
```

写到 `tensorboard` + `csv`。 

## 12. 测试要点

- `test_node_count_loss_anchor`：手工构造 P 有 0 / 1 / 2 个节点，loss 给出对应值；
- `test_action_loss_hydrogenic_quantization`：解析 Coulomb 能级下 `L_action_BS ≈ 0`；
- `test_action_loss_wrong_n_penalty`：用 1s 能量但给 n=2 target，loss 明显增大；
- `test_asymptotic_loss_zero_at_decay`：解析 `r · exp(-r)` 应给 `L_asym ≈ 0`；
- `test_smooth_loss_zero_on_constant`：c 是 constant 时 d² = 0 → loss = 0；
- `test_pde_loss_zero_on_analytic`：解析 1s 喂入应给 `L_PDE < 1e-3`（已有 V1 测试基底）。

---

## 13. **(2026-05) Deprecation Notice & Pointer to §15**

> 本节是规范化补丁。任何与本节冲突的上文（§1–§12）以本节为准。

经过 4 轮 1000-epoch 训练（详见 §15 §0），下列局部形状监督 loss 被实证为
**与 `L_PDE / L_ortho / L_action_BS / L_lambda_prior / L_decay_consistency / L_virial`
的全局信息冗余，且互相竞争梯度**：

| Loss | 状态 | 新默认权重 | 说明 |
|---|---|---|---|
| `L_node_pos` (hydrogenic node position) | **DEPRECATED** | 0.0 | 仅作为 monitor，不进 backward |
| `L_node_cross` (hydrogenic node crossing) | **DEPRECATED** | 0.0 | 仅作为 monitor |
| `L_sign` (global sign pattern) | **DEPRECATED** | 0.0 | 仅作为 monitor |
| `L_lobe_ratio` (radial lobe area ratio) | **降权保留** | 5.0（曾 30.0） | 提供互补积分信号 |
| `L_shape` (analytic cos²) | **降权固定** | 2.0（曾 5–25） | 不再 ramp |
| `L_action_BS` (Bohr-Sommerfeld) | **从 0 启用** | 0.5 (warmup) | 替代上述被废弃 loss 的"区分 n"职责 |
| `L_node` (smoothed node count) | **从 0 启用** | 0.5 | 整数标量，本来一直就有，从未真正打开 |
| `L_decay_consistency` (新) | 启用 | 10.0 | 参见 §15 |
| `L_virial` (新，通用形式) | 启用 | 1.0 | 参见 §15 |
| `L_PDE / L_decay / L_virial` | **per-sample 归一化** | — | 详见 §15 §2.2 |

详细推理、接口规范、调度协议在 `prompts/15_loss_redesign_global_constraints.md`。
实施时按 §15 进行，本文件 §7（action loss）与 §10（防退化机制表）的权重/warmup 阈值
被 §15 覆盖。
