# 05 — Physics Losses

## 1. 总览

V2 在 Stage 1 共用 5 个 loss（NIST **不参与梯度**）：

| Loss | 数学 | 维度 | 默认权重 |
|------|------|------|----------|
| `L_PDE` | `‖ H_D ψ − E_orb ψ ‖²` (V1 复用) | scalar | **10.0** |
| `L_ortho` | `‖ S − I ‖_F²` (V1 复用) | scalar | 1.0 |
| `L_node` | 节点计数惩罚（n - l - 1） | scalar | 0.1 |
| `L_asym` | 远端衰减一致性 `P(r_max) ≈ 0` & log-slope ≈ -λ | scalar | 0.01 |
| `L_smooth` | B-spline 系数二阶差分 `Σ (c_{k+1} - 2c_k + c_{k-1})²` | scalar | 1e-4 |

Stage 2 增加：

| Loss | 数学 | 维度 | 默认权重 |
|------|------|------|----------|
| `L_NIST_res` | `Huber(E_pred + Δ_res − E_target)` | scalar | 1.0 (随 stage 2 ramp 起来) |

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

## 7. NIST 损失（Stage 2）

直接复用 V1 `losses/nist_scalar_loss.py`：

```python
from rc_diracnet.losses.nist_scalar_loss import NISTScalarHuberLoss
```

但在 V2 里这是 **作用在残差**上的损失。流程：

```
Stage 1 终点保存 E_orb_stage1_table [B, N_orb] (per row)
Stage 2:
   E_orb_a = orbital_energy_from_dirac(...)    # 仍按 Dirac 算
   E_orb_sum = Σ_a occ_a · E_orb_a             # Koopmans 求和
   Δ_res = LevelResidualHead(h_cond, J, π, term)   # bounded ±50 meV
   E_pred = E_orb_sum + Δ_res
   L_NIST_res = Huber(E_pred − E_target)
```

注意 **不是** "把残差表 detach 后只算 Δ_res 的 Huber"。E_orb 仍然是从 P,Q 算出的可微量；
我们 **只** 让 `Δ_residual` 和（如果选了）`lr=1e-6` 的 KAN 主路径 一起学。

## 8. 损失权重与 ramp 协议（Stage 1 → Stage 2）

```python
# Stage 1
weights_stage1 = {
    "pde": 10.0,
    "ortho": 1.0,
    "node": 0.1,
    "asym": 0.01,
    "smooth": 1e-4,
    # nist not in loss
}

# Stage 2
weights_stage2 = {
    "pde": 10.0,       # 保持，防 cheating
    "ortho": 1.0,
    "node": 0.1,
    "asym": 0.01,
    "smooth": 1e-4,
    "nist": ramp(epoch, start=0.01, end=1.0, n_warmup=20),   # 线性 ramp
}
```

NIST 权重 ramp 是关键——直接给 1.0 会让 NIST 梯度瞬间压制 PDE，重蹈 V1 R9 覆辙。
"线性 ramp 20 个 epoch"是经验值，可调。

## 9. PDE-only 阶段的 **关键防退化机制**

> 为什么 Stage 1 PDE-only 不会坍缩到 n=1?

| 防退化点 | 实现位置 | 效果 |
|----------|----------|------|
| 跨轨道正交（`L_ortho`） | `losses/orthonormality_loss.py` | 不同 n 的 P 在内积下相互排斥 |
| KAN 显式 `n` 输入 | `kan_hypernet.KANCoeffNet` | 网络对不同 n 可以输出不同 c |
| `λ` 初始 `Z_eff/n` | `models/dirac_net.py` 在 KAN 输出后 clamp | 不同 n 的 env 衰减率不同 |
| 节点损失 `L_node` | `losses/node_count_loss.py` | 强制 n=k 必须有 k-1 节点 |
| 渐近损失 `L_asym` | `losses/asymptotic_loss.py` | 远端衰减一致 |
| `c_0 = 0` 强制 | `readout/bspline_readout.py` | 起点完全由 env 控制 |

**这 6 点联手**，确保即使在 PDE-only + Rayleigh quotient 下，每个 n 都被锁定到其本征解附近，
不再坍缩到基态。这是 V2 相对 V1 PDE-only 方案的根本进步。

## 10. 数值监控（强烈推荐）

每个 batch 记录：

```python
metrics = {
    "L_PDE": float,
    "L_ortho": float,
    "L_node": float,
    "L_asym": float,
    "L_smooth": float,
    "L_NIST_monitor": float,             # 不参与梯度，但要看
    "per_orb_node_pred": [...],           # 用于 sanity check
    "lambda_drift": (λ_pred / (Z_eff/n)).histogram,
    "c_norm": ||c||_2,
}
```

写到 `tensorboard` + `csv`。 

## 11. 测试要点

- `test_node_count_loss_anchor`：手工构造 P 有 0 / 1 / 2 个节点，loss 给出对应值；
- `test_asymptotic_loss_zero_at_decay`：解析 `r · exp(-r)` 应给 `L_asym ≈ 0`；
- `test_smooth_loss_zero_on_constant`：c 是 constant 时 d² = 0 → loss = 0；
- `test_pde_loss_zero_on_analytic`：解析 1s 喂入应给 `L_PDE < 1e-3`（已有 V1 测试基底）。
