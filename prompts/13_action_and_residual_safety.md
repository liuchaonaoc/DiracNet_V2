# 13 — Action Loss + Residual Safety Extension

> 本文件把两段新增设计讨论固化为可执行提示词：
>
> 1. MIT 2026 *On computing quantum waves exactly from classical action* 对 V2 的启发；
> 2. `Δ_residual` 可能形成新 cheating channel 的风险与约束。

## 1. 作用量思想应该怎样进入 V2

### 1.1 不要直接加入“作用量极值”loss

MIT 论文说明 Schrödinger 波可以从 classical action + density 的合适形式中精确计算出来。
这对时间含问题、多路径干涉、隧穿非常重要。

但 V2 当前求的是：

```text
1D radial bound state
stationary eigenfunction
real-valued P(r), Q(r)
```

在这个定态极限里，Hamilton-Jacobi + density 的完整系统退化成与
`Hψ = Eψ` 等价的局部条件。也就是说：

```text
action stationarity ≈ PDE residual
```

所以直接把“作用量本身”加进 loss，会与 `L_PDE` 高度重复，不提供新的物理信息。

### 1.2 应加入 Bohr-Sommerfeld 作用量量子化

V2 真正缺的是“这个本征态是第几个 n”的全局约束。对应可微 scalar loss：

```text
∫_{r_-}^{r_+} p_r(r) dr = π · (n - l - 1/2)
```

其中：

```text
p_r²(r) = 2 · (E_orb - V_eff(r) - l(l+1)/(2r²))
```

这条约束与 `L_node` 互补：

| 约束 | 信息 |
|------|------|
| `L_PDE` | 局部满足本征方程 |
| `L_node` | 波函数过零点数量 |
| `L_action_BS` | 全局作用量积分对应目标 n |

三者一起可以显著降低 PDE-only 阶段的 n-collapse 风险。

## 2. `L_action_BS` 实现提示词

生成文件：

```text
rc_diracnet_v2/losses/action_loss.py
```

实现接口：

```python
class BohrSommerfeldActionLoss(nn.Module):
    """Global radial action quantisation loss.

    L = mean_active_orbitals(∫p_r dr - π(n-l-1/2))²
    """

    def __init__(self, eps: float = 1e-12, smooth: str = "clamp") -> None:
        ...

    def forward(
        self,
        E_orb: Tensor,        # [B, N_orb]
        V_eff: Tensor,        # [B, N_grid]
        n_idx: Tensor,        # [B, N_orb]
        l_idx: Tensor,        # [B, N_orb]
        orb_mask: Tensor,     # [B, N_orb]
        grid: RadialGrid,
    ) -> Tensor:
        ...
```

关键实现：

```python
r = grid.r.to(E_orb.device).to(E_orb.dtype)
l = l_idx.to(E_orb.dtype)
centrifugal = (l * (l + 1.0)).unsqueeze(-1) / (2.0 * r.view(1, 1, -1) ** 2)
v_total = V_eff.unsqueeze(1).to(E_orb.dtype) + centrifugal
p_sq = 2.0 * (E_orb.unsqueeze(-1) - v_total)
p_r = (p_sq.clamp_min(0.0) + eps).sqrt()
S = grid.integrate(p_r, dim=-1)
target = math.pi * (n_idx.to(E_orb.dtype) - l - 0.5).clamp_min(0.0)
err = S - target
return (err.pow(2) * orb_mask.to(err.dtype)).sum() / orb_mask.sum().clamp_min(1.0)
```

### 2.1 Warmup 规则（**已被 §15 §2.1.A.2 覆盖**）

> 原规则（依据 `L_PDE < 1e-2`）在 4 轮实测中从未触发，因为 detach-E 后 PDE 稳定在 1.0–1.5。
> 新规则改为基于 epoch 的线性 ramp，并把 `w_action` 从 0.05 升到 **0.5**——
> 在新 loss 拓扑里 action 是主菜而不是配菜。

```python
# 见 prompts/15 §2.1 A.2 的 pseudo code:
#   epoch < action_warmup_start_epoch (=30)  ->  w_action = 0
#   epoch in [start, end] (=100)             ->  w_action = w_max * t (linear)
#   epoch >= end                             ->  w_action = w_max     (=0.5)
```

`L_node` 同步打开：
```python
if epoch < 10:
    w_node = 0.0
if epoch >= 10:
    w_node = 0.5    # 新方案：从 0.1 提到 0.5（接管废弃 node_pos/node_cross 的部分预算）
```

`L_action_BS` 不在训练第一步打开，避免 `E_orb` 尚未稳定时 turning point 处梯度噪声过大。
turning point 的 `p_sq.clamp_min(0)` 保留；如果 epoch 30-50 仍出现 total_loss 尖刺，
按 §15 §7 把 `action_warmup_end_epoch` 从 100 延长到 200。

### 2.2 门禁

Stage 1 gate 增加：

```yaml
L_action_BS_threshold: 1.0e-3
```

Gate 输出必须包含：

```text
action_error_max
L_action_BS
```

## 3. `Δ_residual` 为什么危险

`Δ_residual` 如果没有约束，会变成 V1 `Δ_term` 的新版本：

```text
E_orb 错
P(r) 错
λ 错
但 E_orb + Δ_residual 贴到 NIST
```

这不是学习物理，而是标量校准掩盖波函数错误。

## 4. V2 的 residual 安全协议

### 4.1 主结果永远是 `E_orb-only`

主预测：

```text
E_pred_main = Σ_a occ_a · E_orb_a
```

可选校准：

```text
E_pred_calibrated = E_pred_main + Δ_residual
```

报告顺序必须是：

1. `E_orb-only` raw/aligned RMS/MAE；
2. `E_orb + Δ_residual` raw/aligned RMS/MAE；
3. residual safety metrics；
4. OOD/LOO；
5. physics gate。

### 4.2 residual 容量分档

| 场景 | `delta_max` |
|------|-------------|
| hydrogenic Phase 1 | 1-5 meV |
| light atoms | 20 meV |
| heavy atoms | 50-100 meV，仅 ablation，不作为主结果 |

默认配置：

```yaml
stage2:
  enabled: false
  delta_max_meV: 5.0
  w_nist_max: 0.3
  residual_ratio_threshold: 0.2
```

### 4.3 禁止结构

禁止：

- `zn_bias_table`;
- `(Z, n)` 直接查表；
- 每个 manifest row 一个 learnable bias；
- 让原始 `term_id` 变成纯 lookup 而没有结构化物理特征；
- Stage 2 解冻 KAN 主路径作为默认设置。

允许：

- 用 `h_cond` + structured term features；
- 用 `(J, parity, spin multiplicity, L, jj tags)`；
- zero-init `LevelResidualHead`;
- small bounded residual。

### 4.4 residual safety 判据

任一成立则判为 `CHEATING_RISK`：

```text
E_orb-only 高误差，但 calibrated 低误差
median |Δ| / median |E_orb - E_target| > 0.2
max |Δ| 接近 delta_max
leave-one-n / leave-one-Z / leave-one-ion 崩溃
Stage 2 后 |cos| / λ / L_PDE / L_action_BS 退化
residual 结构等价 lookup
```

实现 helper：

```python
def residual_safety_verdict(metrics: dict[str, float]) -> str:
    if not metrics["physics_gate_pass"]:
        return "CHEATING_RISK"
    if metrics["delta_ratio_median"] > metrics.get("delta_ratio_threshold", 0.2):
        return "CHEATING_RISK"
    if metrics["delta_abs_max_mev"] > 0.9 * metrics["delta_cap_mev"]:
        return "CHEATING_RISK"
    if metrics["loo_rms_mev"] > metrics.get("loo_rms_threshold_mev", 200.0):
        return "CHEATING_RISK"
    if metrics["e_orb_only_mae_mev"] > metrics.get("e_orb_only_required_mae_mev", 50.0):
        return "CALIBRATION_ONLY_NOT_PHYSICS_PASS"
    return "ACCEPT_CALIBRATION"
```

## 5. 修改清单

实现本扩展时同步修改：

```text
prompts/01_architecture.md      # 加 L_action, E_orb-only 主结果
prompts/05_physics_losses.md    # 加 action loss 与 residual safety
prompts/06_two_stage_training.md# Stage 2 降级为 optional calibration
prompts/07_evaluation.md        # evaluator 强制双指标与 residual safety
prompts/09_project_layout.md    # 加 action_loss.py / test_v2_residual_safety.py
prompts/10_sprint_plan.md       # Sprint 2 增 action, Sprint 3 改 calibration
prompts/11_test_plan.md         # 加 action tests 与 residual safety tests
```

## 6. 版本命名

| 版本 | 含义 | 允许作为主结果？ |
|------|------|------------------|
| `V2.0-physics` | Stage 1 only；PDE/action/node/ortho/asym/smooth；NIST 只评估 | 是 |
| `V2.1-calibrated` | V2.0 通过门禁后，加受限 `Δ_residual` | 否，只能作为次指标 |
| `V2.x-ablation` | 放宽 residual / 解冻 KAN / heavy atom large residual | 否，只能解释机制 |

## 7. 给代码生成 Agent 的硬性要求

1. `NISTScalarHuberLoss` 不得出现在 Stage 1 的 backward 图中。
2. `BohrSommerfeldActionLoss` 必须在 Stage 1 支持 warmup 开启。
3. `v2_evaluate.py` 必须同时输出 `E_orb-only` 和 calibrated 两套指标。
4. `LevelResidualHead` 默认不启用；启用时 `delta_max_meV` 默认 5。
5. 任何 calibrated 结果都必须跑 `residual_safety_verdict`。
6. `residual_safety_verdict != ACCEPT_CALIBRATION` 时，报告不得写 `PASS`。
