# 15 — Loss Redesign: Global Constraints + Per-Sample Normalisation + Curriculum

> 本文件汇总 V2 在 4 轮 hydrogenic 1000-epoch 训练后形成的 **损失拓扑重设计** 方案。
> 与 §05（physics losses）、§06（two-stage training）、§13（action + residual safety）
> 并列；优先级处在 §05 之上：当本文件与 §05 描述冲突时，以本文件为准。
>
> 任何动手改 trainer / loss / config 之前，必须先读完本文件 + ADR-017 / ADR-018。

---

## 0. 触发本次重设计的实证

> Phase 1 hydrogenic 集（Z ∈ {1, 2, 3}, n ∈ {1..4}, 共 12 行）4 轮 1000-epoch 训练对比。

| 轮 | 关键改动 | MAE (meV) | mean_signed | max\|err\| | 备注 |
|---|---|---|---|---|---|
| R1 (baseline) | PDE detach + decay + virial + 形状监督砍半 | **1168** | +1165 | 3654 | 历史最优；n-collapse 完全治愈 |
| R2 | `lam_log_res_clamp 0.2→0.4`，`λ_prior 300→80` | 1364 | +1363 | 4528 | λ 全线漂离 `Z/n`（"自反一致性"自由度被解锁） |
| R3 | `c_norm 1.0→0.3`，`smooth 0.2→0.05`（释放表达力） | 1388 | +1388 | 3310 | **深态改善 / 浅态毁灭** 的典型跷跷板 |
| R4 | R3 + 回退 R2 改动 | 1390 ± 噪声 | +1390 | 3300 | 跷跷板持续，全局调参边际收益枯竭 |

四轮证明：**单一全局权重旋钮已不能同时让 (Z=3, n=2) 深态与 (Z=1, n=2) 浅态都改善**。
继续在 §05 既有 loss 谱系内调参不会带来新增收益。

诊断指向三条结构性问题：

1. **局部约束冗余**：`node_pos / node_cross / sign / lobe_ratio` 共 4 项都是对单点或单瓣的解析形状监督；它们与 `L_PDE` 是冗余信息（解析解必然满足 PDE），同时它们之间互相竞争梯度。
2. **跨样本能量尺度差 16×**：Z=3 n=1 的特征能量 −4.5 Ha，Z=1 n=4 的 −0.03 Ha。把绝对能量量纲的 `L_PDE / L_virial / L_decay` 直接 `.mean()`，n=1 的 100 meV 误差对应的反向梯度被 n=4 的 1 eV 误差完全淹没。
3. **缺乏全局"是第几个本征态"的整数约束**：除了 `L_node`（节点计数），目前**没有任何 loss 在告诉网络"这是第 n 个解"**。`L_action_BS`（Bohr-Sommerfeld）按 §05 设计本应承担此角色，但 4 轮训练里权重一直为 0、从未启用。

---

## 1. 物理动机：从「局部点对点拟合」转向「全局积分约束」

### 1.1 局部 vs 全局：物理信息含量比较

| 约束 | 信息维度 | 与 PDE 是否独立 | 跨 n 区分度 |
|---|---|---|---|
| `L_PDE` (Dirac residual) | 逐点能量量纲² | — | 弱（同一 V 下所有本征态都满足） |
| `L_ortho` | 配对积分 | ✓ | 强（不同 n 必须正交） |
| `L_node` | 整数标量 | ✓ | 强（节点数 = n − l − 1） |
| `L_action_BS` | 全局积分标量 | ✓ | **极强**（直接编码 n） |
| `L_node_pos` (废弃) | 多个 r 点 | ✗（PDE 已隐含） | 中 |
| `L_node_cross` (废弃) | 多个 r 段 | ✗（与 node_pos 高度相关） | 中 |
| `L_sign` (废弃) | 整体符号 | ✗（cos² 已包含） | 低 |
| `L_lobe_ratio` (降权保留) | O(n) 个积分 | 半独立 | 中 |
| `L_shape` (降权保留) | 单个 cos² | 半独立 | 强（但代价大） |

**结论**：`L_action_BS` 是被严重低估的"全局 n 锚"。打开它后，可以**直接砍掉**3 个废弃项与降低 2 个保留项的权重，腾出整组 "形状监督预算" 给真正必要的物理约束。

### 1.2 为什么 Bohr-Sommerfeld 自然解决跨样本竞争

Bohr-Sommerfeld 残差 `S − π(n − l − 1/2)` 的量级是 **O(1)**，与 Z、n 无关——因为 `S = π(n − l − 1/2)` 对所有束缚态都是同一个量级。
这意味着 `L_action_BS` **天生具备跨样本梯度均衡**，不像 `L_PDE` 需要除以 `E_char²` 才能避免梯度被高 Z 主导。

把 `L_action_BS` 作为 stage1 的主物理约束之一，等于把"梯度跷跷板"问题在结构层面而不是数值层面化解。

### 1.3 浅态 vs 深态的物理本质

R3 实验告诉我们：**释放 c_norm/smooth 会引入远端高频** → 浅态（缓慢衰减）对远端噪声**剧烈敏感**，深态（紧致）反而需要更细的局部结构。
这不是参数调节问题，而是 **不同 n 的本征态对正则化的需求方向相反**。
解决路径只能是 **解耦不同 n 的优化轨迹**，即课程学习。

---

## 2. 三条治理路线（A + B + C 同时启用）

### 2.1 A — 用 Bohr-Sommerfeld action 替换局部形状监督

#### A.1 Loss 重组矩阵

| Loss | R4 权重 | **新方案权重** | 改动 |
|---|---|---|---|
| `L_PDE` (detach E) | 1.0 | **1.0** | 保留 |
| `L_ortho` | 1.0 | **1.0** | 保留 |
| `L_node` | 0.0 | **0.5** | **打开**（节点计数，已存在但权重为 0） |
| `L_action_BS` | 0.0 | **0.5 (warmup)** | **打开**（核心新启用） |
| `L_node_pos` | 15.0 | **0.0** | **废弃** |
| `L_node_cross` | 12.5 | **0.0** | **废弃** |
| `L_sign` | 12.5 | **0.0** | **废弃** |
| `L_lobe_ratio` | 15.0 | **5.0** | 降权保留 |
| `L_shape` | 2.5 (12.5 after warmup) | **2.0** (固定) | 降权固定 |
| `L_lambda_prior` | 300.0 | **300.0** | 保留（外部锚） |
| `L_decay_consistency` | 10.0 | **10.0** | 保留（λ ↔ E 自洽） |
| `L_virial` | 1.0 | **1.0** | 保留（全局动能-势能） |
| `L_asym` | 0.01 | **0.01** | 保留 |
| `L_smooth` | 0.2 | **0.2** | 保留 |
| `L_c_norm` | 1.0 | **1.0** | 保留 |
| `L_factor_amp` | 3.0 | **3.0** | 保留 |

> **三项废弃**会通过 evaluator 与 unit test 保留计算（仅作 monitor），但不进入 backward。
> 这是 §05 中 §10「关键防退化机制」表格里"节点损失"那一行的物理替代品：
> `L_action_BS + L_node + L_ortho + λ_prior` 4 件套同样覆盖防 n-collapse 任务，
> 且每件都是 **真正全局**、与 PDE 信息独立。

#### A.2 `L_action_BS` 的 warmup 协议（区别于 §05 §7.4）

§05 原协议是"`L_PDE < 1e-2` 时打开"。但实测 `L_PDE` 在带 detach 之后稳定在 1.0–1.5 量级，永远到不了 1e-2。新协议改为基于 epoch + 节点 loss：

```python
# Pseudo (整合到 TwoStageTrainer)
if epoch < cfg.stage1.warmup.action_warmup_start_epoch:
    w_action_bs = 0.0
elif epoch < cfg.stage1.warmup.action_warmup_end_epoch:
    t = (epoch - start) / (end - start)
    w_action_bs = cfg.stage1.weights.action * t          # 线性 ramp
else:
    w_action_bs = cfg.stage1.weights.action
```

推荐：`action_warmup_start_epoch = 30`，`action_warmup_end_epoch = 100`，
`weights.action = 0.5`（远高于 §05 §7.4 的 0.05；那个值是基于"配菜"假设，
新方案里 action 是主菜）。

### 2.2 B — Per-sample 相对化归一化

#### B.1 通用公式

对所有"能量量纲²"的 loss，**先**逐 (batch, orbital) 除以特征尺度的平方，**再**做 mean：

```python
def _per_sample_normalise(residual_sq_per_orb: Tensor,    # [B, N_orb]
                          E_char_per_orb: Tensor,         # [B, N_orb], > 0
                          orb_mask: Tensor,
                          eps: float = 1.0e-6) -> Tensor:
    norm = E_char_per_orb.clamp_min(eps).pow(2)
    weighted = residual_sq_per_orb / norm
    mask = orb_mask.to(weighted.dtype)
    return (weighted * mask).sum() / mask.sum().clamp_min(1.0)
```

`E_char_per_orb` 的来源（按可获得性排序，**第一个能拿到就用**）：

1. **`λ_pred² / 2`**（动力学相关、模型自洽、对任意势成立；推荐默认）
2. **`Z²_eff / (2 n²)`**（hydrogenic 解析尺度；仅对类氢系统可用，但作为兜底极稳）
3. **`|E_orb_detached|`**（永远可拿，但训练初期可能极小或正，需 clamp）

实施推荐：`E_char = 0.5 * λ.detach()².clamp_min(λ_min²)`。
理由：`λ` 由 KAN 输出，包含网络对该 orbital 尺度的"信念"；detach 避免 normaliser 自己反向求导稀释梯度；用 `λ_min²` 防早期 λ→0。

#### B.2 哪些 loss 需要相对化

| Loss | 原始残差量纲 | 归一化 | 新接口 |
|---|---|---|---|
| `L_PDE` | 能量² × 长度 (after integrate) | ✓ `/ E_char²` | `DiracPDELoss(per_sample_normalise=True)` |
| `L_decay_consistency` | (λ² + 2E)² ~ 能量² | ✓ `/ E_char²` | `DecayConsistencyLoss(per_sample_normalise=True)` |
| `L_virial` | (2T − ⟨r∇V⟩)² ~ 能量² | ✓ `/ E_char²` | `VirialLoss(per_sample_normalise=True)` |
| `L_action_BS` | (S − π(n−l−½))² ~ O(1)² | ✗ 已经无量纲 | 不变 |
| `L_node` | (n_count − target)² ~ O(1)² | ✗ 已经无量纲 | 不变 |
| `L_ortho` | (⟨ψ_i\|ψ_j⟩)² ~ O(1) | ✗ 已经无量纲 | 不变 |
| `L_lambda_prior` | log(λ/λ_ref)² ~ O(1) | ✗ 已经无量纲 | 不变 |
| `L_shape` | 1 − cos²(P_pred, P_an) ~ [0, 1] | ✗ 已经归一 | 不变 |
| `L_lobe_ratio` | log² | ✗ 已经无量纲 | 不变 |
| `L_smooth`, `L_c_norm`, `L_factor_amp` | 系数空间 | ✗ 与样本能量无关 | 不变 |

> 关键：**只有真正带能量量纲的 3 个 loss 需要归一化**，其它无量纲损失维持原状。
> 这避免了"为归一化而归一化"造成 KAN 训练数值不稳。

#### B.3 期望效果

- `L_PDE_per_sample` 在 (Z=1, n=4) 和 (Z=3, n=1) 上的梯度量级应进入同一个 decade
- 不再出现"为了降低 Z=3 n=1 的大绝对残差，把 Z=1 n=4 的小相对残差进一步推大"
- 可监控指标：`std(per_orb_pde_normalised) / mean(...)` 应显著低于不归一化版本

### 2.3 C — Curriculum learning（按 n 阶梯）

#### C.1 硬课程（推荐起步）

为 stage1 划分 4 个子阶段（n_max 渐进开放）：

| 子阶段 | epoch 区间 | 活跃 n | 占总 epoch 比例 |
|---|---|---|---|
| 1a | [0, 0.20·N] | n ≤ 1 | 20% |
| 1b | [0.20·N, 0.45·N] | n ≤ 2 | 25% |
| 1c | [0.45·N, 0.75·N] | n ≤ 3 | 30% |
| 1d | [0.75·N, N] | n ≤ 4 | 25% |

实施位置：`TwoStageTrainer._physics_losses` 中加入 curriculum mask：

```python
def _curriculum_mask(self, batch, epoch: int, n_epochs: int) -> Tensor:
    n_idx = batch["config_shells"][..., 0].to(self.device)     # [B, N_orb]
    schedule = self.cfg.stage1.curriculum.schedule              # e.g. [0.20, 0.45, 0.75, 1.00]
    n_max_per_stage = self.cfg.stage1.curriculum.n_max          # e.g. [1, 2, 3, 4]
    progress = (epoch + 1) / max(1, n_epochs)
    n_max = n_max_per_stage[-1]
    for thr, nm in zip(schedule, n_max_per_stage):
        if progress <= thr:
            n_max = nm
            break
    return (n_idx <= n_max)                                     # bool, [B, N_orb]
```

`orb_mask_effective = orb_mask & curriculum_mask`，传给所有需要 mask 的 loss。

#### C.2 跨阶段反向漂移保护

每次 `n_max` 增加（进入下一子阶段）时，对之前已收敛的 (Z, n) **临时提高 `λ_prior` 权重 ×2**，
直到下一子阶段的 1/3 时间过后逐步衰减回基线。这防止"开放 n=3 后 n=1,2 反向退化"。

```python
def _lambda_prior_curriculum_boost(self, epoch: int, n_epochs: int) -> float:
    """Return multiplier on lambda_prior weight, decays 2.0 → 1.0 after each stage transition."""
    progress = (epoch + 1) / max(1, n_epochs)
    schedule = self.cfg.stage1.curriculum.schedule
    for thr in schedule[:-1]:
        if 0.0 <= (progress - thr) < (schedule[1] - schedule[0]) / 3.0:
            decay = 1.0 - 3.0 * (progress - thr) / (schedule[1] - schedule[0])
            return 1.0 + max(0.0, decay)
    return 1.0
```

#### C.3 子阶段进入门禁

进入 1b（开放 n=2）前必须满足：

- `|cos(P_pred, P_an)|` 在所有 (Z, n=1) 上 > 0.99
- `|λ_pred − Z/1| / Z < 5%` 在所有 (Z, n=1) 上
- `L_PDE_per_sample` 末 5 epoch 平均 < `L_PDE_per_sample` 第 5 epoch 的 1/10

任意一项不达标 → 当前子阶段 epoch 数自动延长 50%（不超过 stage1 总 epoch 的 30%），再检测。

类似规则应用到 1b → 1c, 1c → 1d。

#### C.4 数据准备

不需要重新切分 dataset；只在 batch 级用 mask 实现。
manifest 维持全量，确保即使在 1a，KAN 见到的 (Z, n=2..4) 输入仍然 forward 过，但不进 backward。
这样 KAN 对 (Z, n=2..4) 的输出在 1b 启动时不会是完全冷启动状态。

---

## 3. 配套机制 D — 自适应权重 (lite GradNorm，可选)

> 这是补充建议，**不作为 L1 实施目标**，但写下来作为后续 fallback。

启发：即使 A+B+C 都做了，仍可能出现"shape 与 action 互相打架"的情况。
此时引入 GradNorm-lite：

```python
# 每 100 step 调用一次
def _adjust_weights_via_gradnorm(losses_dict, shared_params, target_ratio=1.0):
    # 1. compute per-loss grad-norm on shared_params (e.g. KAN main MLP)
    grad_norms = {}
    for name, loss in losses_dict.items():
        grads = torch.autograd.grad(loss, shared_params, retain_graph=True)
        grad_norms[name] = sum(g.detach().norm() for g in grads).item()
    # 2. compute target: geometric mean
    G_mean = exp(mean(log(grad_norms.values())))
    # 3. nudge weights: w *= (G_mean / G_i)^alpha   (alpha=0.1, 小步走)
    for name in losses_dict:
        w_new = w_current[name] * (G_mean / grad_norms[name]) ** 0.1
        w_current[name] = clamp(w_new, w_current[name]/2, w_current[name]*2)
```

仅在 stage1 的 1d 子阶段（最后 25%）启用，且不参与梯度（detach grad_norms 后只用值）。

---

## 4. 接口规范（实施时严格遵循）

### 4.1 Loss 类的新签名

```python
# rc_diracnet_v2/losses/pde_loss.py
class DiracPDELoss(nn.Module):
    def __init__(self, small_component_scale=None, detach_energy=True,
                 per_sample_normalise=True, e_char_eps=1.0e-6):
        ...
    def forward(self, P, Q, dPdr, dQdr, E_orb, V_eff, kappa, r_grid, grid,
                orb_mask, E_char_per_orb=None) -> Tensor:
        """
        If per_sample_normalise=True and E_char_per_orb is provided ([B, N_orb] > 0),
        the per-orbital integrated residual is divided by E_char² before averaging.
        If E_char_per_orb is None, falls back to |E_orb.detach()| (clamped).
        """
        ...

# 同样的扩展应用到 VirialLoss / DecayConsistencyLoss
```

### 4.2 Trainer 必须传入 `E_char_per_orb`

```python
# rc_diracnet_v2/training/two_stage_trainer.py
def _physics_losses(self, batch, out):
    ...
    lam = out["lam"]                              # [B, N_orb]
    lam_min = float(self.cfg.envelope.lambda_min)
    E_char = 0.5 * lam.detach().pow(2).clamp_min(lam_min**2)   # [B, N_orb]
    
    # curriculum mask
    if getattr(self.cfg.stage1, "curriculum", None) is not None:
        orb_mask = orb_mask & self._curriculum_mask(batch, self.epoch, self.n_epochs)
    
    return {
        "pde": DiracPDELoss(detach_energy=True, per_sample_normalise=True)(
            ..., E_char_per_orb=E_char),
        "decay_consistency": DecayConsistencyLoss(per_sample_normalise=True)(
            out["lam"], out["E_orb"], orb_mask, E_char_per_orb=E_char),
        "virial": VirialLoss(per_sample_normalise=True)(
            ..., E_char_per_orb=E_char),
        "action": BohrSommerfeldActionLoss()(
            out["E_orb"], out["v_eff"], n_idx, l_idx, orb_mask, self.model.grid),
        "node": NodeCountLoss()(wf["P"], (n_idx - l_idx - 1).clamp_min(0), orb_mask),
        ...
    }
```

### 4.3 config 新增字段

```yaml
stage1:
  curriculum:
    enabled: true
    schedule: [0.20, 0.45, 0.75, 1.00]        # 子阶段进度上界
    n_max:    [1, 2, 3, 4]                    # 对应活跃最大 n
    lambda_prior_boost: 2.0                   # 进入新子阶段时的临时乘数
    transition_check:
      cos_threshold: 0.99
      lambda_rel_threshold: 0.05
      pde_drop_ratio: 10.0                    # 末段 PDE 必须 < 起段 / 10
      max_extension_ratio: 0.5                # 子阶段最长延长 50%
  warmup:
    action_warmup_start_epoch: 30
    action_warmup_end_epoch: 100
    shape_pretrain_epochs: 60                 # 保留旧字段
    shape_weight_after_pretrain: 2.0          # 注意：从 12.5 大幅下调
  weights:
    pde: 1.0
    ortho: 1.0
    node: 0.5                                 # 打开
    action: 0.5                               # 打开
    node_pos: 0.0                             # 废弃
    node_cross: 0.0                           # 废弃
    sign: 0.0                                 # 废弃
    lobe_ratio: 5.0                           # 降权
    shape: 2.0                                # 降权固定
    lambda_prior: 300.0
    decay_consistency: 10.0
    virial: 1.0
    asym: 0.01
    smooth: 0.2
    factor_amp: 3.0
    c_norm: 1.0
  per_sample_normalise:
    enabled: true
    e_char_source: lambda_squared             # {lambda_squared, hydrogenic_Z2n2, e_orb_abs}
    e_char_eps: 1.0e-6
```

---

## 5. 训练调度时间线（1000-epoch 总预算示例）

```
Epoch [   0,   30]:  warmup       — action=0, shape=12.5→2.0 linear, curriculum n_max=1
Epoch [  30,  100]:  action ramp  — action 0→0.5 linear, curriculum n_max=1
Epoch [ 100,  200]:  1a 末段      — curriculum n_max=1 stable
Epoch [ 200,  450]:  1b           — curriculum n_max=2, λ_prior boost ×2 → 1× over first 1/3 of 1b
Epoch [ 450,  750]:  1c           — curriculum n_max=3, λ_prior boost ×2 → 1× over first 1/3 of 1c
Epoch [ 750, 1000]:  1d           — curriculum n_max=4, λ_prior boost ×2 → 1× over first 1/3 of 1d
                                  — optional: GradNorm-lite kicks in
```

每个子阶段结束前自动跑 `_check_curriculum_gate`；不达标则当前子阶段延长（封顶 50%）。

---

## 6. 单元测试（写到 tests/test_v2_loss_redesign.py）

1. `test_action_loss_hydrogenic_vanishes`：氢类解析 E + V_eff = -Z/r 喂入 → `L_action_BS < 1e-3`
2. `test_action_loss_wrong_n_penalty`：用 1s 的 E 喂入 n=2 target → loss 显著大于 1s target 的版本
3. `test_per_sample_normalise_balances_gradient`：构造 Z=1 n=4 与 Z=3 n=1 同 batch，比较归一化前后 `dL/dλ` 的方差
4. `test_curriculum_mask_progression`：调用 `_curriculum_mask` 在 epoch 0, 200, 500, 800 应分别返回只激活 n=1, n≤2, n≤3, n≤4
5. `test_lambda_prior_boost_decays`：进入新子阶段后 mid-stage 时 boost 应回到 1.0
6. `test_pde_loss_falls_back_when_E_char_none`：不传 `E_char_per_orb` 时 PDE 行为退化为旧版（detach E_orb 后取 |E_orb|）

---

## 7. 已知风险与回滚

| 风险 | 表象 | 缓解 |
|---|---|---|
| `L_action_BS` 在 turning point 处梯度爆炸 | epoch 30-50 出现 `total_loss` 尖刺 | clamp p_sq 的 min 不变；如尖刺持续，把 action warmup_end 拉长到 200 |
| 课程切换瞬间 `L_ortho` 跳升 | 进入 1b 时 ortho 翻几倍 | 切换瞬间将 ortho 权重 ×2，持续 20 epoch |
| Per-sample 归一化导致 `lambda_min` 下界过紧 | `E_char` 全部贴 `λ_min²/2` | 把 `λ_min` 从 0.05 降到 0.02 |
| 课程门禁卡在 1a 出不去 | 1a 延长 50% 后 cos 仍不到 0.99 | 降低 1a 门禁到 cos > 0.95（只对 1a），并在 stage1 末尾再做一次 1a→1d 全门禁 |
| GradNorm-lite 触发后 weight 漂移 | shape weight 一夜涨到 50 | clamp 任何 weight 调整在 [0.5×, 2×] 单 step 限幅；本来就只在 1d 启用 |

**回滚路径**：每个改动 (A / B / C / D) 独立开关，可单独关闭。回到 R1 baseline 的等价 config 必须能通过单元测试 `test_legacy_R1_config_still_works`。

---

## 8. 验收指标（成功标准）

stage1 结束时（1000 epoch）：

1. **必须达成**：`MAE_overall < 800 meV`（R1 是 1168；目标至少 ↓ 30%）
2. **必须达成**：`max_signed_per_Z` 各 Z 不超过 +2000 meV（R1 Z=3 是 +3654）
3. **必须达成**：`mean(λ_rel_err)` 在所有 n ≤ 3 上 < 5%
4. **理想**：mean_signed 接近 0（不再系统性 underbinding；说明 ψ 形状真正改善而非 Rayleigh 上界）
5. **理想**：`L_action_BS_per_orb` 末 100 epoch 平均 < 0.1 在所有 (Z, n) 上

未达成 1-3 中任意一项 → 本方案 fail，回到 R1 baseline，重新设计。

---

## 9. 与现有文档的接口

| 文档 | 修改 |
|---|---|
| `prompts/05_physics_losses.md` | 文末追加 §13 "Deprecation Notice & Pointer to §15"；权重表标注 `node_pos/node_cross/sign` 为 deprecated |
| `prompts/06_two_stage_training.md` | §2.1 stage1 loss 公式加入 curriculum + per-sample 归一化注释；§2.4 门禁加入子阶段过渡门禁 |
| `prompts/13_action_and_residual_safety.md` | §2.1 warmup 规则更新为本文 §2.1 A.2 的版本 |
| `docs/design_rationale.md` | 新增 ADR-017 (per-sample normalisation)、ADR-018 (curriculum learning)、ADR-019 (deprecate local shape losses) |

任何与本文档冲突的旧描述均以本文档为准。
