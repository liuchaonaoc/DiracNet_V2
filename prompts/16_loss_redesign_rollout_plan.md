# 16 — §15 Loss Redesign: 三切片执行计划

> §15 同时改了三个独立维度（损失重组 / per-sample 归一化 / curriculum）。
> 一次性全上的风险是：若新结果不理想，无法定位是哪个维度的问题。
> 本文件把 §15 拆成 **3 个独立、可单独验证、可单独回滚** 的切片。
>
> 当前进度：见底部 §4 "状态追踪"。每完成一个切片就更新状态行。

---

## 0. 共同基线 (Reference Baseline = R1)

| 指标 | 值 |
|---|---|
| Config | `clamp 0.2 + λ_prior 300 + c_norm 1.0 + smooth 0.2 + decay 10 + virial 1` |
| MAE (12 行 hydrogenic) | **1168 meV** |
| mean_signed | +1165 meV |
| max\|err\| | 3654 meV (Z=3 n=2) |
| `lambda_rel_err` Z=3 n=4 | 0.182（贴 clamp 边界） |

任何切片的成功最低门槛 = **MAE < 1168**。低于这条线就回滚当前切片，不进入下一切片。

---

## 1. 切片 1 — 损失重组（"action 取代局部形状监督"）

### 1.1 目标
验证：**只换 loss 配方、不动归一化也不动课程**，靠 `action_BS + node` 替代 3 个废弃局部 loss，
单独能否撼动 1168 meV 这条线。

### 1.2 实施清单（全部在 `rc_diracnet_v2_project/` 内）

#### 1.2.1 config 改动 (`configs/default.yaml`)

```yaml
stage1:
  weights:
    pde: 1.0                  # 不变
    ortho: 1.0                # 不变
    node: 0.5                 # ← 从 0.0 打开
    node_pos: 0.0             # ← 废弃（保留 monitor）
    node_cross: 0.0           # ← 废弃
    sign: 0.0                 # ← 废弃
    lobe_ratio: 5.0           # ← 从 15.0 降权
    action: 0.5               # ← 从 0.0 打开
    lambda_prior: 300.0       # 不变
    shape: 2.0                # ← 从 2.5 略降
    decay_consistency: 10.0   # 不变
    virial: 1.0               # 不变
    asym: 0.01                # 不变
    smooth: 0.2               # 不变
    factor_amp: 3.0           # 不变
    c_norm: 1.0               # 不变
  warmup:
    shape_pretrain_epochs: 60                  # 沿用
    pde_weight_during_shape_pretrain: 0.0      # 沿用
    shape_weight_after_pretrain: 2.0           # ← 从 12.5 大幅下调 (= 不再 ramp)
    node_start_epoch: 10                       # 沿用
    node_weight_after_warmup: 0.5              # ← 从 0.2 提到 0.5（与 node 主权重一致）
    action_warmup_start_epoch: 30              # ← 新字段
    action_warmup_end_epoch: 100               # ← 新字段
    action_start_epoch: 10                     # 沿用但被新字段覆盖
    action_pde_threshold: 1.0e-1               # 沿用但被新字段覆盖（不再用 PDE 门）
    action_weight_after_warmup: 0.5            # ← 从 0.05 提到 0.5
```

#### 1.2.2 trainer 改动 (`rc_diracnet_v2/training/two_stage_trainer.py`)

加 `action` 项到加权和（之前 `action_w` 已经存在但权重一直是 0）：

```python
# 在 stage1 epoch loop 内现有 action_w 计算之上新增：
def _action_warmup_weight(self, epoch: int) -> float:
    """Linear ramp action weight from 0 to weights.action.

    epoch in [0, start) -> 0
    epoch in [start, end] -> linear ramp
    epoch >= end -> full weight
    """
    w_max = float(getattr(self.cfg.stage1.weights, "action", 0.0))
    start = int(getattr(self.cfg.stage1.warmup, "action_warmup_start_epoch", 30))
    end = int(getattr(self.cfg.stage1.warmup, "action_warmup_end_epoch", 100))
    if epoch < start:
        return 0.0
    if epoch >= end:
        return w_max
    t = (epoch - start) / max(1, end - start)
    return w_max * t
```

替换/补强现有 `action_w` 计算逻辑（旧的基于 `running_avg["L_PDE"] < 1e-2` 的判断永远不触发）。

epoch 摘要日志加上 `avg_action`（其它 avg_* 已经有）。

#### 1.2.3 单元测试 (`tests/test_v2_action_warmup.py` 新建)

- `test_action_warmup_zero_before_start`：epoch=0..29 → 0.0
- `test_action_warmup_linear_in_ramp`：epoch=65 → ~0.5 * (65-30)/(100-30) = 0.25
- `test_action_warmup_full_after_end`：epoch=100..1000 → w_max

#### 1.2.4 不动的文件
- 所有 loss 类（`pde_loss.py`, `decay_consistency_loss.py`, `virial_loss.py`, `action_loss.py`, `node_count_loss.py` 等）
- `dirac_net_v2.py`
- `scheduler.py`
- `v2_phase1_stage1_pde_only.yaml` （继续 override `n_epochs: 1000`）

### 1.3 验证流程
1. `pytest tests/test_v2_action_warmup.py -v` 必须通过
2. `pytest tests` 全套必须通过（无回归）
3. 3-epoch smoke：`python scripts/v2_train_stage1_only.py --config configs/v2_smoke.yaml`
   - 检查日志里出现 `avg_action` 列；`avg_action` 数值有限；不出现 NaN
4. 1000-epoch full：`python scripts/v2_train_stage1_only.py --config configs/v2_phase1_stage1_pde_only.yaml`
5. evaluate：`python scripts/v2_evaluate.py --config configs/v2_phase1_stage1_pde_only.yaml --manifest data_cache/manifest_hydrogenic_v2.parquet`

### 1.4 切片 1 通过标准
- **必须**：MAE < 1168 meV
- **必须**：mean_signed 不恶化超过 +200 meV
- **必须**：`max|err|` 不恶化超过 +500 meV
- **理想**：`L_action_BS` 末 100 epoch 平均 < 0.1（说明 action 真的收敛了）

不通过 → 回滚 config（git checkout 单 file），进入诊断模式，**不进切片 2**。

---

## 2. 切片 2 — Per-sample 归一化

### 2.1 目标
验证：在切片 1 基础上加 per-sample 归一化，能否进一步消除 "n=1 vs n=4 跷跷板"。

### 2.2 实施清单
#### 2.2.1 loss 接口扩展
- `losses/pde_loss.py`：`DiracPDELoss.__init__` 加 `per_sample_normalise=False, e_char_eps=1.0e-6`；`forward` 加 `E_char_per_orb=None`
- `losses/decay_consistency_loss.py`：同上
- `losses/virial_loss.py`：同上
- 实现 helper `_per_sample_normalise(residual_per_orb, E_char_per_orb, orb_mask, eps)`，可共享到 `losses/_utils.py`

#### 2.2.2 trainer 改动
- 在 `_physics_losses` 顶部计算 `E_char = 0.5 * out["lam"].detach().pow(2).clamp_min(lam_min**2)`
- 三个 loss 调用加 `E_char_per_orb=E_char, per_sample_normalise=True`

#### 2.2.3 config 改动 (`default.yaml`)
```yaml
stage1:
  per_sample_normalise:
    enabled: true
    e_char_source: lambda_squared
    e_char_eps: 1.0e-6
```

#### 2.2.4 单元测试 (`tests/test_v2_per_sample_normalise.py` 新建)
- `test_pde_loss_normalised_lower_for_high_z`：相同 ψ 误差比例下，归一化版应给 Z=1 和 Z=3 接近的 loss
- `test_per_sample_normalise_zero_when_residual_zero`
- `test_per_sample_normalise_gradient_balance`：构造 batch 含 (Z=1, n=4) 和 (Z=3, n=1)，归一化后 `‖dL/dλ_orb‖` 在两 orbital 上量级接近

### 2.3 验证流程（同切片 1，加一步）
3.5. 跑 100-epoch sanity run；记录 `std(per_orb_normalised_pde) / mean(...)` 应显著 < 切片 1 同期值

### 2.4 切片 2 通过标准
- **必须**：MAE < 切片 1 MAE
- **必须**：`std(per_orb_pde) / mean(per_orb_pde)` 至少降低 30%
- **理想**：`max|err|` per Z 的方差降低（不再一头超大）

不通过 → 关掉 `per_sample_normalise.enabled`，回到切片 1 状态。

---

## 3. 切片 3 — Curriculum learning

### 3.1 目标
解决"浅态/深态需求方向相反"的最后一道病灶。

### 3.2 实施清单
#### 3.2.1 trainer 改动
- 实现 `_curriculum_mask(batch, epoch, n_epochs) -> Tensor[B, N_orb] bool`
- 实现 `_lambda_prior_curriculum_boost(epoch, n_epochs) -> float`
- 在 `_physics_losses` 把 `orb_mask` 替换为 `orb_mask & curriculum_mask`
- 在 `run_stage1` 把 `lambda_prior` 项的权重乘上 boost
- 实现 `_check_curriculum_gate(...)`，子阶段切换前调用

#### 3.2.2 config 改动 (`default.yaml`)
```yaml
stage1:
  curriculum:
    enabled: true
    schedule: [0.20, 0.45, 0.75, 1.00]
    n_max: [1, 2, 3, 4]
    lambda_prior_boost: 2.0
    transition_check:
      cos_threshold: 0.99
      lambda_rel_threshold: 0.05
      pde_drop_ratio: 10.0
      max_extension_ratio: 0.5
```

#### 3.2.3 单元测试 (`tests/test_v2_curriculum.py` 新建)
- `test_curriculum_mask_progression`：epoch=0/200/500/800 各阶段 n_max 正确
- `test_lambda_prior_boost_decays`：进入 1b 时 boost=2.0，1b 1/3 时间后回到 1.0
- `test_curriculum_disabled_passthrough`：`enabled: false` 时 mask 全 True

### 3.3 验证流程
3-epoch smoke 在课程下不行（数据太少）；改为 60-epoch smoke 看课程切换 (1a end at 12, 1b end at 27)。

### 3.4 切片 3 通过标准
- **必须**：MAE < 800 meV (§15 §8 验收线)
- **必须**：max_signed per Z 各 Z < +2000 meV
- **必须**：mean(λ_rel_err) 在 n ≤ 3 上 < 5%

---

## 4. 状态追踪

| 切片 | 状态 | MAE | mean_signed | max\|err\| | 日期 | 备注 |
|---|---|---|---|---|---|---|
| R1 baseline | DONE | 1168 | +1165 | 3654 | 2026-05-21 | 参考线 |
| 切片 1 (loss 重组) | TODO | — | — | — | — | — |
| 切片 2 (per-sample 归一) | BLOCKED on 切片 1 | — | — | — | — | — |
| 切片 3 (curriculum) | BLOCKED on 切片 2 | — | — | — | — | — |

---

## 5. 回滚协议

每个切片对应一个 git commit（建议）。失败时：
- 切片 1 失败 → `git revert` 切片 1 commit → 回到 R1
- 切片 2 失败 → `git revert` 切片 2 commit → 回到切片 1 (新 baseline)
- 切片 3 失败 → `git revert` 切片 3 commit → 回到切片 2

如果不用 git，则保留 `configs/default.yaml.bak_R1`, `configs/default.yaml.bak_slice1`, ... 作为备份。

---

## 6. 一句话总结
> **切片 1 检验"action 是否真能取代局部形状监督"，切片 2 检验"归一化是否真能消除跨样本跷跷板"，切片 3 检验"课程学习是否能解决最后一道方向冲突"。三步独立可证伪，任何一步失败都不污染下一步的诊断。**
