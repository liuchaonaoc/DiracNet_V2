# V2 Design Rationale

> 把"为什么这样选 / 不那样选"的关键决策固化下来，避免后续被遗忘或来回反复。
> 任何与本文件结论冲突的修改，要先在这里写一个 ADR (Architecture Decision Record)。

## ADR-001 — 用 B-spline 而不是 RC tanh 或 monomial polynomial

**Status**: Accepted (V2.0)

**Context**：
V1 的 RC `tanh(a·f(r)+b)` 在 7 种 `f` 与 D_res=512 下仍无法张成 Laguerre L_{n-1}^1，
n ≥ 3 的形状余弦稳定在 0 附近（B7-B9 ablation）。V1 B10 引入 monomial polynomial
`Σ c_k r^k` 后形状余弦回升到 0.7-0.9，但高阶单项式数值病态、对 K 不稳定。

**Decision**: 用 B-spline（cubic + log-spaced 32 节点）取代 RC + monomial。

**Consequences**:
- ✓ 局部支撑：每个 B_k 只在小区间非零，conditioning 远好于 r^k
- ✓ 完备：任意光滑函数可在均匀细化下任意逼近
- ✓ 自然多尺度：log-spaced 节点同步覆盖核区与价层
- ✗ 不是物理基底：需要 envelope `r^γ exp(-λr)` 提供长尾衰减；端点处需要约束 c_0=0

**Alternatives rejected**:
- Laguerre `L_n^l(2λr)`：仅类氢精确；多电子需多 ζ；实现复杂
- Sturmian：α 非线性；调优困难
- 纯 DVR / FE：N_grid 自由度（512+）→ 泛化差
- Random tanh：已经在 V1 验证无法张成 Laguerre

---

## ADR-002 — (λ, c) 由 KAN 一次性输出，envelope 不再持有 λ-MLP

**Status**: Accepted (V2.0)

**Context**：
V1 中 `λ` 由 envelope 内部 MLP 出，`c`（或 RC W）由 readout 内部 MLP 出。
两个 MLP 不共享上下文 → 训练时通过 loss 隐式协同 → "λ 漂离 Z/n 而 c 跟着补偿"
的 cheating 通道（V1 retrospective Finding 1）。

**Decision**: 单一 KAN 接收 `(h_cond, Z, n, l, κ, occ, ...)` 后输出 `(λ_log_residual, c_0..c_{K-1})`。

**Consequences**:
- ✓ 二者来源同一个网络 → 训练时显式耦合而非隐式补偿
- ✓ 显式吃 `(Z, n, l)` → 不同 n 拿到不同 c（破除 V1 retrospective Finding 3 的几何退化）
- ✓ KAN 边样条可视化 → 可解释 λ(Z, n) 与 c_k(Z, n) 的形状
- ✗ 训练数据少时 KAN 比 MLP 更易崩；备选 MLPCoeffNet 已保留

---

## ADR-003 — NIST 在 Stage 1 绝对不接梯度

**Status**: Accepted (V2.0)

**Context**：
V1 R9 同时让 `w_pde=0.01, w_nist=10.0` 进入梯度。NIST 标量梯度（单数字误差）压制
PDE 张量梯度（per-grid 残差）→ 模型把能量数字调对，把波函数让出去 → cheating。

**Decision**: Stage 1 只跑 `L_PDE + L_ortho + L_node + L_asym + L_smooth`；NIST 仅监控。
Stage 2 才接 NIST，且只通过 `Δ_residual` head（bounded ±50 meV）作用。

**Consequences**:
- ✓ Stage 1 收敛点必然是 PDE 的物理本征解（在跨壳层正交 + 节点约束下）
- ✓ Stage 2 的 Δ_residual 容量 50 meV 不足以伪造能量；最多吃 multiplet 分裂
- ✓ Stage 2 训练时 PDE 仍在梯度里，物理压力不松懈
- ✗ 总训练时长翻倍（两阶段）；可接受

**Alternatives rejected**:
- 联合训练 + 调小 `w_nist`：V1 B16 尝试过（relative-Huber），仍未解决 cheating
- 联合 + curriculum：V1 B14 尝试过，n=2 加进来后 n=1 立刻退化
- 完全不用 NIST：单粒子方程不含多体相关，NIST level 不可能精确复现

---

## ADR-004 — Stage 2 用 `Δ_residual` bounded ±50 meV，不用 `zn_bias_table`

**Status**: Accepted (V2.0)

**Context**：
V1 R10b 引入 `zn_bias_table[Z, n]` 作为可学习偏置 → 训练误差降到 60 meV，
但 LOO（hold out 5s/6s）误差爆到 8.2 eV → 退化为纯 lookup，泛化为零。
V1 retrospective Recommendation 3 明确否决该方案。

**Decision**: Δ_residual 是 `delta_max · tanh(MLP(h_cond, J, π, term))`，
全局 MLP，不基于 (Z, n) lookup。`delta_max = 50 meV`。

**Consequences**:
- ✓ 泛化到 unseen (Z, n)：MLP 在 h_cond 空间外推，比 lookup 强
- ✓ 容量受限：50 meV 上界让模型不能伪造能量
- ✓ 物理合理：term-dependent multiplet splitting 在轻原子量级 ≤ 50 meV
- ✗ 对重原子 multiplet（如过渡金属 d-d 分裂）可能不够 → Phase 2 时 `delta_max → 100 meV`

---

## ADR-005 — Stage 1 用解析对照硬门禁，违反即不能进入 Stage 2

**Status**: Accepted (V2.0)

**Context**：
V1 中"先看 MAE，再事后跑 Phase-A 解析诊断"是反模式——R9 报出 170 meV 后才发现 cheating。
V2 必须把诊断前移到训练循环里。

**Decision**: `check_stage1_gate(model, hydrogenic_test_set)` 函数在 trainer 内每 epoch 调用，
通过条件 3 项 (cos > 0.99, ΔE < 1 meV, |λ−Z/n|/(Z/n) < 5%) 必须全过才允许保存 `stage1_passed.pt`，
Stage 2 训练时同样的 gate 每 epoch 复查；违反触发回滚。

**Consequences**:
- ✓ 任何 cheating 路径会在 gate 上立刻失败
- ✓ 物理对照是绝对客观的（不能被 metrics 包装）
- ✗ 训练循环更慢（每 epoch 多一次 forward + Laguerre 计算）；可以 cache `P_analytic`

---

## ADR-006 — γ 采用低 Z 近似 `|κ|`（V2.0），高 Z 修正延后

**Status**: Provisional (V2.0)

**Context**：
严格 Dirac γ = `sqrt(κ² - (Zα)²)`，但 Z α 在低 Z 区域很小（H: 0.007, Ne: 0.073）。
V1 已经用 `|κ|`，且能精确复现 hydrogenic 1s 能量到 0.1 meV。

**Decision**: Phase 1 沿用 `γ = |κ|`；Phase 2 (Z > 30) 时改为严格公式。

**Consequences**:
- ✓ Phase 1 简化，与 V1 兼容
- ✗ Phase 2 需要切换公式；提前在 `envelope.py` 留 `use_relativistic_gamma` flag

---

## ADR-007 — 不在 V2 第一版做 Hartree-Fock SCF 外环

**Status**: Accepted (V2.0)

**Context**：
多电子原子的精确处理需要 `V_eff = -Z/r + V_H[ρ] + V_xc[ρ]` 自洽 + 交换-相关。
V2 第一版仅做 `V_eff = -Z/r`（裸核 Coulomb），即 hydrogenic 单粒子方程。

**Decision**: V2 第一版 `V_eff = -Z_eff / r`，其中 `Z_eff` 是预估的屏蔽核电荷。
多电子相关效应留给 Stage 2 的 `Δ_residual` head 吸收（虽然这吸收有限）。

**Consequences**:
- ✓ Phase 1 hydrogenic 测试可严格通过
- ✓ Phase 2 NIST 任务可作为 "Δ_residual 学多体修正" 的 benchmark
- ✗ Phase 2 的精度上限 ~ 0.5 eV（单 ζ + 屏蔽 Z_eff 的 HF 极限）；需 V3+ 引入 SCF
- ✗ 重原子 multiplet 残差超 50 meV 会触发回滚 → 提示用户升级到 SCF

---

## ADR-008 — 节点损失用 `tanh(P/σ)` 的差分平滑近似

**Status**: Accepted (V2.0)

**Context**：
节点 = `P(r)` 的过零点。直接 `count_signed_change` 不可微。
解决：用 `s(r) = tanh(P(r) / σ)`，节点数 ≈ `0.5 · Σ |s_{i+1} - s_i|`。

**Decision**: `σ = median(|P|) · 0.01`，自适应；节点数误差用 Huber(δ=0.5)。

**Consequences**:
- ✓ 全程可微
- ✗ σ 选择有 art 成分；过大检测不出节点，过小数值噪声放大
- ✗ 训练初期 P 还没成型，节点损失给的梯度可能 noisy

**Mitigations**: 训练前 10 个 epoch `w_node = 0`，让形状先稳定，再开节点压力。

---

## ADR-009 — 强制 `c_0 = 0`，让 envelope 完全决定 `r → 0` 极限

**Status**: Accepted (V2.0)

**Context**：
B-spline 在 `r_min` 处通过端点节点重复 (p+1) 次，使 `B_k(r_min) = 0` 当 k > 0。
但 `B_0(r_min) = 1`，所以 `f(r_min) = 1 + c_0`，这会破坏 `P(r→0) ~ r^γ`。

**Decision**: 在 `BSplineReadout` 中显式 `c[..., 0] = 0`。

**Consequences**:
- ✓ `P(r_min) = env(r_min)` 严格成立
- ✓ 损失一个自由度，但 c_0 在物理上本来就是 redundant（envelope 已经覆盖）

---

## ADR-010 — 不在 V2 第一版做 Hungarian 匹配 / [B, M] 接口

**Status**: Accepted (V2.0)

**Context**：
V1 retrospective 已说明 `[B, M]` 接口与 NIST 数据语义不匹配。V1.3 切到 `[B]` 标量
levelwise。V2 沿用 levelwise。

**Decision**: V2 第一版仅做 levelwise (`[B]` scalar)。

**Consequences**:
- ✓ 接口简单
- ✓ Hungarian 模块复用 V1 (`utils/hungarian.py`) 仅做 Phase 2 多 term 时备用
- ✗ 同组态 fine-structure 必须靠 Δ_residual 区分 —— 这是已知设计权衡

---

## ADR-011 — Stage 2 默认冻结 KAN 主路径

**Status**: Accepted (V2.0)，可配置 ablation

**Context**：
若 KAN 在 Stage 2 仍以正常 lr 训练，NIST 梯度会绕过 Δ_residual head 直接修改 KAN，
重蹈 V1 R9 覆辙。

**Decision**: 默认 `freeze_kan_main = true`；ablation 用 `lr_kan = 1e-6` 试一下小幅微调。

**Consequences**:
- ✓ Stage 2 训练目标几乎只有 Δ_residual，权重少 → 收敛快、风险低
- ✗ KAN 在 Stage 2 不能再吸收新数据中暴露的物理修正；后续可以二轮训练（"unfreeze + lower lr"）

---

## ADR-012 — `w_nist` 在 Stage 2 用线性 ramp

**Status**: Accepted (V2.0)

**Context**：
Stage 2 起步时直接给 `w_nist = 1.0`，NIST 梯度瞬间淹没 PDE → 第一个 epoch 就违反 gate → 回滚。

**Decision**: `w_nist(t) = w_nist_max · min(1.0, t / t_warmup)`，`t_warmup = 20 epochs`。

**Consequences**:
- ✓ Δ_residual 有 20 个 epoch 缓冲适应
- ✓ 即使违反 gate 触发回滚，`w_nist_max ×= 0.5` 重试，自动找平衡

---

## ADR-013 — 默认 `align_ground = false`（绝对能）

**Status**: Accepted (V2.0)

**Context**：
V1 V1.3 切到 absolute energy 后才解决 NIST 损失为 0 的退化（Step A）。
V2 沿用 absolute 模式。

**Decision**: `losses.align_ground: false`；manifest 写绝对能（Hartree）。

**Consequences**:
- ✓ 模型必须 fit 绝对能量
- ✗ batch 内不同 ion 的能量量级差很大（H 1s -0.5 vs Li 1s -4.5），需要 sampler `group_by_ion`

---

## ADR-014 — 数值统一用 float32；积分权重 float64 预算

**Status**: Accepted (V2.0)，沿用 V1

**Context**：
GPU 上 float32 显著快于 float64；但积分权重在 `r_min ≈ 1e-4` 处需要 float64 精度。

**Decision**: `RadialGrid` 的 `dr, jac` 用 float64 预算后下采到 float32；
所有 forward/backward 用 float32。

**Consequences**:
- ✓ 训练快
- ✗ 极少数情况（高 Z + 高 n）可能在能量量级 1e-6 量级出现精度损失；可配置切换 float64

---

## ADR-015 — 优先 `efficient-kan` 库；备选纯 MLP

**Status**: Accepted (V2.0)

**Context**：
KAN 实现复杂；自写难度大且易出错。`efficient-kan`（Liu 团队官方）已开源。

**Decision**: V2 首先用 `efficient-kan`；同时实现 `MLPCoeffNet` 备选，通过 `cfg.kan.kind` 切换。

**Consequences**:
- ✓ 不重复造轮子
- ✗ 加一个外部依赖；CI 必须支持安装
- ✗ KAN 训练若数值不稳，可切回 MLP 继续

---

## 备忘（未决问题）

| 问题 | 当前选择 | 何时决定 |
|------|---------|---------|
| Phase 2 重原子是否需要 `delta_max > 50 meV` | 50 meV 起步 | Sprint 4 Stage 2 中观察 max\|Δ_res\| |
| `K = 32` 是否够覆盖 n ≤ 10 | 估计够 (`K ≥ n + l`) | Sprint 3 ablation |
| 是否引入 RKB / DKB | 不做（低 Z 不需要） | Phase 3 高 Z 数据时再评估 |
| 是否需要 multi-zeta envelope (多 λ 求和) | 不做（单 envelope + spline 已够） | Phase 2 重原子数据时再评估 |
| KAN 边样条 `grid_size` 上调 | 8 起步 | Sprint 3 ablation |
| Stage 1 训练完后是否合并 c 与 envelope 到一组参数 | 不合并 | – |

---

## 设计文档自检清单

每次修改 V2 架构前，对照下面 5 条：

- [ ] 这次修改会让 NIST 重新进入 Stage 1 梯度吗？→ **不允许**
- [ ] 这次修改会让 Δ_residual 容量超过 50 meV 吗？→ **慎重，更新 ADR-004**
- [ ] 这次修改会改变 `c_0 = 0` 强制吗？→ **更新 ADR-009**
- [ ] 这次修改会让门禁阈值变松吗？→ **不允许**（除非有更严格的替代）
- [ ] 这次修改 V1 retrospective 提到过类似尝试吗？→ **必读 retrospective 再确认**
