# DiracNet V2 — `envelope × B-spline + KAN(λ, c)`

> 与 `DiracNet_V1/rc_diracnet_project/` 并列存放；V2 是 **新代号**，不在
> V1 仓库内增量；V1 保持现状以便回滚比对。

## 0. 一句话总览

V2 把"波函数读出"从 V1 的 **随机非线性特征 (RC tanh) + 多项式** 替换为
**显式物理基底**：
```
P_a(r) = env_a(r) · ( 1 + Σ_{k=1}^{K} c_{a,k} · B_k(r) )
env_a(r) = r^{γ_a} · exp(-λ_a · r)
Q_a(r) = (1/(2c)) · ( dP_a/dr + κ_a · P_a / r )      ← kinetic balance
```
其中 `(λ_a, c_a)` 由一个 **KAN hyper-network** 从离散量子标签 `(Z, n, l, κ, occ, J, π, term)`
直接预测，B-spline 基底 `{B_k(r)}` 在径向网格上一次性预算。

训练分两个独立阶段：

| 阶段 | 梯度来源 | NIST 作用 | 目的 |
|------|----------|-----------|------|
| **Stage 1 (PDE-only)** | `L_PDE + L_ortho + L_node + L_asym` | **仅作为评估指标**，不进入 loss | 让波函数与单电子能级 `E_orb` 收敛到真物理本征解（避免 V1 中"NIST 与 PDE 同步训练"暴露出的 cheating 链路） |
| **Stage 2 (NIST residual fine-tune)** | `L_PDE + λ_nist · L_NIST_residual` | 仅对 **窄带 Δ_residual head** 提供梯度；主网络 (λ, c-KAN) 冻结或低 lr | 学习"多体相关 / 多重态分裂"等单粒子方程无法覆盖的残差 |

## 1. 与 V1 的差异点

| 维度 | V1 (B9 / B10) | V2 |
|------|---------------|----|
| 径向基底 | RC `tanh(a·f(r)+b)` 固定随机 → 多项式 `Σ c_k r^k` (B10) | **B-spline** `Σ c_k B_k(r)`，可调阶数与节点 |
| (λ, c) 来源 | MLP(`h_cond`) 出 (λ, c) | **KAN** 直接吃量子数 + `h_cond`，单调/分段更易刻画 |
| h_cond 角色 | 不可或缺，承担"原子身份"和"shape" | 仅作为辅助上下文，可被 KAN 主路径吸收 |
| NIST 训练时角色 | 与 PDE 同步参与 loss，PDE 权重小（R9 cheating） | **Stage 1 隔离**；Stage 2 仅作残差 |
| Δ_term | `delta_scale · tanh(MLP)` 全程在线，吸收偏置 | Stage 1 **不存在**；Stage 2 才引入，强约束 ±50 meV |
| 反作弊 | 事后通过 `diagnose_analytic_levels.py` 体检 | **训练时门禁**：Stage 1 结束必须通过解析 `|cos|>0.99` & `|ΔE|<1 meV` 才能进入 Stage 2 |

## 2. 文档导航

| 文件 | 内容 |
|------|------|
| `prompts/00_overview.md` | 设计哲学、V1 的三条教训、为什么是 KAN+B-spline |
| `prompts/01_architecture.md` | 端到端前向流程、张量形状约定 |
| `prompts/02_bspline_basis.md` | B-spline 数学公式 + 工程实现（含解析 1/2 阶导） |
| `prompts/03_kan_hypernet.md` | KAN 结构、输入特征、(λ, c) 输出头 |
| `prompts/04_envelope_kinetic.md` | env 解析形 + kinetic balance 接入 |
| `prompts/05_physics_losses.md` | PDE / Ortho / Node / Asymptotic 四类 loss |
| `prompts/06_two_stage_training.md` | Stage 1/Stage 2 训练协议（**重点**） |
| `prompts/07_evaluation.md` | 解析对照 + 反作弊门禁 |
| `prompts/08_data_pipeline.md` | hydrogenic / NIST manifest 准备 |
| `prompts/09_project_layout.md` | 目录树 + 文件清单 |
| `prompts/10_sprint_plan.md` | 4 个 Sprint 的里程碑与验收 |
| `prompts/11_test_plan.md` | 必备的单元/集成测试 |
| `prompts/12_port_from_v1.md` | 可直接复用的 V1 模块 + 改写表 |
| `docs/design_rationale.md` | 每一条关键设计决策的"为什么这样" |

## 3. 立即开始

1. 阅读 `prompts/00_overview.md` 与 `docs/design_rationale.md`，理解 V1 → V2 的设计跃迁动机；
2. 按 `prompts/09_project_layout.md` 建立 `rc_diracnet_v2/` 包；
3. 按 `prompts/12_port_from_v1.md` 把可复用模块（`grid`, `dirac_operator`, `quantum_encoder`, `lowdin_orthonormalize`,
   `nist_loader`, `level_encoder`, 等）拷贝到 V2；
4. 按 `prompts/10_sprint_plan.md` 的顺序实现 B-spline → KAN → readout → losses → trainer；
5. 在每个 Sprint 末尾跑 `prompts/11_test_plan.md` 中的对应单元测试；
6. **不要跳过 Stage 1 验收门禁**（见 `07_evaluation.md`），否则 Stage 2 的"NIST 微调"会重新滑入 V1 的 cheating 链路。

## 4. 期望成果（与 V1 R9 / B16 / B9-LOO 对比）

| 指标 | V1 R9 | V1 B16 | V1 B9-LOO | **V2 Stage 1** | **V2 Stage 2** |
|------|-------|--------|-----------|----------------|----------------|
| H 1s shape `|cos|` | 0.996 | 0.703 | – | **> 0.999** | > 0.999 |
| H 5s shape `|cos|` | 0.451 | 0.733 | – | **> 0.97** | > 0.97 |
| 全 15 行 NIST RMS (eV) | 0.17* (cheat) | 2.35 | LOO 高 | **< 0.1** | **< 0.005** |
| LOO（hold-out 5s）RMS (eV) | – | – | high | **< 0.2** | < 0.05 |
| 解析 λ 误差 (`|λ_pred−Z/n| / (Z/n)`) | 70% drift | 30% drift | – | **< 5%** | < 5% |

\* V1 R9 的 0.17 eV 是"假赢"。V2 的同等指标必须伴随物理门禁通过才计有效。

## 5. 风险与缓解

| 风险 | 缓解 |
|------|------|
| B-spline 在 r→0 处不严格满足 `P ~ r^γ` | env 已经显式给出 `r^γ`；spline factor 强制 `B_k(r→0)=0` 通过端点节点设计 |
| KAN 训练不稳定 | (a) 全部 spline 节点使用相同 grid；(b) 残差初始化（λ MLP 末层 zero-init）；(c) 学习率分组（KAN-main < KAN-edge < bias） |
| Stage 1 PDE-only 出现 n-collapse（Rayleigh 收敛到基态） | (a) 跨壳层硬正交（Löwdin 完整模式）；(b) 节点计数惩罚 `L_node`；(c) λ 初始锁定 `Z_eff/n`；(d) KAN 显式吃 n → 不同 n 拿到不同 c |
| Stage 2 重新滑入 cheating | (a) 主网络 lr=0 或 1e-6；(b) Δ_residual bounded ±50 meV；(c) 每 epoch 复跑 `|cos| > 0.99` 门禁，违反即回滚 |
| B-spline 自由度过多 | K=32-64；可调；通过 `L_smooth` (二阶差分) 软约束 |

## 6. 备注

- V1 中的 `physics/`, `utils/grid.py`, `losses/pde_loss.py`, `losses/orthonormality_loss.py`, `losses/nist_scalar_loss.py`,
  `data/`, `encoders/quantum_encoder.py`, `data/level_encoder.py`, `readout/orthogonalizer.py` 大多可以原样复制到 V2，
  详见 `prompts/12_port_from_v1.md`。
- V1 中的 `reservoir/basis_generator.py`, `reservoir/envelope.py`, `readout/wavefunction_readout.py`,
  `models/rc_diracnet_levelwise.py` 是要重写的核心。
- 本提示词集合 **不会自己生成 Python 代码**；它的目标是把"如何实现 V2"完整地写下来，
  让任意工程师 / LLM-Agent 拿到后能逐文件落地。
