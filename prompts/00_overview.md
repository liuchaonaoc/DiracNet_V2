# 00 — Overview / V2 设计哲学

> 阅读优先级：**最高**。任何动手写代码的人都应先读完本文 + `docs/design_rationale.md`，
> 再去看 §01 起的工程化细节。

## 1. 我们要解的问题

> 给定原子 / 离子的离散量子标签 `(Z, ion_charge, parent_config, level_config, J, parity, term)`，
> 输出每个活跃轨道 `a` 的相对论径向旋量 `(P_a(r), Q_a(r))`，从而：
>
> 1. **物理上**：满足 Dirac 单粒子方程 `H_D ψ_a = E_orb_a ψ_a`，跨壳层正交归一；
> 2. **数据上**：标量预测 `E_pred = Σ_a occ_a · E_orb_a (+ Δ_residual)` 与 NIST `level_eV` 吻合。

V1 在 Phase 1（类氢沙盒）上拿到 170 meV MAE，但 **Phase-A 解析诊断证实是 "cheating"**
（详见 `DiracNet_V1/rc_diracnet_project/docs/V1_3_sprint1_retrospective.md`）：

- `λ` 漂离物理先验 `Z/n` 三倍以上；
- RC `tanh` 基底无法张成 Laguerre L_{n-1}^1(2λr)，所以 `n ≥ 3` 的形状余弦 < 0.1；
- Löwdin 归一化 + 有界 `Δ_term (≤0.1 Ha)` 联手吃掉数百 meV 的能量误差。

## 2. V2 的四条核心立场

### 2.1 假设类 (hypothesis class) 必须 **结构性包含真解**

类氢解析解：
```
P_{n,l}^{exact}(r) ∝ r^{l+1} · exp(-λ_n r) · L_{n-l-1}^{2l+1}(2λ_n r)
λ_n = Z / n   (NR)   λ_n = Z α / sqrt(n² - (Zα)²)  (Dirac)
```

V1 的假设类（`tanh(a·f(r)+b)` 的有限随机线性组合）**结构上不包含** Laguerre 多项式；
即使配 PDE 损失也只能在投影意义下逼近，而投影系数依赖 random `(a, b)`，
高 n 的形状余弦稳定在 0 附近。

V2 用 **envelope × B-spline**：
- `r^γ · exp(-λr)` 已经显式包含 1s/2s 的全部"指数+幂律"主导项；
- B-spline 在 `(r_min, r_max)` 上的有限元基底，可以以可控误差逼近 Laguerre 多项式
  （4 阶 cubic spline + K ≥ 16 节点足以让 `|cos|` 收敛到 1）；
- 把 `(λ, c_k)` 留给 KAN 学习。

### 2.2 训练时 **不允许 NIST 与 PDE 形成"耦合作弊"**

V1 的 R9 同时让 `PDE (w=0.01)` 与 `NIST (w=10)` 进入梯度。后者占绝对主导，
迫使网络 "把能量调到 NIST，把波函数留给一堆与物理无关的形变"。配合
`Löwdin renorm + Δ_term` 这条暗道，模型可以在波函数错的离谱下还报出小 MAE。

V2 的严格做法：

```
Stage 1:  loss = w_pde · L_PDE + w_ortho · L_ortho + w_node · L_node
                + w_asym · L_asym + w_action · L_action_BS
          # NIST 只在 metric log 里出现，不连 backward
          # L_action_BS 是 Bohr-Sommerfeld 作用量量子化约束，用于防 n-collapse
          
Stage 2:  可选 calibration，不是主结果
          主报告必须先给 E_orb-only；Δ_residual 只能作为受限残差实验
          # 主网络 (λ, c-KAN) 冻结；Δ_residual 有严格幅度、占比、OOD 门禁
```

### 2.3 评估必须 **比对解析解 / 跨数据集泛化**

Stage 1 完成后必须执行三项硬门禁：

1. **解析对照**：在 H/He⁺/Li²⁺ 的 1s..5s 上计算 `|cos(P_model, P_analytic)|`，要求 `> 0.99` for all (Z, n)；
2. **λ 一致性**：KAN 预测的 `λ_a` 满足 `|λ_pred - Z/n| / (Z/n) < 5%`；
3. **能量一致性**：`|E_orb_model - (-Z²/2n²)| < 1 meV`。

任意一项不过，不许进入 Stage 2。

### 2.4 `E_orb-only` 是主结果，`Δ_residual` 只是受限校准

V2 默认主基准是：

```
E_pred_main = Σ_a occ_a · E_orb_a
```

`Δ_residual` **不属于主物理模型**，只能作为 Stage 2 calibration experiment：

```
E_pred_calibrated = E_pred_main + Δ_residual
```

报告任何 calibrated 指标时，必须同时报告：

- `E_orb-only` 的 raw/aligned MAE/RMS；
- `E_orb + Δ_residual` 的 raw/aligned MAE/RMS；
- `max |Δ_residual|` 与 `median |Δ_residual|`；
- `|Δ_residual| / |E_orb - E_target|` 的占比；
- leave-one-n / leave-one-Z / leave-one-ion 的 OOD 结果；
- Stage 2 后的 `|cos|`, `λ drift`, `L_PDE`, `L_action_BS` 门禁。

如果 `Δ_residual` 改善能量但 `E_orb-only` 仍错、波函数门禁退化，结果判为 cheating。

## 3. 为什么是 B-spline，不是别的

| 候选 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| Random tanh (V1) | 表达力理论无上限 | 有限维下不张成 Laguerre；与物理基底正交性差 | ❌ |
| Monomial `r^k` (V1 B10) | 包含 Laguerre 多项式 | 高 k 病态，需手工裁剪；与 `e^{-λr}` 耦合时数值爆炸 | ❌ |
| Laguerre `L_n^l(2λr)` | 精确解析（类氢） | 多电子下不再是 Laguerre；需多 ζ 才能逼近 | △ |
| Sturmian `S_{nl}(αr)` | 多电子原子结构计算的现代标准 | 实现复杂；非线性 α | △ |
| **B-spline** | 局部支撑 → 数值稳定；任意函数光滑逼近；阶/节点自由 | 不是"物理基底"，需要 envelope 提供长尾衰减 | ✅ |
| 纯网格点值 (DVR/FE) | 完全无参 | 自由度 = N_grid（几百~几千），泛化差 | ❌ |

**结论**：B-spline 是"工程稳定 + 数学完备 + 自由度可控"的最优折衷。

## 4. 为什么是 KAN，不是 MLP

KAN (Kolmogorov-Arnold Networks, 2024) 把 MLP 的线性权重替换成 **每条边的可学习
1D 样条函数**：
```
MLP: y = W · σ(x)
KAN: y = Σ_i φ_i(x_i)        其中 φ_i 是 B-spline
```

对 V2 的关键好处：

1. **物理依赖通常是 1D 的**：`λ ≈ Z/n`、`c_k` 与 `Z` 的关系都是 (近似) 单调或分段，
   MLP 的 `σ(W·x+b)` 把它扭成隐空间，可解释性差；KAN 边样条直接读出 `λ(Z, n)` 形状。
2. **样本少时鲁棒**：V2 Phase 1 数据只有 ~15 行；MLP 容易过拟合到无意义的形状，
   KAN 的稀疏 1D 表示在低数据下泛化更好（已有多篇 2024-2025 实验）。
3. **可解释性**：训练完后我们能 print 出 `λ vs Z` 的边样条曲线，验证是否合理。

KAN 不是必需的——如果工程觉得 KAN 不稳定，**保留备选**：用 `MLP + monotonic projection` 替代（详见 §03）。

## 5. 工程化原则

1. **接口隔离**：B-spline / KAN / readout / loss 各自独立模块，可单独单测；
2. **数值稳定优先**：所有解析导数显式写出（避免 autograd 链断；V1 已建立的纪律）；
3. **物理先验保底**：所有 hyper-net 的"残差头"末层 zero-init，初值即解析先验；
4. **门禁前置**：每个 Sprint 必须先过单元测试 + 解析对照，再开始 Long-run；
5. **复用 V1**：所有与本次架构调整无关的模块（grid、Dirac 算子、quantum encoder、
   level encoder、NIST loader、ortho 损失、scalar Huber 损失）**直接复制**，不重写。

## 6. 与 V1 retrospective 的对应

| V1 retro 中的 Finding | V2 的应对 |
|----------------------|-----------|
| Finding 1: "cheating channels"（λ 漂、shape 错、Δ_term 兜底） | Stage 1 锁 NIST 梯度 + Stage 2 主网络 freeze + λ 门禁 |
| Finding 2: 多项式 readout 修复假设类 | B-spline 是多项式的"工程化升级"，局部支撑更稳 |
| Finding 3: 联合优化的几何退化 | KAN 接受 (Z, n, l) 显式输入，不再依赖联合 MLP 自己分辨 |
| Recommendation 1: 永远先跑解析对照 | 写入 `07_evaluation.md` 的 Stage 1 验收门禁 |
| Recommendation 2: 加 Z=4..10 数据破除退化 | 在 `08_data_pipeline.md` 中写入 Phase 1+ manifest |
| Recommendation 3: 不再用 `zn_bias_table` | V2 主结果只看 `E_orb-only`；`Δ_residual` 仅作为受限校准实验，不能当主结果 |
| Recommendation 4: 默认 polynomial / 替换 RC | 直接弃用 RC，仅 B-spline |
| Recommendation 5: 高 n 用 Laguerre | B-spline + 节点对数分布等价的覆盖能力 |

## 6.1 与 MIT 作用量论文的取舍

MIT 2026 的 *On computing quantum waves exactly from classical action* 说明可以从 classical action + density
严格重构 Schrödinger 波。在 V2 的**束缚定态径向问题**中，完整 Hamilton-Jacobi + density 框架会退化成
与 Schrödinger/Dirac PDE 等价的局部条件，因此不应把“作用量本身”重复加入 loss。

V2 只吸收其中对本任务有新增信息的一部分：**Bohr-Sommerfeld 作用量量子化**

```
∫_{r_-}^{r_+} p_r(r) dr = π · (n - l - 1/2)
```

它是全局、可微、携带整数 n 信息的 scalar constraint，可与 `L_node` 一起防止 PDE-only 阶段的 n-collapse。

## 7. 不在 V2 第一版考虑的事项

为了避免战线过长，**以下都放到 V3+**：

- 多电子 Hartree-Fock SCF 外环（V2 仍是单粒子方程 + occ 加权 Koopmans 求和）
- Slater 积分 / CSF 装配 / Racah 代数（即 V1 的"方案 C"）
- DKB (Dual Kinetic Balance) 高 Z 修正
- Breit / QED 残差
- 周期 / 分子拓展
- 时间含 Hamilton-Jacobi + density 多路径传播（MIT 论文完整威力，留到 V3 的光跃迁 / 隧穿 / continuum）

V2 的目标：**在原子谱学的单粒子任务上，把假设类、训练动力学、评估门禁三件事一次性做对。**
