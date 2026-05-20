# 03 — KAN HyperNet for `(λ, c)` Prediction

## 1. KAN 速览

Kolmogorov-Arnold Networks (Liu et al., 2024) 把 MLP 的"linear weight + nonlinear activation"
颠倒过来：每条边都是一个 **可学习的 1D 函数**（通常用 B-spline 参数化），
节点只做求和。一个 `L` 层 KAN 的运算 (输入 `x ∈ R^{d_in}` → 输出 `y ∈ R^{d_out}`)：

```
x^{(0)} = x
For l = 1..L:
    x^{(l)}_i = Σ_j φ^{(l)}_{ij}(x^{(l-1)}_j)        i = 1..d_l
                                                       d_l = layer-l 输出维度
                                                       φ^{(l)}_{ij}: R → R (B-spline)
y = x^{(L)}
```

每个 `φ_{ij}` 用 B-spline + 一个可学习的 SiLU 或 sigmoid 残差混合：

```
φ(x) = w_b · silu(x) + w_s · Σ_g c_g · spline_g(x)
```

主要超参数：
- `grid_size`: spline 节点数（每条边 5..20）
- `spline_order`: 通常 3
- `noise_scale`: 初始化扰动
- `grid_range`: spline 定义域

## 2. 在 V2 里 KAN 解决什么

把"离散量子标签 + h_cond 上下文"映射到"每个轨道的 (λ_residual, c_k)"。
形式上：

```
KANCoeffNet:
   inputs:  h_cond [B, D_cond] + per_orb_phys_features [B, N_orb, D_phys]
   outputs: lam_log_residual [B, N_orb]
            c_bspline       [B, N_orb, K]
```

### 2.1 输入特征 (per_orb_phys_features)

| 名称 | 维度 | 描述 |
|------|------|------|
| `Z` | 1 | 原子序数（连续标量）|
| `Z_eff` | 1 | 有效核电荷（用 Slater 屏蔽估算或 `Z - charge - σ_screening`）|
| `n` | 1 | 主量子数 (continuous) |
| `n_star` | 1 | 有效量子数（Slater 经验） |
| `l` | 1 | 轨道角动量 |
| `kappa` | 1 | Dirac 量子数 |
| `occ` | 1 | 该轨道占据数 |
| `is_outer` | 1 | bool ∈ {0,1}，是否最外壳层 |
| **小计** | **D_phys = 8** | |

`Z_eff(per orbital)` 推荐用 **Clementi-Raimondi 表 + 学习残差**：

```
Z_eff_init = Z - Σ_{j: shell < shell_a} σ_{ja} · n_j        # 离线表
KAN 可以学到 σ_{ja} 的修正
```

如果觉得屏蔽规则不写也行，把 `Z_eff` 留给 KAN 自己拟合。

### 2.2 输入特征 (h_cond)

复用 V1 的 `GlobalQuantumEncoder + LevelFeatureEncoder` 输出（256 维），
作为 **batch 全局上下文**，concat 到每个轨道的 per_orb feature 上：

```python
def forward(h_cond, per_orb_features):
    # h_cond [B, D_cond]
    # per_orb_features [B, N_orb, D_phys]
    h_b = h_cond.unsqueeze(1).expand(-1, N_orb, -1)            # [B, N_orb, D_cond]
    x = torch.cat([h_b, per_orb_features], dim=-1)             # [B, N_orb, D_cond + D_phys]
    x = x.reshape(B * N_orb, -1)
    out = self.kan(x)                                          # [B*N_orb, 1 + K]
    out = out.view(B, N_orb, 1 + K)
    lam_log_res = out[..., 0]
    c           = out[..., 1:]
    return lam_log_res, c
```

### 2.3 输出

| 名称 | 维度 | 范围 | 解释 |
|------|------|------|------|
| `lam_log_res` | `[B, N_orb]` | clamp(-2, 2) | `λ = (Z_eff/n) · exp(lam_log_res)` |
| `c` | `[B, N_orb, K]` | unbounded but `L_smooth` 控制 | B-spline 系数 |

末层 zero-init → 初值 `lam_log_res = 0, c = 0`，从解析先验起步。

### 2.4 强制 `c[..., 0] = 0`

如 `02_bspline_basis.md` §2.3 说，端点 `r_min` 处只让 envelope 决定 → 屏蔽 0 号系数：

```python
c_eff = c.clone()
c_eff[..., 0] = 0.0
return lam_log_res, c_eff
```

或者更优雅地，让 KAN 输出 `K` 维，前面 `K-1` 用，第一个常数 0（既省一个参数也禁了 gradient）。

## 3. KAN 实现选择

### 3.1 推荐用现成库：`efficient-kan`

```bash
pip install efficient-kan
```

```python
from efficient_kan import KAN

self.kan = KAN(
    layers_hidden=[D_cond + D_phys, 128, 64, 1 + K],
    grid_size=8,
    spline_order=3,
    scale_noise=0.01,
    scale_base=1.0,
    scale_spline=1.0,
    base_activation=torch.nn.SiLU,
    grid_eps=0.02,
    grid_range=[-1, 1],   # 注意输入需归一化到这区间
)
```

**输入归一化**：KAN 的 `grid_range` 默认 `[-1, 1]`，所以 `(Z, n, ...)` 必须先归一化：

```python
def normalize_per_orb(x):
    # x [B, N_orb, D_phys]
    Z = x[..., 0] / 110.0          # → [0, 1]
    Z_eff = x[..., 1] / 30.0
    n = x[..., 2] / 10.0
    n_star = x[..., 3] / 10.0
    l = x[..., 4] / 7.0
    kappa = x[..., 5] / 10.0
    occ = x[..., 6] / 14.0
    is_outer = x[..., 7]
    return torch.stack([Z, Z_eff, n, n_star, l, kappa, occ, is_outer], dim=-1) * 2 - 1
```

### 3.2 备选：纯 MLP fallback

如果 KAN 训练不稳定（V2 应当对这种风险有预案），可以保留一个等价 API 的 MLP 备选：

```python
class MLPCoeffNet(nn.Module):
    def __init__(self, d_in, d_hidden, d_out_lam, d_out_c):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_hidden), nn.SiLU(),
        )
        self.head_lam = nn.Linear(d_hidden, d_out_lam)
        self.head_c   = nn.Linear(d_hidden, d_out_c)
        nn.init.zeros_(self.head_lam.weight); nn.init.zeros_(self.head_lam.bias)
        nn.init.zeros_(self.head_c.weight);   nn.init.zeros_(self.head_c.bias)
```

通过 `cfg.kan.kind = "kan" | "mlp"` 切换。**Sprint 1 用 MLP 走通整链路，
Sprint 2 把 KAN 接入并对比**。

## 4. 推荐架构超参

```yaml
kan:
  kind: kan        # or mlp
  layers_hidden: [D_cond + 8, 128, 64, 1 + K]   # K = bspline.n_basis
  grid_size: 8
  spline_order: 3
  scale_spline: 0.1
  freeze_main_after_stage1: true                # Stage 2 时 main path 冻结
  lr_factor: 1.0                                # 与 base lr 的相对比例
```

## 5. Stage 2 冻结策略

进入 Stage 2 时：

```python
for p in model.kan.parameters():
    p.requires_grad_(False)
# 只让 LevelResidualHead 学
```

或者用 **超低学习率** (1e-6) 而不是完全冻结，让 KAN 仍能跟随 NIST 残差做微调，
但远远慢于 `Δ_residual`。两种策略都跑一遍 ablation。

## 6. 反作弊监测：KAN 输出健全性

每个 epoch 记录：
- `λ_pred / (Z_eff / n)` 的分布：应在 `0.95..1.05` 之间（Stage 1 后）；
- `||c||_2` 的分布：应不超过 `O(1)`；过大说明 KAN 在补偿物理误差；
- `c_k` 的稀疏度：B-spline 系数应大致集中在 `k ≈ n` 附近；
- Stage 2 时 `c_k` 与 Stage 1 末的差异：应 ≤ 5%。

任意一项偏离 → 把它打入 `tensorboard / logs/kan_health.csv`，触发警报。

## 7. 可视化 (强烈推荐)

KAN 的优势之一是可解释。训练完后画：

1. **`λ(Z, n)` 边样条** — 期望接近 1/n 形状；
2. **`c_k(Z, n)` 边样条** — 期望对于 H/He+/Li2+ 应该接近 hydrogenic Laguerre 系数。

提供 `scripts/visualize_kan.py`：

```python
def plot_edge_spline(kan, layer, edge_in, edge_out, x_range):
    spl = kan.layers[layer]
    xs = torch.linspace(*x_range, 200)
    ys = spl.forward_edge(xs, edge_in, edge_out)
    plt.plot(xs, ys)
```

## 8. 与 V1 `envelope.py` + `wavefunction_readout.py` MLP 的对比

| 维度 | V1 | V2 |
|------|----|----|
| λ 网络 | `MLP(h_cond + kappa_embed + n_embed) → 1`，1 个 MLP | KAN 的输出第 0 维 |
| c 网络 | `MLP(h_cond) → [2 · N_orb · D_res]`，逐 batch 全连接 | KAN 的输出第 1..K 维 |
| 参数共享 | h_cond 给所有轨道；通过 reshape 拆分 | per-orbital concat，KAN 显式吃每个轨道的 (Z, n, l) |
| 物理可解释性 | 黑盒 | 边样条可视化 |
| 物理先验 | λ 通过 zero-init residual MLP；c 通过 `+1` anchor | (λ, c) 同时 zero-init residual |

V2 的最大改动是 **(λ, c) 由同一个 KAN 一次给出**，避免 V1 中 envelope.λ 与 readout.c
在 loss 下相互"补偿对方误差"造成的隐式协同（这一点在 V1 retrospective Finding 1 中
特别提到）。
