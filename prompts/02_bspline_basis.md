# 02 — B-spline Radial Basis

## 1. 数学

### 1.1 节点序列

定义 **node** (knot) 序列 `t_0 ≤ t_1 ≤ ... ≤ t_{m-1}`，其中 `m = K + p + 1`，
`K = n_basis`（B-spline 基底数），`p = spline_order`（默认 cubic, `p = 3`）。

为了让基底在 `r_min..r_max` 上完整定义，端点节点需要 **重复 `p+1` 次**：

```
t_0 = t_1 = ... = t_p = r_min
t_{m-1-p} = ... = t_{m-1} = r_max
内部节点 t_{p+1}, ..., t_{m-2-p}  在 (r_min, r_max) 上以 log-spaced 分布
```

总共 K 个 B-spline 基函数 `B_0(r), ..., B_{K-1}(r)`，满足：
- `B_k(r) ≥ 0`；
- `Σ_k B_k(r) = 1`（partition of unity，仅在内部成立；端点处可能略小）；
- 局部支撑：`B_k(r) ≠ 0` 仅在 `r ∈ [t_k, t_{k+p+1}]`；
- 端点条件：因为重复节点，`B_0(r_min) = 1`，`B_{K-1}(r_max) = 1`，其它在端点 = 0。

### 1.2 Cox-de Boor 递推 (用于 reference impl)

```
B_k^0(r) = 1   if t_k ≤ r < t_{k+1}, else 0

B_k^p(r) = (r - t_k) / (t_{k+p} - t_k) · B_k^{p-1}(r)
         + (t_{k+p+1} - r) / (t_{k+p+1} - t_{k+1}) · B_{k+1}^{p-1}(r)
```

### 1.3 解析导数（关键，用于 PDE）

```
d B_k^p / dr = p · [ B_k^{p-1} / (t_{k+p} - t_k) − B_{k+1}^{p-1} / (t_{k+p+1} - t_{k+1}) ]

d²B_k^p / dr² 用同一递推对 dB^{p-1}/dr 再做一次差分
```

工程上更常用的等价形式：把整个 B-spline 看作 `K + p` 个支撑节点上的逐段 polynomial，
然后用 scipy / 自写 segment-wise 多项式 + 解析微分。**禁止** 使用 `torch.autograd.grad`
对预算好的 buffer 求导（B-spline 是固定基底，与 V1 的 reservoir 等价）。

## 2. 工程实现要求

### 2.1 模块签名

```python
# rc_diracnet_v2/basis/bspline_basis.py

class BSplineBasis(nn.Module):
    """Fixed B-spline basis on (r_min, r_max) with K basis functions.
    
    Buffers (after `precompute(r_grid)`):
        knots   [K + p + 1]
        B       [K, N_grid]    B_k(r_i)
        dB      [K, N_grid]    B'_k(r_i)
        d2B     [K, N_grid]    B''_k(r_i)
    """
    
    def __init__(
        self,
        n_basis: int = 32,
        order: int = 3,
        r_min: float = 1.0e-4,
        r_max: float = 50.0,
        knot_scheme: str = "log",         # {"log", "linear", "loglinear", "physics"}
        boundary_clamp: bool = True,      # True ⇒ 端点重复 p+1 次
    ) -> None:
        ...
    
    @torch.no_grad()
    def precompute(self, r_grid: Tensor) -> None:
        """Compute B [K, N_grid], dB, d2B as float32 buffers."""
        ...
    
    def forward(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return (B, dB, d2B)."""
        if not self._ready:
            raise RuntimeError("call precompute(r_grid) first")
        return self.B, self.dB, self.d2B
```

### 2.2 节点策略

| 方案 | 节点位置 | 适用 |
|------|----------|------|
| `"log"` | 内部 K-2 个节点在 `(r_min, r_max)` 上 log-spaced | 默认；类氢 1s-5s 与 r ≈ 25 同步覆盖 |
| `"linear"` | 内部节点 linear | 仅供 ablation |
| `"loglinear"` | 与 V1 `RadialGrid.scheme="loglinear"` 一致 | 与 grid 一致时数值最稳 |
| `"physics"` | 节点按 `r = n²/(2Z)` 经验布点（核 → 价 → Rydberg 三段） | 数据驱动；推荐 K ≥ 24 |

**默认起步用 `"log"` + `K = 32` + `order = 3`**；后续 ablation 再调。

### 2.3 端点行为

为了让 B-spline factor `f(r) = 1 + Σ c_k B_k(r)` 在 `r → 0` 不破坏 `r^γ` 的解析行为：

```
约束：B_k(r_min) = 0   for k > 0
     B_0(r_min) ≈ 1
```

通过 **端点节点重复 `p+1` 次** 自动满足上述条件。注意这意味着 `f(r_min) = 1 + c_0`；
如果想让 envelope 完全主导起点，可以额外约束 `c_0 = 0` (mask 第 0 个系数)：

```python
# 在 readout 中：
c[..., 0] = 0   # forced zero, no gradient
```

推荐 **强制 `c_0 = 0`**，让 envelope 100% 决定 `r → 0` 的极限行为。

### 2.4 解析二阶导数测试

单元测试必须包含数值对照：

```python
def test_bspline_derivative_consistency():
    basis = BSplineBasis(n_basis=8, order=3)
    r = torch.linspace(0.01, 10.0, 200)
    basis.precompute(r)
    B, dB, d2B = basis()
    # 用中心差分对照
    dB_fd = (B[:, 2:] - B[:, :-2]) / (r[2:] - r[:-2])
    assert torch.allclose(dB[:, 1:-1], dB_fd, atol=1e-3)
    # 二阶同理
```

## 3. 实现要点

### 3.1 推荐路径：直接用 scipy 离线预算 + Tensor 包装

```python
import numpy as np
from scipy.interpolate import BSpline

def _build_bspline_eval(knots, order, r):
    K = len(knots) - order - 1
    B = np.zeros((K, len(r)))
    dB = np.zeros_like(B)
    d2B = np.zeros_like(B)
    for k in range(K):
        coef = np.zeros(K); coef[k] = 1.0
        spl = BSpline(knots, coef, order, extrapolate=False)
        B[k]   = spl(r,   nu=0)
        dB[k]  = spl(r,   nu=1)
        d2B[k] = spl(r,   nu=2)
    return B, dB, d2B
```

然后在 `__init__` 里 build knots → `precompute(r_grid)` 调上面函数，转 tensor 并 register_buffer。

### 3.2 完全用 PyTorch 写（可选）

如果不想引入 scipy，写一份纯 PyTorch 的 Cox-de Boor：

```python
def cox_de_boor(r, knots, order):
    """Return [K, N] tensor of B_k^order(r)."""
    K = len(knots) - order - 1
    # build B^0
    B = torch.zeros(K + order, len(r))
    for k in range(K + order):
        mask = (knots[k] <= r) & (r < knots[k+1])
        B[k] = mask.float()
    # recursion
    for p in range(1, order + 1):
        Bp = torch.zeros(K + order - p, len(r))
        for k in range(K + order - p):
            left  = (r - knots[k]) / (knots[k+p] - knots[k] + 1e-12) * B[k]
            right = (knots[k+p+1] - r) / (knots[k+p+1] - knots[k+1] + 1e-12) * B[k+1]
            Bp[k] = left + right
        B = Bp
    return B   # [K, N]
```

但 scipy 版稳定且可对照 SciPy docstring 验证，建议起步直接用 scipy。

### 3.3 显式不可微

`BSplineBasis` 内部不持有任何 `nn.Parameter`；所有节点位置 / 阶数都是常量。
**B-spline 本身的"可学习性"完全经由 readout 的 `c_k` 系数实现**（即 KAN 的输出）。

## 4. 推荐配置（写进 `default.yaml`）

```yaml
bspline:
  n_basis: 32
  order: 3
  knot_scheme: log
  r_min: 1.0e-4              # 必须 ≥ grid.r_min
  r_max: 50.0                # 必须 ≤ grid.r_max
  force_c0_zero: true        # f(r_min) = 1 (env-only at origin)
```

## 5. 与 V1 `reservoir/basis_generator.py` 的关系

| V1 | V2 |
|----|----|
| `RCBasisGenerator(d_res=512)` 固定随机 `(a, b)` | `BSplineBasis(n_basis=32)` 固定确定性节点 |
| 7 种 `f(r) ∈ {r, log r, 1/r, sqrt r, r², r³, r⁴}` 通过 `tanh` 复合 | 局部 piecewise polynomial，自然带 (K-1) 阶导数 |
| 512 维但表达力受限 | 32 维但完备（在数值意义下） |
| precompute → buffer | precompute → buffer（同接口） |
| `phi, dphi, d2phi` | `B, dB, d2B`（同形状 `[K, N_grid]`） |

⇒ **API 完全可以 mirror V1 的 `RCBasisGenerator`，只是基底定义变了**。
V1 中调用方 `WavefunctionReadout(... phi, dphi, ...)` 的写法可以保留，
只把传入张量从 `phi` 换成 `B` 即可。这意味着 `wavefunction_readout.py` 的
核心 einsum 结构（`einsum("bok,kn->bon", W, phi)`）也是可复用的。

## 6. 数值健全性建议

- `r_min < knots[0] + 1e-6`：保证 r_min 落在第一个 B-spline 支撑内；
- `r_max > knots[-1] - 1e-6`：同上；
- 内部节点最小间距 `≥ 1e-3`：防止 `(t_{k+p} - t_k) → 0`；
- 训练前打印 `B.sum(dim=0)`：应在内部点接近 1，端点处略小是允许的。
