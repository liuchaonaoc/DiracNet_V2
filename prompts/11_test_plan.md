# 11 — Test Plan

> 纪律：**每个 Sprint 结束前，对应单元/集成测试必须 ALL GREEN**。
> 不允许"等下个 Sprint 一起补测试"。

## 1. 测试分层

| 层级 | 范围 | 工具 | 频率 |
|------|------|------|------|
| L0 单元 | 一个文件 / 一个类 | pytest | 每次提交 |
| L1 模块 | 多文件协作（如 envelope + readout） | pytest | 每次提交 |
| L2 集成 | 完整 forward / loss / trainer | pytest | Sprint 结尾 |
| L3 端到端 | 小数据训练 + 评估 | pytest + 真训练 | Sprint 结尾 |
| L4 解析对照 | 物理正确性 | gate.py | 训练中每 epoch |

## 2. L0 - L1 单元测试

### 2.1 `test_v2_bspline_basis.py`

```python
def test_bspline_partition_of_unity_interior():
    """Σ B_k(r) ≈ 1 在内部 r 上"""
    basis = BSplineBasis(n_basis=32, order=3, r_min=1e-4, r_max=50.0)
    r = torch.linspace(1.0, 30.0, 200)
    basis.precompute(r)
    B, _, _ = basis()
    s = B.sum(dim=0)
    assert (s - 1.0).abs().max() < 1e-3, "B-spline POU violated"

def test_bspline_derivative_consistency():
    """解析导数 vs 数值导数"""
    basis = BSplineBasis(n_basis=16)
    r = torch.linspace(0.1, 20.0, 500)
    basis.precompute(r)
    B, dB, d2B = basis()
    dB_fd = (B[:, 2:] - B[:, :-2]) / (r[2:] - r[:-2])
    assert torch.allclose(dB[:, 1:-1], dB_fd, atol=1e-2, rtol=1e-2)
    d2B_fd = (dB[:, 2:] - dB[:, :-2]) / (r[2:] - r[:-2])
    assert torch.allclose(d2B[:, 1:-1], d2B_fd, atol=1e-1, rtol=1e-1)

def test_bspline_endpoint_zero():
    """B_k(r_min) = 0 for k > 0  (forces env to dominate at origin)"""
    basis = BSplineBasis(n_basis=32, order=3, boundary_clamp=True)
    r = torch.tensor([1e-4, 50.0])
    basis.precompute(r)
    B, _, _ = basis()
    assert B[1:, 0].abs().max() < 1e-6
    assert B[:-1, -1].abs().max() < 1e-6
```

### 2.2 `test_v2_envelope.py`

```python
def test_envelope_h1s_analytic():
    """λ=1, κ=-1, Z=1 → env(r) = r · exp(-r) exactly"""
    env = AnalyticEnvelope(alpha=ALPHA)
    r = torch.linspace(0.01, 20.0, 200)
    lam = torch.tensor([[1.0]])   # [B=1, N_orb=1]
    kappa = torch.tensor([[-1]])
    Z = torch.tensor([1])
    e, de, d2e = env(r, lam, kappa, Z)
    expected = r * torch.exp(-r)
    assert torch.allclose(e[0, 0], expected, atol=1e-4)

def test_envelope_lam_scaling():
    """λ=2 → e(r) = r · exp(-2r)"""
    ...
```

### 2.3 `test_v2_bspline_readout.py`

```python
def test_readout_init_equals_env():
    """c=0 时 P == env"""
    readout = BSplineReadout()
    K = 16
    B_, N_orb, N = 2, 3, 200
    c = torch.zeros(B_, N_orb, K)
    env = torch.randn(B_, N_orb, N)
    denv = torch.randn_like(env)
    d2env = torch.randn_like(env)
    B_b = torch.randn(K, N); dB_b = torch.randn_like(B_b); d2B_b = torch.randn_like(B_b)
    kappa = torch.tensor([[-1, -2, -3]] * B_)
    r = torch.linspace(0.1, 10.0, N)
    mask = torch.ones(B_, N_orb, dtype=torch.bool)
    
    out = readout(c, env, denv, d2env, B_b, dB_b, d2B_b, kappa, r, mask)
    assert torch.allclose(out["P"], env)

def test_readout_force_c0_zero():
    """c_0 即使非零也被强制为 0 (when force_c0_zero=True)"""
    ...

def test_readout_kinetic_balance_h1s():
    """Analytic H 1s spinor → Q ≈ (Zα/2) P (Pauli limit)"""
    ...
```

### 2.4 `test_v2_kan_coeff_net.py`

```python
def test_mlp_coeff_zero_init():
    """末层 zero-init → 输出全 0"""
    net = MLPCoeffNet(d_in=264, d_hidden=128, d_out_lam=1, d_out_c=32)
    x = torch.randn(4, 16, 264)
    lam_log_res, c = net(x)
    assert lam_log_res.abs().max() < 1e-6
    assert c.abs().max() < 1e-6

def test_kan_coeff_normalize_inputs():
    """输入归一化在 [-1, 1] 区间"""
    ...

def test_kan_coeff_gradient_flow():
    """backward 不 NaN"""
    ...
```

### 2.5 `test_v2_node_count_loss.py`

```python
def test_node_count_zero_nodes():
    """P = exp(-r) (no nodes) → n_pred ≈ 0"""
    r = torch.linspace(0.1, 10.0, 200)
    P = torch.exp(-r).view(1, 1, -1)
    mask = torch.ones(1, 1, dtype=torch.bool)
    loss = NodeCountLoss()
    n_required = torch.tensor([[0]])
    L = loss(P, n_required, mask)
    assert L < 0.1

def test_node_count_one_node():
    """P = (1 - r) * exp(-r) (1 node at r=1) → n_pred ≈ 1"""
    r = torch.linspace(0.1, 10.0, 200)
    P = ((1.0 - r) * torch.exp(-r)).view(1, 1, -1)
    mask = torch.ones(1, 1, dtype=torch.bool)
    loss = NodeCountLoss()
    n_required = torch.tensor([[1]])
    L = loss(P, n_required, mask)
    assert L < 0.1

def test_node_count_penalty():
    """P 是 1s 但 n_required=2 → 大 loss"""
    ...
```

### 2.6 `test_v2_asymptotic_loss.py`

```python
def test_asymptotic_zero_on_decaying():
    """P(r) = r·exp(-r) → L_asym ≈ 0"""
    r = torch.linspace(0.1, 50.0, 500)
    P = (r * torch.exp(-r)).view(1, 1, -1)
    mask = torch.ones(1, 1, dtype=torch.bool)
    loss = AsymptoticTailLoss()
    L = loss(P, r, mask)
    assert L < 1e-3

def test_asymptotic_high_on_non_decaying():
    """P(r) ≡ 1 → 大 loss"""
    ...
```

### 2.7 `test_v2_smooth_loss.py`

```python
def test_smooth_zero_on_constant():
    """c_k = c → Σ d² = 0"""
    c = torch.ones(2, 3, 32) * 0.5
    mask = torch.ones(2, 3, dtype=torch.bool)
    L = BSplineSmoothLoss()(c, mask)
    assert L < 1e-10

def test_smooth_high_on_oscillating():
    """c_k = (-1)^k → 高 loss"""
    ...
```

### 2.8 `test_v2_pde_loss.py`（复用 V1）

```python
def test_pde_loss_h1s_analytic():
    """Analytic H 1s → L_PDE < 1e-3"""
    # 复用 V1 已有测试 `tests/test_phase1_orbital_energy.py`
    ...
```

### 2.9 `test_v2_per_orb_features.py`

```python
def test_per_orb_features_h1s():
    """Z=1, config='1s1' → features (1, ~1, 1, 1, 0, -1, 1, 1)"""
    batch = {
        "Z": torch.tensor([1]),
        "config_shells": torch.tensor([[[1, 0, 1, 1]]]).long(),   # (n=1, l=0, 2j=1, occ=1)
        "orb_mask": torch.tensor([[True]]),
    }
    feat = build_per_orbital_features(batch, n_orb_max=1)
    assert feat.shape == (1, 1, 8)
    assert feat[0, 0, 0] == 1.0   # Z
    assert feat[0, 0, 2] == 1.0   # n
    assert feat[0, 0, 4] == 0.0   # l
    assert feat[0, 0, 5] == -1.0  # kappa
    assert feat[0, 0, 6] == 1.0   # occ
    assert feat[0, 0, 7] == 1.0   # is_outer
```

## 3. L2 集成测试

### 3.1 `test_v2_dirac_net_v2_forward.py`

```python
def test_full_forward_pass():
    """完整 forward 无 NaN，输出形状正确"""
    cfg = load_test_config()
    model = DiracNetV2(cfg)
    batch = make_dummy_batch(cfg)
    out = model(batch)
    assert out["E_orb"].shape == (cfg.batch_size, cfg.readout.n_orb_max)
    assert torch.isfinite(out["P"]).all()
    assert torch.isfinite(out["E_orb"]).all()

def test_forward_with_kan_zero_init_equals_env():
    """KAN/MLP zero-init → P == env"""
    model = DiracNetV2(cfg)
    batch = make_dummy_batch(cfg)
    out = model(batch)
    # 因为 c=0, c_0=0 强制, 所以 P 应该接近 r^γ * exp(-(Z/n) r)
    ...
```

### 3.2 `test_v2_gate_pass_on_analytic.py`

```python
def test_gate_pass_when_model_outputs_analytic_solution():
    """手工构造一个 model 输出解析 P → gate PASS"""
    class AnalyticDiracNet(nn.Module):
        def forward(self, batch):
            ...
            return {
                "P": analytic_P_from_batch(batch),
                "Q": kinetic_balance(P),
                "E_orb": batch_analytic_energies,
                "lam": batch["Z"] / batch["n"]
            }
    model = AnalyticDiracNet()
    test_loader = make_hydrogenic_loader()
    assert check_stage1_gate(model, test_loader, thresholds=DEFAULT_THRESHOLDS)
```

### 3.3 `test_v2_gate_fail_on_random.py`

```python
def test_gate_fail_on_random_init():
    """随机初始化网络（c 非零）门禁应 FAIL"""
    cfg = load_test_config()
    model = DiracNetV2(cfg)
    # 给 c 加随机扰动
    for p in model.kan.parameters():
        p.data += torch.randn_like(p.data) * 0.5
    assert not check_stage1_gate(model, hydrogenic_loader, DEFAULT_THRESHOLDS)
```

### 3.4 `test_v2_two_stage_trainer_smoke.py`

```python
def test_stage1_smoke():
    """3 epoch Stage 1，验证 L_PDE 单调下降，不 NaN"""
    cfg = make_smoke_config(n_epochs=3)
    trainer = TwoStageTrainer(...)
    history = trainer.run_stage1(...)
    assert all(torch.isfinite(torch.tensor(h["L_PDE"])) for h in history)
    assert history[-1]["L_PDE"] < history[0]["L_PDE"]

def test_stage2_rollback():
    """模拟 NIST loss 把模型推到违反 gate → rollback 触发"""
    ...
```

## 4. L3 端到端测试

### 4.1 `test_v2_full_smoke.py`

```python
def test_e2e_smoke():
    """3 epoch Stage 1 + 2 epoch Stage 2，5 行数据
    
    不要求性能；只要：
        - 全程不 NaN
        - 训练完成 checkpoint 写盘
        - evaluator 运行成功
    """
    ...
```

### 4.2 `test_v2_phase1_stage1_passes_gate.py`（**关键**）

```python
@pytest.mark.slow
@pytest.mark.physics
def test_phase1_stage1_passes_gate():
    """完整 Phase 1 Stage 1 训练，最终通过 gate"""
    cfg = load("configs/v2_phase1_stage1_pde_only.yaml")
    cfg.training.n_epochs = 300
    
    # 训练
    run_stage1(cfg)
    
    # 加载 checkpoint
    model = load_checkpoint("checkpoints/v2_phase1_stage1/best.pt")
    
    # gate
    test_loader = make_hydrogenic_loader_v2()
    assert check_stage1_gate(model, test_loader, cfg.stage1.gate)
```

**这个测试不在 CI 跑（耗时），但 Sprint 2 验收手动跑过**。

## 5. L4 物理对照（训练中）

参见 `07_evaluation.md` §2-4。`check_stage1_gate` 函数：

- 在 trainer 中每个 epoch 调用；
- 写入 `logs/v2_*/gate.csv` 时间序列；
- 触发 stage 切换 / rollback。

## 6. CI 集成

`.github/workflows/v2_ci.yml`：

```yaml
name: V2 CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - run: pip install -e .[dev]
      - run: pytest tests/ -v -m "not slow" --maxfail=1
```

`tests/test_v2_phase1_stage1_passes_gate.py` 标记 `@pytest.mark.slow`，
仅在 manual trigger 跑。

## 7. 单元测试 fixture 设计

`tests/conftest.py`：

```python
@pytest.fixture
def default_grid():
    return RadialGrid(r_min=1e-4, r_max=50.0, n_grid=256, scheme="loglinear")

@pytest.fixture
def default_bspline(default_grid):
    bs = BSplineBasis(n_basis=32, order=3)
    bs.precompute(default_grid.r)
    return bs

@pytest.fixture
def hydrogenic_h_1s_batch():
    """Single-row H 1s batch (B=1, N_orb=1)."""
    return {
        "Z": torch.tensor([1]),
        "charge": torch.tensor([0]),
        "nele": torch.tensor([1]),
        "config_shells": torch.tensor([[[1, 0, 1, 1]] + [[0,0,0,0]]*31]).long(),
        "shell_mask": torch.tensor([[True] + [False]*31]),
        "kappa": torch.tensor([[-1] + [0]*15]),
        "orb_mask": torch.tensor([[True] + [False]*15]),
        "J": torch.tensor([1]),
        "parity": torch.tensor([0]),
        "term_id": torch.tensor([0]),
        "E_target": torch.tensor([-0.5]),    # Hartree
    }
```

## 8. 测试数量目标

- **Sprint 1 完成**：≥ 12 单元测试 (B-spline 4 + envelope 3 + readout 4 + per_orb 1)
- **Sprint 2 完成**：≥ 20 单元 + 2 集成
- **Sprint 3 完成**：≥ 30 单元 + 4 集成 + 1 端到端
- **Sprint 4 完成**：≥ 35 单元 + 5 集成 + 2 端到端

任何 Sprint 验收时测试数下滑 = 视为未通过验收。
