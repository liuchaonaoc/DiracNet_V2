# 06 — Two-Stage Training Protocol (核心)

> 本文件是 V2 的灵魂。任何对 "PDE / NIST 关系" 的修改都必须经过这里的纪律审查。

## 1. 设计目标

把 V1 retrospective 中描述的 **"NIST 与 PDE 同步训练 → cheating"** 问题
**结构性消除**。具体做法：

1. **Stage 1**: 完全没有 NIST 梯度。波函数与单粒子能级只通过物理约束（PDE + 正交 + 节点 + 渐近）训练。
   NIST 仅作为评估指标记录在 log 中。
2. **Stage 2**: 在 Stage 1 通过解析对照门禁后，启用一个 **窄带 Δ_residual head**（强约束 ±50 meV）
   + 主网络冻结或低 lr，让 NIST 残差只能通过这个 head 学习。
3. **门禁回滚**: Stage 2 中每个 epoch 复跑解析对照；若 `|cos|` 跌破 0.99，回滚到最近一个通过门禁的 ckpt。

## 2. Stage 1 — PDE-only 训练

### 2.1 训练目标

```
L_stage1 = w_pde   · L_PDE
         + w_ortho · L_ortho
         + w_node  · L_node
         + w_asym  · L_asym
         + w_smooth · L_smooth
```

NIST **不** 在 `L_stage1` 里。仍然每个 batch 计算 `L_NIST_monitor` 写日志，但
**绝对不接梯度**（实现上 `with torch.no_grad():` 包住 NIST 计算分支）。

### 2.2 优化器与学习率

```yaml
optimizer:
  name: adamw
  lr_kan: 3.0e-4              # KAN 主路径
  lr_kan_edges: 5.0e-4        # KAN 边样条系数（独立学习率，可调）
  lr_encoder: 1.0e-4          # h_cond 编码器
  lr_residual: 0.0            # Stage 1 不训练 residual head
  weight_decay: 1.0e-4
scheduler:
  name: warmup_cosine
  warmup_steps: 200
  total_steps_stage1: 5000
```

### 2.3 训练循环

```python
def train_stage1(model, train_loader, val_loader, n_epochs, cfg):
    losses = {
        "pde": DiracPDELoss(),
        "ortho": OrthonormalityLoss(),
        "node": NodeCountLoss(),
        "asym": AsymptoticTailLoss(),
        "smooth": BSplineSmoothLoss(),
    }
    nist_loss_monitor = NISTScalarHuberLoss(delta=cfg.losses.nist_huber_delta,
                                            align_ground=False)

    optimizer = build_optimizer(model, cfg.optimizer, stage=1)
    scheduler = build_scheduler(optimizer, cfg.scheduler)
    
    weights = cfg.losses.weights_stage1
    
    for epoch in range(n_epochs):
        for batch in train_loader:
            out = model(batch)        # forward: env, P, Q, E_orb, ...
            
            # ── compute physics losses (with grad)
            l_pde   = losses["pde"](...)
            l_ortho = losses["ortho"](...)
            l_node  = losses["node"](...)
            l_asym  = losses["asym"](...)
            l_smooth= losses["smooth"](out["c_raw"], batch["orb_mask"])
            
            L = (weights["pde"]   * l_pde
               + weights["ortho"] * l_ortho
               + weights["node"]  * l_node
               + weights["asym"]  * l_asym
               + weights["smooth"]* l_smooth)
            
            # ── compute NIST loss for monitoring (NO grad)
            with torch.no_grad():
                E_pred_monitor = (out["E_orb"] * batch["occ"] * batch["orb_mask"]).sum(-1)
                l_nist_mon = nist_loss_monitor(
                    E_pred_monitor, batch["E_target"], batch["Z"], batch["charge"]
                )
            
            optimizer.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()
            scheduler.step()
            
            log_metrics({
                "L_PDE": l_pde.item(),
                "L_ortho": l_ortho.item(),
                "L_node": l_node.item(),
                "L_asym": l_asym.item(),
                "L_smooth": l_smooth.item(),
                "L_NIST_monitor": l_nist_mon.item(),
                ...
            })
        
        # ── per-epoch validation
        val_metrics = validate(model, val_loader)
        if check_stage1_gate(model, hydrogenic_test_set, thresholds=cfg.stage1.gate):
            save_checkpoint("stage1_passed.pt")
            print("[Stage 1] gate passed; ready to switch to Stage 2.")
            return
    
    print("[Stage 1] reached n_epochs without passing gate.")
```

### 2.4 Stage 1 通过条件（门禁）

定义在 `cfg.stage1.gate`：

```yaml
stage1:
  gate:
    cos_threshold: 0.99          # |cos(P_model, P_analytic)| 全部 (Z, n)
    e_orb_meV_threshold: 1.0      # |E_orb − E_analytic| 全部 (Z, n)
    lambda_rel_threshold: 0.05    # |λ - Z_eff/n| / (Z_eff/n)
    L_PDE_threshold: 1.0e-3
    L_ortho_threshold: 1.0e-4
    L_node_threshold: 1.0e-2
    test_set: data_cache/manifest_hydrogenic_v2.parquet   # H/He+/Li2+ 1s..5s
```

`check_stage1_gate` 函数：

```python
def check_stage1_gate(model, test_loader, thresholds) -> bool:
    """Compare model to analytic hydrogenic solutions on the test set."""
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            P_model = out["P"][..., 0, :]         # active orbital
            for i, (Z, n) in enumerate(batch_metadata(batch)):
                P_ref = analytic_hydrogenic_P(r_grid, Z, n)
                cos = abs(cosine(P_model[i], P_ref))
                e_err = abs(out["E_orb"][i, 0] - (-Z**2 / (2 * n**2)))
                lam_err = abs(out["lam"][i, 0] - Z/n) / (Z/n)
                if cos < thresholds.cos_threshold: return False
                if e_err > thresholds.e_orb_meV_threshold / 27211.4: return False
                if lam_err > thresholds.lambda_rel_threshold: return False
    return True
```

### 2.5 Stage 1 不能通过的常见原因 + 处理

| 症状 | 可能原因 | 处理 |
|------|----------|------|
| `L_PDE` 不下降 | KAN init 异常 / `c_0` 没置零 | 检查初始化、强制 c_0=0、降低 lr |
| `L_PDE` 下降但 `|cos|` 卡在 0.9 | KAN 过拟合到错形状 / B-spline 节点不够 | `K: 32 → 48`；增加 `L_smooth` |
| n=2 一直坍缩到 n=1 形状 | `L_node` 权重不够 / λ 没拉开 | `w_node: 0.1 → 1.0`；强制 λ_n = Z/n（freeze） |
| `λ` 跑飞 | `lam_log_res` clamp 没生效 | 检查 clamp(-2, 2) 是否运行 |
| 远端 P 不衰减 | B-spline 节点没到 r_max / `L_asym` 不够 | 增加节点、提高 r_max、`w_asym: 0.01 → 0.1` |

## 3. Stage 2 — NIST Residual Fine-Tune

### 3.1 训练目标

```
L_stage2 = w_pde       · L_PDE              (保持物理压力，防滑回)
         + w_ortho     · L_ortho
         + w_node      · L_node
         + w_asym      · L_asym
         + w_smooth    · L_smooth
         + w_nist(t)   · L_NIST_res
```

`w_nist(t)` 是 ramp：

```
w_nist(t) = w_nist_max · min(1.0, t / t_warmup)
t_warmup = 20 epochs
w_nist_max = 1.0
```

PDE 权重 **不降**，让模型在 fine-tune 时仍受物理约束。

### 3.2 哪些参数 trainable / frozen

| 模块 | Stage 1 | Stage 2 (default) | Stage 2 (ablation) |
|------|---------|-------------------|---------------------|
| `QuantumEncoder` | trainable | frozen (lr=0) | lr=1e-5 |
| `LevelFeatureEncoder` | trainable | frozen | lr=1e-5 |
| `KANCoeffNet (main)` | trainable | frozen | lr=1e-6 |
| `AnalyticEnvelope` | (无可学习参数) | – | – |
| `BSplineReadout` | (无可学习参数) | – | – |
| `LevelResidualHead` | **不存在** | **新增 + trainable** | trainable |

```python
def setup_stage2(model, cfg):
    # freeze main params
    for p in model.kan.parameters():        p.requires_grad_(False)
    for p in model.encoder.parameters():    p.requires_grad_(False)
    for p in model.level_encoder.parameters(): p.requires_grad_(False)
    
    # add residual head
    model.residual_head = LevelResidualHead(
        d_cond=cfg.encoder.d_cond,
        delta_max=cfg.stage2.delta_max,        # 50 meV → 1.84e-3 Ha
    )
    
    optimizer = AdamW(
        model.residual_head.parameters(),
        lr=cfg.optimizer.lr_residual,
        weight_decay=cfg.optimizer.weight_decay,
    )
    return optimizer
```

### 3.3 `LevelResidualHead`

```python
class LevelResidualHead(nn.Module):
    """Bounded learnable residual for term-dependent fine structure.
    
    Δ = delta_max * tanh(MLP(h_cond, J_emb, parity_emb, term_emb))
    """
    
    def __init__(self, d_cond: int, delta_max: float = 1.84e-3) -> None:
        super().__init__()
        self.delta_max = float(delta_max)   # 50 meV → ≈ 1.84 mHa
        self.mlp = nn.Sequential(
            nn.Linear(d_cond, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward(self, h_cond: Tensor) -> Tensor:
        return self.delta_max * torch.tanh(self.mlp(h_cond).squeeze(-1))
```

注意 V2 的 `delta_max` 比 V1 的 `delta_scale=0.1 Ha` 小 **54 倍**。原因：

- V1 `delta_scale=0.1 Ha` 可以吃掉整个 H 1s 的 100 meV 误差（→ cheating）；
- V2 期望 Stage 1 已经把波函数练对到 < 1 meV，**Δ_residual 只是补 term-dependent
  多重态结构**，这通常 ≤ 50 meV（轻原子量级）。

### 3.4 训练循环（Stage 2）

```python
def train_stage2(model, train_loader, hydrogenic_loader, val_loader, n_epochs, cfg):
    losses = {
        "pde": DiracPDELoss(),
        "ortho": OrthonormalityLoss(),
        "node": NodeCountLoss(),
        "asym": AsymptoticTailLoss(),
        "smooth": BSplineSmoothLoss(),
        "nist": NISTScalarHuberLoss(delta=cfg.losses.nist_huber_delta,
                                    align_ground=cfg.losses.align_ground),
    }
    
    optimizer = setup_stage2(model, cfg)
    scheduler = build_scheduler(optimizer, cfg.scheduler)
    
    last_good_ckpt = "stage1_passed.pt"
    
    for epoch in range(n_epochs):
        w_nist = ramp(epoch, cfg.stage2.warmup_epochs, max_w=cfg.stage2.w_nist_max)
        
        for batch in train_loader:
            out = model(batch)
            E_pred_orb = (out["E_orb"] * batch["occ"] * batch["orb_mask"]).sum(-1)
            Δ_res = model.residual_head(out["h_cond"])
            E_pred = E_pred_orb + Δ_res
            
            L = (w_pde   * losses["pde"](...)
               + w_ortho * losses["ortho"](...)
               + w_node  * losses["node"](...)
               + w_asym  * losses["asym"](...)
               + w_smooth* losses["smooth"](out["c_raw"], batch["orb_mask"])
               + w_nist  * losses["nist"](E_pred, batch["E_target"],
                                           batch["Z"], batch["charge"]))
            optimizer.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()
            scheduler.step()
        
        # ── physics gate (rollback safeguard)
        if not check_stage1_gate(model, hydrogenic_loader, cfg.stage1.gate):
            print(f"[Stage 2 epoch {epoch}] physics gate VIOLATED; rolling back.")
            model.load_state_dict(torch.load(last_good_ckpt)["model"])
            # 同时 reduce w_nist_max ×0.5，下个 epoch 再尝试
            cfg.stage2.w_nist_max *= 0.5
            continue
        else:
            last_good_ckpt = save_checkpoint(f"stage2_epoch_{epoch}.pt")
        
        # ── log
        val_metrics = validate(model, val_loader)
        log_metrics(val_metrics)
```

### 3.5 Stage 2 通过条件

```yaml
stage2:
  pass:
    rms_meV: 50         # 全数据集（带 Δ_res 后）
    median_meV: 20
    cos_threshold: 0.99 # 不允许波函数退化
    lambda_drift: 0.05
```

## 4. **关于 PDE 与 NIST 关系的严格论证**

> 用户在请求中特别说："**需要慎重考虑 PDE 残差和 NIST 损失函数的关系**"。

下面是 V2 给出的最终关系定义（写入 `docs/design_rationale.md` 也是这一段）：

### 4.1 V1 的失败模式（再申）

V1 R9 同时跑 `w_pde = 0.01`, `w_nist = 10.0` →
- NIST 的标量梯度推 `E_pred = Σ E_orb + Δ_term` 一个方向；
- PDE 的张量梯度推 `(P, Q)` 朝向 Dirac eigenfunction 另一个方向；
- 中间通过 `Löwdin renorm + Δ_term` 这条暗道相互转化；
- 最终：能量数字看起来对，波函数完全错。

### 4.2 V2 的纪律

```
Stage 1:  NIST → 不进 backward          (信息单向：仅监控)
          PDE  → 全力 backward          (波函数必须物理)
          
解析对照门禁:  λ 一致 + |cos| > 0.99 + E_orb 误差 < 1 meV  (硬约束)
          
Stage 2:  NIST → 只能通过 Δ_residual head 流回梯度       (容量 ±50 meV)
          KAN main → 冻结或 lr=1e-6                       (波函数被锁定)
          PDE  → 仍参与 backward                         (防止滑回)
          
每 epoch 物理回滚:  门禁违反 → 回到上一个通过 ckpt + 降低 w_nist_max
```

⇒ **NIST 在 V2 中的全部权力被压在 50 meV 的 Δ_residual 上**；剩下的物理由 PDE 严格规约。

### 4.3 为什么这样能解 V1 没解开的问题

| V1 失败原因 | V2 对应措施 |
|-------------|-------------|
| NIST 梯度主导，PDE 让步 | Stage 1 NIST 不接梯度，物理优先 |
| Δ_term 吸收 ~50-100 meV → 假能量 | Stage 2 Δ_residual 严格 ≤ 50 meV，必要时设 ≤ 20 meV |
| λ 漂离 Z/n → 形状错 | KAN 显式吃 (Z, n)，门禁要求 `|λ−Z/n|/(Z/n) < 5%` |
| RC 假设类不含 Laguerre → cos < 0.1 for n≥3 | B-spline 在 32 节点下可表 Laguerre 至 |cos|≈1 |
| Löwdin renorm 把形状错误归一化掩盖 | 解析对照门禁直接对 P_model 与 P_analytic 求 cos |
| Phase-A 诊断"事后发现" | 训练时门禁，trainer 内自动检测 + 回滚 |

### 4.4 备选方案：完全不用 NIST

V2 是否真的需要 Stage 2？

**如果你的目标是"复现 NIST 数据"**：必要，因为单粒子 Dirac 方程不含多电子相关，
PDE 收敛点必然偏离 NIST level（multi-electron correlation gap）。

**如果你的目标是"做正确的单粒子量子力学"**：Stage 1 就够，Stage 2 可跳过。
NIST 仅作为 final evaluation。

V2 配置文件提供两个 mode：

```yaml
# configs/v2_phase1_pde_only.yaml      # 仅 Stage 1
# configs/v2_phase1_full.yaml          # Stage 1 + Stage 2
```

## 5. 训练时间预估

| Stage | Phase 1 (15 行 hydrogenic) | Phase 2 (NIST 全量 36k 行) |
|-------|---------------------------|---------------------------|
| Stage 1 | 100-300 epoch × ~10 s = 15-50 min | 30-50 epoch × ~10 min = 5-8 h |
| Stage 2 | 20-50 epoch × ~10 s = 5-15 min | 10-20 epoch × ~10 min = 1.5-3 h |

总训练时间：Phase 1 ≤ 1 h，Phase 2 ≤ 12 h。

## 6. 训练监控仪表板

`logs/v2_*/` 下每个 run 自动生成：

```
metrics.csv            # 所有 batch-level loss 与梯度范数
gate.csv               # 每 epoch 的解析对照结果（|cos|, ΔE, λ_drift）
lambda_drift.png       # 每个 (Z, n) 的 λ 漂移
nist_residual.png      # NIST_pred - NIST_target 的箱型图
cos_per_n.png          # |cos(P_model, P_analytic)| 随 epoch 的轨迹
stage_transition.txt   # 进入 Stage 2 的时间戳与状态
```

每个 run 必须配套 **Markdown 总结** (`results/v2_*/SUMMARY.md`)，
描述：
- 通过 / 未通过门禁
- 波函数形状偏离最大的行
- 与 V1 R9 / B16 / B9-LOO 的对比表

## 7. 与 V1 LevelwiseTrainer 的关系

V1 `training/levelwise_trainer.py` 把 4 个 loss (`pde, nist, ortho, wkb`) 一锅炖。
V2 必须 **新写** `training/two_stage_trainer.py`，但内部可以调用 V1 已有的
`losses/*.py` 实例。trainer 的责任：

1. 决定 `loss = ...` 里是否包含 NIST 梯度（`stage` 状态机）；
2. 调用 `check_stage1_gate` 自动门禁；
3. 维护 `last_good_ckpt` 以便 Stage 2 回滚。

代码骨架在 §2.3 + §3.4 已经给出，照实现即可。
