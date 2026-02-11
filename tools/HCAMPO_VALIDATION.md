# H-CAMPO 验证指南

本文档记录 H-CAMPO（Hierarchical CAMPO）实现的三层验证流程。

---

## 验证架构

```
┌─────────────────────────────────────────────────────────────┐
│  L1: 数学验证 (✅ 完成)                                      │
│  - 退化正确性: plan_found=False → H-CAMPO == CAMPO         │
│  - 分解一致性: A_plan + A_exec == A_total                   │
│  - 隔离正确性: NO_PLAN_ID 不污染统计                        │
│  工具: tools/hcampo_sanity_check.py                         │
│  依赖: torch + numpy (无需模型)                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  L2: 数据流验证 (✅ 完成)                                   │
│  - 数组形状一致性                                           │
│  - plan_end_pos 边界有效性                                  │
│  - Clamp 逻辑正确性                                         │
│  - Metrics 计算稳定性                                       │
│  工具: tools/hcampo_dataflow_check.py                       │
│  依赖: fake tokenizer (无需模型)                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  L3: 行为验证 (⏸️ 待启动)                                   │
│  - plan_found_rate: 模型是否产出 <plan> 结构              │
│  - plan_unique_ratio: P 的基数是否合理                     │
│  - E[S|P] 可估计性: 每桶样本数 M 是否足够                  │
│  工具: tools/hcampo_smoke_test.py                           │
│  依赖: base 模型 (1B-2B 推荐)                               │
└─────────────────────────────────────────────────────────────┘
```

---

## L1: 数学验证（无需模型）

### 目标
验证 H-CAMPO 的核心数学性质：
1. **退化正确性**：当 `plan_found=False` 时，H-CAMPO 必须严格等于 CAMPO
2. **分解一致性**：$A_{plan} + A_{exec} = A_{total}$ 恒成立
3. **隔离正确性**：`NO_PLAN_ID` 样本不参与 plan 统计

### 运行
```bash
python tools/hcampo_sanity_check.py
```

### 预期输出
```
[A] plan_found=False fallback max|diff| = 0
[B] decomposition identity max| (A_plan + A_exec) - A_campo | = 2.38419e-07
✓ All sanity checks passed!
  - plan_found=False samples degrade to CAMPO correctly
  - Decomposition A_plan + A_exec = A_total holds
  - NO_PLAN_ID samples are isolated from plan statistics
```

### 状态
✅ **通过** (2026-02-10)

---

## L2: 数据流验证（无需模型）

### 目标
验证 plan 解析管线的工程健壮性：
- 数组长度一致性（plan_end_pos/plan_found/plan_mode/plan_id）
- `plan_end_pos` 边界检查（必须 < response_length）
- `NO_PLAN_ID` 正确分配（plan_found=False 时）
- Metrics 计算不崩溃且值域合理

### 运行
```bash
python tools/hcampo_dataflow_check.py
```

### 预期输出
```
============================================================
H-CAMPO Data Flow Validation
============================================================

[1] Input validation
  Batch size: 16
  Max seq len: 128
  Response lengths: min=8, max=128, mean=65.4

[2] Parsing plan boundaries...
  ✓ Array shapes match batch size
  ✓ plan_end_pos within valid ranges

[3] Plan detection statistics
  plan_found_rate: 43.75% (7/16)
  plan_mode=ratio: 9 (56.2%)
  plan_mode=tag: 7 (43.8%)

[4] Extracting plan IDs...
  ✓ plan_ids array length matches
  ✓ NO_PLAN_ID correctly assigned to plan_found=False samples
  Unique plan_ids (total): 8
  Unique valid plan_ids: 7

[5] Computing plan metrics...
  ✓ All metrics in valid ranges [0, 1]

[6] Validating plan_end_pos clamping...
  ✓ Clamped plan_end_pos all within valid ranges

============================================================
✓ All data flow checks passed!
============================================================
```

### 状态
✅ **通过** (2026-02-10)

---

## L3: 行为验证（需要 base 模型）

### 目标
验证**离散变量 P 的稳定性**，即：
- 模型是否产出稳定的 `<plan>...</plan>` 结构
- `plan_found_rate` 是否足够高（推荐 > 0.3）
- `plan_unique_ratio` 是否合理（推荐 0.3-0.7）
- $E[S|P]$ 是否可估计（每桶样本数 M > 2）

### 前置条件
1. **模型**：1B-2B 参数（推荐 Qwen2.5-1.5B-Instruct / SmolLM-1.7B）
2. **数据**：10-20 条 GSM8K/MATH 样本（JSONL 格式）
3. **环境**：单卡 GPU，支持 vLLM（推荐）

### 设置步骤

#### 1. 配置路径
编辑 `tools/hcampo_smoke_test.py`：
```python
MODEL_PATH = "models/Qwen2.5-1.5B-Instruct"  # 你的模型路径
TRAIN_FILE = "data/gsm8k_mini.jsonl"         # 你的数据路径
```

#### 2. 生成配置
```bash
python tools/hcampo_smoke_test.py
```

#### 3. 运行训练
```bash
python -m rl_train.src.main_campo \
    --config-path smoke_test_hcampo/smoke_config.yaml
```

### 关键指标

训练日志中关注以下 H-CAMPO 指标：

| 指标 | 目标范围 | 说明 |
|------|----------|------|
| `hcampo/plan_found_rate` | **> 0.3** | <plan> 结构检测成功率 |
| `hcampo/plan_mode_tag_rate` | > 0.5 | 真正的 tag 检测（非 fallback） |
| `hcampo/plan_unique_ratio_per_uid` | **0.3-0.7** | 计划多样性（太低=无信息，太高=高基数） |
| `hcampo/plan_mode_ratio_rate` | < 0.5 | ratio fallback 比例（越低越好） |

### 预期运行时间
- 2 batches × 2 prompts × 4 responses = 16 rollouts
- 约 **3-5 分钟**（单卡，1.5B 模型）

### 诊断指南

#### 问题 1: `plan_found_rate < 0.1`
**原因**：模型不产出 `<plan>` 标签  
**解决方案**：
1. 添加 System Prompt：
   ```
   You must structure your response as:
   <plan>Your solving strategy</plan>
   <solution>Step-by-step solution</solution>
   ```
2. 或用 SFT 数据微调（包含 plan 结构）
3. 或接受 ratio fallback（调高 `tau_max` 到 512）

#### 问题 2: `plan_unique_ratio > 0.9`
**原因**：P 基数太高，每个 plan 几乎唯一  
**解决方案**：
1. 增大 `n_resp_per_prompt`（如 8/16）
2. 检查 `dual_sig` 是否过于细粒度
3. 考虑切换到 `route_enum` 模式（方案2）

#### 问题 3: `plan_unique_ratio < 0.1`
**原因**：所有 plan 几乎一样（废话 plan）  
**解决方案**：
1. 检查 prompt 是否太宽泛
2. 提高 temperature（如 1.2）增加多样性
3. info-gate 会自动关闭（$w_g \to 0$），退化为 CAMPO

### 状态
⏸️ **待启动** - 需要用户提供 base 模型

---

## 快速参考

### 完整验证流程
```bash
# L1: 数学验证（30 秒）
python tools/hcampo_sanity_check.py

# L2: 数据流验证（30 秒）
python tools/hcampo_dataflow_check.py

# L3: 行为验证（3-5 分钟，需要模型）
python tools/hcampo_smoke_test.py
python -m rl_train.src.main_campo --config-path smoke_test_hcampo/smoke_config.yaml
```

### 文件清单
```
tools/
├── hcampo_sanity_check.py      # L1: 数学验证（独立，无依赖）
├── hcampo_dataflow_check.py    # L2: 数据流验证（需要 plan_utils）
├── hcampo_smoke_test.py        # L3: Smoke test 配置生成器
└── HCAMPO_VALIDATION.md        # 本文档

rl_train/src/
├── plan_utils.py               # Plan 解析核心逻辑
├── campo_ray_trainer.py        # H-CAMPO 集成点
└── config/campo_trainer.yaml   # H-CAMPO 默认配置

verl/trainer/ppo/
├── core_algos.py               # compute_hcampo_advantage()
└── ray_trainer.py              # AdvantageEstimator.H_CAMPO
```

---

## 附录：关键不变量

### 数学不变量
1. **Fallback 等价性**：
   ```python
   if plan_found[i] == False:
       assert hcampo_adv[i] == campo_adv[i]
   ```

2. **分解一致性**（样本级）：
   ```python
   if plan_found[i] == True:
       assert A_plan[i] + A_exec[i] ≈ (S[i] - μ_g) / σ_g
   ```

3. **统计隔离**：
   ```python
   # NO_PLAN_ID 不参与 uid_plan_mu/uid_delta 计算
   if plan_id[i] == NO_PLAN_ID:
       assert (uid, plan_id[i]) not in uid_plan_mu
   ```

### 工程不变量
1. **数组长度**：
   ```python
   len(plan_end_pos) == len(plan_found) == len(plan_id) == bsz
   ```

2. **边界有效性**：
   ```python
   0 <= plan_end_pos[i] < response_lengths[i]
   ```

3. **NO_PLAN_ID 语义**：
   ```python
   plan_found[i] == False  ⟺  plan_id[i] == "__NO_PLAN__"
   ```

---

## 后续工作

完成 L3 验证后，可选的改进方向：

1. **方案2：Guided Decoding**（推荐）
   - 用 vLLM 的 `guided_choice` 强制输出 route_id
   - 让 P 变成真正的稳定离散变量（基数可控）
   - 参考：`vllm.SamplingParams(guided_choice=["algebra", "geometry", ...])`

2. **增强 Plan 提示**
   - System prompt 引导生成 `<plan>` 结构
   - Few-shot 示例（推理链带 plan 标签）
   - SFT 数据增强（标注 plan 段）

3. **动态 Gate 调优**
   - 根据 `plan_found_rate` 自动调整 `plan_found_threshold`
   - 根据 `plan_unique_ratio` 自动调整 `shrinkage_kappa`
   - A/B test H-CAMPO vs CAMPO 在不同任务上的表现

---

**最后更新**：2026-02-10  
**维护者**：GitHub Copilot
