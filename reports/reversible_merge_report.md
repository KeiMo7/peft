# 可逆适配器合并（Reversible Adapter Merge）验证报告

## 1. 问题与根因

灰度方案把适配器合并进基座权重以省显存，回退时再拆分。现有 `merge_adapter()` /
`unmerge_adapter()` 是一对**原地**操作：

- 合并：`base_weight += delta_weight`（`src/peft/tuners/lora/layer.py:1110`）
- 拆分：`base_weight -= delta_weight`（`src/peft/tuners/lora/layer.py:1138`），delta 现场重算

在 float16 / bfloat16 下，每次加法和减法都发生一次新的舍入，拆分后的权重并不等于合并前的权重，
且误差随往返轮次线性累积。基线复现脚本 `artifacts/work/baseline_drift.py` 实测（非零随机适配器）：

| 精度 | 单次往返 权重最大偏差 | 10 次往返 权重最大偏差 | 增长 |
|---|---|---|---|
| float32 | 1.49e-08 | 1.49e-08 | 1.0×（不累积） |
| float16 | 1.22e-04 | 6.10e-04 | **5.0×** |
| bfloat16 | 9.77e-04 | 9.77e-03 | **10.0×（线性累积）** |

这正是"反复合并拆分几轮后差异越来越大、回退对不上"的直接原因。

## 2. 方案：快照式可逆合并

新增 `src/peft/tuners/reversible_merge.py`，在合并前对每个受影响基座参数保存**逐位快照**，
拆分时直接用快照覆盖还原，而不是再减一次重算的 delta：

- 还原是 `copy_(snapshot)`，**任意轮次后基座权重与输出的往返误差严格为 0**；
- 不需要重新加载原始基座检查点即可在现场完成回退（满足"不能靠重载基座假装可逆"）；
- 半精度下提供 `upcast_merge=True`：合并算术在 float32 工作副本上执行一次再 cast 回原 dtype，
  基座参数与快照 dtype 不变；
- 合并时记录合并权重的校验和（float64 的 sum / sumsq + shape/dtype），拆分前检测外部篡改；
- 所有簿记状态只存在于合并/拆分时刻，推理热路径上合并层仍走纯 base layer 前向，与现有合并完全一致。

公开 API（全部为**新增**方法，现有 `merge_adapter` / `unmerge_adapter` 签名与返回结构不变）：

- `PeftModel.merge_adapter_reversible(adapter_names=None, *, upcast_merge=True)`
- `PeftModel.unmerge_adapter_reversible()`
- `PeftModel.reversible_merge_status() -> dict`（哪些层已合并、合并了哪些适配器、往返次数）
- 同样的三个方法也挂在 `BaseTuner` 上（`src/peft/tuners/tuners_utils.py`）。

## 3. 实测数字（`reports/measurements.json`）

测试模型：两层 Linear（32→32→16）+ ReLU，LoRA r=8/alpha=16，非零随机适配器，输入 shape (4,32)。

### 3.1 往返一次 / 十次的最大权重偏差

| 精度 | 可逆：单次权重偏差 | 可逆：10 次权重偏差 | 旧路径：单次 | 旧路径：10 次 |
|---|---|---|---|---|
| float32 | **0.00e+00** | **0.00e+00** | 1.49e-08 | 1.49e-08 |
| float16 | **0.00e+00** | **0.00e+00** | 1.22e-04 | 6.10e-04 |
| bfloat16 | **0.00e+00** | **0.00e+00** | 9.77e-04 | 9.77e-03 |

可逆路径每一轮 `drift_per_cycle` 都是全 0，上界为**零**且与轮次无关；输出最大偏差同样为 0
（逐位还原，见 `single_round_trip_max_output_diff`）。

### 3.2 合并路径 vs 未合并路径输出偏差（带分布，未藏进容差）

| 精度 | max | p99 | p90 | p50 | mean | 断言容差 atol | allclose |
|---|---|---|---|---|---|---|---|
| float32 | 2.38e-07 | 2.01e-07 | 1.19e-07 | 4.47e-08 | 5.16e-08 | 1e-6 | ✅ |
| float16 | 4.88e-04 | 4.88e-04 | 2.44e-04 | 6.10e-05 | 1.04e-04 | 5e-3 | ✅ |
| bfloat16 | 7.81e-03 | 7.81e-03 | 3.61e-03 | 9.77e-04 | 1.26e-03 | 2e-2 | ✅ |

该差异来自浮点结合律（`(W+BA)x` 与 `Wx+B(Ax)` 的运算顺序不同）以及半精度合并权重的单次舍入，
与旧合并路径的合并精度相同，**不是往返误差**——拆分后输出逐位回到合并前。

### 3.3 保存 / 加载往返

合并态 `save_pretrained` 写出两个 sidecar 文件（仅在确有合并层时）：
`reversible_merge_state.json`（层名、适配器、校验和、往返次数）与
`reversible_merge_state.safetensors`（基座参数快照）。在全新构造的模型（模拟另一台机器，
只携带合并后的权重）上 `PeftModel.from_pretrained` 后再 `unmerge_adapter_reversible`：

- `exact_weight_restore: true`（权重逐位还原）
- `exact_output_restore: true`（输出逐位一致）
- 未使用该特性时不产生任何新文件，检查点格式与之前完全一致。

## 4. 需求逐条覆盖

| # | 要求 | 实现 / 测试 |
|---|---|---|
| 一.1 | 可逆操作、偏差有明确上界 | 快照还原，上界为 0；`test_single_round_trip_restores_weights_and_outputs_exactly` |
| 一.2 | 十次往返不累积 | `test_ten_round_trips_have_zero_drift`（fp32/fp16/bf16，逐位相等） |
| 一.3 | 合并与未合并推理容差内一致 | `test_merged_output_matches_unmerged_path_within_tolerance`，容差与实测见 3.2 |
| 一.4 | 拆分恢复含未涉及层 | `test_unmerge_touches_only_targeted_layers_and_keeps_others` |
| 一.5 | 保存加载后仍可拆分 | `test_reload_and_unmerge_on_fresh_model_restores_exact_weights` |
| 一.6 | 合并状态可查询 | `reversible_merge_status()`；`test_round_trip_count_is_queryable` |
| 一.7 | 不支持层提前拒绝、无部分合并 | 量化层/分组卷积预检；`test_quantized_layer_rejected...`、`test_grouped_conv_rejected...` |
| 一.8 | 半精度补偿 | `upcast_merge` 在 fp32 工作副本执行合并算术（说明见第 5 节） |
| 一.9 | 失败可诊断到层与原因 | `ReversibleMergeError` 消息含层名+原因；`test_error_message_names_layer_and_reason` |
| 一.10 | 关闭能力时原行为逐条一致 | 新方法独立、sidecar 惰性；`TestReversibleMergeDoesNotChangeExistingBehavior` |
| 一.11 | 多次往返后可保存并在别处继续拆分 | `test_ten_round_trips_then_save_load_then_unmerge` |
| 二.1 | 不靠重载基座假装可逆 | 快照随模型/sidecar 携带，回退在无原始检查点现场完成 |
| 二.2 | 给出实测最大偏差与分布 | 见 3.1 / 3.2 与 `measurements.json` |
| 二.3 | 合并原子、失败整体回滚 | 预检阶段零修改；执行期异常逐层回滚；`test_failure_mid_model_rolls_back_already_merged_layers` |
| 二.4 | 现有入口调用方式/返回不变 | 仅新增方法，旧方法零改动（diff 为纯新增 154 行 + 新文件） |
| 二.5 | 不改变未合并路径数值 | `test_unmerged_forward_path_numerically_unchanged` |
| 三.1 | 不支持类型报层名原因且模型可用 | 拒绝后断言所有层未合并且前向输出有限 |
| 三.2 | 拆分时发现外部篡改要拒绝 | 校验和比对；`test_external_modification_of_merged_weights_is_detected`，模型保持合并态 |
| 三.3 | 非有限值中止并回滚 | `test_non_finite_adapter_aborts_and_rolls_back` |
| 三.4 | 中途失败不留部分合并 | 同二.3，回滚包含已处理的层 |
| 四.1-4 | 现有语义/精度速度/保存格式/热路径不变 | 复用各方法自身 `layer.merge()`；合并后前向为纯 base 前向；无 sidecar 时检查点不变 |

## 5. 关于半精度补偿（upcast_merge）的诚实说明

实测发现：在当前 PyTorch（2.14.1, CPU）的舍入实现下，对 Linear 层的标量加法
`fl(W+Δ)` 与 `fl32(W+Δ)` 再 cast 回半精度的结果在本测试数据上逐元素相同——
IEEE 就近舍入使"先加后舍"与"fp32 加后舍回"在这些量级上等价。因此 upcast 开关在
**合并权重数值**上的可测增益在当前硬件/数据下为 0，报告不夸大为"显著降低误差"。

它仍然保留的理由：(1) 数学上更严格，覆盖次正规数等边界；(2) 把整个 `layer.merge()`
（含 delta 的 matmul、卷积、DoRA 等变体路径）放到 fp32 工作副本，对其他层类型/后端有意义；
(3) 本特性解决的核心问题——**往返误差累积**——不依赖该开关，快照还原在任何精度下都严格为 0。

## 6. 测试执行记录

新测试（`tests/test_reversible_merge.py`，25 项，日志 `reports/pytest_reversible_merge.log`）：

```
25 passed, 3 warnings in 1.86s
```

原有合并/卸载回归（`reports/pytest_existing_merge.log`）：

```
pytest tests/test_custom_models.py -k "merge or unload"
4755 passed, 746 skipped, 19342 deselected in 25.00s
```

更宽的保存/加载回归：

```
pytest tests/test_custom_models.py -k "merge or unload or save or load"
10895 passed, 1509 skipped, 12439 deselected in 94.06s
```

其他：`test_tuners_utils.py` + `test_low_level_api.py` 358 passed；
`test_mixed.py` + `test_lora_variants.py` 108 passed；
IA³ / OFT / LoRA 三方法 5 轮可逆往返均逐位还原；`ruff check` 与 `ruff format --check` 全部通过。

## 7. 产物清单

- `src/peft/tuners/reversible_merge.py`：可逆合并核心（registry、快照、校验和、sidecar、原子回滚）
- `src/peft/tuners/tuners_utils.py`：`BaseTuner` 上三个新增公开方法
- `src/peft/peft_model.py`：`PeftModel` 包装方法 + save/load sidecar 挂载
- `tests/test_reversible_merge.py`：25 项测试
- `reports/measurements.json`：全部实测数值
- `reports/legacy_drift_baseline.json`：旧路径漂移基线
- `reports/pytest_*.log`：实际测试输出
- `artifacts/work/`：复现与测量脚本
