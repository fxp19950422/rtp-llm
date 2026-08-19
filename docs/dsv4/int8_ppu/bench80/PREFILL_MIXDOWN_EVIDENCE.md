# D1 混跑退化与 prefill 探针证据（2026-08-19 03:3x，serve pid 920009 会话）

## 现象（D1 自然 EOS 版，results/d1_natural/）

| 指标 | 值 | 80 基线 | 说明 |
|---|---|---|---|
| duration | 1509.84 s | 179 s | 8.4× |
| output throughput | 265.51 tok/s | 2676.48 | 10.1× 差距 |
| 完成率 | 400880/480000 = 83.5% | 100% | 自然 EOS 提前停（主口径已修 ignore_eos 重跑） |
| mean TPOT | 45.95 ms | 39.17 | 仅 1.17× |
| ITL P50/P95/P99/Max | 37.58/110.70/174.88/594.05 ms | — | decode 步进被 prefill 长打断 |
| running(steady) | mean 120.0 / max 128（前端在途） | 110.6/128 | 前端满载，但引擎侧实际 decode 序列 ≈ 吞吐×TPOT ≈ 12 条 |
| KV usage（bench 中段 RANK0） | pool0 0.58–0.76%，global ≤0.38% | full 16.4% | KV 远未成为瓶颈 |
| input throughput | 847.77 tok/s（≈全程 prefill 主导） | — | duration ≈ 320 条 × ~4.7 s/条 prefill |

## 空闲态 prefill 探针（prefill_probe.py，同 serve 会话）

| 探针 | wall | 单条延迟 | 全局吞吐 |
|---|---|---|---|
| ~1129 tok × 1 | 1.00 s | 1.00 s | ~1.1k tok/s |
| ~4504 tok × 1 | 1.83 s | 1.83 s | ~2.2k tok/s |
| ~4504 tok × 8 并发 | 3.04 s | 2.91–3.04 s（均衡） | **~11.8k tok/s** |

## 结论（写入 RESULTS.md §分析）

1. **prefill 本身可并行扩展**（8 并发 3s 内全部完成，无全局串行锁）；单条 4504 tok ≈ 1.83 s（~2.4k tok/s/rank）。
2. **decode 混跑时 prefill 吞吐塌方至 ~848 tok/s（12× 退化）**：rtp-llm 无 chunked prefill，4000+ tok 单次全长 forward 独占 rank，期间该 rank decode 停摆（ITL P95 110ms/P99 175ms 实证），DP8 各 rank 轮流被长 prefill 阻塞 → 有效 decode 并发坍缩至 ~12 条（KV 占用 0.3–0.8% 与之互证）。
3. 80 SGLang 侧 chunked_prefill 将长输入切片与 decode 交错 → decode 不停摆（running 110.6、TPOT 39ms 稳定）。此为**结构性差异的量化实锤**，非配置错误（75 无对应参数，已列入无法对齐项 §4.4）。
4. 主口径（ignore_eos=100% 完成率）重跑后，吞吐量级预期与自然版接近（decode 并发坍缩不变），数字以重跑为准。

## ignore_eos 机制实证

- 不带：completion 188/300（自然 EOS 停）
- `extra_configs: {"ignore_eos": true}`：completion 300/300（跑满）
- rtp-llm `ChatCompletionRequest` 无顶层 ignore_eos 字段（pydantic 静默丢弃），须走 `extra_configs` 嵌套（openai/api_datatype.py:215）；80 sglang 原生认顶层字段。
