# RESULTS: 75 节点 rtp-llm vs 80 节点 SGLang 同口径性能对比

> 状态:  D1/D2/P1 压测进行中（本文档随轮次更新）
> 设备:  75 = 8×ZW-M890P (144GiB, rtp-llm TP1/DP8/EP8)；80 = 8×ZW-M890P (SGLang 0.5.13, TP8/CP8/EP8)
> 模型:  DeepSeek-V4-Flash W8A8-INT8（同一 ckpt 指纹，见 CONFIG_MAPPING.md §1）
> 配置映射与无法对齐项: 见 [CONFIG_MAPPING.md](CONFIG_MAPPING.md)

## 0. 结论摘要（压测完成后填写）

- Decode 吞吐/TPOT 差距: 待填
- Prefill TTFT 差距（TP 口径 / CP 口径分别）: 待填
- MTP 接受率/接受长度对比: 待填
- KV 显存效率: 待填

## 1. Decode 主口径（conc128 / in4000 / out1500 / n320）

双轮互补口径说明: rtp-llm 的 ignore_eos 仅在非流式路径生效（流式路径 ignore_eos/min_new_tokens 均被丢弃，
实证见 PREFILL_MIXDOWN_EVIDENCE.md）。故: **D1-stream 轮**（results/d1/）取延迟时序（TPOT/ITL，
每 token 流式回传计时准确），自然 EOS 提前停使输出完成率 85.0%（吞吐口径偏低但可复核）；
**D1-nostream 轮**（results/d1_nostream/）取吞吐/Duration/Completed（ignore_eos 生效，100% 完成；
非流式聚合响应使 TPOT/ITL 失真不采用）。两轮同一 serve 会话（pid 920009）、同负载、间隔 <1h，
D1-natural 预演轮（results/d1_natural/，TPOT 45.95ms / 265.51 tok/s / 83.5%）与主轮一致性 <2%（重复性证据）。

### 1.2 延迟与吞吐（80 §1.2 同构）

| 指标 | 80 SGLang TP8-MTP | 75 rtp-llm D1-stream | 75 rtp-llm D1-nostream | Δ% (75 vs 80) |
|---|---|---|---|---|
| Output throughput (tok/s) | 2676.48 | 268.62 (85% 完成口径) | 317.05 (100% 口径) | -88.2% (nostream vs 80) |
| TPOT mean (ms) | 39.17 | 46.12 | （失真不采用） | +17.7% |
| TPOT median (ms) | 40.15 | 46.59 | | +16.0% |
| TPOT P99 (ms) | 59.88 | 56.37 | | -5.9% |
| ITL P50/P95/P99/Max (ms) | — | 37.57 / 106.18 / 143.04 / 269.48 | | P95≈3× P50: prefill 独占 rank 打断 decode |
| TTFT mean (ms) | 7595 | 426,527（排队含，不可比） | 478,004（=E2E，非流式聚合不可比） | 口径不同不可比 |
| Duration (s) | 179 | 1518.85 | 1513.95 | 8.5×（nostream） |
| Completed / total | 320/320 | 320/320（output 407,989/480,000 = 85.0%） | 320/320（output 480,000/480,000 = 100.0%） | |

注:
- 80 该轮为 MTP 开启口径；75 D1 为无投机口径（公平的 decode 基线），75 的 MTP 对照见 §2。
- TTFT 口径: 80 为满并发稳态下闭式轮的队列头延迟；75 侧 320 条一次性入队、非闭式，TTFT 含排队时长，
  两口径不可直接比。75 单条 prefill 延迟见 §3 与 PREFILL_MIXDOWN_EVIDENCE.md 空闲探针。
- 吞吐差距主因不是 decode 慢（TPOT 仅 +17.7%），而是无 chunked prefill 导致 prefill/decode 混跑互相阻塞:
  input throughput 842.7 tok/s 全程被 prefill 主导，duration ≈ 320 条 × ~4.7s/条 prefill；详见
  [PREFILL_MIXDOWN_EVIDENCE.md](PREFILL_MIXDOWN_EVIDENCE.md)（空闲 8 并发 prefill 探针 11.8k tok/s vs 混跑 848 tok/s，
  12× 退化实锤）。

### 1.3 运行时指标（80 §1.3 同构，取 D1-stream 轮）

| 指标 | 80 | 75 | 说明 |
|---|---|---|---|
| running batch 全局 (mean/max) | 110.6 / 128 | 120.1 / 128（前端在途口径，稳态窗 n=1204） | 稳态窗: running ≥ 0.5×max_concurrency |
| running per-rank | — (TP8 单 batch) | 15.0 (120.1/8，DP8 换算) | 前端在途口径；引擎侧有效 decode 序列 ≈ 12（KV 占用互证，见证据文档） |
| full KV usage mean/max | 0.164 / 0.210 | 0.56 / 1.11 (%)（RANK0 mean 0.56/max 0.80；8-rank 峰值 1.11） | kvc pool gid0-2（CSA/HCA/INDEXER 三池镜像同值）；0.3–0.8% 占用与有效并发 ≈12 互证 |
| SWA KV usage mean/max | 0.323 / 0.520 | 0.060 / 0.109 (%)（RANK0 mean 0.060/max 0.073；8-rank 峰值 0.109） | kvc pool gid6（SWA/FIXED） |
| max_total_num_tokens | 3,260,160 (TP8 全局, per-GPU 407,520) | FULL 池 per-rank 5513 blocks × 256 = 1,411,328；全局 8× = 11,290,624 | per-GPU 容量为 80 的 3.46×（DP8 每 rank 独立持有全部层 KV）；三 FULL 池各存一份同 token 数，等效单池容量 |
| CUDA graph 命中 | 覆盖 128 | capture bs = [1,2,4,8,12,16,20,24,32] ×8 rank；运行期 eager/fallback 0 条 | 稳态 per-rank running ≈15 ≤ 32，覆盖充分 |
| cache hit (reuse_len) | — | 0%（10/10 请求 aux_info.reuse_len=0；同一长前缀连续两次 431.3→431.5ms 无加速） | 无前缀缓存实证 |

## 2. MTP 对照（D2, GEN=3, 同负载）—— 无法完成：大 prefill 死锁缺陷

**结论: D2 bench（conc128/in4000/out1500/n320）无法执行**。MTP 开启时 prefill ≥ ~400 token
触发引擎主循环死锁（全 rank 集合通信死等），in4000 负载必触发。以下为缺陷证据链与功能侧数据。

### 2.1 死锁缺陷证据链（5 次复现 + 2 轮排除实验）

| 实验 | 配置变量 | 请求 | 结果 |
|---|---|---|---|
| 第 3/4/5 次 serve | GEN=3, STREAM_ASYNC=1 | smoke 8000 / 8000 / 1000 | 死锁（链路断、进程活、kvc 独立线程活） |
| 实验 A | GEN=3, **STREAM_ASYNC=0** | smoke 1000 | 死锁（10:42 RANK 2 fallback 后全 rank sp metrics 停） |
| 实验 B | **GEN=1**, STREAM_ASYNC=1 | smoke 1000 | 死锁（10:51 RANK 1+2 同刻 fallback） |
| 阈值定位 | GEN=1 | smoke 100 / 200 / 400 | **过(1.6s) / 过(1.6s) / 死** |

- 死锁边界: **prefill ≤ ~200 token（单 KV block 内）正常，≥ 400（跨 256-tok block）死锁**——与 block 边界高度吻合但未确认因果
- 排除项: GEN 档位无关（1/3 均死）、STREAM_ASYNC/DROP_BROAD_SYNC 无关（0/1 均死）
- 死锁机制（日志证据）: stream 的 MTP device-state fallback（`accept/propose_tokens_gpu_missing`，
  fake-prefill draft sampler 无 token_ids 所致，`success_count=0` 结构性降级）→ legacy verify 路径
  → 全 8 rank `MtpExecutor::process` 的 sp metrics 同刻停止（如 09:11:47→48，len=1000 stream
  seq_len=1005 fallback 后）→ 集合通信死等。paris（14 tok，同样 fallback 但降级普通 decode）可完成
- 原始证据: `logs_d2/engine.log.hang_len1000_0911`（8.7MB）+ `rank2_gdb_bt_hang_0911.txt` +
  `logs_d2_exp/engine.log.expB_hang400`（实验 A/B 归档）

### 2.2 MTP 功能侧数据（短 input 验证有效，供参考不可比）

| 指标 | 80 TP8-MTP | 75 rtp-llm（R6 轮，input ≤ 20 tok） |
|---|---|---|
| accept length | 2.702 | 不可比（input 口径不同） |
| 接受率 | 0.851 | token/iter ≈ 2.1–2.4（GEN=1 上限 2；128 tok/61–69 iter，logs_serve_mtp_async/access_r0_s*.log） |
| bs=1 TPOT | — | 20.05ms vs 非 MTP 29.02ms（R6 修复验证轮） |

注: 75 的 MTP 为单 draft 层自回归复用（CONFIG_MAPPING §4.2）；80 为 3-token draft 树。
D2 并发口径接受率/吞吐无法测得——缺陷修复前 MTP 仅在 prefill ≤ ~200 token 场景可用。

## 3. Prefill 主口径（5 档 ×10 条, conc1, out1）

### Mean TTFT (ms)

| input_len | 80 TP8-MTP | 80 CP8 | 75 rtp-llm P1 (TP1/DP8/EP8) | Δ% vs 80 TP8 | Δ% vs 80 CP8 |
|---|---|---|---|---|---|
| 4K | 282.38 | 170.31 | 待填 | 待填 | 待填 |
| 8K | 539.28 | 284.40 | 待填 | 待填 | 待填 |
| 16K | 1083.62 | 548.72 | 待填 | 待填 | 待填 |
| 64K | 4966.19 | 2218.27 | 待填 | 待填 | 待填 |
| 100K | 8551.98 | 3617.77 | 待填 | 待填 | 待填 |

- 每档 10/10 成功才计入；失败档位如实记录。
- 75 无 CP 口径（PPU 未验证），仅 TP 口径对比两列。
- prefill 独立 server（serve_prefill.sh），与 D1/D2 分别重启。

## 4. 环境与公平性检查记录

| 项 | 值 |
|---|---|
| 75 镜像 | rtp_llm_ppu:0.2.0_0.2.0_2026_08_16_00_49_6a9733d36 |
| 75 代码 | a9812ec + 24 files diff（bench80 插桩 2 处，env-gated 默认关） |
| 每轮重启 | 是（kill_server -> serve -> health） |
| DeepGEMM JIT | 首启 ~25min 正常，复用 /ssd/1/dpskv4/dg_cache |
| 每卡显存基线 | D1 运行中：8×（145.4–147.2 GiB / 147456 MiB），rank 进程均 146.5 GiB（env/ppu_smi_d1_run.txt）；D2（MTP，RESERVER 12288）：无稳态快照（bench 未执行，死锁缺陷见 §2.1）；启动期 KV pool 79315MiB/4635 blocks/rank（engine.log 归档）；P1: 待快照 |
| 权重指纹 | config.json md5 30ee012da4d1fdf1446b2a5a28dee2f2, 109 shards, 292871459192 B |

## 5. 原始数据清单

见 CONFIG_MAPPING.md §5。
