# CONFIG_MAPPING: 80 (SGLang) ↔ 75 (rtp-llm) 参数对照

> 75 节点 rtp-llm vs 80 节点 SGLang 同口径对比测试的配置映射。
> 「实际生效值」以 engine.log 启动期日志为准（见 §3 日志证据）。

## 1. 负载口径（sglang bench_serving，逐字一致）

| 参数 | 80 值 | 75 值 | 一致性 |
|---|---|---|---|
| backend | openai-chat (vllm-chat) | vllm-chat | 一致（同为 openai chat completions 协议） |
| endpoint | /v1/chat/completions | /v1/chat/completions | 一致 |
| Decode: max-concurrency | 128 | 128 | 一致 |
| Decode: random-input-len | 4000 | 4000 | 一致 |
| Decode: random-output-len | 1500 | 1500 | 一致 |
| Decode: random-range-ratio | 1 | 1 | 一致 |
| Decode: num-prompts | 320 | 320 | 一致 |
| Prefill 档位 input-len | 4000/8000/16000/64000/100000 | 同 | 一致 |
| Prefill: output-len / concurrency / prompts | 1 / 1 / 10 | 同 | 一致 |
| bench 客户端版本 | sglang 0.5.13（80 完整安装） | sglang v2.1.0 内部 fork（git 1568a59，本地源码 + venv） | **客户端差异声明**：PyPI 无 0.5.13（最高 0.5.10.post1），回退 75 本机 sglang 源码树（与 80 客户端同源内部版系）。bench_serving 负载生成逻辑一致 |
| random dataset 采样 | 0.5.13：均匀随机整数 token | v2.1.0 fork：从 ShareGPT_V3 采样自然语言 token（repeat/truncate 到目标长度） | **输入 token 总数逐字一致**（Decode 1,280,000 / Prefill 按档位×10），但 token 分布不同（自然语言 vs 均匀随机）→ MoE routing 分布可能不同，结果解读时注明 |
| random 数据源 | 内置生成 | `--dataset-path /ssd/1/dpskv4/dataset/ShareGPT_V3_unfiltered_cleaned_split.json`（fork 默认从 HF 下载同款文件 `anon8231489123/ShareGPT_Vicuna_unfiltered`，离线环境改为本地注入，零口径差异）+ HF_HUB_OFFLINE=1 |

## 2. 服务端配置映射

### 2.1 并行拓扑（结构性差异，无法对齐）

| 项 | 80 SGLang | 75 rtp-llm | 说明 |
|---|---|---|---|
| Decode 主口径 | TP8（+MTP） | TP1 / DP8 / EP8 | PPU FlashMLA 稀疏 kernel 要求 h_q%64==0，DSV4 64 头不可切分 → TP 必须=1 |
| Prefill 口径 | CP8（另有 TP8 对照） | TP1 / DP8 / EP8 | rtp-llm 本树 PPU prefill CP 未验证（P2 待定项） |
| experts/卡 | 256/8=32 | 256/8=32 | EP 粒度一致 |
| KV 池拓扑 | 单池 max_total_num_tokens=3,260,160（TP8 全局） | 7 池 HybridPool per-rank（DSV4CacheConfigHelper） | 详见 §3 换算 |

### 2.2 量化 / KV

| 项 | 80 值 | 75 rtp-llm env | 实际生效 |
|---|---|---|---|
| 权重量化 | W8A8-INT8（compressed-tensors） | checkpoint config.json 自动检测 | 同一 ckpt（md5 30ee012d...） |
| KV dtype | FP8（框架默认） | `FP8_KV_CACHE=1` | engine.log AttentionFP8 |
| block tokens | 64（sglang page） | 256 物理 / 64 kernel（DSV4 helper） | 池容量换算见 §3 |
| 上下文长度 | 131072 | D1/D2：未设 MAX_SEQ_LEN（engine 默认 8192，负载峰值 4000+1500=5500 < 8192，结果有效）；P1：显式 `MAX_SEQ_LEN=131072`（100k 档必需，否则 511_LONG_PROMPT_ERROR） | 详见 §4.9 |

### 2.3 性能开关（75 侧）

| env | 值 | 作用 | 80 侧对应 |
|---|---|---|---|
| `RTP_LLM_STREAM_ASYNC=1` | decode 异步 dispatch overlap | TPOT 46→35ms | sglang 内建 overlap |
| `RTP_LLM_DROP_BROAD_SYNC=1` | 去除宽同步 | 与上项配套 | — |
| `ENABLE_CUDA_GRAPH=1` | decode CUDA graph | capture success 日志核对 | `--cuda-graph-max-bs`（80 踩坑#1） |
| `DECODE_CAPTURE_CONFIG` | D1="1,2,4,8,12,16,20,24,32"；D2="1,2,4,8,12,16,20"（MTP draft+verify 双 graph 显存更高，32 档 smoke 即 OOM；20 已覆盖 per-rank 峰值 16-19） | 图 bs 覆盖 per-rank 峰值（conc128/DP8≈16-19）；80 踩坑#1 对策 | 80: cuda-graph-max-bs 覆盖 128 |
| `USE_DEEPEP_LOW_LATENCY=1` | DeepEP LL（masked GEMM） | MoE dispatch 低延迟 | sglang deepep low-latency mode |
| `WARM_UP=1` | 启动 warmup | — | — |
| `RESERVER_RUNTIME_MEM_MB` | D1=4096（实际生效 7372——MemoryEvaluationHelper 有内部下限）；D2=12288（OOM 修复链：8192 边际仅 820MiB 仍 OOM；12288 真增量 +4.9GB，KV pool 缩至 79315MiB/4635 blocks，实测峰值使用率仅 0.8%，容量 118万tok/rank 仍为 80 per-GPU 的 2.9×） | 预留运行时显存 | — |
| `CONCURRENCY_LIMIT=160` | 全局并发上限（conc128 + 余量） | — | sglang `--max-running-requests 128` 等效语义 |
| `RTP_LLM_KVC_LOG_INTERVAL_SEC=5` | bench80 插桩①：kvc raw 日志 5s（默认 180 不变） | 运行时指标 | 80: extract_runtime_metrics.py 每秒轮询 |
| `RTP_LLM_SP_METRICS_LOG_SEC=5` | bench80 插桩②：SP 接受率日志 5s（默认 0=关） | D2 MTP | 80: sglang server metrics |
| D2: `SP_TYPE=mtp` `GEN_NUM_PER_CIRCLE=3` `SP_MODEL_TYPE=deepseek_v4_mtp` `SP_CHECKPOINT_PATH=$MODEL_DIR` | ≈steps2/topk1/draft3（单 draft 层自回归复用） | MTP 对照组 | 80: num_speculative_steps=2, speculative_eagle_topk=1, num_draft_tokens=3 |
| D2: `RTP_LLM_DEVICE_INPUT=1` | MTP 输入张量驻留 GPU（否则 ensureModelInputsOnCuda 空转，device fast path 静默回退 host assembly）——L0 对齐必需（mtp_gap_analysis.md R1） | MTP 对照组 | sglang 内建 |
| D2 LL buffer | max_generate_batch_size×(GEN+1)（CONCURRENCY_LIMIT=160 时足够 conc128） | — | sglang spec num draft tokens buffer |

## 3. KV 池容量与等价 max_total_num_tokens 换算（75 侧）

engine.log `DSV4 pool desc`（每 rank，7 池）：

| gid | region | type | layers | tokens_per_block | blocks/rank | 说明 |
|---|---|---|---|---|---|---|
| 0 | CSA_KV | FULL(paged) | 21 | 256 | 5513 | paged 主 KV |
| 1 | HCA_KV | FULL(paged) | 20 | 256 | 5513 | paged 主 KV |
| 2 | INDEXER_KV | FULL(paged) | 21 | 256 | 5513 | 索引 KV |
| 3 | INDEXER_STATE | SWA/FIXED | 21 | 256 | 5513 | 固定 slot 状态 |
| 4 | CSA_STATE | SWA/FIXED | 21 | 256 | 5513 | 固定 slot 状态 |
| 5 | HCA_STATE | SWA/FIXED | 20 | 256 | 5513 | 固定 slot 状态 |
| 6 | SWA_KV | SWA/FIXED | 43 | 256 | 5513 | 滑窗 KV |

- 80 口径 full KV ≈ gid0+1+2（paged token 容量 3×5513×256 ≈ 4.23M tokens/rank）
- 80 口径 SWA KV ≈ gid6（容量同上但语义为滑窗 ring）
- 80 的 max_total_num_tokens=3,260,160 为 TP8 全局单池口径；75 为 per-rank 7 池，报告按「paged token 容量 per-rank ×8」与「usage 峰值」对比

## 4. 无法对齐项（报告单列）

1. **并行拓扑**：80 TP8（decode）/CP8（prefill）↔ 75 TP1/DP8/EP8。TP>1 在 PPU 上不可行（FlashMLA h_q 约束）；DP8 意味着每 rank 独立 batch、LL buffer / KV 池独立。
2. **MTP**：75 ckpt 仅 1 个 MTP 层（num_nextn_predict_layers=1），GEN=3 需单层自回归复用 → 接受率随深度衰减；80 为标准 3-token draft 树。开关对齐（STREAM_ASYNC/DEVICE_INPUT 等，见 mtp_gap_analysis.md）+ R6 fake-prefill 投票修复后 MTP 已转正收益（bs=1 TPOT 20.05ms vs 非 MTP 29.02ms，短 input 场景）；**但 D2 轮发现 MTP + prefill ≥ ~400 tok（跨 256-tok block）触发全 rank 主循环死锁**（GEN=1/3、STREAM_ASYNC=0/1 均复现，≤200 tok 正常）——并发 bench 口径无法执行，详见 RESULTS.md §2.1 缺陷证据链。历史 R6 验证轮仅覆盖 input ≤ 20 tok，未暴露此缺陷。
3. **CP**：80 CP8 prefill ↔ 75 未验证（prefill_cp_config/PREFILL_CP env 在 PPU 上无验证记录），P2 轮视时间验证。
4. **chunked prefill**：rtp-llm 无对应参数；80 有 chunked_prefill_size。
5. **KV dtype**：80 主口径框架默认 FP8；75 需显式 FP8_KV_CACHE=1（一致性达成，但机制不同）。
6. **端点**：rtp-llm 无 /v1/completions → bench 走 /v1/chat/completions（80 openai-chat 同为 chat 端点；但 80 报告的 decode 口径走 /generate 裸文本时，输入 token 计数有 chat template 开销差异，本文档在结果中注明）。
7. **前缀缓存**：rtp-llm 当前未启用 → cache 专项 C1/C2 免做。D1 双证据实证：响应字段 reuse/cached 全 0 + 前缀对延迟无加速（prefix-1st 431.3ms vs prefix-2nd-same 431.5ms），见 results/d1_nostream/reuse_probe.json。
8. **调度**：80 max_running_requests 精确 128；75 CONCURRENCY_LIMIT=160 是前端准入上限，实际 running 由 KV/调度决定（monitor 序列实证）。
9. **max_seq_len**：rtp-llm 未设 MAX_SEQ_LEN 时 engine 默认 8192（model_config.py:847），而 ckpt 名义 131072。D1/D2 负载峰值 5500 未受影响；P1 100k 档需显式 MAX_SEQ_LEN=131072。80 侧 sglang 默认取 ckpt 值。

## 5. 原始数据路径

- serve/engine 日志：`/ssd/1/dpskv4/dsv4_work/bench80/logs_{d1,d2,p1}/`；多轮共用 LOGDIR 的归档变体：`logs_d1/engine.log`（stream 轮）+ `engine.log.d1_natural`（重复性轮，含启动期 pool/capture 元数据）+ `engine.log.d1_nostream`（nostream 轮）；`logs_d2/engine.log.hang_len1000_0911`（MTP 死锁缺陷证据，8.7MB）+ `rank2_gdb_bt_hang_0911.txt`（gdb 栈）+ `serve.log.hang_len1000_0911`；实验轮 `logs_d2_exp/`（实验 A/B: `engine.log.expB_hang400`、`serve.log.expB`）；D2 无 bench 轮（缺陷无法执行，见 RESULTS §2.1）
- bench 输出：`/ssd/1/dpskv4/dsv4_work/bench80/results/{d1,d1_natural,d1_nostream,d2,d2_nostream,p1}/bench_*.json{,l}` + stdout
- 运行时指标：`results/*/runtime_metrics.json`、`worker_status_series.jsonl`、`reuse_probe.json`（probe 走 /v1/chat/completions——rtp-llm 无 /generate 端点；早期 404 版已废弃）
- 环境快照：`/ssd/1/dpskv4/dsv4_work/bench80/env/`（镜像/commit/权重指纹）
- 脚本源（staging）：`RTP-LLM/bench80_staging/` → 容器内 `sync_to_ssddir.sh` 同步至 /ssd
