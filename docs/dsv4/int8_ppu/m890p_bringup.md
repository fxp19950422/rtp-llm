# DeepSeek-V4-Flash W8A8-INT8 on PPU **M890P** — bring-up 结果

> 日期 2026-08-17　设备 8×ZW-M890P（SM8.9, 144GiB/卡）　镜像 cu130 / torch 2.9.0 / CUB(CCCL) 3.0.1
> 前置：[README.md](README.md) · [progress.md](progress.md) · [next_steps.md](next_steps.md)
>
> 一句话：**端到端打通**。8 卡 serve 拉起（start 49s）、`1+1=?`→`The answer is 2`、
> `capital of France`→`Paris`、俳句/数学推理均相干正确，**采样器 514 NaN/inf 未复现**。
> 即 next_steps.md 的 P0 是 ZW810E（SM8.0/无 `_scaled_mm`/大量 torch 回退）特有，M890P 上不出现。

## 1. M890P vs ZW810E 关键差异（实测）

| 维度 | ZW810E | M890P |
|---|---|---|
| 卡 | 16×96GiB SM8.0 | **8×144GiB SM8.9** |
| 拓扑 | TP1/DP16/EP16（16 exp/卡） | **TP1/DP8/EP8**（32 exp/卡） |
| SDK/torch | v2.1.0 / torch2.6 / CUB2.x | **cu130 / torch2.9 / CCCL3.0.1** |
| `torch._scaled_mm` | ✗（要 SM≥8.9） | **✓** |
| triton `float8e4nv` | ✗ | **✓** |
| deep_gemm int8 族 | ✓ | ✓（`int8_paged_mqa_logits` 等） |
| tilelang | ✗ | ✗（mHC 仍走 torch 回退） |
| 656B slot 宽度 | 656B | **656B 不变**（`verify_fp8_656_correct`/`slots_656`/`swa_dequant_656` 均 PASS，padding 已清零） |
| serve 启动 | ~1205s | **49s**（模型在本地 /ssd + 8 卡） |
| 首 token | ✗（P0 NaN） | **✓ 相干正确** |

warm 基线：TTFT≈0.9–1.1s，decode≈2–2.7 tok/s（torch 回退下限）。

## 2. 为在 M890P 上把 dsv4 分支编起来/跑起来所做的改动（均未提交）

代码（`github-opensource` 子模块 & `internal_source`）：
- `internal_source/deps/pip.bzl` — aliyun 主索引（artlab 走隧道会截断），加 find-links/retries。
- `github-opensource/3rdparty/cub_compat.h` — CUB_VERSION 守卫（CCCL3.x 才 alias `cub::Max/Min/...`）。
- `github-opensource/3rdparty/cuda_config/cuda_configure.bzl` — 注释触发 `@local_config_cuda` 重生成（CUDA13 cccl→flat 头文件）。
- `github-opensource/WORKSPACE` — xgrammar remote 由不可达 gitlab 改到可达 `code.alibaba-inc.com`（同 commit 557becfb），并去掉 submodule `--depth=1`（浅拉取钉住提交会失败）。
- `github-opensource/BUILD` — 新增 `using_ppu_cuda` config_setting（`using_cuda` 的严格超集，避免与 `using_ppu` 的 select 歧义）。
- `github-opensource/BUILD.pytorch` + `internal_source/BUILD.pytorch` — PPU(CUDA13) 分支不链已被删除的 `-lnvToolsExt`（NVTX 头-only、SDK 的 libnvtx3interop 软链断裂、torch2.9 无 NVTX DT_NEEDED）。
- `internal_source/.internal_bazelrc` — `build:ppu` 的 `TF_CUDA_VERSION` 12.6→13.0。
- `internal_source/deps/http.bzl` + `internal_source/bazel/arch_select.bzl` — 新增 `torch_2.9_py310_ppu`（cu130 PPU wheel，sha 校验），PPU torch 2.6→2.9（ABI 必须与运行时一致，否则 `librtp_compute_ops.so` undefined c10:: 符号）。
- `internal_source/deps/git.bzl` — 给 `flashinfer_ppu` 加 CUDA13 可见性补丁 0014/0015 + `patch_cmds` 把 `cub::Max/Min()`→`cuda::maximum/minimum<>()`（CCCL3 删了 `cub::Max/Min`）。
- `github-opensource/rtp_llm/models_py/modules/dsv4/fp8/_indexer_score.py` + `dsv4_kernel_jit_warmup.py` — `fp8_mqa_logits` 按 wheel 实际 arity 调用（M890P wheel 去掉了尾参 `max_seqlen_k`）。

环境/资产（非 git）：
- 容器 `/usr/local/PPU_SDK/CUDA_SDK/include` 建 `cccl/{cuda,cub,thrust}`→扁平软链（镜像缺，nvidia.Dockerfile 契约）。
- 容器 conda 装 `jsonschema`（运行时缺）。
- `/ssd/1/dpskv4/model/encoding/encoding_dsv4.py` — 从 sglang 同源脚本装入（模型下载漏了 encoding/ 目录，chat renderer 必需）。
- `dsv4_work/dsv4_env_m890p.sh`、`dsv4_work/run_serve_m890p.sh`（8 卡）、`dsv4_work/probe_m890p_caps.py`。
- 产物备份 `/ssd/2/artifacts/rtp_llm-0.2.0-dsv4int8-cu130-torch2.9-a9812ec-M890P.whl`（sha 校验）。

## 3. 下一步（B10 优化，端到端已打通后）

890P 补齐了 810E 缺的能力，可把「为在 810E 跑通而写的 torch 回退」换回原生 kernel，直接提 decode 吞吐。参考 sglang 的 DSV4 实现：
- **`_scaled_mm` 可用** → FP8 线性/输出投影从 torch 回退切 `torch._scaled_mm`（`_is_ppu_device` 分支开关）。
- **triton `float8e4nv` 可用** → compressor 写入、indexer score、656B pack/dequant 的 triton fp8e4nv kernel 打开（progress.md 列的 torch 回退热点）。
- **`cublas_gemm_bf16_bf16_fp32`** 回退（a9812ec）在 890P 上确认是否可用真 kernel。
- 打开 `ENABLE_CUDA_GRAPH`（decode capture）与 MTP 投机（P2）。
- 精度定量基线：按 progress.md §3「hash 路由层严格对齐 + 打分路由层看分布/困惑度」对齐 sglang。

### 3.1 已完成：`_is_ppu_device` 在 M890P 上恒为 False

`compressor.py:_is_ppu_device` 判据是 `"PPU" in props.name`，而 M890P 的 `props.name == "ZW-M890P"` **不含 `PPU` 子串**，因此 indexer score / compressor / 656B 的 PPU torch 回退分支（indexer.py:667/1129/1365、compressor.py:922）在 890P 上**从未被走到** —— 上表的 torch-回退替换实际已隐式生效，不需要改代码。这也是 P0 采样器 NaN 在 890P 不复现的原因之一。

### 3.2 已完成：DeepEP LOW_LATENCY + INT8 masked GEMM（decode 通信瓶颈）

**定位**：`py-spy record --gil` 抓 rank0 的 64-token decode，`deep_ep/buffer.py:dispatch` 单独占 **72.7%** GIL 时间，MoE 层合计 76.8% —— decode 是**通信瓶颈**，不是算力瓶颈。根因是 normal-mode dispatch 返回 `num_recv_tokens_per_expert_list` 是 **Python list**，即每个 MoE 层都有一次 device→host 同步，40 层串起来吃掉整个 decode step。

**改动**（三处，均未提交）：
1. `deepep_wrapper.py:calc_low_latency_max_token_per_rank` 的量化白名单加 `INT8_PER_CHANNEL_COMPRESSED`（原先只有 FP8 系列 + W4A8_INT4 + fp4，INT8 直接 `raise ValueError("Unsupported quantization config")`）。cu130 whl **原生支持 INT8**：`Buffer.low_latency_dispatch` 有 `use_int8`/`quant_size` 形参，`deep_gemm` 有 `m_grouped_gemm_int8_int8_bf16_nt_masked` / `int8_paged_mqa_logits`。
2. 新增 `moe/strategies/deepep_low_latency.py` — `DeepEPLowLatencyStrategy`：`low_latency_dispatch` → 3 个 INT8 masked grouped GEMM → `low_latency_combine`。per-expert Python 循环被 masked 布局取代，recv 计数以**设备张量**直接喂 `masked_m`，无同步。
3. `moe/strategies/base.py`：PPU（无 `fp8_fp4_mega_moe`）的 EP 回退顺序改为 `("deepep_low_latency", "deepep")`，并允许显式 force 这两个做 A/B。

**kernel 契约（实测确定，勿凭猜改）**——`dsv4_work/probe_int8_masked_gemm.py` 扫了 16 种组合，只有一种被接受，其余全 `AssertionError`：

```
lhs      = ([E,M,K] int8, [E,M,1] fp32)   # per-token 激活 scale
rhs      = ([E,N,K] int8, [E,N,1] fp32)   # per-output-channel 权重 scale
out      =  [E,M,N] bf16
masked_m =  [E] int32
out[e,m,n] = (Σ_k lhs[e,m,k]*rhs[e,n,k]) * lhs_s[e,m] * rhs_s[e,n]
```

`rhs` 这个 layout **就是 DSV4 checkpoint 的原生 layout**（`W.v4_routed_w*_w` 已是 `[E_local, out, in]` int8 + `[E_local, out, 1]` fp32），所以路由权重原样喂进 kernel，**不需要重量化/重排**。`probe_int8_masked_numerics.py` 另外验证了：dequant 约定 cosine 0.999999、`masked_m` 之后的行**不被写**（故 down-proj 输出 slab 必须预先 zero）、per-token 对称量化 vs bf16 参考 cosine 0.999944。

**两个与 `DeepEPStrategy` 的语义差异（踩过）**：
- **不能 pad topk**。normal 路径把 V4 的 topk=6 垫到 8（`intranode.cu` 只 switch `{2,4,8,16}`），但 LL buffer 是按 `moe_k` sizing 的，垫宽会让 dispatch 和 combine 失配 → 改为断言宽度等于 `wrapper.num_topk`。
- **router weight 由 `low_latency_combine` 施加**，不是像 `Expert.forward` 那样在 down-proj 之前乘。两边都乘会把权重平方。

**实测收益**（8 卡 EP8/DP8，`WARM_UP=1`）：

| 指标 | NORMAL | LOW_LATENCY | 变化 |
|---|---|---|---|
| GIL 样本总数（20s@200Hz） | 2413 | 667 | **−72%** |
| `deep_ep` dispatch 占 GIL | **72.7%** | **1.8%** | −70.9pp |
| MoE 层合计占 GIL | 76.8% | 15.0% | −61.8pp |
| attention 占 GIL | 9.9% | 38.8% | 瓶颈已转移 |
| decode tok/s（16–64 tok） | 2.43–2.73 | **2.82–2.97** | **+9%～+19%** |
| warm TTFT | 900–1100 ms | **610–750 ms** | **−30%** |

生成质量复验通过（Paris / 三原色 / 俳句 / 45min-60km 换算均正确，无 NaN）。

> **调试记录**：首版输出全是乱码。原因是 scratch buffer 缓存键只有 `(device, shape, dtype)`，而 `gate` 与 `up` 形状 dtype 完全相同 → 拿到同一块 tensor，w3 的 GEMM 覆盖了 w1 的结果，等价于 `silu_mul_split(up, up)`。缓存键加 `slot` 名后即恢复正常。**任何按形状缓存复用 buffer 的地方都要带一个身份标签。**

### 3.3 ⚠️ 更正：3.2 的 LL 收益只在 decode 成立，prefill 会崩

上面 3.2 的测量全部用的是短 prompt（≤32 token），因此遗漏了一个**严重缺陷**：

```
Assertion error deep_ep.cpp:1150
  'x.size(0) == topk_idx.size(0) and x.size(0) <= num_max_dispatch_tokens_per_rank'
```

LL buffer 是按 `ll_num_max_token_per_rank`（当前 **32**）预分配的，DeepEP 在 C++ 层硬断言，**超出直接 abort 整个 rank**（不是抛异常）。decode 天然在预算内，prefill 不是 —— 59 词的 prompt 就把 8 卡服务打挂了。通用 router（`deepep_low_latency_router.py:prepare`）有这个容量断言，我漏了。

**分块不是解法（已实测否定）**。把超容量的 token 拆成 cap 大小的多轮 dispatch，确实不再 abort，但：dispatch 是**集合操作**，而 DP=8 下每个 rank 带的 token 数不同 → 各 rank 算出的轮数不一致 → 最长 rank 多出来的那几轮没有对端。实测：同一个 59 词 prompt，TTFT **43766 ms**（集合超时）且输出损坏（先输出无关内容再复述 prompt）。对照 NORMAL 模式同 prompt：TTFT **1250 ms**、摘要正确。

**正确的切分是按阶段，不是按分块** —— Qwen3.5 PPU 就是这么做的，它同时注册两个 strategy：

| strategy | router | executor |
|---|---|---|
| `w8a8_int8_ep_low_latency_deepgemm` | `deepep_low_latency_int8_router` | `deepgemm_int8_masked_executor` |
| `w8a8_int8_dp_normal_deepgemm` | `deepep_normal_router` | `deepgemm_int8_hybrid_executor` |

**当前状态**：`USE_DEEPEP_LOW_LATENCY=0`（服务跑 NORMAL，已验证长短 prompt 均正确）。`DeepEPLowLatencyStrategy` 保留并在超容量时**明确报错**（而不是 abort 或返回错数据），等待接入分阶段选择后才能上线。

### 3.4 从 GLM-4.7 / Qwen3.5 PPU（均 ZW810E, SM8.0）可直接千拿的东西

两个项目都已在 810E 上做完了我们正在做的事，代码在：

- `origin/feature/qwen35-ppu-w8a8-int8`（INT8 W8A8 全栈）
- `origin/feature/glm47-ppu-rdma-mlx5-mtpfix-arenaenv`（PD 分离 + 容量调优）
- `origin/feature/rtp-ppu-base`（通用 PPU runtime）

**重要前提（用户强调）**：他们在 SM8.0，我们在 SM8.9。但实际查验的结果是：他们 `kernels/ppu/` 下的 Triton kernel **零 SM 能力分支**（无 `get_sm()` / 无 capability 守卫），因此可直接移植；需要警惕的不是 kernel 本身，而是他们“**因为 810E 不支持而没选的路**”（例如全程不碰 FP8 路径）——那些在 890P 上不应该继承。

按收益排序：

1. **`quant_size=hidden_size` 解开了 INT8 走线**（`deepep_low_latency_int8_router.py:132`）。我之前止步于“无法确认 dispatch emit 的 int8 scale layout”，实际答案是：`use_int8=True, quant_size=hidden_size`（**不是默认 128**）→ 整行 per-token scale，正好是 masked GEMM 要的 `[E,M,1]`。dispatch 字节数腰斩，并且省掉本地第一次量化。
2. **`per_token_quant_int8_masked(x, masked_m)`**（`kernels/ppu/int8_quant.py`）——带 `masked_m` 的单 Triton kernel，直接替掉我那个约 7 个 torch kernel 且会扫到垃圾行的纯 torch 量化器。它产出的 scale 形状 `(*shape[:-1], 1)` 独立印证了我探测出的 layout。
3. **`silu_and_mul_masked_per_token_quant_int8_fwd`**（`kernels/ppu/fused_silu_mul_int8_quant.py`）—— SiLU×mul + per-token INT8 量化融成一个 kernel，只遍历 `masked_m` 内的有效行。**不能直接用**：它没有 SwiGLU clamp，而 DSV4 靠 `clamp_limit=swiglu_limit`；且它假定 `up|gate` 拼在一个 `[E,M,2*inter]` 张量里，而 DSV4 的 w1/w3 是分开的。要写一个 DSV4 变体。
4. **他们只用 2 个 GEMM，我用了 3 个**。他们的 `w1` 是 `[E, 2*inter, hidden]`（up‖gate 拼接），一次 grouped GEMM 同时出 gate 和 up。DSV4 把 w1/w3 分开存 → 我做了两次。合计：他们 3 个 kernel（2 GEMM + 1 fused）vs 我 11+ 个。
5. **`configure_deep_gemm_num_sms` SM 预算**。M890P 单卡只有 **39 个 SM**（探针日志 `num_of_sm=39`），DeepEP 和 DeepGEMM 抢 SM 的问题对我们比对他们更尖锐。他们每个 masked GEMM 都包在 `with configure_deep_gemm_num_sms(...)` 里，我没做。
6. **CUDA graph 在 PPU 上是可行的，我之前的结论错了**。Qwen3.5-VL 在 PPU 上 `capture shape 1/2/4/8/16/32/64/128 均完成`，且是 **MTP + CUDA Graph + Reuse + DeepEP 同时开**。他们踩的坑（`CUDA Graph 动态 padding 元数据残留`、`PCCL CUDA Graph batch 2/4 首次 replay 非法访存`、`FA3 零长度 query / replay 卡死`）才是真正需要逐个解的；`sglang` 那句 `MLA seems not support piecewise graph on PPU` 只否定了 **piecewise** graph，不是整图。
7. **MTP 是 BS1 最大的杆**。Qwen3.5-VL 同口径：并发 1 的 TPS 145.3 vs 75.8（**decode 1.96×**）、并发 8 是 1.89×、并发 32 是 1.46×。我们现在就是 BS1 为主的场景。
8. **`DEEPEP_LL_NUM_MAX_TOKEN` 容量语义**（GLM commit `c9d9e0b0`）：LL buffer 随它**线性**预分配（2048 → 13757 MB/卡，和整个 decode KV pool 一样大），且 MoE dispatch chunk = 值 × TP_SIZE。我们现在 32 → 276 MB，在极小端；要让 prefill 能进 LL 就得抬这个值并接受显存代价。
