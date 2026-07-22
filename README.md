# MixAttn

MixAttn 基于 FlashAttention-3，主要用于在 NVIDIA Hopper（SM90a）上实验和调优 MIX-WGMMA attention forward kernel。目前的自动调优程序位于 `hopper/cxx-tests/tune`。

## Tune 使用说明

### 运行环境

调优程序会为每个候选配置生成独立 CUDA 源码，通过 CMake 编译可执行文件，然后在 GPU 上运行 benchmark。因此需要：

- 支持 SM90a 的 NVIDIA Hopper GPU；
- CUDA Toolkit、CMake 3.26 或更高版本；
- 安装了 CUDA 版 PyTorch 的 Python/Conda 环境；
- 项目中的 `hopper/cutlass`、`hopper/custom_api` 等依赖完整可用。

CMake 按以下顺序查找 PyTorch 环境：

1. CMake 变量 `FA_VENV_DIR`；
2. 环境变量 `VIRTUAL_ENV`；
3. 环境变量 `CONDA_PREFIX`；
4. 项目上级目录中的 `.venv`。

建议先激活对应环境：

```bash
cd /home/xiebaokang/projects/cuda/test/microbenchmarks/MixAttn/hopper/cxx-tests/tune
conda activate <your-env>
```

### 启动调优

`tune.py` 当前提供 Python API，没有 argparse 命令行参数。可以修改文件末尾的 `tune(...)` 调用，也可以直接运行：

```bash
cd /home/xiebaokang/projects/cuda/test/microbenchmarks/MixAttn/hopper/cxx-tests/tune

python - <<'PY'
from tune import tune
from utils import DType, Mode

tune(
    shape=(1, 16, 32768, 64),  # (B, H, S, D)
    dtype=DType.FP16,          # DType.FP16 或 DType.FP8
    causal=False,
    num_consumer_limit=(1, 3),
    stage_limit=(1, 3),
    rank=32,
    jobs=16,
    mode=Mode.RADICAL,
)
PY
```

常用参数如下：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `shape` | 必填 | attention 形状 `(B, H, S, D)` |
| `dtype` | 必填 | `DType.FP16` 或 `DType.FP8` |
| `causal` | 必填 | 是否启用 causal mask |
| `smem_limit` | `232448` | 每个 CTA 的 shared-memory 筛选上限，单位为 byte |
| `reg_limit` | `262144` | 每个 CTA 的 register storage 筛选上限，单位为 byte |
| `num_consumer_limit` | `(1, 3)` | consumer warpgroup 数量范围 |
| `stage_limit` | `(1, 3)` | K/V pipeline stage 数量范围 |
| `bn_rate` | `0.6` | 每组候选保留的 BN 区间比例 |
| `rank` | `15` | 每个阶段保留/送入后续阶段的候选数量；第一阶段会从有效 base configs 中随机抽取该数量 |
| `jobs` | `16` | 并行编译任务数 |
| `arch` | `90a` | CUDA architecture |
| `coarse_register_usage_level` | `5` | 前三阶段传给 ptxas 的 register-usage level |
| `final_register_usage_level` | `10` | 最终阶段使用的 register-usage level |
| `benchmark_timeout_seconds` | `120` | 单个可执行文件的运行超时 |
| `mode` | `Mode.RADICAL` | `RADICAL` 使用峰值 live-range 资源模型；`KEEP` 使用更保守的累加模型 |
| `src_dir` | `tune/src` | 生成源码和 build tree 的目录 |
| `result_dir` | `tune/results` | 最终 JSON 结果目录 |

### 调优流程

调优分为四步：

1. `stage 1 / base structure`：生成并测试 BM、BN、stage、P/Q SMEM-register 划分和 consumer 数量。
2. `stage 2 / execution schedule`：调优 `use_scheduler_barrier` 和 `rescale_o_before_gemm`。
3. `stage 3 / register allocation`：调优 producer/consumer 的 `setmaxnreg` 分配。
4. `final / full register usage`：以最终 register-usage level 重新编译并测试胜出配置，写入结果 JSON。

中间阶段会保留相同 base structure 中 TFLOPS 最高的配置。编译警告、失败配置和 benchmark 超时会记录下来，不会进入下一阶段。

### 配置名称

生成的 executable 名称包含完整配置，例如：

```text
bm256_bn96_st3_prd24_cra240_p6_q0_nc2_sb0_rb0
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `bm` / `bn` | Q tile 的 M 大小和 K/V tile 的 N 大小 |
| `st` | K/V pipeline stage 数 |
| `prd` | producer warpgroup 释放后的寄存器数 |
| `cra` | 每个 consumer warpgroup 分配的寄存器数 |
| `p` | P 放入 SMEM 的 MMA-K tile 数；FP16 每 tile 16 列，FP8 每 tile 32 列 |
| `q` | Q 放入寄存器的 MMA-K tile 数 |
| `nc` | consumer warpgroup 数量 |
| `sb` | 是否使用 warp scheduler barrier |
| `rb` | 是否在 GEMM 前 rescale O |

### FP8 P SMEM/寄存器混合路径

FP8 和 FP16 都会生成完整的 P prefix 候选：

```text
p_smem_k_tiles = 0 ... BN / MMA_K
```

其中 FP8 的 `MMA_K=32`，FP16 的 `MMA_K=16`。因此在当前
`BN <= 256` 的搜索范围内，FP8 的 `p` 最大为 8：

```text
p = 0       ：P 全部保留在寄存器中，PV 使用纯 RS WGMMA
0 < p < max ：P prefix 使用 SS，剩余部分使用 RS
p = max     ：P 全部写入 SMEM，PV 使用纯 SS WGMMA
```

FP16 使用 STSM 将寄存器中的 P 写入 SMEM。Hopper 的 STSM
不能直接表达 FP8 的字节布局，因此 FP8 使用独立的 SIMT store 路径：

1. 将 QK accumulator 重排为 PV operand A 的线程所有权；
2. 将 FP32 softmax 结果转换为 FP8；
3. 每个线程按照 `partition_A(identity)` 得到的坐标写入 swizzled SMEM；
4. 在 SS-PV 前完成 shared-memory fence 和 warp-local 同步。

FP8 P prefix 会根据 V 支持的布局分解为一个或多个 SW128、SW64 和
SW32 区域，每个区域使用独立的 SMEM descriptor。例如：

```text
p=1：SW32
p=2：SW64（或 V 仅支持 SW32 时使用 SW32）
p=3：SW64 + SW32（或完整 SW32 区域）
p=4：SW128（或按 V 支持的较窄 swizzle 分解）
```

配置生成器会为 FP8 枚举 `range(BN // 32 + 1)`，随后再根据 SMEM、
寄存器容量及其他资源约束过滤候选。旧 result JSON 和已经生成的 CUDA
源码不会自动加入新的 `p` 候选。修改候选生成规则后应重新运行 tune；如果
只修改了 kernel，可以在验证旧结果时使用 `--recompile` 重新构建原有配置。

`ptxas` 的 `insufficient register resources` / `Potential Performance Loss`
表示 WGMMA 可能被串行化，是性能警告而不是数值错误。当前调优流程会将
这类目标记录为 `performance_warning`，并排除在成功候选之外；仍可直接对
生成的 executable 使用 `--verify` 检查正确性。

MIX-WGMMA repeated-M 路径满足：

```text
ComputeM = num_consumer * 64
RepeatM  = BM / ComputeM
```

因此 BM 必须是 `num_consumer * 64` 的正整数倍。

### BM 上限

Hopper 单个 TMA tensor-map box 的 M 方向不能超过 256。Q load 和 O store 对更大的逻辑 BM 使用两条 TMA：前缀 256 行，加上最多 256 行的尾部，因此 config 生成器限制：

```text
BM <= 512
```

第二、第三阶段也会过滤旧结果中的 `BM > 512` 配置。例如 BM384 会分别使用 `256 + 128` 两条 Q-load TMA 和两条 O-store TMA；两条 Q load 共用同一个 transaction barrier，消费者仍在完整 Q tile 就绪后开始计算。超过 512 目前仍会在编译期拒绝；若要继续扩大，需要推广为三条及以上 TMA。

### 输出文件

默认目录结构为：

```text
hopper/cxx-tests/tune/
├── src/test_b<B>_h<H>_s<S>_d<D>_<dtype>_<mode>/
│   ├── cuda_source/       # 每个 config 生成的 .cu
│   └── build/             # 按 register-usage level 划分的构建目录和 executable
└── results/
    └── bench_result_test_b<B>_h<H>_s<S>_d<D>_<dtype>_<mode>.json
```

结果 JSON 包含 shape、dtype、causal、成功/失败数量、排名、完整 `TConfig`、executable 路径、耗时、TFLOPS，以及估算的 shared-memory/register 占用。

### 正确性验证

使用 `verify.py` 将调优 kernel 与 LibTorch `scaled_dot_product_attention` 对比。默认验证排名第一的配置，并复用 JSON 中记录的 executable：

```bash
python verify.py \
  results/bench_result_test_b1_h16_s32768_d64_fp16_noncausal.json
```

验证所有保留配置：

```bash
python verify.py \
  results/bench_result_test_b1_h16_s32768_d64_fp16_noncausal.json \
  --all
```

源码修改后强制重新编译：

```bash
python verify.py <result.json> --all --recompile --jobs 16
```

测试不同序列边界或 tail mask：

```bash
python verify.py <result.json> --all --seqlen 30720 --jobs 16
```

其他常用选项：

```text
--index N                 只验证 top_results 中第 N 个配置（从 1 开始）
--seed N                  随机种子，默认 1234
--atol X / --rtol X       覆盖误差阈值
--timeout SEC             单个验证程序超时，默认 120 秒
--register-usage-level N  重新编译时的 ptxas level，默认 10
--build-dir PATH          指定验证构建目录
```

FP16 默认使用 `atol=0.02, rtol=0.02`，FP8 默认使用 `atol=0.1, rtol=0.1`。

### 常见问题

- CMake 找不到 Torch：确认已经激活正确环境，或配置 `FA_VENV_DIR` 指向包含 PyTorch 的环境。
- 编译出现 `Potential Performance Loss`：该目标会记录为 performance warning，并从成功候选中排除。
- benchmark 超时：先对相应 executable 执行 `--verify --seed=1234`，区分单次 kernel 问题和 benchmark 性能问题。
- 修改 kernel 后验证仍使用旧结果：添加 `--recompile`；只添加 `--all` 会优先复用 JSON 中已有 executable。
