# MIX-WGMMA P/Q 驻留优化：调优结果与性能来源

## 1. 结论

本文分析当前 `tune` 目录中的 20 组实验：

```text
B=1, H=16, S=30720
D={64,96,128,192,256}
dtype={FP16,FP8}
causal={true,false}
```

数据来源：

- `results_cache/`：完整 tune 后保存的 top-30；
- `results/`：为缺失或不充分的 p0_q0 基线补跑的候选；
- `src/.../build/rl10/bin/`：上述 JSON 指向的最终二进制。

本文将“ours”定义为每个 workload 在两目录合并后最快的非零 p/q 配置，
将“baseline”定义为最快的 p0_q0 配置。所有 workload 现在都有 baseline。

综合 JSON、同环境复跑、资源模型和 SASS，结论如下：

1. **该优化有效，但不是通用默认优化。**  
   同环境两轮复测中，20 组有 16 组的均值为正，几何平均加速 1.0068x，
   中位数 1.0109x。真正稳定且超过约 2% 的案例主要是 D64 FP16
   non-causal、D64 FP8 non-causal、D192 FP16 causal/non-causal 和
   D192 FP8 non-causal。

2. **最初的“用一种片上资源换另一种，从而解锁更大 BN”目标确实实现了，
   最强证据是 D192 FP16。**  
   p0_q0 的 `BM128×BN128×stage2` 需要 248832 B SMEM，超过 232448 B
   上限；q4 恰好释放 16384 B，使其降至 232448 B。最终 baseline 的
   BN96 被提升到 BN128，同环境复测 causal / non-causal 分别快
   2.14% / 1.80%，SASS 也从 `HGMMA.64x96x16` 变成
   `HGMMA.64x128x16`。

3. **该方法还能与流水线调优直接协同。**  
   D192 FP8 non-causal 中，p0_q0 的 `BN128×stage3` 需要 248832 B
   SMEM；q5 释放 20480 B，使 stage3 合法。相对最优 p0_q0 stage2，
   复测快 2.33%。这不是更大 BN，而是用 SMEM→register 转移换取更深
   pipeline。

4. **D128 FP8 是“软解锁”案例。**  
   `BM192×BN128×stage3` 的 p0_q0 consumer register requirement
   恰好等于 245760 B budget；p3 将其降至 233472 B，换取 18 KiB
   P-SMEM。它不是从非法变合法，而是把 BN128 从 register cliff 拉开。
   最终 BN96→128，复测提升约 1.5%～1.7%。

5. **资源变均衡或 tile 变大本身并不保证更快。**  
   D96 FP16 同样用 p3 将 BN96 提升到 BN128，但复测没有收益；
   D128 FP16 causal 的 p8 虽能让 `BM192×BN128` 合法，最新 baseline
   却找到更快的 `p0_q0 BM128×BN192`，复测中 p8 反而慢约 6%。

6. **固定 BM/BN 时也存在独立的 p/q 收益。**  
   D64 FP16 non-causal 的完全同参 p1q2 相对 p0_q0 三轮均快，
   平均约 2.81%；D96 FP8 non-causal 的同参 q1 平均约 1.79%。
   SASS 中 GMMA 数量和 shape 不变，变化来自 SS/RS operand path、
   Q 的 LDG/PRMT、P 的 store 和 SMEM/register 峰值。这是实现后出现的
   第二类收益，不是“大 BN”收益。

因此，这项工作的最佳定位是：

> 一种由资源感知编译器选择的 mixed-residency WGMMA scheduling：
> 联合决定 P/Q 在 SMEM 与 registers 中的驻留比例，并与 BM、BN、
> pipeline stage、consumer 数和 setmaxnreg 一起调优。

## 2. p、q 的含义

FP16 的 `MMA_K=16`，FP8 的 `MMA_K=32`。

```text
p_smem_k_tiles:
    把 P 的前 p 个 MMA-K tiles 从 registers 转移到 SMEM；
    对应 PV 从 RS WGMMA 变为 SS WGMMA。

q_reg_k_tiles:
    把 Q 的后 q 个 MMA-K tiles 从 SMEM 转移到 registers；
    对应 QK 从 SS WGMMA 变为 RS WGMMA。
```

主要资源方向：

| 参数增大 | SMEM | registers | 新增代价 |
| --- | --- | --- | --- |
| p | 增加 | 通常减少 P persistent fragment | P convert/store、fence、sync |
| q | 减少 | 增加 persistent Q fragment | global LDG、PRMT/layout conversion |

每增加一个 q tile，Q SMEM 的原始容量减少：

```text
BM × MMA_K × sizeof(element)
```

每增加一个 p tile，P SMEM 的原始容量增加：

```text
PBlockM × MMA_K × sizeof(element)
```

实际总 SMEM 还受 swizzle、成员对齐和 union padding 影响。


## 3. 最优 ours 与最优 baseline

加速定义为：

```text
speedup = baseline_time / ours_time
```

| Case | ours：BM×BN/stage/pq | baseline：BM×BN/stage | ours ms | p0_q0 ms | 加速 |
| --- | --- | --- | ---: | ---: | ---: |
| D64 FP16 NC | 192×128/s3/p1q2 | 192×128/s3 | 11.727 | 11.976 | **+2.12%** |
| D64 FP16 C | 192×128/s3/p1q1 | 192×128/s2 | 5.762 | 5.763 | +0.02% |
| D64 FP8 NC | 192×128/s3/p0q1 | 192×160/s3 | 8.555 | 8.944 | **+4.54%** |
| D64 FP8 C | 192×128/s3/p0q1 | 192×128/s3 | 4.241 | 4.242 | +0.02% |
| D96 FP16 NC | 192×128/s3/p3q0 | 192×96/s3 | 16.054 | 15.790 | **-1.65%** |
| D96 FP16 C | 192×128/s3/p3q0 | 192×96/s3 | 8.422 | 8.155 | -3.17%（不稳定） |
| D96 FP8 NC | 192×128/s2/p0q1 | 192×128/s3 | 10.717 | 10.841 | +1.15% |
| D96 FP8 C | 192×128/s3/p0q1 | 192×128/s2 | 5.280 | 5.352 | +1.37% |
| D128 FP16 NC | 128×128/s2/p0q4 | 128×128/s2 | 19.765 | 19.928 | +0.83% |
| D128 FP16 C | 192×128/s2/p8q0 | 128×192/s2 | 11.745 | 11.046 | **-5.95%** |
| D128 FP8 NC | 192×128/s3/p3q0 | 192×96/s3 | 12.448 | 12.665 | +1.74% |
| D128 FP8 C | 192×128/s3/p3q0 | 192×96/s3 | 6.255 | 6.348 | +1.49% |
| D192 FP16 NC | 128×128/s2/p0q4 | 128×96/s2 | 29.084 | 29.607 | +1.80% |
| D192 FP16 C | 128×128/s2/p0q4 | 128×96/s2 | 17.788 | 18.170 | **+2.14%** |
| D192 FP8 NC | 128×128/s3/p0q5 | 128×128/s2 | 17.343 | 17.746 | **+2.33%** |
| D192 FP8 C | 128×128/s2/p0q1 | 128×128/s2 | 8.825 | 8.889 | +0.73% |
| D256 FP16 NC | 128×64/s2/p0q9 | 128×64/s2 | 39.975 | 40.383 | +1.02% |
| D256 FP16 C | 128×64/s2/p0q5 | 128×64/s2 | 24.817 | 25.662 | +3.40%（幅度不稳定） |
| D256 FP8 NC | 128×128/s2/p0q1 | 128×128/s2 | 21.086 | 21.242 | +0.74% |
| D256 FP8 C | 128×128/s2/p0q1 | 128×128/s2 | 11.006 | 10.947 | -0.54% |

### 测量限制

CUDA benchmark 可以直接运行，但 NVML 报 driver/library version mismatch，
无法读取或锁定 GPU clocks。长时间顺序运行时绝对时间有明显漂移，因此：

- 每个 benchmark 自身的 200 次 launch 平均仍有效；
- ours/base 相邻执行并反转顺序，降低了系统漂移影响；
- 大于 2%、两轮方向一致的结果可信度较高；
- 小于 1% 统一视为持平；
- D96 FP16 causal、D256 FP16 causal 等两轮差异较大的组需要在锁频后重测；
- 本表适合机制判断，不应作为最终发布数字。

## 4. 如何判断收益确实来自 p/q

“最优 ours 对最优 baseline”允许 BM、BN、stage、register allocation 和调度
开关都变化，它回答的是“加入该搜索维度后最终能否更快”，不能单独证明
p/q 的直接因果作用。

为此，又选择了 8 组已有完全同参 p0 sibling 的配置：

```text
BM、BN、stage、prd、cra、num_consumer、scheduler barrier、rescale
全部相同，只把 p/q 改成 0/0。
```

每对交错执行三轮。以下为三轮 `p0_time / pq_time` 的平均：

| Case | p/q | 同参加速 | 判断 |
| --- | ---: | ---: | --- |
| D64 FP16 NC | p1q2 | **+2.81%** | 三轮均为 +2.5%～+3.1%，强因果证据 |
| D64 FP8 C | p0q1 | +3.62% | 同参有效，但最优 p0 用另一 cra 后总体持平 |
| D96 FP8 NC | p0q1 | **+1.79%** | 三轮均为正，p/q 直接收益 |
| D96 FP8 C | p0q1 | -0.32% | 持平且方向不一致 |
| D128 FP16 NC | p0q4 | +0.05% | 完全持平；最优表中的 +0.83%主要来自其他参数 |
| D256 FP16 NC | p0q9 | +1.03% | 小幅直接收益 |
| D256 FP8 NC | p0q1 | +0.98% | 小幅直接收益 |
| D256 FP8 C | p0q1 | 不稳定 | 有异常值，不能归因 |

这说明：

- D64 FP16 NC、D96 FP8 NC 的收益可以明确归因给 p/q；
- D128 FP16 NC 的最优差值不能归因给 q4 本身；
- 某个 p/q 可以改善一个较差的 register allocation，但全局最优 p0 可能通过
  调整 cra 达到同样性能，例如 D64 FP8 causal；
- 因此 p/q 必须与 setmaxnreg 联合调优，并保留同参消融。

## 5. SASS：机器码到底改变了什么


### 5.1 固定 tile 的直接 p/q 收益

| Case | 版本 | GMMA | SS/RS | 静态指令 | LDG | PRMT | STS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D64 FP16 NC | p0q0 | 24 | 8/16 | 2056 | 5 | 4 | 4 |
|  | p1q2 | 24 | 6/18 | 2304 | 53 | 54 | 6 |
| D96 FP8 NC | p0q0 | 14 | 6/8 | 2352 | 7 | 52 | 18 |
|  | p0q1 | 14 | 4/10 | 2584 | 47 | 62 | 18 |
| D256 FP16 NC | p0q0 | 80 | 64/16 | 5248 | 10 | 8 | 32 |
|  | p0q9 | 80 | 28/52 | 6336 | 442 | 444 | 32 |

三个共同点：

1. GMMA 数量和 shape 不变；
2. 非零 q 将部分 SS-QK 改成 RS-QK，非零 p 将部分 RS-PV 改成 SS-PV；
3. 优化版本的静态指令、LDG 和 PRMT 反而更多。

因此，固定 tile 的性能来源绝不是“少做了计算”或“指令数减少”。额外的
load/repack/store 换来了不同的 operand delivery 和 live-range：

- Q 不再全部依赖 TMA→SMEM→SS-QK；
- P 不再全部作为 persistent register operand 进入 RS-PV；
- SMEM operand bandwidth、register operand bandwidth、barrier 等待和
  编译器 schedule 被重新分配；
- 哪条路径原本位于关键路径，决定了该交换是否获益。

没有 NCU 时，SASS 能证明路径变化，但不能进一步声称具体是哪一种 stall
下降。需要后续检查 eligible warps、MIO/shared throttle、long scoreboard、
GMMA issue 和 tensor pipe utilization。

### 5.2 更大 BN 的 SASS 证据：D192 FP16

最优 p0_q0：

```text
BM128 × BN96 × stage2
QK: HGMMA.64x96x16
```

q4：

```text
BM128 × BN128 × stage2
QK: HGMMA.64x128x16
```

non-causal 静态 SASS：

| 版本 | GMMA 总数 | 主要 shape | SS/RS | LDG | PRMT |
| --- | ---: | --- | ---: | ---: | ---: |
| BN96 p0q0 | 72 | 48×64x96x16 + 24×64x192x16 | 48/24 | 10 | 8 |
| BN128 q4 | 80 | 48×64x128x16 + 32×64x192x16 | 32/48 | 202 | 204 |

对完整 `S=30720`：

```text
BN96 需要 320 个 N tiles
BN128 需要 240 个 N tiles
```

每个 QK GMMA 的 N 宽度从 96 增加到 128，同样的 QK FLOPs用更少的
外层 N-tile iterations 完成。PV 每个 tile 的 K 分段增加，但乘以 tile 数后
总 PV 数学工作量不变。主要减少的是：

- QK GMMA issue 数；
- pipeline acquire/release；
- 在线 softmax 状态更新次数；
- mask、循环控制和调度摊销。

这就是“更大 BN”应有的机器码和循环层证据。

### 5.3 更大的数值 BN 不一定是更好的 WGMMA

D64 FP8 non-causal 是一个反直觉但很重要的例子：

```text
baseline: BN160, 30 条静态 QGMMA
ours:     BN128, 12 条静态 QGMMA
```

baseline 的 BN160 被分解出大量 `QGMMA.64x32x32`，而 BN128 使用原生
`QGMMA.64x128x32`。ours 虽然数值 BN 更小，复测却稳定快约 4.5%。

因此编译器不应只最大化 BN，而应最大化“有效 WGMMA shape”：

- 是否能映射到 64x128/64x192/64x256 等高效指令；
- 是否被拆成多个较窄 GMMA；
- 每个完整序列的动态 GMMA issue 数；
- 更大 BN 是否引入额外 P conversion、mask 浪费或寄存器压力。

## 6. 为什么 SMEM/register 的变化会带来性能收益

### 6.1 首先是可行域变化，而不是抽象的“更均衡”

如果 p0_q0 已经合法，并且 p/q 没有改变 tile、stage、occupancy 或真实 stall，
仅让两个资源百分比更接近不会自动提速。

资源交换最有价值的情况是跨越离散边界：

```text
非法 tile → 合法 tile
BN96 → BN128
stage2 → stage3
发生 spill/serialization → 不发生
较窄/分解 GMMA → 原生宽 GMMA
```

这些变化可能带来阶跃式收益，而不是连续的小变化。

### 6.2 register cliff

consumer budget 为：

```text
consumer_reg_alloc × 128 threads × num_consumer × 4 B
```

当 source-level peak 或 ptxas 实际需求接近该值时：

- 编译器调度自由度下降；
- P、Q、O、score 和 softmax state 的 live ranges 更难重叠；
- 可能出现 spill 或 ptxas WGMMA serialization/performance warning；
- 即使没有 spill，也可能使 operand preparation 位于关键路径。

p 将 P prefix 写入 SMEM，可缩短一部分 P register live range；但它同时增加
store/fence/sync。只有前者比后者更重要时才会获益。

### 6.3 SMEM cliff

当 SMEM 接近 232448 B：

- 更大的 BN 或更多 stage 会被直接过滤；
- 即使 CTA 合法，也可能限制 residency；
- 更宽的 Q SMEM descriptor 增加 shared operand 流量。

q 将部分 Q 保存在 registers，释放 SMEM；代价是 Q 的 LDG/PRMT 和
RS-QK register operand。长序列中，同一个 Q tile 会遍历大量 K/V blocks，
一次 Q-register materialization 可以被复用，较容易摊薄。

### 6.4 occupancy 不是当前已证明的主因

JSON 中 SMEM/reg 变化是确定的，但没有 NCU achieved occupancy。很多配置
本身已因 registers 或 SMEM 限制为低 CTA residency，因此少几 KiB 未必改变
active CTAs。

只有当理论 occupancy calculator 和 NCU 都显示：

```text
active CTAs/SM 或 active warps/SM 上升
```

才能把收益归因于 occupancy。否则更稳妥的解释是：

```text
tile/stage 被解锁，或 SS/RS operand critical path 改善。
```

## 7. 哪些情况达到了最初目标

最初目标是：

> 当 SMEM 或 registers 一侧先达到上限、另一侧仍有余量时，通过 P/Q
> 驻留转换，使更大的 WGMMA BN 合法并获得性能。

### 7.1 强成功：D192 FP16 causal / non-causal

目标配置：

```text
BM128 × BN128 × stage2 × p0q0
SMEM = 248832 B > 232448 B
```

使用 q4：

```text
释放 Q SMEM = 128 × 4 × 16 × 2 = 16384 B
新 SMEM     = 232448 B
register    = 212992 → 229376 B，仍低于 245760 B consumer budget
```

结果：

```text
best p0_q0: BN96
ours:       BN128
SASS:       HGMMA.64x96x16 → HGMMA.64x128x16
复测:       NC +1.80%，C +2.14%
```

这组同时满足“资源一侧超限、另一侧有余量”“更大 BN”“更宽 WGMMA”
和“真实性能提升”，是最完整的成功案例。

### 7.2 软成功：D128 FP8 causal / non-causal

目标 `BM192×BN128×stage3`：

```text
p0q0 register requirement = 245760 B
consumer budget           = 245760 B
p3q0 register requirement = 233472 B
SMEM                       = 175104 → 193536 B
```

它不是硬超限，而是处于 register cliff。p3 增加 18432 B P-SMEM，换取
12288 B register model 余量。

结果：

```text
best p0_q0: BN96
ours:       BN128
复测:       NC +1.74%，C +1.49%
```

这是符合设计方向但幅度较小的软成功。

### 7.3 流水线成功：D192 FP8 non-causal

目标 `BM128×BN128×stage3`：

```text
p0q0 SMEM = 248832 B > 232448 B
q5 释放   = 128 × 5 × 32 × 1 = 20480 B
q5 SMEM   = 228352 B
```

结果：

```text
best p0_q0: stage2
ours:       stage3
复测:       +2.33%
```

它没有增大 BN，但证明该方法可以成为流水线编译器的资源变换，与 stage
调优协同。

### 7.4 资源目标达到但性能失败：D96 FP16

`BM192×BN128×stage3` 的 p0_q0 registers 正好顶住 245760 B budget，
p3 将其降到 233472 B，并把 BN96 提高到 BN128。

但是复测 non-causal 回退 1.65%，causal 结果也不稳定且总体回退。SASS
显示 p3 增加 P store 和 SS-PV，额外成本超过了更大 BN 的循环收益。

这组说明编译器不能以“资源更均衡”或“BN 更大”作为最终 objective；
它们只能用于生成候选，最后仍需 cost model 或实测。

### 7.5 硬解锁但全局失败：D128 FP16 causal

同一个 `BM192×BN128×stage2`：

```text
p0q0 register requirement = 270336 B，非法
p8q0 register requirement = 233472 B，合法
p8q0 SMEM                 = 232448 B
```

p8 确实完成 register→SMEM 硬转移，但最新 p0_q0 tuner 找到：

```text
BM128 × BN192 × stage2
```

该 baseline 的 BN 更大，SASS 使用 `HGMMA.64x192x16`，复测比 p8 快约
6%。所以“解锁某个更大 BM tile”不等于“战胜全局最优 baseline”。

## 8. 判断达到最初目标所需的指标

一个案例只有同时满足以下证据链，才能宣称“资源交换解锁更大 BN并提速”。

### 8.1 资源可行性

对目标 `(BM,BN,stage,nc)` 同时计算：

```text
SMEM_p0q0 > limit 或 RegPeak_p0q0 > consumer budget
SMEM_pq   <= limit 且 RegPeak_pq   <= consumer budget
```

同时记录 ptxas：

```text
register count
spill stores/loads
local memory
WGMMA serialization / Potential Performance Loss warning
dynamic SMEM
```

### 8.2 tile / pipeline 确实跨界

至少出现一项：

```text
best BN 增大
best BM 增大
pipeline stage 增加
consumer WG 增加
理论 active CTA 数增加
```

### 8.3 SASS 确实使用目标 WGMMA

不能只看配置名中的 BN。必须检查：

```text
HGMMA/QGMMA 的 MxNxK
宽 GMMA 是否被分解
SS/RS 比例
完整序列估算的动态 GMMA issue 数
LDG / STS / PRMT / fence / sync 增量
```

### 8.4 性能必须超过额外转换成本

同环境交错复测，并分别报告：

```text
全局最优 ours vs 全局最优 p0_q0
固定其他参数的 pq vs p0_q0
固定 tile、只改变 pipeline 的消融
固定 pipeline、只改变 tile 的消融
```

建议至少 5 个独立进程、锁频、报告 median、p10/p90。小于 1% 视为持平。

### 8.5 NCU 恢复后应检查

重点指标类别：

```text
achieved occupancy / active warps
eligible warps and issued warps
long scoreboard
MIO/shared-memory throttle
register dependency / math-pipe throttle
tensor pipe utilization
shared load/store wavefronts
DRAM/L2 traffic
GMMA dynamic instruction count
```

可证伪预测：

- 更大 BN 成功时，N-tile iteration 数和 QK GMMA issue 应下降；
- q 成功时，Q SMEM traffic 应下降，RS-QK 与 global→register load 上升；
- p 成功时，RS-PV/register operand 压力应下降，P store/sync 上升；
- 若 occupancy 不变，则不应把收益归因于 occupancy；
- 若 stall 没有沿预期下降，收益可能来自其他调度参数或测量噪声。

## 9. 编译器集成建议

该优化最适合加入流水线/tile 编译器，而不是固定开启。

联合搜索维度：

```text
BM, BN
pipeline stages
num_consumer
producer_reg_dealloc
consumer_reg_alloc
p_smem_k_tiles
q_reg_k_tiles
scheduler barrier
rescale schedule
```

推荐两阶段搜索。

### 阶段一：资源边界求解

先寻找能够跨越离散边界的最小 p/q：

```text
使非法 BN 合法的最小 p 或 q
使 stage2→stage3 的最小 q
消除 register warning 的最小 p
由 q 释放空间、再由 p 消耗空间的 Pareto 组合
```

利用单调方向剪枝：

```text
p 增大：P-SMEM 上升，P-register requirement 通常下降
q 增大：Q-SMEM 下降，Q-register requirement 上升
```

若一个候选同时使用更多 SMEM、更多 registers，tile/stage 又不更好，可直接
从 Pareto frontier 删除。

### 阶段二：性能选择

资源合法不等于性能更好。cost model 至少需要：

```text
实际 GMMA shape，而不只是数值 BN
ceil(S / BN)
QK 与 PV 的动态 GMMA issue 估计
P store/fence/sync 成本
Q LDG/PRMT 成本
register headroom
SMEM headroom
occupancy upper bound
causal 边界浪费
```

对模型无法区分的少量 Pareto 候选再做 benchmark。

### 推荐消融

| 版本 | p/q | tile/stage | 回答的问题 |
| --- | --- | --- | --- |
| A | p0q0 | 独立完整调优 | 官方 baseline |
| B | 非零 p/q | 固定 A 的全部其他参数 | operand-path 直接收益 |
| C | 非零 p/q | 允许改变 BM/BN | 解锁更大 WGMMA 的收益 |
| D | 非零 p/q | 允许改变 pipeline stage | 与流水线协同的收益 |
| E | 非零 p/q | 全部联合调优 | 编译器最终收益 |

## 10. 最终评价

当前实验已经找到符合最初目标的真实场景：

- D192 FP16：SMEM→register 转移硬解锁 BN128 和
  `HGMMA.64x128x16`，获得约 1.8%～2.1%；
- D128 FP8：register→SMEM 转移让 BN128 离开 register cliff，获得约
  1.5%～1.7%；
- D192 FP8 non-causal：SMEM→register 转移硬解锁 stage3，获得约 2.3%。

也找到了重要反例：

- D96 FP16 的 BN128 没有战胜 P-store/SS-PV 成本；
- D128 FP16 causal 解锁的 tile 没有战胜全局最优 p0_q0 BN192；
- 若只看资源平衡或配置名 BN，会误判 D64 FP8 non-causal；实际 SASS
  shape 比数值 BN 更重要。

因此，最有说服力的贡献不是“p/q 总能提高性能”，而是：

> p/q 提供了一个双向、可组合的片上资源变换。它能够扩展 WGMMA tile 和
> software-pipeline 的合法搜索空间；编译器再结合真实 GMMA shape、
> operand-path 成本和资源边界，从这些新候选中选择真正更快的配置。
