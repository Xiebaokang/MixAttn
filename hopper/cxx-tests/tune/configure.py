from __future__ import annotations

from dataclasses import replace

from utils import TConfig, Mode


MMA_M = 64
MAX_MMA_N = 256
WARP_GROUP_THREADS = 128
MIN_PRODUCER_REGISTERS = 24

# Stage-3 alternatives. Base allocations are deliberately omitted because
# tune.py compares these new results against the retained stage-2 winner.
REGISTER_ALLOCATION = {
    1: (
        (48, 256),
        (64, 256),
    ),
    2: (
        (32, 240),
        (24, 232),
        (32, 232),
        (24, 224),
        (32, 224),
        (24, 216),
        (32, 216),
        (24, 208),
        (32, 208),
        (24, 200),
        (32, 200),
    ),
    3: (
        (24, 160),
        (24, 152),
        (40, 152),
        (24, 144),
        (40, 144),
    ),
}

# The first-stage allocation is intentionally separate from
# REGISTER_ALLOCATION: the latter contains only the alternatives introduced in
# stage 3, so the base executable is not compiled a second time.
BASE_REGISTER_ALLOCATION = {
    1: (56, 256),
    2: (24, 240),
    3: (32, 160),
    4: (24, 120),
}


def _ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    return (value + divisor - 1) // divisor


def _align_up(value: int, alignment: int) -> int:
    return _ceil_div(value, alignment) * alignment


def _mma_k(elem_width: int) -> int:
    if elem_width not in (1, 2):
        raise ValueError("elem_width must be 1 (FP8) or 2 (FP16/BF16)")
    return 32 if elem_width == 1 else 16


def _validate_common(
    hd: int,
    elem_width: int,
    reg_limit: int,
    mode: Mode,
) -> None:
    mma_k = _mma_k(elem_width)
    if hd <= 0 or hd % mma_k != 0:
        raise ValueError(f"HD must be a positive multiple of {mma_k}")
    if hd > 256:
        raise ValueError("USE_MIX_WGMMA requires HD <= 256")
    if reg_limit <= 0:
        raise ValueError("reg_limit must be positive")
    if not isinstance(mode, Mode):
        raise TypeError("mode must be a Mode value")


def _validate_closed_range(
    name: str,
    bounds: tuple[int, int],
    minimum: int,
    maximum: int,
) -> tuple[int, int]:
    if (
        not isinstance(bounds, tuple)
        or len(bounds) != 2
        or any(not isinstance(value, int) for value in bounds)
    ):
        raise TypeError(f"{name} must be a tuple[int, int]")
    lower, upper = bounds
    if lower > upper:
        raise ValueError(f"{name} lower bound must not exceed its upper bound")
    if lower < minimum or upper > maximum:
        raise ValueError(f"{name} must be within [{minimum}, {maximum}]")
    return lower, upper


def _pipeline_storage_bytes(stage: int) -> int:
    # Three standalone barriers, five k-stage TMA pipelines and one scheduler
    # integer. Every member is aligned to 16 bytes in PipelineStorage.
    return 3 * 16 + 5 * (2 * stage * 8) + 16


def _swizzle_alignment(elements: int, elem_width: int) -> int:
    byte_width = elements * elem_width
    if byte_width % 128 == 0:
        return 1024
    if byte_width % 64 == 0:
        return 512
    if byte_width % 32 == 0:
        return 256
    return 128


def _smem_size_bytes(
    bm: int,
    bn: int,
    hd: int,
    stage: int,
    elem_width: int,
    p_smem_k_tiles: int,
    q_reg_k_tiles: int,
) -> int:
    """Mirror FlashAttnFwdSm90 SharedStorage for the fixed-length MIX path."""
    mma_k = _mma_k(elem_width)
    q_smem_cols = hd - q_reg_k_tiles * mma_k
    p_smem_cols = p_smem_k_tiles * mma_k
    if q_smem_cols < 0 or p_smem_cols > bn:
        raise ValueError("invalid mixed Q/P tile partition")

    v_tma_alignment = _swizzle_alignment(hd, elem_width)
    v_mma_alignment = (
        _swizzle_alignment(bn, elem_width)
        if elem_width == 1 else v_tma_alignment
    )
    p_alignment = (
        _swizzle_alignment(bn, elem_width) if p_smem_cols > 0 else 128
    )
    mainloop_alignment = max(
        128, v_tma_alignment, v_mma_alignment, p_alignment
    )
    offset = 0

    def add_array(size: int, alignment: int) -> None:
        nonlocal offset
        if size == 0:
            return
        offset = _align_up(offset, alignment)
        offset += _align_up(size, alignment)

    v_storage = bn * hd * stage * elem_width
    add_array(v_storage, v_mma_alignment)              # smem_v
    if elem_width == 1:
        add_array(v_storage, v_tma_alignment)          # FP8 smem_vt
    add_array(bm * q_smem_cols * elem_width, 128)     # smem_q
    add_array(bn * hd * stage * elem_width, 128)      # smem_k
    offset += 1                                       # zero-sized smem_qv member
    add_array(bm * p_smem_cols * elem_width, p_alignment)
    mainloop_storage = _align_up(offset, mainloop_alignment)

    output_storage = _align_up(bm * hd * 2, 128)      # FP8 output is BF16
    smem_v_member_size = _align_up(v_storage, v_mma_alignment)
    epilogue_padding = max(0, output_storage - smem_v_member_size)
    padding_member_size = epilogue_padding or 1
    mainloop_offset = _align_up(padding_member_size, mainloop_alignment)
    mainloop_union_member = _align_up(
        mainloop_offset + mainloop_storage, mainloop_alignment
    )
    tensor_storage = _align_up(
        max(output_storage, mainloop_union_member), mainloop_alignment
    )
    return _align_up(
        tensor_storage + _pipeline_storage_bytes(stage), mainloop_alignment
    )


def _consumer_registers_per_thread(
    bm: int,
    hd: int,
    bn: int,
    elem_width: int,
    p_smem_k_tiles: int,
    q_reg_k_tiles: int,
    num_consumer: int,
    mode: Mode,
) -> int:
    """Estimate source-visible 32-bit fragment registers per consumer thread.

    KEEP charges the complete converted P fragment. RADICAL follows the live
    C++ implementation and charges only the persistent register suffix after
    the P shared-memory prefix.
    """
    num_mma_threads = WARP_GROUP_THREADS * num_consumer
    acc_o_elements = bm * hd
    acc_s_elements = bm * bn
    if acc_o_elements % num_mma_threads != 0:
        raise ValueError("BM * HD must be divisible by NumMmaThreads")
    if acc_s_elements % num_mma_threads != 0:
        raise ValueError("BM * BN must be divisible by NumMmaThreads")

    acc_o = acc_o_elements // num_mma_threads
    acc_s = acc_s_elements // num_mma_threads
    mma_k = _mma_k(elem_width)
    p_reg_cols = bn - p_smem_k_tiles * mma_k
    if p_reg_cols < 0:
        raise ValueError("P SMEM tiles exceed the PV K dimension")
    p_live_cols = p_reg_cols if mode is Mode.RADICAL else bn
    p_elements = bm * p_live_cols // num_mma_threads
    q_reg_elements = MMA_M * (q_reg_k_tiles * mma_k) // WARP_GROUP_THREADS
    p_regs = _ceil_div(p_elements * elem_width, 4)
    q_regs = _ceil_div(q_reg_elements * elem_width, 4)

    k_n_rows = 2 * (2 * bm // num_mma_threads)
    softmax_peak_regs = 4 * k_n_rows + 1
    return acc_o + acc_s + p_regs + q_regs + softmax_peak_regs


def _producer_registers_per_thread() -> int:
    """Return the source-visible producer lower bound.

    The producer owns pipeline/TMA control state but no persistent matrix
    fragment. Its exact ptxas allocation is compiler-dependent, so the Python
    model can only enforce the C++ architectural minimum; compile-time C75xx
    filtering remains the authoritative spill/performance check.
    """
    return MIN_PRODUCER_REGISTERS


def _allocated_registers_per_cta(
    producer_reg_dealloc: int,
    consumer_reg_alloc: int,
    num_consumer: int,
) -> int:
    return WARP_GROUP_THREADS * (
        producer_reg_dealloc + num_consumer * consumer_reg_alloc
    )


def _allocation_is_valid(
    *,
    consumer_required: int,
    producer_reg_dealloc: int,
    consumer_reg_alloc: int,
    num_consumer: int,
    reg_limit: int,
) -> bool:
    values = (producer_reg_dealloc, consumer_reg_alloc)
    if any(value < 24 or value > 256 or value % 8 != 0 for value in values):
        return False
    if _producer_registers_per_thread() > producer_reg_dealloc:
        return False
    if consumer_required > consumer_reg_alloc:
        return False
    return _allocated_registers_per_cta(
        producer_reg_dealloc, consumer_reg_alloc, num_consumer
    ) <= reg_limit


def _unique_configs(configs: list[TConfig]) -> list[TConfig]:
    result: list[TConfig] = []
    seen: set[str] = set()
    for config in configs:
        name = config.name()
        if name not in seen:
            seen.add(name)
            result.append(config)
    return result


def mix_wgmma_base_configs(
    HD: int,
    elem_width: int = 2,
    causal: bool = False,
    smem_limit: int = 232_448,
    reg_limit: int = 65_536,
    num_consumer_limit: tuple[int, int] = (1, 4),
    stage_limit: tuple[int, int] = (1, 3),
    mode: Mode = Mode.RADICAL,
) -> list[TConfig]:
    """Generate stage-1 structures using the fixed base register allocation."""
    _validate_common(HD, elem_width, reg_limit, mode)
    if not isinstance(causal, bool):
        raise TypeError("causal must be bool")
    if smem_limit <= 0:
        raise ValueError("smem_limit must be positive")
    num_consumer_lower, num_consumer_upper = _validate_closed_range(
        "num_consumer_limit", num_consumer_limit, 1, 4
    )
    stage_lower, stage_upper = _validate_closed_range(
        "stage_limit", stage_limit, 1, 3
    )

    del causal  # causal/non-causal schedulers have equal SharedStorage size
    mma_k = _mma_k(elem_width)
    q_total_tiles = HD // mma_k
    configs: list[TConfig] = []

    for num_consumer in range(num_consumer_lower, num_consumer_upper + 1):
        bm = MMA_M * num_consumer
        producer_reg_dealloc, consumer_reg_alloc = (
            BASE_REGISTER_ALLOCATION[num_consumer]
        )
        if _allocated_registers_per_cta(
            producer_reg_dealloc, consumer_reg_alloc, num_consumer
        ) > reg_limit:
            continue

        for stage in range(stage_lower, stage_upper + 1):
            staged_tensor_count = 3 if elem_width == 1 else 2
            kv_bytes_per_bn = staged_tensor_count * HD * stage * elem_width
            bn_max = min(
                MAX_MMA_N,
                smem_limit // kv_bytes_per_bn // mma_k * mma_k,
            )
            for bn in range(mma_k, bn_max + 1, mma_k):
                # FP8 row-major V uses the transpose path. For non-64-aligned
                # HD, its current transpose layout requires BN % 64 == 0.
                if elem_width == 1 and HD % 64 != 0 and bn % 64 != 0:
                    continue

                if HD == 64 and bn < 128 or HD == 128 and bn < 128 or HD == 256 and bn < 64:
                    continue

                p_total_tiles = bn // mma_k
                for p_smem_k_tiles in range(p_total_tiles + 1):
                    for q_reg_k_tiles in range(q_total_tiles + 1):
                        smem_size = _smem_size_bytes(
                            bm, bn, HD, stage, elem_width,
                            p_smem_k_tiles, q_reg_k_tiles,
                        )
                        if smem_size > smem_limit:
                            continue
                        consumer_required = _consumer_registers_per_thread(
                            bm, HD, bn, elem_width, p_smem_k_tiles,
                            q_reg_k_tiles, num_consumer, mode,
                        )
                        if not _allocation_is_valid(
                            consumer_required=consumer_required,
                            producer_reg_dealloc=producer_reg_dealloc,
                            consumer_reg_alloc=consumer_reg_alloc,
                            num_consumer=num_consumer,
                            reg_limit=reg_limit,
                        ):
                            continue
                        configs.append(TConfig(
                            kBlockM=bm,
                            kBlockN=bn,
                            kStage=stage,
                            producer_reg_dealloc=producer_reg_dealloc,
                            consumer_reg_alloc=consumer_reg_alloc,
                            p_smem_k_tiles=p_smem_k_tiles,
                            q_reg_k_tiles=q_reg_k_tiles,
                            num_consumer=num_consumer,
                            use_scheduler_barrier=0,
                        ))
    return _unique_configs(configs)


def mix_wgmma_second_configs(configs: list[TConfig]) -> list[TConfig]:
    """Generate scheduler-barrier alternatives for multi-consumer winners.

    A single consumer warpgroup has no inter-warpgroup execution order to
    enforce, and the C++ scheduler barrier is intentionally a 2--4 WG ring.
    Its only meaningful configuration is therefore ``sb0``.
    """
    if not isinstance(configs, list):
        raise TypeError("configs must be a list[TConfig]")
    result = []
    for config in configs:
        if not isinstance(config, TConfig):
            raise TypeError("configs must contain only TConfig values")
        if config.use_scheduler_barrier not in (0, 1):
            raise ValueError("use_scheduler_barrier must be 0 or 1")
        if config.use_scheduler_barrier == 0 and config.num_consumer >= 2:
            result.append(replace(config, use_scheduler_barrier=1))
    return _unique_configs(result)


def mix_wgmma_third_configs(
    configs: list[TConfig],
    HD: int,
    elem_width: int,
    reg_limit: int = 65_536,
    mode: Mode = Mode.RADICAL,
) -> list[TConfig]:
    """Expand register-allocation alternatives for stage-2 winners.

    num_consumer==4 intentionally has no stage-3 alternatives; its base
    allocation is retained by tune.get_best_result().
    """
    _validate_common(HD, elem_width, reg_limit, mode)
    if not isinstance(configs, list):
        raise TypeError("configs must be a list[TConfig]")

    result: list[TConfig] = []
    for config in configs:
        if not isinstance(config, TConfig):
            raise TypeError("configs must contain only TConfig values")
        if config.num_consumer == 4:
            continue
        if config.num_consumer not in REGISTER_ALLOCATION:
            raise ValueError(
                f"no register allocations for num_consumer={config.num_consumer}"
            )
        if config.kBlockM != MMA_M * config.num_consumer:
            raise ValueError("kBlockM must equal 64 * num_consumer")

        consumer_required = _consumer_registers_per_thread(
            config.kBlockM, HD, config.kBlockN, elem_width,
            config.p_smem_k_tiles, config.q_reg_k_tiles,
            config.num_consumer, mode,
        )
        for producer_reg_dealloc, consumer_reg_alloc in (
            REGISTER_ALLOCATION[config.num_consumer]
        ):
            if (
                producer_reg_dealloc == config.producer_reg_dealloc
                and consumer_reg_alloc == config.consumer_reg_alloc
            ):
                continue
            if not _allocation_is_valid(
                consumer_required=consumer_required,
                producer_reg_dealloc=producer_reg_dealloc,
                consumer_reg_alloc=consumer_reg_alloc,
                num_consumer=config.num_consumer,
                reg_limit=reg_limit,
            ):
                continue
            result.append(replace(
                config,
                producer_reg_dealloc=producer_reg_dealloc,
                consumer_reg_alloc=consumer_reg_alloc,
            ))
    return _unique_configs(result)
