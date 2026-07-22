from __future__ import annotations

from dataclasses import replace

from utils import TConfig, Mode


MMA_M = 64
MAX_MMA_N = 256
# Hopper limits one tensor-map box dimension to 256 elements.  Q load and O
# store split a larger logical BM tile into a 256-row prefix plus one tail, so
# the two-transaction implementation supports BM up to 512.
MAX_TMA_M = 512
WARP_GROUP_THREADS = 128
MIN_PRODUCER_REGISTERS = 24
REGISTER_BYTES = 4
OUTPUT_ELEMENT_BYTES = 2
# Covers pipeline state, descriptors, indices and other scalar consumer state
# that is not represented by the tensor-fragment formulas below.
CONSUMER_CONTROL_REGISTER_RESERVE = 8

# Stage-3 alternatives. Base allocations are deliberately omitted because
# tune.py compares these new results against the retained stage-2 winner.
REGISTER_ALLOCATION = {
    1: (
        (56, 256),
        (48, 256),
        (64, 256),
    ),
    2: (
        (24, 240),
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
        (32, 160),
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
    4: (24, 112),
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
    num_consumer: int,
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
    compute_m = MMA_M * num_consumer
    p_block_m = bm if bm == compute_m else compute_m
    add_array(p_block_m * p_smem_cols * elem_width, p_alignment)
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


def _consumer_register_bytes(
    bm: int,
    hd: int,
    bn: int,
    elem_width: int,
    p_smem_k_tiles: int,
    q_reg_k_tiles: int,
    num_consumer: int,
    mode: Mode,
) -> int:
    """Estimate aggregate consumer register storage in bytes.

    KEEP is deliberately conservative: it adds every modeled fragment and
    temporary even when their live ranges do not overlap.  RADICAL models the
    peak of the mainloop and epilogue live ranges.  The latter can still be
    lower than ptxas allocation because compiler scheduling and inlining are
    not visible to this source-level model.
    """
    num_mma_threads = WARP_GROUP_THREADS * num_consumer
    compute_m = MMA_M * num_consumer
    if bm < compute_m or bm % compute_m != 0:
        raise ValueError("BM must be a positive multiple of ComputeM")

    mma_k = _mma_k(elem_width)
    p_reg_k_tiles = bn // mma_k - p_smem_k_tiles
    if p_reg_k_tiles < 0:
        raise ValueError("P SMEM tiles exceed the PV K dimension")

    def fragment_regs(elements: int, element_bytes: int) -> int:
        return _ceil_div(
            elements * element_bytes,
            num_mma_threads * REGISTER_BYTES,
        )

    # Persistent mainloop fragments.
    acc_o_regs = fragment_regs(bm * hd, REGISTER_BYTES)
    q_regs = fragment_regs(
        bm * q_reg_k_tiles * mma_k,
        elem_width,
    )
    softmax_rows = 4 * bm // num_mma_threads
    softmax_slice_rows = softmax_rows // (bm // compute_m)
    softmax_state_regs = 2 * softmax_rows       # row_max + row_sum

    # One ComputeM slice is processed at a time. tSrS is FP32. The converted
    # P suffix is persistent until its PV WGMMA completes. A SMEM P prefix is
    # converted one MMA-K tile at a time into a short-lived register fragment.
    acc_s_regs = fragment_regs(compute_m * bn, REGISTER_BYTES)
    p_suffix_regs = fragment_regs(
        compute_m * p_reg_k_tiles * mma_k,
        elem_width,
    )
    p_prefix_temp_regs = (
        fragment_regs(compute_m * mma_k, elem_width)
        if p_smem_k_tiles > 0
        else 0
    )

    # max_get_scale can hold the returned scale and the previous-max copy.
    # Both cover one M repeat, not the complete logical BM tile.
    softmax_temp_regs = 2 * softmax_slice_rows

    # The epilogue converts FP32 O to FP16/BF16 before R2S. FP8 attention also
    # writes BF16, hence OUTPUT_ELEMENT_BYTES is independent of elem_width.
    output_fragment_regs = fragment_regs(
        bm * hd,
        OUTPUT_ELEMENT_BYTES,
    )
    control_regs = CONSUMER_CONTROL_REGISTER_RESERVE

    if mode is Mode.KEEP:
        registers_per_thread = (
            control_regs
            + acc_o_regs
            + q_regs
            + softmax_state_regs
            + acc_s_regs
            + p_suffix_regs
            + p_prefix_temp_regs
            + softmax_temp_regs
            + output_fragment_regs
        )
    else:
        persistent_mainloop_regs = (
            control_regs + acc_o_regs + q_regs + softmax_state_regs
        )
        # QK(current), PV(previous), and softmax(current) overlap in the
        # repeated-M steady state.
        qk_pv_softmax_peak = (
            persistent_mainloop_regs
            + acc_s_regs
            + p_suffix_regs
            + softmax_temp_regs
        )
        # After the previous PV drains, the current P is materialized. Prefix
        # conversion and suffix conversion have disjoint source-level lives.
        p_materialize_peak = (
            persistent_mainloop_regs
            + acc_s_regs
            + max(p_prefix_temp_regs, p_suffix_regs)
            + softmax_slice_rows
        )
        # O and converted O overlap until R2S. The next RS-Q prefetch starts
        # afterwards, so Q is intentionally absent from this phase.
        epilogue_convert_peak = (
            control_regs
            + acc_o_regs
            + output_fragment_regs
            + softmax_rows
        )
        epilogue_q_prefetch_peak = control_regs + q_regs + softmax_rows
        registers_per_thread = max(
            qk_pv_softmax_peak,
            p_materialize_peak,
            epilogue_convert_peak,
            epilogue_q_prefetch_peak,
        )

    # setmaxnreg allocations and tuning alternatives are 8-register aligned.
    registers_per_thread = _align_up(registers_per_thread, 8)
    return registers_per_thread * num_mma_threads * REGISTER_BYTES


def _producer_registers_per_thread() -> int:
    """Return the source-visible producer lower bound.

    The producer owns pipeline/TMA control state but no persistent matrix
    fragment. Its exact ptxas allocation is compiler-dependent, so the Python
    model can only enforce the C++ architectural minimum; compile-time C75xx
    filtering remains the authoritative spill/performance check.
    """
    return MIN_PRODUCER_REGISTERS


def _allocated_register_bytes_per_cta(
    producer_reg_dealloc: int,
    consumer_reg_alloc: int,
    num_consumer: int,
) -> int:
    return REGISTER_BYTES * WARP_GROUP_THREADS * (
        producer_reg_dealloc + num_consumer * consumer_reg_alloc
    )


def _allocation_is_valid(
    *,
    consumer_required_bytes: int,
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
    consumer_allocated_bytes = (
        consumer_reg_alloc * WARP_GROUP_THREADS * num_consumer * REGISTER_BYTES
    )
    if consumer_required_bytes > consumer_allocated_bytes:
        return False
    return _allocated_register_bytes_per_cta(
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
    reg_limit: int = 262_144,
    bn_rate: float = 0.5,
    num_consumer_limit: tuple[int, int] = (1, 4),
    stage_limit: tuple[int, int] = (1, 3),
    mode: Mode = Mode.RADICAL,
) -> list[TConfig]:
    """Generate base structures, including repeated-M MIX-WGMMA mappings."""
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
    # The custom tuning API fixes V_colmajor=false and MIX-WGMMA fixes
    # IntraWGOverlap=true, so this is the current C++ default heuristic:
    # HD > 128 && (!Is_FP8 || V_colmajor) && IntraWGOverlap.
    default_rescale_o_before_gemm = int(HD > 128 and elem_width != 1)
    configs: list[TConfig] = []

    for num_consumer in range(num_consumer_lower, num_consumer_upper + 1):
        producer_reg_dealloc, consumer_reg_alloc = (
            BASE_REGISTER_ALLOCATION[num_consumer]
        )
        if _allocated_register_bytes_per_cta(
            producer_reg_dealloc, consumer_reg_alloc, num_consumer
        ) > reg_limit:
            continue

        compute_m = MMA_M * num_consumer
        # RepeatM is derived from BM, rather than being an independent tuning
        # parameter.  These are necessary upper bounds: every candidate keeps
        # a BM x HD output accumulator in registers and a BM x HD output tile
        # in shared memory.  The exact resource models below reject candidates
        # that exceed the remaining S/P/Q/K/V requirements.
        consumer_register_bytes = (
            consumer_reg_alloc * WARP_GROUP_THREADS * num_consumer
            * REGISTER_BYTES
        )
        bm_reg_upper = consumer_register_bytes // (HD * REGISTER_BYTES)
        bm_smem_upper = smem_limit // (HD * 2)
        bm_upper = min(bm_reg_upper, bm_smem_upper, MAX_TMA_M)

        for bm in range(compute_m, bm_upper + 1, compute_m):
            # RepeatM==1 is the original MIX-WGMMA mapping. Larger values keep
            # the same consumer warpgroup count and assign each warpgroup
            # multiple logical 64-row Q slices while K/V remains resident.
            for stage in range(stage_lower, stage_upper + 1):
                staged_tensor_count = 3 if elem_width == 1 else 2
                kv_bytes_per_bn = staged_tensor_count * HD * stage * elem_width
                bn_max = min(
                    MAX_MMA_N,
                    smem_limit // kv_bytes_per_bn // mma_k * mma_k,
                )

                tmp_cfgs: list[TConfig] = []
                max_bn = 0
                for bn in range(mma_k, bn_max + 1, mma_k):
                    # FP8 row-major V uses the transpose path. For non-64-aligned
                    # HD, its current transpose layout requires BN % 64 == 0.
                    if elem_width == 1 and HD % 64 != 0 and bn % 64 != 0:
                        continue

                    p_total_tiles = bn // mma_k
                    # Both FP16 and FP8 support the complete P-prefix range.
                    # FP8 uses ordinary SIMT stores and decomposes the prefix
                    # into the widest layouts supported by V: SW128, SW64,
                    # then SW32.
                    p_smem_candidates = range(p_total_tiles + 1)
                    for p_smem_k_tiles in p_smem_candidates:
                        for q_reg_k_tiles in range(q_total_tiles + 1):
                            smem_size = _smem_size_bytes(
                                bm, bn, HD, stage, elem_width,
                                p_smem_k_tiles, q_reg_k_tiles, num_consumer,
                            )
                            if smem_size > smem_limit:
                                continue
                            consumer_required_bytes = _consumer_register_bytes(
                                bm, HD, bn, elem_width, p_smem_k_tiles,
                                q_reg_k_tiles, num_consumer, mode,
                            )
                            if not _allocation_is_valid(
                                consumer_required_bytes=consumer_required_bytes,
                                producer_reg_dealloc=producer_reg_dealloc,
                                consumer_reg_alloc=consumer_reg_alloc,
                                num_consumer=num_consumer,
                                reg_limit=reg_limit,
                            ):
                                continue

                            if max_bn < bn:
                                max_bn = bn
                            tmp_cfgs.append(TConfig(
                                kBlockM=bm,
                                kBlockN=bn,
                                kStage=stage,
                                producer_reg_dealloc=producer_reg_dealloc,
                                consumer_reg_alloc=consumer_reg_alloc,
                                p_smem_k_tiles=p_smem_k_tiles,
                                q_reg_k_tiles=q_reg_k_tiles,
                                num_consumer=num_consumer,
                                use_scheduler_barrier=0,
                                rescale_o_before_gemm=default_rescale_o_before_gemm,
                            ))
                # delete those bn values that are less than half of max_bn.
                min_bn = (((max_bn * (1 - bn_rate)) + mma_k-1) // mma_k) * mma_k
                for cfg in tmp_cfgs:
                    if cfg.kBlockN >= min_bn:
                        configs.append(cfg)
    return _unique_configs(configs)


def mix_wgmma_second_configs(configs: list[TConfig]) -> list[TConfig]:
    """Generate joint scheduler/rescale alternatives for stage-1 winners.

    A single consumer warpgroup has no inter-warpgroup execution order to
    enforce, and the C++ scheduler barrier is intentionally a 2--4 WG ring.
    It therefore keeps only ``sb0``. Every mapping tests both rescale
    placements; multi-consumer mappings additionally test both scheduler
    choices, including their interaction with the rescale placement.
    """
    if not isinstance(configs, list):
        raise TypeError("configs must be a list[TConfig]")
    result = []
    for config in configs:
        if not isinstance(config, TConfig):
            raise TypeError("configs must contain only TConfig values")
        # Do not derive stage-2 variants beyond the two-transaction Q/O TMA
        # implementation's logical-M limit.
        if config.kBlockM > MAX_TMA_M:
            continue
        if config.use_scheduler_barrier not in (0, 1):
            raise ValueError("use_scheduler_barrier must be 0 or 1")
        if config.rescale_o_before_gemm not in (0, 1):
            raise ValueError("rescale_o_before_gemm must be 0 or 1")
        if config.num_consumer == 1 and config.use_scheduler_barrier != 0:
            raise ValueError("a single consumer warpgroup requires sb0")
        scheduler_values = (0, 1) if config.num_consumer >= 2 else (0,)
        for use_scheduler_barrier in scheduler_values:
            for rescale_o_before_gemm in (0, 1):
                candidate = replace(
                    config,
                    use_scheduler_barrier=use_scheduler_barrier,
                    rescale_o_before_gemm=rescale_o_before_gemm,
                )
                # if candidate != config:
                result.append(candidate)
    return _unique_configs(result)


def mix_wgmma_third_configs(
    configs: list[TConfig],
    HD: int,
    elem_width: int,
    reg_limit: int = 262_144,
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
        # Keep stage 3 safe when its input comes from a historical tuning
        # result rather than the newly capped base-config generator.
        if config.kBlockM > MAX_TMA_M:
            continue
        if config.num_consumer == 4:
            continue
        if config.num_consumer not in REGISTER_ALLOCATION:
            raise ValueError(
                f"no register allocations for num_consumer={config.num_consumer}"
            )
        compute_m = MMA_M * config.num_consumer
        if config.kBlockM < compute_m or config.kBlockM % compute_m != 0:
            raise ValueError(
                "kBlockM must be a positive multiple of 64 * num_consumer"
            )

        consumer_required_bytes = _consumer_register_bytes(
            config.kBlockM, HD, config.kBlockN, elem_width,
            config.p_smem_k_tiles, config.q_reg_k_tiles,
            config.num_consumer, mode,
        )
        for producer_reg_dealloc, consumer_reg_alloc in (
            REGISTER_ALLOCATION[config.num_consumer]
        ):
            # if (
            #     producer_reg_dealloc == config.producer_reg_dealloc
            #     and consumer_reg_alloc == config.consumer_reg_alloc
            # ):
            #     continue
            if not _allocation_is_valid(
                consumer_required_bytes=consumer_required_bytes,
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

# if __name__ == "__main__":
#     cfgs = mix_wgmma_base_configs(HD=128, elem_width=2, mode=Mode.RADICAL, num_consumer_limit=(2, 3))
#     for cfg in cfgs:
#         print(cfg)
