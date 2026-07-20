from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random

from utils import Shape, DType, Mode, TConfig, BenchResult
from interface import run_interface
from configure import (
    mix_wgmma_base_configs,
    mix_wgmma_second_configs,
    mix_wgmma_third_configs,
)


def get_best_result(
    origin_ret: list[BenchResult],
    new_ret: list[BenchResult],
) -> list[BenchResult]:
    """Keep the fastest result for every structure and rebuild the ranking."""
    best_by_base: dict[str, BenchResult] = {}
    for item in (*origin_ret, *new_ret):
        base_name = item.config.base_name()
        current = best_by_base.get(base_name)
        if current is None or item.tflops > current.tflops:
            best_by_base[base_name] = item

    ordered = sorted(best_by_base.values(), key=lambda item: item.tflops, reverse=True)
    return [replace(item, rank=index) for index, item in enumerate(ordered, 1)]


def _run_stage(
    *,
    stage_name: str,
    shape: Shape,
    dtype: DType,
    causal: bool,
    configs: list[TConfig],
    arch: str,
    jobs: int,
    register_usage_level: int,
    rank: int,
    timeout_seconds: float,
    mode: Mode,
    src_dir: str | Path | None,
    result_dir: str | Path | None = None,
) -> list[BenchResult]:
    if not configs:
        print(f"{stage_name}: no configs, skip")
        return []
    print(
        f"{stage_name}: {len(configs)} configs, "
        f"register-usage-level={register_usage_level}"
    )
    return run_interface(
        shape=shape,
        dtype=dtype,
        causal=causal,
        configs=configs,
        arch=arch,
        jobs=jobs,
        register_usage_level=register_usage_level,
        rank=rank,
        timeout_seconds=timeout_seconds,
        mode=mode,
        src_dir=src_dir,
        result_dir=result_dir,
    )


def tune(
    shape: Shape,
    dtype: DType,
    causal: bool,
    smem_limit: int = 232_448,
    reg_limit: int = 262_144,
    num_consumer_limit: tuple[int, int] = (2, 3),
    stage_limit: tuple[int, int] = (1, 3),
    bn_rate: float = 0.5,
    arch: str = "90a",
    jobs: int = 20,
    rank: int = 15,
    coarse_register_usage_level: int = 5,
    final_register_usage_level: int = 10,
    mode: Mode = Mode.KEEP,
    benchmark_timeout_seconds: float = 120.0,
    src_dir: str | Path | None = None,
    result_dir: str | Path | None = None,
) -> list[BenchResult]:
    """Run structure, scheduler, register-allocation and final tuning passes."""
    if len(shape) != 4 or any(value <= 0 for value in shape):
        raise ValueError("shape must be a positive (B, H, S, D) tuple")
    if not isinstance(dtype, DType):
        raise TypeError("dtype must be a DType value")
    if not isinstance(mode, Mode):
        raise TypeError("mode must be a Mode value")
    if not isinstance(causal, bool):
        raise TypeError("causal must be bool")
    if rank <= 0:
        raise ValueError("rank must be positive")
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    if benchmark_timeout_seconds <= 0:
        raise ValueError("benchmark_timeout_seconds must be positive")
    if not 0 <= coarse_register_usage_level <= 10:
        raise ValueError("coarse_register_usage_level must be in [0, 10]")
    if not 0 <= final_register_usage_level <= 10:
        raise ValueError("final_register_usage_level must be in [0, 10]")

    result_dir = Path(__file__).resolve().parent / "results" if result_dir is None else result_dir
    elem_width = 1 if dtype is DType.FP8 else 2
    base_configs = mix_wgmma_base_configs(
        HD=shape[-1],
        elem_width=elem_width,
        causal=causal,
        smem_limit=smem_limit,
        reg_limit=reg_limit,
        bn_rate=bn_rate,
        num_consumer_limit=num_consumer_limit,
        stage_limit=stage_limit,
        mode=mode,
    )

    # base_configs = random.sample(base_configs, rank)
    if not base_configs:
        raise ValueError("base config generation returned no valid configs")

    ret1 = _run_stage(
        stage_name="stage 1 / base structure",
        shape=shape,
        dtype=dtype,
        causal=causal,
        configs=base_configs,
        arch=arch,
        jobs=jobs,
        register_usage_level=coarse_register_usage_level,
        rank=rank,
        timeout_seconds=benchmark_timeout_seconds,
        mode=mode,
        src_dir=src_dir,
    )
    if not ret1:
        return []

    schedule_configs = mix_wgmma_second_configs([item.config for item in ret1])
    schedule_results = _run_stage(
        stage_name="stage 2 / execution schedule",
        shape=shape,
        dtype=dtype,
        causal=causal,
        configs=schedule_configs,
        arch=arch,
        jobs=jobs,
        register_usage_level=coarse_register_usage_level,
        rank=len(schedule_configs),
        timeout_seconds=benchmark_timeout_seconds,
        mode=mode,
        src_dir=src_dir,
    )
    ret2 = get_best_result(ret1, schedule_results)

    register_configs = mix_wgmma_third_configs(
        configs=[item.config for item in ret2],
        HD=shape[-1],
        elem_width=elem_width,
        reg_limit=reg_limit,
        mode=mode,
    )
    register_results = _run_stage(
        stage_name="stage 3 / register allocation",
        shape=shape,
        dtype=dtype,
        causal=causal,
        configs=register_configs,
        arch=arch,
        jobs=jobs,
        register_usage_level=coarse_register_usage_level,
        rank=len(register_configs),
        timeout_seconds=benchmark_timeout_seconds,
        mode=mode,
        src_dir=src_dir,
    )
    ret3 = get_best_result(ret2, register_results)
    if not ret3:
        return []

    return _run_stage(
        stage_name="final / full register usage",
        shape=shape,
        dtype=dtype,
        causal=causal,
        configs=[item.config for item in ret3],
        arch=arch,
        jobs=jobs,
        register_usage_level=final_register_usage_level,
        rank=rank,
        timeout_seconds=benchmark_timeout_seconds,
        mode=mode,
        src_dir=src_dir,
        result_dir=result_dir,
    )


if __name__ == "__main__":
    # fp16
    tune(shape=(1, 16, 30720, 64), dtype=DType.FP16, causal=False)
    tune(shape=(1, 16, 30720, 128), dtype=DType.FP16, causal=False)
    tune(shape=(1, 16, 30720, 64), dtype=DType.FP16, causal=True)
    tune(shape=(1, 16, 30720, 128), dtype=DType.FP16, causal=True)
    # fp8
    tune(shape=(1, 16, 30720, 64), dtype=DType.FP8, causal=False)
    tune(shape=(1, 16, 30720, 128), dtype=DType.FP8, causal=False)
    tune(shape=(1, 16, 30720, 64), dtype=DType.FP8, causal=True)
    tune(shape=(1, 16, 30720, 128), dtype=DType.FP8, causal=True)

