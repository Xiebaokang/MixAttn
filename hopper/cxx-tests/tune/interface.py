from __future__ import annotations

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Iterable

from utils import Shape, DType, TConfig, CompileResult, BenchResult


HERE = Path(__file__).resolve().parent
ATTN_TEMPLATE = HERE / "attn_temp.cu"
CMAKE_UTILS_DIR = HERE.parent / "cmake"

TCONFIG_FIELDS = (
    "kBlockM",
    "kBlockN",
    "kStage",
    "producer_reg_dealloc",
    "consumer_reg_alloc",
    "p_smem_k_tiles",
    "q_reg_k_tiles",
    "num_consumer",
    "use_scheduler_barrier",
)

PTXAS_PERF_LOSS_RE = re.compile(
    r"ptxas\s+info\s*:\s*\(C75\d{2}\).*Potential Performance Loss[^\r\n]*",
    re.IGNORECASE,
)
BENCH_RESULT_RE = re.compile(
    r"time\s*=\s*([0-9.eE+-]+)\s*ms,\s*"
    r"throughput\s*=\s*([0-9.eE+-]+)\s*TFLOPS"
)
def _replace_one(text: str, patterns: Iterable[str], replacement: str, name: str) -> str:
    for pattern in patterns:
        updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if count == 1:
            return updated
    raise RuntimeError(f"cannot replace {name} in attn_temp.cu")


def _render_case(temp_src: str, shape: Shape, dtype: DType, causal: bool) -> str:
    if len(shape) != 4 or any(value <= 0 for value in shape):
        raise ValueError("shape must be a positive (B, H, S, D) tuple")
    if not isinstance(dtype, DType):
        raise TypeError("dtype must be a DType value")
    if not isinstance(causal, bool):
        raise TypeError("causal must be bool")

    batch, heads, seqlen, head_dim = shape
    source = temp_src
    for name, value in {
        "kBatch": batch,
        "kSeqlen": seqlen,
        "kNumHeads": heads,
        "kHeadDim": head_dim,
    }.items():
        source = _replace_one(
            source,
            [rf"(constexpr\s+int\s+{name}\s*=\s*)\d+(\s*;)"],
            rf"\g<1>{value}\g<2>",
            name,
        )

    calls = {
        (DType.FP8, False): (
            'benchmark<cute::float_e4m3_t, at::kFloat8_e4m3fn, false>'
            '("FP8", options);'
        ),
        (DType.FP8, True): (
            'benchmark<cute::float_e4m3_t, at::kFloat8_e4m3fn, true>'
            '("FP8", options);'
        ),
        (DType.FP16, False): (
            'benchmark<cutlass::half_t, at::kHalf, false>("FP16", options);'
        ),
        (DType.FP16, True): (
            'benchmark<cutlass::half_t, at::kHalf, true>("FP16", options);'
        ),
    }
    main = re.search(
        r"int\s+main\s*\([^)]*\)\s*\{.*?^\}",
        source,
        re.DOTALL | re.MULTILINE,
    )
    if main is None:
        raise RuntimeError("cannot find main() in attn_temp.cu")
    main_text = re.sub(
        r"^\s*benchmark<.*?\);\s*$", "", main.group(0), flags=re.MULTILINE
    )
    main_text = main_text.replace(
        "  return 0;", f"  {calls[(dtype, causal)]}\n\n  return 0;"
    )
    main_text = re.sub(r"\n{3,}", "\n\n", main_text)
    return source[:main.start()] + main_text + source[main.end():]


def _render_config(temp_src: str, config: TConfig) -> str:
    match = re.search(r"struct\s+TConfig\s*\{.*?\};", temp_src, re.DOTALL)
    if match is None:
        raise RuntimeError("cannot find TConfig in attn_temp.cu")

    struct = match.group(0)
    values = asdict(config)
    missing = set(TCONFIG_FIELDS) - values.keys()
    if missing:
        raise ValueError(f"TConfig is missing fields: {sorted(missing)}")
    for name in TCONFIG_FIELDS:
        value = values[name]
        if not isinstance(value, int):
            raise TypeError(f"TConfig.{name} must be int")
        struct = _replace_one(
            struct,
            (
                rf"(\bint\s+{name}\s*=\s*)-?\d+(\s*;)",
                rf"(\buint32_t\s+{name}\s*=\s*)-?\d+(\s*;)",
            ),
            rf"\g<1>{value}\g<2>",
            name,
        )
    return temp_src[:match.start()] + struct + temp_src[match.end():]


def _create_source_with_state(
    temp_src: str,
    config: TConfig,
    cu_src_dir: str | Path,
) -> tuple[Path, bool]:
    cu_src_dir = Path(cu_src_dir)
    cu_src_dir.mkdir(parents=True, exist_ok=True)
    cu_file_path = cu_src_dir / f"{config.name()}.cu"
    rendered = _render_config(temp_src, config)
    generated = not cu_file_path.exists() or cu_file_path.read_text() != rendered
    if generated:
        cu_file_path.write_text(rendered)
    return cu_file_path, generated


def create_source(
    temp_src: str,
    config: TConfig,
    cu_scr_dir: str | Path | None = None,
) -> Path:
    """Render one config-specific CUDA source and return its path.

    Existing files are left untouched when their contents already match, so a
    cache hit does not change the source timestamp and trigger a rebuild.
    """
    cu_scr_dir = HERE / "cuda_source" if cu_scr_dir is None else Path(cu_scr_dir)
    cu_file_path, _ = _create_source_with_state(temp_src, config, cu_scr_dir)
    return cu_file_path


def _write_cmake_project(
    project_dir: Path,
    cu_files: list[Path],
    register_usage_level: int,
    runtime_dir: Path,
) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    source_lines = "\n".join(
        f'add_single_source_executable("{source.resolve()}" {register_usage_level})'
        for source in cu_files
    )
    content = f"""cmake_minimum_required(VERSION 3.26)
set(PROJ_NAME new_tune_fa3_tests)
set(CMAKE_UTILS_DIR "{CMAKE_UTILS_DIR.resolve()}")
include(${{CMAKE_UTILS_DIR}}/cmakePrologue.cmake)
include(${{CMAKE_UTILS_DIR}}/utilFuncs.cmake)
project(${{PROJ_NAME}} LANGUAGES CXX CUDA)
include(${{CMAKE_UTILS_DIR}}/flashVenv.cmake)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "{runtime_dir.resolve()}")

{source_lines}
"""
    cmake_path = project_dir / "CMakeLists.txt"
    if not cmake_path.exists() or cmake_path.read_text() != content:
        cmake_path.write_text(content)


def build_source(
    register_usage_level: int = 10,
    cu_build_dir: str | Path | None = None,
    *,
    cu_files: Iterable[str | Path],
    arch: str = "90a",
    jobs: int = 16,
) -> dict[str, Path]:
    """Compile a batch of CUDA sources and return successful executables.

    Each register-usage level owns an independent CMake build tree. All config
    targets are submitted in one build command so ``jobs`` controls one build
    scheduler instead of launching competing Ninja processes.
    """
    if not 0 <= register_usage_level <= 10:
        raise ValueError("register_usage_level must be in [0, 10]")
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    cu_files = [Path(path).resolve() for path in cu_files]
    if not cu_files:
        return {}
    if any(not path.is_file() for path in cu_files):
        missing = [str(path) for path in cu_files if not path.is_file()]
        raise FileNotFoundError(f"CUDA source does not exist: {missing}")
    targets = [path.stem for path in cu_files]
    if len(set(targets)) != len(targets):
        raise ValueError("CUDA source stems must be unique")

    cu_build_dir = HERE / "build" if cu_build_dir is None else Path(cu_build_dir)
    level_dir = cu_build_dir / f"rl{register_usage_level}"
    project_dir = cu_build_dir / f"project_rl{register_usage_level}"
    runtime_dir = level_dir / "bin"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    _write_cmake_project(project_dir, cu_files, register_usage_level, runtime_dir)

    configure = [
        "cmake", "-Wno-dev", "-S", str(project_dir), "-B", str(level_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
    ]
    print("+", " ".join(configure), flush=True)
    subprocess.run(configure, check=True)

    common_build = [
        "cmake", "--build", str(level_dir),
        "--target", "fa3_prepare_scheduler", "--parallel", "1",
    ]
    print("+", " ".join(common_build), flush=True)
    subprocess.run(common_build, check=True)

    report_path = level_dir / "compile_index.json"
    previous: dict[str, dict] = {}
    if report_path.exists():
        try:
            previous = {
                item["target"]: item for item in json.loads(report_path.read_text())
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            previous = {}

    source_by_target = dict(zip(targets, cu_files))
    fingerprints = {
        target: sha256(source_by_target[target].read_bytes()).hexdigest()
        for target in targets
    }

    def build_one(target: str) -> tuple[str, int, list[str], str, bool, float]:
        """Build exactly one target; outer ThreadPool controls concurrency."""
        executable = runtime_dir / target
        old_mtime = executable.stat().st_mtime_ns if executable.is_file() else None
        command = [
            "cmake", "--build", str(level_dir),
            "--target", target, "--parallel", "1",
        ]
        start = perf_counter()
        process = subprocess.run(
            command, text=True, capture_output=True, check=False
        )
        elapsed = perf_counter() - start
        output = "\n".join(
            part for part in (process.stdout.strip(), process.stderr.strip())
            if part
        )
        warnings = [
            match.group(0) for match in PTXAS_PERF_LOSS_RE.finditer(output)
        ]
        new_mtime = executable.stat().st_mtime_ns if executable.is_file() else None
        rebuilt = old_mtime != new_mtime
        return target, process.returncode, warnings, output, rebuilt, elapsed

    worker_count = min(jobs, len(targets))
    print(
        f"building {len(targets)} config target(s) with "
        f"{worker_count} parallel worker(s)",
        flush=True,
    )
    build_results: dict[str, tuple[int, list[str], str, bool, float]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(build_one, target): target for target in targets}
        for completed, future in enumerate(as_completed(futures), 1):
            target, returncode, warnings, output, rebuilt, elapsed = future.result()
            build_results[target] = (
                returncode, warnings, output, rebuilt, elapsed
            )
            if returncode != 0:
                state = "compile failed"
            elif warnings:
                state = "performance warning"
            elif rebuilt:
                state = "compiled"
            else:
                state = "cached binary"
            print(
                f"[{completed}/{len(targets)}] {target}: "
                f"{state} ({elapsed:.1f}s)",
                flush=True,
            )
            if returncode != 0:
                print(output or "  compiler produced no output", flush=True)
            elif warnings:
                for warning in warnings:
                    print(f"  {warning}", flush=True)

    successful: dict[str, Path] = {}
    report_by_target = dict(previous)
    for target in targets:
        source = source_by_target[target]
        executable = runtime_dir / target
        fingerprint = fingerprints[target]
        returncode, warnings, output, rebuilt, elapsed = build_results[target]
        old = previous.get(target)
        if (
            not warnings
            and not rebuilt
            and old is not None
            and old.get("source_sha256") == fingerprint
            and old.get("status") == "performance_warning"
        ):
            warnings = list(old.get("warnings", []))

        if returncode != 0 or not executable.is_file():
            status = "compile_failed"
        elif warnings:
            status = "performance_warning"
        else:
            status = "success"
            successful[target] = executable.resolve()
        report_by_target[target] = {
            "target": target,
            "source": str(source),
            "source_sha256": fingerprint,
            "register_usage_level": register_usage_level,
            "status": status,
            "warnings": warnings,
            "elapsed_seconds": elapsed,
        }
    report = [report_by_target[target] for target in sorted(report_by_target)]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return successful


def compile_interface(
    shape: Shape,
    dtype: DType,
    causal: bool,
    configs: list[TConfig],
    arch: str = "90a",
    jobs: int = 16,
    register_usage_level: int = 10,
    build_dir: str | Path | None = None,
) -> list[CompileResult]:
    if not configs:
        raise ValueError("config list is empty")
    if not 0 <= register_usage_level <= 10:
        raise ValueError("register_usage_level must be in [0, 10]")

    case_name = _case_name(shape, dtype, causal)
    build_dir = HERE / case_name if build_dir is None else Path(build_dir)
    cu_scr_dir = build_dir / "cuda_source"
    cu_build_dir = build_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    temp_src = _render_case(ATTN_TEMPLATE.read_text(), shape, dtype, causal)
    unique_configs: list[TConfig] = []
    config_by_name: dict[str, TConfig] = {}
    for config in configs:
        name = config.name()
        if name in config_by_name:
            if config_by_name[name] != config:
                raise ValueError(f"different configs share source name {name!r}")
            continue
        config_by_name[name] = config
        unique_configs.append(config)

    sources: list[Path] = []
    generated_count = 0
    for config in unique_configs:
        source, generated = _create_source_with_state(temp_src, config, cu_scr_dir)
        sources.append(source)
        generated_count += int(generated)
    print(
        f"CUDA sources: {generated_count} generated, "
        f"{len(sources) - generated_count} cached"
    )

    executables = build_source(
        register_usage_level=register_usage_level,
        cu_build_dir=cu_build_dir,
        cu_files=sources,
        arch=arch,
        jobs=jobs,
    )
    compiled = [
        CompileResult(config=config, exec_file=executables[config.name()])
        for config in unique_configs
        if config.name() in executables
    ]
    print(f"compiled {len(compiled)}/{len(unique_configs)} configs successfully")
    return compiled


def _save_bench_results(
    result_path: Path,
    results: list[BenchResult],
    failed: list[dict],
    total_configs: int,
    successful_count: int,
    shape: Shape | None,
    dtype: DType | None,
    causal: bool | None,
) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_configs": total_configs,
        "successful": successful_count,
        "failed": len(failed),
        "ranked": len(results),
        "top_results": [
            {
                "rank": item.rank,
                "config": asdict(item.config),
                "executable": str(item.exec_file),
                "time_ms": item.time_ms,
                "tflops": item.tflops,
            }
            for item in results
        ],
        "failed_results": failed,
    }
    metadata = (shape, dtype, causal)
    if any(value is not None for value in metadata):
        if any(value is None for value in metadata):
            raise ValueError("shape, dtype and causal must be provided together")
        assert shape is not None and dtype is not None and causal is not None
        _case_name(shape, dtype, causal)
        payload = {
            "shape": {
                "B": shape[0],
                "H": shape[1],
                "S": shape[2],
                "D": shape[3],
            },
            "dtype": "fp16" if dtype is DType.FP16 else "fp8",
            "causal": causal,
            **payload,
        }
    result_path.write_text(json.dumps(payload, indent=2) + "\n")


def bench_interface(
    compiled: list[CompileResult],
    rank: int = 15,
    result_path: str | Path | None = None,
    timeout_seconds: float = 120.0,
    *,
    shape: Shape | None = None,
    dtype: DType | None = None,
    causal: bool | None = None,
) -> list[BenchResult]:
    if rank <= 0:
        raise ValueError("rank must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if not compiled:
        return []

    successful: list[tuple[CompileResult, float, float]] = []
    failed: list[dict] = []
    for index, item in enumerate(compiled, 1):
        executable = Path(item.exec_file)
        print(f"[{index}/{len(compiled)}] {executable.name}", flush=True)
        if not executable.is_file():
            failed.append({
                "config": asdict(item.config),
                "executable": str(executable),
                "error": "executable not found",
            })
            continue
        try:
            process = subprocess.run(
                [str(executable.resolve())],
                cwd=executable.parent,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
            )
            output = "\n".join(
                part for part in (process.stdout.strip(), process.stderr.strip()) if part
            )
            if process.returncode != 0:
                raise RuntimeError(
                    f"exit code {process.returncode}: {output or 'no output'}"
                )
            time_ms, tflops = parse_result(output)
            successful.append((item, time_ms, tflops))
            print(f"  time={time_ms:.3f} ms, tflops={tflops:.2f}")
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            failed.append({
                "config": asdict(item.config),
                "executable": str(executable),
                "error": str(error),
            })
            print(f"  failed: {error}")

    successful.sort(key=lambda item: item[2], reverse=True)
    results = [
        BenchResult(
            rank=index,
            config=item.config,
            exec_file=item.exec_file,
            time_ms=time_ms,
            tflops=tflops,
        )
        for index, (item, time_ms, tflops) in enumerate(successful[:rank], 1)
    ]
    if result_path is not None:
        _save_bench_results(
            Path(result_path), results, failed, len(compiled), len(successful),
            shape, dtype, causal,
        )
    return results


def parse_result(output: str) -> tuple[float, float]:
    """Return the last ``(time_ms, TFLOPS)`` pair printed by a benchmark."""
    matches = BENCH_RESULT_RE.findall(output)
    if not matches:
        raise RuntimeError(f"cannot parse benchmark output: {output!r}")
    time_ms, tflops = map(float, matches[-1])
    if time_ms <= 0 or tflops < 0:
        raise RuntimeError(
            f"invalid benchmark result: time_ms={time_ms}, tflops={tflops}"
        )
    return time_ms, tflops


def _case_name(shape: Shape, dtype: DType, causal: bool) -> str:
    if len(shape) != 4 or any(value <= 0 for value in shape):
        raise ValueError("shape must be a positive (B, H, S, D) tuple")
    if not isinstance(dtype, DType):
        raise TypeError("dtype must be a DType value")
    if not isinstance(causal, bool):
        raise TypeError("causal must be bool")
    batch, heads, seqlen, head_dim = shape
    mode = "causal" if causal else "noncausal"
    dty = "fp16" if dtype is DType.FP16 else "fp8"
    return f"test_b{batch}_h{heads}_s{seqlen}_d{head_dim}_{dty}_{mode}"


def run_interface(
    shape: Shape,
    dtype: DType,
    causal: bool,
    configs: list[TConfig],
    arch: str = "90a",
    jobs: int = 16,
    register_usage_level: int = 5,
    rank: int = 15,
    timeout_seconds: float = 120.0,
    src_dir: str | Path | None = None,
    result_dir: str | Path | None = None,
) -> list[BenchResult]:
    src_dir = HERE / "src" if src_dir is None else Path(src_dir)
    case_name = _case_name(shape, dtype, causal)
    build_dir = src_dir / case_name
    build_dir.mkdir(parents=True, exist_ok=True)

    compiled = compile_interface(
        shape=shape,
        dtype=dtype,
        causal=causal,
        configs=configs,
        arch=arch,
        jobs=jobs,
        register_usage_level=register_usage_level,
        build_dir=build_dir,
    )
    result_path = (
        Path(result_dir) / f"bench_result_{case_name}.json"
        if result_dir is not None else None
    )
    return bench_interface(
        compiled=compiled,
        rank=rank,
        result_path=result_path,
        timeout_seconds=timeout_seconds,
        shape=shape,
        dtype=dtype,
        causal=causal,
    )

# if __name__ == "__main__":
#     shape = (1, 16, 32768, 64)
#     dtype = DType.FP8
#     causal = True
#     cfg1 = TConfig(128, 128, 2, 24, 240, 1, 1, 2, 0)
#     cfg2 = TConfig(128, 128, 2, 24, 240, 0, 1, 2, 0)
#     cfg3 = TConfig(128, 128, 2, 24, 240, 1, 0, 2, 0)
#     cfg4 = TConfig(128, 128, 2, 24, 240, 0, 0, 2, 0)
#     result_dir = HERE / "results"
#     run_interface(shape=shape, dtype=dtype, causal=causal, configs=[cfg1, cfg2, cfg3, cfg4], result_dir=result_dir)
