#!/usr/bin/env python3
"""Compile and numerically verify new_tune FA3 configurations."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from interface import compile_interface
from utils import CompileResult, DType, Shape, TConfig


HERE = Path(__file__).resolve().parent
CONFIG_FIELDS = {field.name for field in fields(TConfig)}
CASE_NAME_RE = re.compile(
    r"(?:bench_result_)?test_"
    r"b(?P<B>\d+)_h(?P<H>\d+)_s(?P<S>\d+)_d(?P<D>\d+)_"
    r"(?P<dtype>fp8|fp16)_(?P<causal>causal|noncausal)"
)


@dataclass(frozen=True)
class VerifyCase:
    config: TConfig
    executable: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile configurations from a new_tune result JSON and compare "
            "each custom kernel against LibTorch scaled-dot-product attention."
        )
    )
    parser.add_argument(
        "result_json", type=Path,
        help="bench_result_*.json produced by new_tune",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--index", type=int, default=1,
        help="1-based top_results index to verify (default: 1)",
    )
    selection.add_argument(
        "--all", action="store_true", help="verify every entry in top_results"
    )
    parser.add_argument("--arch", default="90a")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--register-usage-level", type=int, default=10)
    parser.add_argument(
        "--recompile", action="store_true",
        help="ignore recorded executables and rebuild from the current sources",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument(
        "--seqlen", type=int,
        help="override S from the result JSON for boundary/tail tests",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--build-dir", type=Path,
        help="verification build directory (default: new_tune/verify_src/<case>)",
    )
    return parser.parse_args()


def _parse_dtype(value: object, source: Path) -> DType:
    if value == "fp16":
        return DType.FP16
    if value == "fp8":
        return DType.FP8
    raise ValueError(f"unsupported dtype in {source}: {value!r}")


def _metadata_from_filename(path: Path) -> tuple[Shape, DType, bool]:
    match = CASE_NAME_RE.fullmatch(path.stem)
    if match is None:
        raise ValueError(
            f"{path} has no shape/dtype/causal metadata and its filename does "
            "not match bench_result_test_b<B>_h<H>_s<S>_d<D>_<dtype>_<mode>.json"
        )
    shape: Shape = tuple(
        int(match.group(name)) for name in ("B", "H", "S", "D")
    )  # type: ignore[assignment]
    dtype = _parse_dtype(match.group("dtype"), path)
    causal = match.group("causal") == "causal"
    return shape, dtype, causal


def _load_metadata(
    payload: dict[str, Any], path: Path,
) -> tuple[Shape, DType, bool]:
    metadata_keys = ("shape", "dtype", "causal")
    present = tuple(key in payload for key in metadata_keys)
    if not any(present):
        return _metadata_from_filename(path)
    if not all(present):
        missing = [key for key in metadata_keys if key not in payload]
        raise ValueError(f"{path} is missing metadata fields: {missing}")

    shape_obj = payload["shape"]
    if not isinstance(shape_obj, dict):
        raise ValueError(f"shape in {path} must be an object with B/H/S/D")
    try:
        shape: Shape = tuple(
            int(shape_obj[name]) for name in ("B", "H", "S", "D")
        )  # type: ignore[assignment]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid shape metadata in {path}") from error
    if any(value <= 0 for value in shape):
        raise ValueError(f"shape values in {path} must be positive")

    dtype = _parse_dtype(payload["dtype"], path)
    causal = payload["causal"]
    if not isinstance(causal, bool):
        raise ValueError(f"causal in {path} must be a JSON boolean")
    return shape, dtype, causal


def _load_config(value: object, path: Path, position: int) -> TConfig:
    if not isinstance(value, dict):
        raise ValueError(f"top_results[{position}].config in {path} must be an object")
    keys = set(value)
    missing = CONFIG_FIELDS - keys
    extra = keys - CONFIG_FIELDS
    if missing or extra:
        raise ValueError(
            f"invalid config fields at top_results[{position}] in {path}: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if any(isinstance(value[name], bool) or not isinstance(value[name], int)
           for name in CONFIG_FIELDS):
        raise ValueError(
            f"all config values at top_results[{position}] in {path} must be integers"
        )
    return TConfig(**{name: value[name] for name in CONFIG_FIELDS})


def load_cases(
    path: Path,
    index: int,
    verify_all: bool,
) -> tuple[Shape, DType, bool, list[VerifyCase]]:
    payload: dict[str, Any] = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    shape, dtype, causal = _load_metadata(payload, path)

    entries = payload.get("top_results")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} contains no top_results")
    if verify_all:
        selected = list(enumerate(entries))
    else:
        if index <= 0 or index > len(entries):
            raise ValueError(f"--index must be in [1, {len(entries)}]")
        selected = [(index - 1, entries[index - 1])]

    cases: list[VerifyCase] = []
    for position, entry in selected:
        if not isinstance(entry, dict) or "config" not in entry:
            raise ValueError(f"top_results[{position}] in {path} has no config")
        config = _load_config(entry["config"], path, position)
        executable_value = entry.get("executable")
        if executable_value is None:
            executable = None
        elif not isinstance(executable_value, str) or not executable_value:
            raise ValueError(
                f"top_results[{position}].executable in {path} must be a path string"
            )
        else:
            executable = Path(executable_value).expanduser()
            if not executable.is_absolute():
                executable = path.parent / executable
            executable = executable.resolve()
        cases.append(VerifyCase(config=config, executable=executable))
    return shape, dtype, causal, cases


def verify_executables(
    compiled: list[CompileResult],
    seed: int,
    atol: float | None,
    rtol: float | None,
    timeout: float,
) -> bool:
    passed = True
    for position, item in enumerate(compiled, 1):
        executable = item.exec_file.resolve()
        command = [str(executable), "--verify", f"--seed={seed}"]
        if atol is not None:
            command.append(f"--atol={atol}")
        if rtol is not None:
            command.append(f"--rtol={rtol}")
        print(
            f"[{position}/{len(compiled)}] {shlex.join(command)}",
            flush=True,
        )
        try:
            process = subprocess.run(
                command,
                cwd=executable.parent,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            passed = False
            print(f"  verification timed out after {timeout:g} seconds")
            continue
        if process.stdout:
            print(process.stdout, end="" if process.stdout.endswith("\n") else "\n")
        if process.returncode != 0:
            passed = False
            print(f"  verification exited with code {process.returncode}")
    return passed


def _case_name(shape: Shape, dtype: DType, causal: bool) -> str:
    batch, heads, seqlen, head_dim = shape
    dtype_name = "fp16" if dtype is DType.FP16 else "fp8"
    mode = "causal" if causal else "noncausal"
    return (
        f"test_b{batch}_h{heads}_s{seqlen}_d{head_dim}_"
        f"{dtype_name}_{mode}"
    )


def prepare_executables(
    *,
    cases: list[VerifyCase],
    shape: Shape,
    dtype: DType,
    causal: bool,
    force_recompile: bool,
    arch: str,
    jobs: int,
    register_usage_level: int,
    build_dir: Path,
) -> list[CompileResult]:
    reusable: dict[str, CompileResult] = {}
    configs_to_compile: list[TConfig] = []
    queued_names: set[str] = set()

    for case in cases:
        name = case.config.name()
        if (
            not force_recompile
            and case.executable is not None
            and case.executable.is_file()
        ):
            reusable[name] = CompileResult(case.config, case.executable)
            continue
        if name not in queued_names:
            queued_names.add(name)
            configs_to_compile.append(case.config)

    compiled_by_name: dict[str, CompileResult] = {}
    if configs_to_compile:
        reason = (
            "forced recompilation"
            if force_recompile else
            "recorded executable missing"
        )
        print(
            f"compiling {len(configs_to_compile)} configuration(s): {reason}",
            flush=True,
        )
        compiled_by_name = {
            item.config.name(): item
            for item in compile_interface(
                shape=shape,
                dtype=dtype,
                causal=causal,
                configs=configs_to_compile,
                arch=arch,
                jobs=jobs,
                register_usage_level=register_usage_level,
                build_dir=build_dir,
            )
        }
    else:
        print(
            f"reusing {len(reusable)} executable(s) recorded in the result JSON",
            flush=True,
        )

    prepared: list[CompileResult] = []
    missing: list[str] = []
    for case in cases:
        name = case.config.name()
        item = compiled_by_name.get(name) or reusable.get(name)
        if item is None:
            missing.append(name)
        else:
            prepared.append(item)
    if missing:
        raise RuntimeError(
            f"failed to prepare {len(missing)} executable(s): {missing}"
        )
    return prepared


def main() -> int:
    args = parse_args()
    try:
        result_path = args.result_json.resolve()
        shape, dtype, causal, cases = load_cases(
            result_path, args.index, args.all
        )
        if args.seqlen is not None:
            if args.seqlen <= 0:
                raise ValueError("--seqlen must be positive")
            shape = (shape[0], shape[1], args.seqlen, shape[3])
        if args.jobs <= 0:
            raise ValueError("--jobs must be positive")
        if not 0 <= args.register_usage_level <= 10:
            raise ValueError("--register-usage-level must be in [0, 10]")
        if not 0 <= args.seed <= 2**64 - 1:
            raise ValueError("--seed must be in [0, 2^64 - 1]")
        if args.atol is not None and args.atol < 0:
            raise ValueError("--atol must be nonnegative")
        if args.rtol is not None and args.rtol < 0:
            raise ValueError("--rtol must be nonnegative")
        if args.timeout <= 0:
            raise ValueError("--timeout must be positive")

        build_dir = (
            args.build_dir.resolve()
            if args.build_dir is not None
            else HERE / "verify_src" / _case_name(shape, dtype, causal)
        )
        compiled = prepare_executables(
            cases=cases,
            shape=shape,
            dtype=dtype,
            causal=causal,
            force_recompile=args.recompile or args.seqlen is not None,
            arch=args.arch,
            jobs=args.jobs,
            register_usage_level=args.register_usage_level,
            build_dir=build_dir,
        )
        return 0 if verify_executables(
            compiled, args.seed, args.atol, args.rtol, args.timeout
        ) else 1
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"verify.py: {error}", file=sys.stderr)
        return 2

# python verify.py results/bench_result_test_b1_h16_s32768_d64_fp16_noncausal.json
# python verify.py result.json --recompile
if __name__ == "__main__":
    raise SystemExit(main())
