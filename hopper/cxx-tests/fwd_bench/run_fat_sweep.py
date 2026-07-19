#!/usr/bin/env python3
import argparse
import csv
import itertools
import math
import shlex
import subprocess
import sys
from pathlib import Path


NCU_METRICS = (
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
)
CSV_FIELDS = ["B", "H", "S", "D", "causal", "tc_utilization"]


def parse_int_list(value):
    return [int(item) for item in value.replace(" ", ",").split(",") if item.strip()]


def parse_float(value):
    value = value.strip().strip('"').replace(",", "")
    if not value or value.lower() in {"n/a", "nan"}:
        return math.nan
    return float(value)


def metric_value_from_row(row, metric_name):
    for idx, cell in enumerate(row):
        if cell.strip().strip('"') != metric_name:
            continue
        for value_idx in range(len(row) - 1, idx, -1):
            try:
                value = parse_float(row[value_idx])
            except ValueError:
                continue
            unit = row[value_idx - 1] if value_idx - 1 > idx else ""
            return value, unit
    return None


def metric_value_from_text_line(line, metric_name):
    if metric_name not in line:
        return None
    fields = line.split(metric_name, 1)[1].strip().split()
    if not fields:
        return None
    try:
        value = parse_float(fields[-1])
    except ValueError:
        return None
    unit = fields[-2] if len(fields) >= 2 else ""
    return value, unit


def parse_ncu_output(output):
    tc_utilization = math.nan

    for line in output.splitlines():
        tc_result = metric_value_from_text_line(
            line, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
        )
        if tc_result is not None:
            value, _unit = tc_result
            tc_utilization = value

    for row in csv.reader(output.splitlines()):
        if not row:
            continue
        tc_result = metric_value_from_row(
            row, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
        )
        if tc_result is not None:
            value, _unit = tc_result
            tc_utilization = value

    return tc_utilization


def format_number(value):
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def median_finite(values):
    finite_values = sorted(value for value in values if not math.isnan(value))
    if not finite_values:
        return math.nan
    mid = len(finite_values) // 2
    if len(finite_values) % 2:
        return finite_values[mid]
    return 0.5 * (finite_values[mid - 1] + finite_values[mid])


def print_subprocess_output(output):
    if not output:
        return
    if isinstance(output, bytes):
        output = output.decode(errors="replace")
    print(output, end="")


def write_rows(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def make_bench_cmd(args, csv_path, B, H, S, D):
    cmd = [
        str(args.bin),
        "--single",
        "--batch-size",
        str(B),
        "--num-heads",
        str(H),
        "--seq-len",
        str(S),
        "--head-dim",
        str(D),
        "--causal",
        str(args.causal),
        "--warmup",
        str(args.warmup),
        "--iter",
        str(args.iter),
    ]
    if csv_path is not None:
        cmd[2:2] = ["--csv", str(csv_path)]
    return cmd


def run_shape(args, B, H, S, D):
    tmp_csv = args.build / f"bench_B{B}_H{H}_S{S}_D{D}.csv"
    cmd = make_bench_cmd(args, tmp_csv, B, H, S, D)
    print(shlex.join(cmd), flush=True)

    try:
        completed = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        print_subprocess_output(exc.stdout)
        print(
            f"FA-T benchmark timed out after {args.timeout_sec}s for B={B} H={H} S={S} D={D}",
            file=sys.stderr,
        )
        return math.nan
    print_subprocess_output(completed.stdout)
    if completed.returncode != 0:
        print(
            f"FA-T benchmark failed for B={B} H={H} S={S} D={D} "
            f"with return code {completed.returncode}",
            file=sys.stderr,
        )
        return math.nan
    return math.nan


def profile_shape(args, B, H, S, D):
    bench_cmd = make_bench_cmd(args, None, B, H, S, D)
    cmd = [
        args.ncu,
        "--metrics",
        NCU_METRICS,
        "--launch-skip",
        str(args.launch_skip),
        "--launch-count",
        str(args.launch_count),
    ] + bench_cmd
    # print(cmd)
    if args.kernel_name:
        cmd[1:1] = [
            "--kernel-name-base",
            "function",
            "--kernel-name",
            args.kernel_name,
        ]

    print(shlex.join(cmd), flush=True)

    try:
        completed = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        print_subprocess_output(exc.stdout)
        print(
            f"ncu timed out after {args.timeout_sec}s for B={B} H={H} S={S} D={D}",
            file=sys.stderr,
        )
        return math.nan
    if args.print_ncu_output:
        print_subprocess_output(completed.stdout)
    tc_utilization = parse_ncu_output(completed.stdout)
    if completed.returncode != 0 or math.isnan(tc_utilization):
        print(
            f"ncu failed or metrics could not be parsed for B={B} H={H} S={S} D={D}",
            file=sys.stderr,
        )
    return tc_utilization


def profile_shape_repeated(args, B, H, S, D):
    values = []
    for repeat_idx in range(1, args.ncu_repeats + 1):
        if args.ncu_repeats > 1:
            print(f"repeat {repeat_idx}/{args.ncu_repeats}", flush=True)
        values.append(profile_shape(args, B, H, S, D))
    if args.ncu_repeats > 1:
        formatted = ", ".join(format_number(value) for value in values)
        print(f"tc_utilization repeats = [{formatted}]", flush=True)
    return median_finite(values)


def iter_shapes(args):
    return itertools.product(
        args.batch_sizes,
        args.num_heads,
        args.seq_lens,
        args.head_dims,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Sweep FA-T or FA3 single-shape benchmark and save tensor-core utilization CSV."
    )
    parser.add_argument("--method", choices=["fat", "fa3"], default="fat")
    parser.add_argument("--dtype", choices=["fp16", "fp8"], default="fp16")
    # parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 2, 4, 8, 16, 32, 64, 128])
    # parser.add_argument("--num-heads", type=parse_int_list, default=[16, 32])
    # parser.add_argument("--seq-lens", type=parse_int_list, default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    # parser.add_argument("--head-dims", type=parse_int_list, default=[64, 128])
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[1])
    parser.add_argument("--num-heads", type=parse_int_list, default=[16])
    parser.add_argument("--seq-lens", type=parse_int_list, default=[8192])
    parser.add_argument("--head-dims", type=parse_int_list, default=[128])
    # parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 2, 4, 8, 16, 32, 64, 128])
    # parser.add_argument("--num-heads", type=parse_int_list, default=[8, 12, 16, 32])
    # parser.add_argument("--seq-lens", type=parse_int_list, default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    # parser.add_argument("--head-dims", type=parse_int_list, default=[64, 128])
    parser.add_argument("--causal", type=int, choices=[0, 1], default=0)
    parser.add_argument("--build", type=Path, default=None)
    parser.add_argument("--bin", type=Path, default=None)

    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iter", type=int, default=10)
    parser.add_argument("--launch-skip", type=int, default=55)
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument("--ncu-repeats", type=int, default=1)
    parser.add_argument("--kernel-name", default="regex:.*(FlashAttnFwd|device_kernel).*")

    parser.add_argument("--profile-ncu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--print-ncu-output", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    return parser


def default_bench_name(method, dtype):
    suffix = "-orig" if method == "fa3" else ""
    return f"{dtype}-fwd-bench{suffix}"


def default_csv_name(method, dtype, causal):
    method_name = "fa3" if method == "fa3" else "fat"
    if dtype == "fp16":
        return f"{method_name}_sweep_{causal}.csv"
    return f"{method_name}_{dtype}_sweep_{causal}.csv"


def main():
    args = build_parser().parse_args()
    script_dir = Path(__file__).resolve().parent
    default_build_dir = script_dir / "build"
    tc_utilization_csv_dir = default_build_dir / default_csv_name(args.method, args.dtype, args.causal)
    if args.build is None:
        args.build = default_build_dir
    if args.bin is None:
        args.bin = args.build / default_bench_name(args.method, args.dtype)
    if args.timeout_sec <= 0:
        args.timeout_sec = None
    if args.ncu_repeats < 1:
        raise ValueError("--ncu-repeats must be >= 1")
    shapes = list(iter_shapes(args))

    if not args.bin.exists():
        raise FileNotFoundError(
            f"benchmark binary not found: {args.bin}\n"
            "Build it first, for example:\n"
            " ninja "
        )
    
    if not args.profile_ncu:
        print(
            "warning: --no-profile-ncu cannot collect tensor-core utilization; "
            "rows will use nan tc_utilization.",
            file=sys.stderr,
        )

    print(f"num shapes = {len(shapes)}")
    print(f"method     = {args.method}")
    print(f"dtype      = {args.dtype}")
    print(f"build      = {args.build}")
    print(f"bin        = {args.bin}")
    print(f"warmup     = {args.warmup}")
    print(f"iter       = {args.iter}")
    print(f"causal     = {args.causal}")
    print(f"profile ncu= {args.profile_ncu}")
    print(f"ncu repeats= {args.ncu_repeats}")
    print(f"print ncu  = {args.print_ncu_output}")
    print(f"timeout s  = {args.timeout_sec}")

    rows = []
    for idx, (B, H, S, D) in enumerate(shapes, 1):
        # xie
        # if B * S != 16384 or H * D != 2048:
        #     continue
        if B * S > 32768 or B * S < 2048:
            continue

        print(f"[{idx}/{len(shapes)}] B={B} H={H} S={S} D={D}", flush=True)
        if args.profile_ncu:
            tc_utilization = profile_shape_repeated(args, B, H, S, D)
            row = {
                "B": B,
                "H": H,
                "S": S,
                "D": D,
                "causal": args.causal,
                "tc_utilization": format_number(tc_utilization),
            }
            rows.append(row)
            print(row, flush=True)
        else:
            run_shape(args, B, H, S, D)
    if args.profile_ncu:
        write_rows(tc_utilization_csv_dir, rows)
        print(f"saved rows = {len(rows)} to {tc_utilization_csv_dir}")

if __name__ == "__main__":
    main()
