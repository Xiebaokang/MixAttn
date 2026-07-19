import argparse
import csv
import torch
import flash_attn_interface


DEFAULT_SHAPES = [(1, 16, 32768, 256, 0)]


def parse_shape(value):
    try:
        fields = tuple(int(field) for field in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape fields must be integers") from error
    if len(fields) != 5:
        raise argparse.ArgumentTypeError("shape must be B,H,S,D,causal")
    if any(field <= 0 for field in fields[:4]) or fields[4] not in (0, 1):
        raise argparse.ArgumentTypeError(
            "B/H/S/D must be positive and causal must be 0 or 1"
        )
    return fields


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=250)
    parser.add_argument("--output", type=str, default="fa3_fp16_results.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shape", type=parse_shape, action="append",
        help="repeatable B,H,S,D,causal override (default: 1,16,32768,128,0)",
    )
    return parser.parse_args()


@torch.no_grad()
def benchmark_one_shape(batchsize, nheads, seqlen, headdim, is_causal, warmup, iters):
    device = "cuda"
    dtype = torch.float16

    # FlashAttention-3 Python interface 一般使用 [B, S, H, D]
    q = torch.randn(
        batchsize, seqlen, nheads, headdim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        batchsize, seqlen, nheads, headdim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn(
        batchsize, seqlen, nheads, headdim,
        device=device,
        dtype=dtype,
    )

    causal = bool(is_causal)

    # 预热
    for _ in range(warmup):
        out = flash_attn_interface.flash_attn_func(
            q, k, v,
            causal=causal,
        )

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        out = flash_attn_interface.flash_attn_func(
            q, k, v,
            causal=causal,
        )
    end.record()

    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / iters

    _ = out

    return avg_ms


def main():
    args = parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    shapes = args.shape or DEFAULT_SHAPES

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows = []

    for batchsize, nheads, seqlen, headdim, is_causal in shapes:
        print(
            f"Running: B={batchsize}, H={nheads}, "
            f"S={seqlen}, D={headdim}, causal={is_causal}"
        )

        try:
            time_ms = benchmark_one_shape(
                batchsize=batchsize,
                nheads=nheads,
                seqlen=seqlen,
                headdim=headdim,
                is_causal=is_causal,
                warmup=args.warmup,
                iters=args.iters,
            )
            print(f"  time_ms = {time_ms:.6f}")

        except Exception as e:
            print(f"  failed: {repr(e)}")
            time_ms = float("nan")

        rows.append({
            "DataType": "FP16-FP32",
            "Comment": "fa3",
            "batchsize": batchsize,
            "nheads": nheads,
            "seqlen": seqlen,
            "headdim": headdim,
            "is_causal": is_causal,
            "time_ms": time_ms,
        })

        torch.cuda.empty_cache()

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "DataType",
                "Comment",
                "batchsize",
                "nheads",
                "seqlen",
                "headdim",
                "is_causal",
                "time_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
