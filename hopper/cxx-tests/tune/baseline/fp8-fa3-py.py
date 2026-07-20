import argparse
import csv
import torch
import flash_attn_interface


SHAPES = [(1, 16, 32768, 256, 0)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--output", type=str, default="fa3_fp8_results.csv")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def benchmark_one_shape(batchsize, nheads, seqlen, headdim, is_causal, warmup, iters):
    device = "cuda"

    # FA3 的 layout 通常是 [B, S, H, D]
    q = torch.randn(
        batchsize, seqlen, nheads, headdim,
        device=device,
        dtype=torch.float16,
    )
    k = torch.randn(
        batchsize, seqlen, nheads, headdim,
        device=device,
        dtype=torch.float16,
    )
    v = torch.randn(
        batchsize, seqlen, nheads, headdim,
        device=device,
        dtype=torch.float16,
    )

    # FP8 输入
    # Hopper 上一般使用 torch.float8_e4m3fn
    q = q.to(torch.float8_e4m3fn)
    k = k.to(torch.float8_e4m3fn)
    v = v.to(torch.float8_e4m3fn)

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

    # 防止极端情况下被优化掉，虽然 PyTorch 调用一般不会
    _ = out

    return avg_ms


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows = []

    for batchsize, nheads, seqlen, headdim, is_causal in SHAPES:
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
            "DataType": "FP8-FP32",
            "Comment": "fa3",
            "batchsize": batchsize,
            "nheads": nheads,
            "seqlen": seqlen,
            "headdim": headdim,
            "is_causal": is_causal,
            "time_ms": time_ms,
        })

        # 释放显存，避免不同 shape 之间显存碎片影响
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