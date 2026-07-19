#!/usr/bin/env python3

import argparse
import csv
import html
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_bool(value):
    value = str(value).strip().lower()
    if value in ("1", "true"):
        return True
    if value in ("0", "false"):
        return False
    raise ValueError(f"invalid boolean value: {value}")


def load_rows(path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "data_type": row["DataType"],
                    "comment": row["Comment"],
                    "batchsize": int(row["batchsize"]),
                    "nheads": int(row["nheads"]),
                    "seqlen": int(row["seqlen"]),
                    "headdim": int(row["headdim"]),
                    "is_causal": parse_bool(row["is_causal"]),
                    "time_ms": float(row["time_ms"]),
                }
            )
    return rows


def calc_attention_flops_full(row):
    batchsize = row["batchsize"]
    nheads = row["nheads"]
    seqlen = row["seqlen"]
    headdim = row["headdim"]
    is_causal = row["is_causal"]
    gemm_flops = 4 * batchsize * seqlen**2 * nheads * headdim
    ffma_flops = 2 * seqlen**2 * nheads * batchsize
    fadd_flops = seqlen**2 * nheads * batchsize
    fmul_flops = seqlen * headdim * nheads * batchsize
    return (gemm_flops + ffma_flops + fadd_flops) // (2 if is_causal else 1) + fmul_flops


def tflops(row):
    return calc_attention_flops_full(row) / (row["time_ms"] * 1e9)


def seqlen_label(seqlen):
    labels = {
        128: "128",
        256: "256",
        512: "512",
        1024: "1k",
        2048: "2k",
        4096: "4k",
        8192: "8k",
        16384: "16k",
        32768: "32k",
        65536: "64k",
    }
    return labels.get(seqlen, str(seqlen))


def rows_by_seqlen(rows, headdim, is_causal):
    return {
        row["seqlen"]: row
        for row in rows
        if row["headdim"] == headdim and row["is_causal"] == is_causal
    }


def make_groups(fa3_py_rows, fa3_cpp_rows, fat_rows):
    head_dims = sorted({row["headdim"] for row in fa3_py_rows + fa3_cpp_rows + fat_rows})
    groups = []
    for headdim in head_dims:
        for is_causal in (False, True):
            fa3_py = rows_by_seqlen(fa3_py_rows, headdim, is_causal)
            fa3_cpp = rows_by_seqlen(fa3_cpp_rows, headdim, is_causal)
            fat = rows_by_seqlen(fat_rows, headdim, is_causal)
            seqlens = sorted(set(fa3_py) & set(fa3_cpp) & set(fat))
            if seqlens:
                groups.append(
                    (
                        headdim,
                        is_causal,
                        [fa3_py[seqlen] for seqlen in seqlens],
                        [fa3_cpp[seqlen] for seqlen in seqlens],
                        [fat[seqlen] for seqlen in seqlens],
                    )
                )
    return groups


def bar_chart(title, fa3_py, fa3_cpp, fat):
    width, height = 640, 390
    left, right, top, bottom = 68, 18, 42, 58
    plot_w = width - left - right
    plot_h = height - top - bottom
    bar_w = 15
    inner_gap = 4
    group_w = 3 * bar_w + 2 * inner_gap

    seqlens = [row["seqlen"] for row in fa3_py]
    series = (
        ("FA3-py", [tflops(row) for row in fa3_py], "#2563eb"),
        ("FA3-cpp", [tflops(row) for row in fa3_cpp], "#dc2626"),
        ("FAT", [tflops(row) for row in fat], "#16a34a"),
    )
    max_y = max(value for _, values, _ in series for value in values) * 1.15
    min_y = 0.0
    step_count = 5
    y_ticks = [max_y * i / step_count for i in range(step_count + 1)]

    def y_map(value):
        return top + (max_y - value) / (max_y - min_y) * plot_h

    def group_center(idx):
        if len(seqlens) == 1:
            return left + plot_w / 2
        usable_w = plot_w - group_w
        return left + group_w / 2 + idx * usable_w / (len(seqlens) - 1)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{width / 2}" y="22" text-anchor="middle" class="title">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
    ]

    for tick in y_ticks:
        y = y_map(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" class="tick">{tick:.0f}</text>')

    for idx, seqlen in enumerate(seqlens):
        center = group_center(idx)
        x0 = center - group_w / 2
        for series_idx, (name, values, color) in enumerate(series):
            x = x0 + series_idx * (bar_w + inner_gap)
            value = values[idx]
            y = y_map(value)
            h = top + plot_h - y
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w}" height="{h:.2f}" fill="{color}" opacity="0.82"/>')
            parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{y - 4:.2f}" text-anchor="middle" class="annot">{value:.0f}</text>')
            parts.append(f'<title>{name} {seqlen}: {value:.2f} TFLOPS</title>')
        parts.append(f'<text x="{center:.2f}" y="{top + plot_h + 20}" text-anchor="middle" class="tick">{seqlen_label(seqlen)}</text>')

    legend_y = top + 15
    for idx, (name, _, color) in enumerate(series):
        x = left + 12 + idx * 88
        parts.append(f'<rect x="{x}" y="{legend_y - 10}" width="14" height="14" fill="{color}" opacity="0.82"/>')
        parts.append(f'<text x="{x + 20}" y="{legend_y + 2}" class="legend">{name}</text>')

    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 16}" text-anchor="middle" class="label">Sequence Length</text>')
    parts.append(f'<text x="16" y="{top + plot_h / 2}" transform="rotate(-90 16 {top + plot_h / 2})" text-anchor="middle" class="label">TFLOPS</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def report_label(dtype):
    return f"H100 {dtype.upper()} FA3-py vs FA3-cpp vs FAT"


def write_html(fa3_py_rows, fa3_cpp_rows, fat_rows, output_path, dtype):
    perf_cards = []
    for headdim, is_causal, fa3_py, fa3_cpp, fat in make_groups(fa3_py_rows, fa3_cpp_rows, fat_rows):
        suffix = f"headdim={headdim}, {'causal' if is_causal else 'non-causal'}"
        perf_cards.append(bar_chart(f"H100 {dtype.upper()}, {suffix}", fa3_py, fa3_cpp, fat))

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{report_label(dtype)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; background: #ffffff; }}
    h1 {{ font-size: 24px; margin: 0 0 8px; }}
    h2 {{ font-size: 18px; margin: 28px 0 12px; }}
    .grid-wrap {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 18px; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 8px; background: white; }}
    svg {{ width: 100%; height: auto; }}
    .axis {{ stroke: #111827; stroke-width: 1.2; }}
    .grid {{ stroke: #d1d5db; stroke-width: 0.8; stroke-dasharray: 3 4; }}
    .ref {{ stroke: #111827; stroke-width: 1; stroke-dasharray: 5 4; opacity: 0.7; }}
    .title {{ font-size: 15px; font-weight: 700; }}
    .tick {{ font-size: 11px; fill: #4b5563; }}
    .label {{ font-size: 12px; fill: #374151; }}
    .legend {{ font-size: 12px; fill: #111827; }}
    .annot {{ font-size: 10px; fill: #111827; }}
  </style>
</head>
<body>
  <h1>{report_label(dtype)}</h1>
  <p>Bar heights use the same full-attention FLOP formula as <code>plot/plot.py</code>.</p>
  <h2>TFLOPS Bar Chart</h2>
  <div class="grid-wrap">
    {''.join(f'<div class="card">{card}</div>' for card in perf_cards)}
  </div>
</body>
</html>
"""
    output_path.write_text(doc)


def draw_text(draw, xy, text, fill="#111827", font=None, anchor=None):
    draw.text(xy, text, fill=fill, font=font, anchor=anchor)


def draw_bar_panel(draw, box, title, fa3_py, fa3_cpp, fat, font, small_font):
    x0, y0, x1, y1 = box
    left, right, top, bottom = 58, 16, 38, 46
    plot_x0, plot_y0 = x0 + left, y0 + top
    plot_x1, plot_y1 = x1 - right, y1 - bottom
    plot_w, plot_h = plot_x1 - plot_x0, plot_y1 - plot_y0
    draw.rectangle(box, outline="#d1d5db", width=1)
    draw_text(draw, ((x0 + x1) / 2, y0 + 16), title, font=font, anchor="mm")
    draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill="#111827", width=1)
    draw.line((plot_x0, plot_y0, plot_x0, plot_y1), fill="#111827", width=1)

    seqlens = [row["seqlen"] for row in fa3_py]
    series = [
        ("FA3-py", [tflops(row) for row in fa3_py], "#2563eb"),
        ("FA3-cpp", [tflops(row) for row in fa3_cpp], "#dc2626"),
        ("FAT", [tflops(row) for row in fat], "#16a34a"),
    ]
    max_y = max(value for _, values, _ in series for value in values) * 1.15
    ylabel = "TFLOPS"

    for idx in range(6):
        value = max_y * idx / 5
        y = plot_y1 - (value / max_y) * plot_h
        draw.line((plot_x0, y, plot_x1, y), fill="#e5e7eb", width=1)
        draw_text(draw, (plot_x0 - 8, y), f"{value:.0f}", font=small_font, fill="#4b5563", anchor="rm")

    group_count = len(seqlens)
    group_gap = plot_w / group_count
    bar_w = min(20, group_gap * 0.18)
    offsets = [-bar_w * 1.15, 0, bar_w * 1.15]

    for group_idx, seqlen in enumerate(seqlens):
        center = plot_x0 + group_gap * (group_idx + 0.5)
        draw_text(draw, (center, plot_y1 + 18), seqlen_label(seqlen), font=small_font, fill="#4b5563", anchor="mm")
        for series_idx, (_, values, color) in enumerate(series):
            value = values[group_idx]
            bar_x0 = center + offsets[series_idx] - bar_w / 2
            bar_x1 = bar_x0 + bar_w
            bar_y0 = plot_y1 - (value / max_y) * plot_h
            draw.rectangle((bar_x0, bar_y0, bar_x1, plot_y1), fill=color)
            draw_text(draw, ((bar_x0 + bar_x1) / 2, bar_y0 - 7), f"{value:.0f}", font=small_font, anchor="mm")

    legend_x = plot_x0 + 10
    legend_y = plot_y0 + 8
    for idx, (name, _, color) in enumerate(series):
        lx = legend_x + idx * 92
        draw.rectangle((lx, legend_y, lx + 12, legend_y + 12), fill=color)
        draw_text(draw, (lx + 17, legend_y + 6), name, font=small_font, anchor="lm")

    draw_text(draw, ((plot_x0 + plot_x1) / 2, y1 - 15), "Sequence Length", font=small_font, fill="#374151", anchor="mm")
    draw_text(draw, (x0 + 18, (plot_y0 + plot_y1) / 2), ylabel, font=small_font, fill="#374151", anchor="mm")


def write_png(fa3_py_rows, fa3_cpp_rows, fat_rows, output_path, dtype):
    groups = make_groups(fa3_py_rows, fa3_cpp_rows, fat_rows)
    panel_w, panel_h = 720, 390
    margin, gap = 36, 22
    title_h, section_h = 54, 34
    width = margin * 2 + panel_w * 2 + gap
    rows = (len(groups) + 1) // 2
    height = margin + title_h + section_h + panel_h * rows + gap * max(rows - 1, 0) + margin
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()

    draw_text(draw, (width / 2, margin + 12), report_label(dtype), font=font, anchor="mm")
    y = margin + title_h
    draw_text(draw, (margin, y), "TFLOPS Bar Chart", font=font, anchor="lm")
    y += section_h
    for idx, (headdim, is_causal, fa3_py, fa3_cpp, fat) in enumerate(groups):
        row, col = divmod(idx, 2)
        x0 = margin + col * (panel_w + gap)
        y0 = y + row * (panel_h + gap)
        title = f"H100 {dtype.upper()}, headdim={headdim}, {'causal' if is_causal else 'non-causal'}"
        draw_bar_panel(draw, (x0, y0, x0 + panel_w, y0 + panel_h), title, fa3_py, fa3_cpp, fat, font, small_font)

    image.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("fp16", "fp8"), default="fp8")
    parser.add_argument("--ours", type=Path, default=None, help="FA3 Python results CSV")
    parser.add_argument("--fa3", type=Path, default=None)
    parser.add_argument("--fat", type=Path, default=None)
    parser.add_argument("--html-out", type=Path, default=None)
    parser.add_argument("--png-out", type=Path, default=None)
    args = parser.parse_args()

    if args.ours is None:
        args.ours = Path(f"fa3_{args.dtype}_results.csv")
    if args.fa3 is None:
        args.fa3 = Path(f"data_fa3_h100_{args.dtype}.csv")
    if args.fat is None:
        args.fat = Path(f"data_ours_h100_{args.dtype}.csv")
    if args.html_out is None:
        args.html_out = Path(f"h100_{args.dtype}_bar_report.html")
    if args.png_out is None:
        args.png_out = Path(f"h100_{args.dtype}_bar_report.png")

    fa3_py_rows = load_rows(args.ours)
    fa3_cpp_rows = load_rows(args.fa3)
    fat_rows = load_rows(args.fat)
    write_html(fa3_py_rows, fa3_cpp_rows, fat_rows, args.html_out, args.dtype)
    write_png(fa3_py_rows, fa3_cpp_rows, fat_rows, args.png_out, args.dtype)
    print(f"Saved {args.html_out}")
    print(f"Saved {args.png_out}")


if __name__ == "__main__":
    main()
