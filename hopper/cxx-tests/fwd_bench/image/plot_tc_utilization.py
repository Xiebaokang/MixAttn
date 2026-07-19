#!/usr/bin/env python3

import argparse
import csv
import html
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_float(value):
    value = value.strip().strip('"').replace(",", "")
    if not value or value.lower() in {"n/a", "nan"}:
        return math.nan
    return float(value)


def parse_bool(value):
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n", ""}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def load_rows(path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "B": int(row["B"]),
                    "H": int(row["H"]),
                    "S": int(row["S"]),
                    "D": int(row["D"]),
                    "causal": parse_bool(row.get("causal", "0")),
                    "tc_utilization": parse_float(row["tc_utilization"]),
                }
            )
    return rows


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


def shape_key(row):
    return row["B"], row["H"], row["S"], row["D"], row["causal"]


def causal_label(is_causal):
    return "causal" if is_causal else "non-causal"


def make_groups(fat_rows, fa3_rows):
    fat_by_shape = {shape_key(row): row for row in fat_rows}
    fa3_by_shape = {shape_key(row): row for row in fa3_rows}
    common_keys = sorted(set(fat_by_shape) & set(fa3_by_shape), key=lambda x: (x[4], x[3], x[1], x[2], x[0]))

    groups_by_causal_hd = {}
    for key in common_keys:
        B, H, S, D, is_causal = key
        groups_by_causal_hd.setdefault((is_causal, H, D), []).append((fat_by_shape[key], fa3_by_shape[key]))

    groups = []
    for (is_causal, H, D), pairs in sorted(
        groups_by_causal_hd.items(), key=lambda item: (item[0][0], item[0][2], item[0][1])
    ):
        pairs.sort(key=lambda pair: pair[0]["S"])
        groups.append((is_causal, H, D, pairs))
    return groups


def finite_values(values):
    return [value for value in values if not math.isnan(value)]


def y_axis_max(values):
    finite = finite_values(values)
    if not finite:
        return 100.0
    return min(100.0, max(10.0, max(finite) * 1.15))


def report_label(dtype):
    return f"H100 {dtype.upper()} Tensor Core Utilization"


def bar_chart(title, pairs):
    width, height = 660, 390
    left, right, top, bottom = 68, 18, 42, 64
    plot_w = width - left - right
    plot_h = height - top - bottom
    bar_w = 18
    gap = 18
    group_w = 2 * bar_w + gap

    seqlens = [fat["S"] for fat, _fa3 in pairs]
    fat_vals = [fat["tc_utilization"] for fat, _fa3 in pairs]
    fa3_vals = [fa3["tc_utilization"] for _fat, fa3 in pairs]
    max_y = y_axis_max(fat_vals + fa3_vals)
    y_ticks = [max_y * i / 5 for i in range(6)]

    def y_map(value):
        if math.isnan(value):
            return top + plot_h
        return top + (max_y - value) / max_y * plot_h

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
        x_fat = center - bar_w - 2
        x_fa3 = center + 2
        for x, value, color, name in (
            (x_fat, fat_vals[idx], "#2563eb", "FA-T"),
            (x_fa3, fa3_vals[idx], "#dc2626", "FA3"),
        ):
            y = y_map(value)
            h = top + plot_h - y
            label = "nan" if math.isnan(value) else f"{value:.1f}"
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w}" height="{h:.2f}" fill="{color}" opacity="0.82"/>')
            parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{max(top + 11, y - 4):.2f}" text-anchor="middle" class="annot">{label}</text>')
            parts.append(f'<title>{name} S={seqlen}: {label}%</title>')
        parts.append(f'<text x="{center:.2f}" y="{top + plot_h + 18}" text-anchor="middle" class="tick">{seqlen_label(seqlen)}</text>')

    legend_y = top + 15
    for idx, (name, color) in enumerate((("FA-T", "#2563eb"), ("FA3", "#dc2626"))):
        x = left + 12 + idx * 82
        parts.append(f'<rect x="{x}" y="{legend_y - 10}" width="14" height="14" fill="{color}" opacity="0.82"/>')
        parts.append(f'<text x="{x + 20}" y="{legend_y + 2}" class="legend">{name}</text>')

    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 18}" text-anchor="middle" class="label">Sequence Length</text>')
    parts.append(f'<text x="16" y="{top + plot_h / 2}" transform="rotate(-90 16 {top + plot_h / 2})" text-anchor="middle" class="label">Tensor Core Utilization (%)</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def write_html(fat_rows, fa3_rows, output_path, dtype):
    cards = []
    for is_causal, H, D, pairs in make_groups(fat_rows, fa3_rows):
        cards.append(bar_chart(f"{causal_label(is_causal)}, H={H}, D={D}", pairs))

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{report_label(dtype)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; background: #ffffff; }}
    h1 {{ font-size: 24px; margin: 0 0 8px; }}
    h2 {{ font-size: 18px; margin: 28px 0 12px; }}
    .grid-wrap {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(620px, 1fr)); gap: 18px; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 8px; background: white; }}
    svg {{ width: 100%; height: auto; }}
    .axis {{ stroke: #111827; stroke-width: 1.2; }}
    .grid {{ stroke: #d1d5db; stroke-width: 0.8; stroke-dasharray: 3 4; }}
    .title {{ font-size: 15px; font-weight: 700; }}
    .tick {{ font-size: 11px; fill: #4b5563; }}
    .label {{ font-size: 12px; fill: #374151; }}
    .legend {{ font-size: 12px; fill: #111827; }}
    .annot {{ font-size: 10px; fill: #111827; }}
  </style>
</head>
<body>
  <h1>{report_label(dtype)}</h1>
  <h2>Tensor Core Utilization</h2>
  <div class="grid-wrap">
    {''.join(f'<div class="card">{card}</div>' for card in cards)}
  </div>
</body>
</html>
"""
    output_path.write_text(doc)


def draw_text(draw, xy, text, fill="#111827", font=None, anchor=None):
    draw.text(xy, text, fill=fill, font=font, anchor=anchor)


def draw_bar_panel(draw, box, title, pairs, font, small_font):
    x0, y0, x1, y1 = box
    left, right, top, bottom = 58, 16, 38, 52
    plot_x0, plot_y0 = x0 + left, y0 + top
    plot_x1, plot_y1 = x1 - right, y1 - bottom
    plot_w, plot_h = plot_x1 - plot_x0, plot_y1 - plot_y0
    draw.rectangle(box, outline="#d1d5db", width=1)
    draw_text(draw, ((x0 + x1) / 2, y0 + 16), title, font=font, anchor="mm")
    draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill="#111827", width=1)
    draw.line((plot_x0, plot_y0, plot_x0, plot_y1), fill="#111827", width=1)

    seqlens = [fat["S"] for fat, _fa3 in pairs]
    fat_vals = [fat["tc_utilization"] for fat, _fa3 in pairs]
    fa3_vals = [fa3["tc_utilization"] for _fat, fa3 in pairs]
    series = [("FA-T", fat_vals, "#2563eb"), ("FA3", fa3_vals, "#dc2626")]
    max_y = y_axis_max(fat_vals + fa3_vals)

    for idx in range(6):
        value = max_y * idx / 5
        y = plot_y1 - (value / max_y) * plot_h
        draw.line((plot_x0, y, plot_x1, y), fill="#e5e7eb", width=1)
        draw_text(draw, (plot_x0 - 8, y), f"{value:.0f}", font=small_font, fill="#4b5563", anchor="rm")

    group_count = len(seqlens)
    group_gap = plot_w / max(group_count, 1)
    bar_w = min(24, group_gap * 0.28)
    offsets = [-bar_w * 0.55, bar_w * 0.55]

    for group_idx, seqlen in enumerate(seqlens):
        center = plot_x0 + group_gap * (group_idx + 0.5)
        draw_text(draw, (center, plot_y1 + 18), seqlen_label(seqlen), font=small_font, fill="#4b5563", anchor="mm")
        for series_idx, (_name, values, color) in enumerate(series):
            value = values[group_idx]
            if math.isnan(value):
                continue
            bar_x0 = center + offsets[series_idx] - bar_w / 2
            bar_x1 = bar_x0 + bar_w
            bar_y0 = plot_y1 - (value / max_y) * plot_h
            draw.rectangle((bar_x0, bar_y0, bar_x1, plot_y1), fill=color)
            draw_text(draw, ((bar_x0 + bar_x1) / 2, bar_y0 - 7), f"{value:.1f}", font=small_font, anchor="mm")

    legend_x = plot_x0 + 10
    legend_y = plot_y0 + 8
    for idx, (name, _values, color) in enumerate(series):
        lx = legend_x + idx * 92
        draw.rectangle((lx, legend_y, lx + 12, legend_y + 12), fill=color)
        draw_text(draw, (lx + 17, legend_y + 6), name, font=small_font, anchor="lm")

    draw_text(draw, ((plot_x0 + plot_x1) / 2, y1 - 15), "Sequence Length", font=small_font, fill="#374151", anchor="mm")
    draw_text(draw, (x0 + 18, (plot_y0 + plot_y1) / 2), "TC Util (%)", font=small_font, fill="#374151", anchor="mm")


def write_png(fat_rows, fa3_rows, output_path, dtype):
    groups = make_groups(fat_rows, fa3_rows)
    panel_w, panel_h = 720, 390
    margin, gap = 36, 22
    title_h, section_h = 54, 34
    width = margin * 2 + panel_w * 2 + gap
    rows = max(1, (len(groups) + 1) // 2)
    height = margin + title_h + section_h + panel_h * rows + gap * max(rows - 1, 0) + margin
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()

    draw_text(draw, (width / 2, margin + 12), report_label(dtype), font=font, anchor="mm")
    y = margin + title_h
    draw_text(draw, (margin, y), "Tensor Core Utilization", font=font, anchor="lm")
    y += section_h
    for idx, (is_causal, H, D, pairs) in enumerate(groups):
        row, col = divmod(idx, 2)
        x0 = margin + col * (panel_w + gap)
        y0 = y + row * (panel_h + gap)
        draw_bar_panel(
            draw,
            (x0, y0, x0 + panel_w, y0 + panel_h),
            f"{causal_label(is_causal)}, H={H}, D={D}",
            pairs,
            font,
            small_font,
        )

    image.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("fp16", "fp8"), default="fp16")
    parser.add_argument("--fat", type=Path, default=None)
    parser.add_argument("--fa3", type=Path, default=None)
    parser.add_argument("--html-out", type=Path, default=None)
    parser.add_argument("--png-out", type=Path, default=None)
    args = parser.parse_args()

    if args.fat is None:
        args.fat = Path("fat_sweep.csv" if args.dtype == "fp16" else "fat_fp8_sweep.csv")
    if args.fa3 is None:
        args.fa3 = Path("fa3_sweep.csv" if args.dtype == "fp16" else "fa3_fp8_sweep.csv")
    if args.html_out is None:
        args.html_out = Path(f"h100_{args.dtype}_tc_utilization.html")
    if args.png_out is None:
        args.png_out = Path(f"h100_{args.dtype}_tc_utilization.png")

    fat_rows = load_rows(args.fat)
    fa3_rows = load_rows(args.fa3)
    write_html(fat_rows, fa3_rows, args.html_out, args.dtype)
    write_png(fat_rows, fa3_rows, args.png_out, args.dtype)
    print(f"Saved {args.html_out}")
    print(f"Saved {args.png_out}")


if __name__ == "__main__":
    main()
