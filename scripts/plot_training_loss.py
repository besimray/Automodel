# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plot training loss from checkpoints/training.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot loss curve from NeMo AutoModel training.jsonl logs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("checkpoints/training.jsonl"),
        help="Path to training jsonl file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/loss_curve.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=20,
        help="Moving-average window size for the smoothed loss curve.",
    )
    return parser.parse_args()


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    out: list[float] = []
    running_sum = 0.0
    for idx, val in enumerate(values):
        running_sum += val
        if idx >= window:
            running_sum -= values[idx - window]
        count = min(idx + 1, window)
        out.append(running_sum / count)
    return out


def write_svg_plot(steps: list[int], losses: list[float], smoothed: list[float], output_path: Path, window: int) -> None:
    width = 1000
    height = 520
    left = 70
    right = 30
    top = 40
    bottom = 60
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_min = min(steps)
    x_max = max(steps)
    y_min = min(min(losses), min(smoothed))
    y_max = max(max(losses), max(smoothed))
    if x_max == x_min:
        x_max += 1
    if y_max == y_min:
        y_max += 1.0

    def sx(x: float) -> float:
        return left + (x - x_min) * plot_w / (x_max - x_min)

    def sy(y: float) -> float:
        return top + (y_max - y) * plot_h / (y_max - y_min)

    raw_points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(steps, losses))
    smooth_points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(steps, smoothed))

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="white" />
  <text x="{width/2:.0f}" y="24" text-anchor="middle" font-family="sans-serif" font-size="20">Training Loss</text>
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" />
  <text x="{left + plot_w/2:.0f}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="14">Step</text>
  <text x="20" y="{top + plot_h/2:.0f}" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 20 {top + plot_h/2:.0f})">Loss</text>
  <polyline fill="none" stroke="#7f8c8d" stroke-width="1.5" opacity="0.45" points="{raw_points}" />
  <polyline fill="none" stroke="#1f77b4" stroke-width="2.5" points="{smooth_points}" />
  <text x="{left + 10}" y="{top + 18}" font-family="sans-serif" font-size="12" fill="#7f8c8d">loss</text>
  <text x="{left + 10}" y="{top + 36}" font-family="sans-serif" font-size="12" fill="#1f77b4">loss (ma{window})</text>
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_text(svg, encoding="utf-8")
        print(f"Wrote SVG plot: {output_path}")
    except PermissionError:
        fallback = Path.cwd() / output_path.name
        fallback.write_text(svg, encoding="utf-8")
        print(f"Could not write to {output_path}; wrote SVG plot to {fallback} instead.")


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    steps: list[int] = []
    losses: list[float] = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "step" in row and "loss" in row:
                steps.append(int(row["step"]))
                losses.append(float(row["loss"]))

    if not steps:
        raise ValueError(f"No step/loss entries found in: {args.input}")

    smoothed = moving_average(losses, args.smooth_window)
    try:
        import matplotlib.pyplot as plt

        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(steps, losses, alpha=0.35, label="loss")
        plt.plot(steps, smoothed, linewidth=2, label=f"loss (ma{args.smooth_window})")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        try:
            plt.savefig(args.output, dpi=150)
            print(f"Wrote plot: {args.output}")
        except PermissionError:
            fallback_png = Path.cwd() / args.output.name
            plt.savefig(fallback_png, dpi=150)
            print(f"Could not write to {args.output}; wrote plot to {fallback_png} instead.")
    except ImportError:
        fallback_output = args.output
        if fallback_output.suffix.lower() != ".svg":
            fallback_output = fallback_output.with_suffix(".svg")
        write_svg_plot(steps, losses, smoothed, fallback_output, args.smooth_window)


if __name__ == "__main__":
    main()
