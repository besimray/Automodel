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

"""Plot training loss from NeMo logs or training.jsonl."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


STEP_LOG_RE = re.compile(r"step\s+(\d+)\s+\|\s+epoch\s+(\d+)\s+\|\s+loss\s+([0-9]*\.?[0-9]+)")
MAX_STEPS_RE = re.compile(r"(?:max_steps:|- Max train steps:)\s*(\d+)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot loss curves from NeMo AutoModel logs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional path to training jsonl file. If omitted, parse logs from --logs-dir.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing plain-text training logs.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="minimax*.log",
        help="Glob used to select logs under --logs-dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("loss_curve_all_runs_epoch.svg"),
        help="Output image path.",
    )
    parser.add_argument(
        "--x-axis",
        choices=["step", "epoch"],
        default="epoch",
        help="X-axis domain.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
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


def _svg_polyline_points(xs: list[float], ys: list[float], sx, sy) -> str:
    return " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(xs, ys))


def write_svg_plot(
    series: list[dict[str, object]],
    output_path: Path,
    window: int,
    x_label: str,
    title: str,
) -> None:
    width = 1900
    height = 520
    left = 70
    right = 620
    top = 40
    bottom = 60
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_x = [x for s in series for x in s["x"]]  # type: ignore[index]
    all_y = [y for s in series for y in s["y_smooth"]]  # type: ignore[index]
    x_min = min(all_x)
    x_max = max(all_x)
    y_min = min(all_y)
    y_max = max(all_y)
    if x_max == x_min:
        x_max += 1.0
    if y_max == y_min:
        y_max += 1.0

    def sx(x: float) -> float:
        return left + (x - x_min) * plot_w / (x_max - x_min)

    def sy(y: float) -> float:
        return top + (y_max - y) * plot_h / (y_max - y_min)

    grid_lines: list[str] = []
    for i in range(11):
        gy = top + i * (plot_h / 10)
        val = y_max - i * (y_max - y_min) / 10
        grid_lines.append(f'<line x1="{left}" y1="{gy:.2f}" x2="{left + plot_w}" y2="{gy:.2f}" stroke="#eee"/>')
        grid_lines.append(
            f'<text x="{left - 10}" y="{gy + 4:.2f}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{val:.2f}</text>'
        )

    for i in range(11):
        gx = left + i * (plot_w / 10)
        val = x_min + i * (x_max - x_min) / 10
        grid_lines.append(f'<line x1="{gx:.2f}" y1="{top}" x2="{gx:.2f}" y2="{top + plot_h}" stroke="#eee"/>')
        grid_lines.append(
            f'<text x="{gx:.2f}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="11" fill="#555">{val:.2f}</text>'
        )

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    lines: list[str] = []
    legend: list[str] = []
    legend_y = top + 20
    for idx, s in enumerate(series):
        color = colors[idx % len(colors)]
        xs: list[float] = s["x"]  # type: ignore[assignment]
        ys: list[float] = s["y_smooth"]  # type: ignore[assignment]
        name: str = s["name"]  # type: ignore[assignment]
        points = _svg_polyline_points(xs, ys, sx, sy)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        legend.append(f'<line x1="{left + plot_w + 16}" y1="{legend_y}" x2="{left + plot_w + 36}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        legend.append(
            f'<text x="{left + plot_w + 42}" y="{legend_y + 4}" font-family="Arial" font-size="11" fill="#222">{name}</text>'
        )
        legend_y += 20

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="white" />
  <text x="{left}" y="24" font-family="Arial" font-size="20" font-weight="bold">{title}</text>
  {"".join(grid_lines)}
  <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333"/>
  {"".join(lines)}
  {"".join(legend)}
  <text x="{left + plot_w/2:.0f}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="14">{x_label}</text>
  <text x="20" y="{top + plot_h/2:.0f}" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 20 {top + plot_h/2:.0f})">Loss</text>
  <text x="{left + plot_w + 16}" y="{top + 8}" font-family="Arial" font-size="12" fill="#222">loss (ma{window})</text>
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


def parse_jsonl_series(path: Path, smooth_window: int) -> dict[str, object]:
    steps: list[int] = []
    losses: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "step" in row and "loss" in row:
                steps.append(int(row["step"]))
                losses.append(float(row["loss"]))
    if not steps:
        raise ValueError(f"No step/loss entries found in: {path}")
    return {"name": path.stem, "x": [float(s) for s in steps], "y_raw": losses, "y_smooth": moving_average(losses, smooth_window)}


def infer_steps_per_epoch(steps: list[int], epochs: list[int]) -> float | None:
    epoch_start_step: dict[int, int] = {}
    for step, epoch in zip(steps, epochs):
        if epoch not in epoch_start_step:
            epoch_start_step[epoch] = step

    epoch_diffs: list[int] = []
    sorted_epochs = sorted(epoch_start_step.keys())
    for prev_epoch, next_epoch in zip(sorted_epochs, sorted_epochs[1:]):
        diff = epoch_start_step[next_epoch] - epoch_start_step[prev_epoch]
        if diff > 0:
            epoch_diffs.append(diff)

    if not epoch_diffs:
        return None
    return float(statistics.median(epoch_diffs))


def infer_log_steps_per_epoch(path: Path) -> float | None:
    steps: list[int] = []
    epochs: list[int] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            step_match = STEP_LOG_RE.search(raw_line)
            if step_match:
                steps.append(int(step_match.group(1)))
                epochs.append(int(step_match.group(2)))
    return infer_steps_per_epoch(steps, epochs)


def parse_log_series(
    path: Path,
    smooth_window: int,
    x_axis: str,
    fallback_steps_per_epoch: float | None = None,
) -> dict[str, object] | None:
    max_steps = None
    steps: list[int] = []
    epochs: list[int] = []
    losses: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            max_steps_match = MAX_STEPS_RE.search(line)
            if max_steps_match:
                max_steps = int(max_steps_match.group(1))
            step_match = STEP_LOG_RE.search(line)
            if step_match:
                steps.append(int(step_match.group(1)))
                epochs.append(int(step_match.group(2)))
                losses.append(float(step_match.group(3)))
    if not steps:
        return None

    if x_axis == "epoch":
        epoch_start_step: dict[int, int] = {}
        for step, epoch in zip(steps, epochs):
            if epoch not in epoch_start_step:
                epoch_start_step[epoch] = step

        sorted_epochs = sorted(epoch_start_step.keys())
        inferred_steps_per_epoch = infer_steps_per_epoch(steps, epochs) or fallback_steps_per_epoch

        if inferred_steps_per_epoch and inferred_steps_per_epoch > 0:
            xs = []
            for step, epoch in zip(steps, epochs):
                epoch_start = epoch_start_step.get(epoch, step)
                within_epoch = max(step - epoch_start, 0)
                xs.append(float(epoch) + (within_epoch / inferred_steps_per_epoch))
        elif max_steps and max_steps > 0:
            # If logs only show epoch=0, approximate epoch-progress from step progression.
            observed_epoch_span = max(epochs) + 1 if epochs else 1
            xs = [observed_epoch_span * (s / max_steps) for s in steps]
        else:
            max_observed_step = max(steps) if steps else 1
            xs = [s / max_observed_step for s in steps]
    else:
        xs = [float(s) for s in steps]

    return {"name": path.stem, "x": xs, "y_raw": losses, "y_smooth": moving_average(losses, smooth_window)}


def main() -> None:
    args = parse_args()
    series: list[dict[str, object]] = []
    if args.input is not None:
        if not args.input.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        series.append(parse_jsonl_series(args.input, args.smooth_window))
    else:
        if not args.logs_dir.exists():
            raise FileNotFoundError(f"Logs directory not found: {args.logs_dir}")
        log_files = sorted(args.logs_dir.glob(args.glob))
        inferred_steps_per_epoch_values = [
            steps_per_epoch
            for log_file in log_files
            if (steps_per_epoch := infer_log_steps_per_epoch(log_file)) is not None
        ]
        fallback_steps_per_epoch = (
            float(statistics.median(inferred_steps_per_epoch_values)) if inferred_steps_per_epoch_values else None
        )
        for log_file in log_files:
            parsed = parse_log_series(log_file, args.smooth_window, args.x_axis, fallback_steps_per_epoch)
            if parsed is not None:
                series.append(parsed)
        if not series:
            raise ValueError(f"No training step/loss lines found in {args.logs_dir}/{args.glob}")

    x_label = "epoch" if args.x_axis == "epoch" else "step"
    title = f"MiniMax run loss curves ({x_label} x-axis)"
    try:
        import matplotlib.pyplot as plt

        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(16, 7))
        for s in series:
            plt.plot(s["x"], s["y_smooth"], linewidth=2, label=s["name"])  # type: ignore[index]
        plt.xlabel(x_label)
        plt.ylabel("Loss")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
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
        write_svg_plot(series, fallback_output, args.smooth_window, x_label, title)


if __name__ == "__main__":
    main()
