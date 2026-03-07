#!/usr/bin/env python3
"""Plot timing sweep results for vectorized_omp runs.

Expected input files:
  results/sweeps_render_food10/summary_threads{T}_ants{N}.csv

Output:
  One pair of plots per ant size:
  - omp_total_breakdown_ants{N}.png
  - omp_calc_only_ants{N}.png
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from exc


SUMMARY_RE = re.compile(r"summary_threads(?P<threads>\d+)_ants(?P<ants>\d+)\.csv$")
EXPECTED_ANTS = [5000, 10000 ,20000, 40000, 80000, 160000]
SweepData = Dict[int, Dict[int, Dict[str, float]]]


def read_summary(path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    in_meta = False

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            row = [cell.strip() for cell in row]
            if row[0] == "metric":
                continue
            if row[0] == "meta_key":
                in_meta = True
                continue
            if in_meta:
                continue
            if len(row) < 4:
                continue
            metrics[row[0]] = float(row[3])  # mean_ms

    return metrics


def load_data(results_dir: Path) -> SweepData:
    data: SweepData = {}
    for path in sorted(results_dir.glob("summary_threads*_ants*.csv")):
        match = SUMMARY_RE.match(path.name)
        if not match:
            continue
        threads = int(match.group("threads"))
        ants = int(match.group("ants"))
        metrics = read_summary(path)
        if "iteration_total" not in metrics:
            continue
        data.setdefault(ants, {})[threads] = metrics
    return data


def metric(metrics: Dict[str, float], key: str) -> float:
    return metrics.get(key, 0.0)


def ordered_ant_sizes(keys: set[int]) -> list[int]:
    ordered = [ants for ants in EXPECTED_ANTS if ants in keys]
    extras = sorted(keys - set(EXPECTED_ANTS))
    ordered.extend(extras)
    return ordered


def plot_total_for_ant(data: SweepData, ants: int, out_path: Path, dpi: int) -> None:
    threads = sorted(data[ants].keys())
    total = [metric(data[ants][t], "iteration_total") for t in threads]
    calc = [metric(data[ants][t], "advance_total") for t in threads]
    render = [metric(data[ants][t], "render") + metric(data[ants][t], "blit") for t in threads]
    other = [max(tt - cc - rr, 0.0) for tt, cc, rr in zip(total, calc, render)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(threads, calc, width=0.65, color="#1f77b4", label="Calc (advance_total)")
    ax.bar(threads, render, width=0.65, bottom=calc, color="#6baed6", label="Render + blit")
    ax.bar(
        threads,
        other,
        width=0.65,
        bottom=[c + r for c, r in zip(calc, render)],
        color="#c6dbef",
        label="Other overhead",
    )

    ax.set_title(f"Vectorized OMP total breakdown (ants={ants})")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Mean time (ms)")
    ax.set_xticks(threads)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_calc_for_ant(data: SweepData, ants: int, out_path: Path, dpi: int) -> None:
    threads = sorted(data[ants].keys())

    ants_advance = [metric(data[ants][t], "ants_advance") for t in threads]
    evap = [metric(data[ants][t], "evaporation") for t in threads]
    update = [metric(data[ants][t], "update") for t in threads]
    adv_total = [metric(data[ants][t], "advance_total") for t in threads]
    other = [max(at - (aa + ev + up), 0.0) for at, aa, ev, up in zip(adv_total, ants_advance, evap, update)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(threads, ants_advance, width=0.65, color="#ff7f0e", label="Ant movement")
    ax.bar(
        threads,
        evap,
        width=0.65,
        bottom=ants_advance,
        color="#fdae6b",
        label="Evaporation",
    )
    ax.bar(
        threads,
        update,
        width=0.65,
        bottom=[a + e for a, e in zip(ants_advance, evap)],
        color="#fd8d3c",
        label="Update",
    )
    ax.bar(
        threads,
        other,
        width=0.65,
        bottom=[a + e + u for a, e, u in zip(ants_advance, evap, update)],
        color="#fdd0a2",
        label="Other compute overhead",
    )

    ax.set_title(f"Vectorized OMP compute-only breakdown (ants={ants})")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Mean time (ms)")
    ax.set_xticks(threads)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    default_results = root / "results" / "sweeps_render_food10"
    default_out = default_results / "plots"

    parser = argparse.ArgumentParser(description="Plot vectorized_omp sweep timing summaries.")
    parser.add_argument("--results-dir", type=Path, default=default_results, help=f"Default: {default_results}")
    parser.add_argument("--output-dir", type=Path, default=default_out, help=f"Default: {default_out}")
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI (default: 150)")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    data = load_data(args.results_dir)
    if not data:
        print(f"No summary sweep files found in {args.results_dir}", file=sys.stderr)
        return 1

    missing_expected = [ants for ants in EXPECTED_ANTS if ants not in data]
    if missing_expected:
        print(f"Warning: missing expected ant sizes in summaries: {missing_expected}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    for ants in ordered_ant_sizes(set(data.keys())):
        out_total = args.output_dir / f"omp_total_breakdown_ants{ants}.png"
        out_calc = args.output_dir / f"omp_calc_only_ants{ants}.png"

        plot_total_for_ant(data, ants, out_total, args.dpi)
        plot_calc_for_ant(data, ants, out_calc, args.dpi)

        generated.append(out_total)
        generated.append(out_calc)

    print("Generated plots:")
    for path in generated:
        print(f"- {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
