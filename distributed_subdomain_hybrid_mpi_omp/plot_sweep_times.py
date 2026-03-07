#!/usr/bin/env python3
"""Plot timing sweep results for hybrid MPI+OpenMP runs.

Expected input files:
  results/sweeps_render_food10/summary_ranks{R}_omp{T}_ants{N}.csv

Output (per ant size and OMP thread count):
  - hybrid_total_breakdown_omp{T}_ants{N}.png
  - hybrid_calc_only_omp{T}_ants{N}.png
  - hybrid_calc_speedup_omp{T}_ants{N}.png
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


SUMMARY_RE = re.compile(r"summary_ranks(?P<ranks>\d+)_omp(?P<omp>\d+)_ants(?P<ants>\d+)\.csv$")
EXPECTED_ANTS = [5000, 10000, 20000, 40000, 80000, 160000]
EXPECTED_MPI_RANKS = list(range(2, 9))
EXPECTED_OMP_THREADS = [2, 3, 4]
SweepData = Dict[int, Dict[int, Dict[int, Dict[str, float]]]]


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
            metrics[row[0]] = float(row[3])

    return metrics


def load_data(results_dir: Path) -> SweepData:
    data: SweepData = {}
    for path in sorted(results_dir.glob("summary_*_ants*.csv")):
        match = SUMMARY_RE.match(path.name)
        if not match:
            continue

        ranks = int(match.group("ranks"))
        omp_threads = int(match.group("omp"))
        ants = int(match.group("ants"))

        if ants not in EXPECTED_ANTS:
            continue
        if ranks not in EXPECTED_MPI_RANKS:
            continue
        if omp_threads not in EXPECTED_OMP_THREADS:
            continue

        metrics = read_summary(path)
        if "iteration_total" not in metrics:
            continue

        data.setdefault(ants, {}).setdefault(omp_threads, {})[ranks] = metrics

    return data


def metric(metrics: Dict[str, float], key: str) -> float:
    return metrics.get(key, 0.0)


def mpi_compute_comm(metrics: Dict[str, float]) -> float:
    return (
        metric(metrics, "mpi_halo_exchange")
        + metric(metrics, "mpi_migration")
        + metric(metrics, "mpi_food_allreduce")
    )


def plot_total_for_config(data: SweepData, ants: int, omp_threads: int, out_path: Path, dpi: int) -> None:
    ranks = sorted(data[ants][omp_threads].keys())

    local_calc = [metric(data[ants][omp_threads][r], "advance_total") for r in ranks]
    mpi_calc = [mpi_compute_comm(data[ants][omp_threads][r]) for r in ranks]
    calc_total = [lc + mc for lc, mc in zip(local_calc, mpi_calc)]
    render_total = [
        metric(data[ants][omp_threads][r], "render")
        + metric(data[ants][omp_threads][r], "blit")
        + metric(data[ants][omp_threads][r], "mpi_render_comm")
        for r in ranks
    ]
    total = [metric(data[ants][omp_threads][r], "iteration_total") for r in ranks]
    other = [max(tt - cc - rr, 0.0) for tt, cc, rr in zip(total, calc_total, render_total)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ranks, calc_total, width=0.65, color="#1f77b4", label="Calc (local + MPI compute comm)")
    ax.bar(ranks, render_total, width=0.65, bottom=calc_total, color="#6baed6", label="Render + blit + render comm")
    ax.bar(
        ranks,
        other,
        width=0.65,
        bottom=[c + r for c, r in zip(calc_total, render_total)],
        color="#c6dbef",
        label="Other overhead",
    )

    ax.set_title(f"Hybrid MPI+OMP total breakdown (ants={ants}, omp={omp_threads})")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Mean time (ms)")
    ax.set_xticks(ranks)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_calc_for_config(data: SweepData, ants: int, omp_threads: int, out_path: Path, dpi: int) -> None:
    ranks = sorted(data[ants][omp_threads].keys())

    local_calc = [metric(data[ants][omp_threads][r], "advance_total") for r in ranks]
    mpi_calc = [mpi_compute_comm(data[ants][omp_threads][r]) for r in ranks]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ranks, local_calc, width=0.65, color="#ff7f0e", label="Local compute (advance_total)")
    ax.bar(
        ranks,
        mpi_calc,
        width=0.65,
        bottom=local_calc,
        color="#9467bd",
        label="MPI compute comm",
    )

    ax.set_title(f"Hybrid MPI+OMP calc-only breakdown (ants={ants}, omp={omp_threads})")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Mean time (ms)")
    ax.set_xticks(ranks)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_calc_speedup_for_config(data: SweepData, ants: int, omp_threads: int, out_path: Path, dpi: int) -> None:
    ranks = sorted(data[ants][omp_threads].keys())
    if not ranks:
        return

    calc_with_comm = [
        metric(data[ants][omp_threads][r], "advance_total") + mpi_compute_comm(data[ants][omp_threads][r])
        for r in ranks
    ]
    calc_without_comm = [metric(data[ants][omp_threads][r], "advance_total") for r in ranks]

    base_idx = 0
    base_ranks = ranks[base_idx]
    base_with = calc_with_comm[base_idx]
    base_without = calc_without_comm[base_idx]

    speedup_with = [(base_with / t) if t > 0.0 else 0.0 for t in calc_with_comm]
    speedup_without = [(base_without / t) if t > 0.0 else 0.0 for t in calc_without_comm]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        ranks,
        speedup_with,
        marker="o",
        linewidth=2.0,
        markersize=6,
        color="#1f77b4",
        label="Speedup calc (with MPI compute comm)",
    )
    ax.plot(
        ranks,
        speedup_without,
        marker="s",
        linewidth=2.0,
        markersize=6,
        color="#2ca02c",
        label="Speedup calc (without comm)",
    )

    ax.set_title(f"Hybrid MPI+OMP calc speedup (ants={ants}, omp={omp_threads}, baseline={base_ranks} ranks)")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Speedup")
    ax.set_xticks(ranks)
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

    parser = argparse.ArgumentParser(description="Plot hybrid MPI+OpenMP sweep timing summaries.")
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.output_dir.glob("hybrid_*_ants*.png"):
        stale.unlink(missing_ok=True)

    generated: list[Path] = []
    for ants in EXPECTED_ANTS:
        if ants not in data:
            print(f"Warning: missing expected ant size in summaries: {ants}", file=sys.stderr)
            continue

        for omp_threads in EXPECTED_OMP_THREADS:
            if omp_threads not in data[ants]:
                print(f"Warning: ants={ants} missing omp_threads={omp_threads}", file=sys.stderr)
                continue

            missing_ranks = [r for r in EXPECTED_MPI_RANKS if r not in data[ants][omp_threads]]
            if missing_ranks:
                print(
                    f"Warning: ants={ants}, omp_threads={omp_threads} missing MPI ranks: {missing_ranks}",
                    file=sys.stderr,
                )

            out_total = args.output_dir / f"hybrid_total_breakdown_omp{omp_threads}_ants{ants}.png"
            out_calc = args.output_dir / f"hybrid_calc_only_omp{omp_threads}_ants{ants}.png"
            out_speedup = args.output_dir / f"hybrid_calc_speedup_omp{omp_threads}_ants{ants}.png"

            plot_total_for_config(data, ants, omp_threads, out_total, args.dpi)
            plot_calc_for_config(data, ants, omp_threads, out_calc, args.dpi)
            plot_calc_speedup_for_config(data, ants, omp_threads, out_speedup, args.dpi)

            generated.append(out_total)
            generated.append(out_calc)
            generated.append(out_speedup)

    if not generated:
        print("No plots generated; dataset is incomplete.", file=sys.stderr)
        return 1

    print("Generated plots:")
    for path in generated:
        print(f"- {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
