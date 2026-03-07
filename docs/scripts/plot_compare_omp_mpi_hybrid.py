#!/usr/bin/env python3
"""Generate comparison plots for OMP-only vs MPI-only vs Hybrid(MPI+OMP).

Default inputs:
- vectorized_omp/results/sweeps_render_food10
- distributed_subdomain_mpi/results/sweeps_render_food10
- distributed_subdomain_hybrid_mpi_omp/results/sweeps_render_food10

Output (per ant size and hybrid OMP thread count):
- compare3_total_omp{T}_ants{N}.png
- compare3_calc_omp{T}_ants{N}.png
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


SUMMARY_OMP_RE = re.compile(r"summary_threads(?P<threads>\d+)_ants(?P<ants>\d+)\.csv$")
SUMMARY_MPI_RE = re.compile(r"summary_(?:threads|ranks)(?P<ranks>\d+)_ants(?P<ants>\d+)\.csv$")
SUMMARY_HYBRID_RE = re.compile(r"summary_ranks(?P<ranks>\d+)_omp(?P<omp>\d+)_ants(?P<ants>\d+)\.csv$")

EXPECTED_ANTS = [5000, 10000, 20000, 40000, 80000, 160000]
EXPECTED_PAR_UNITS = list(range(2, 9))
EXPECTED_HYBRID_OMP = [2, 3, 4]

OmpData = Dict[int, Dict[int, Dict[str, float]]]
MpiData = Dict[int, Dict[int, Dict[str, float]]]
HybridData = Dict[int, Dict[int, Dict[int, Dict[str, float]]]]


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


def load_omp_data(results_dir: Path) -> OmpData:
    data: OmpData = {}
    for path in sorted(results_dir.glob("summary_threads*_ants*.csv")):
        match = SUMMARY_OMP_RE.match(path.name)
        if not match:
            continue

        threads = int(match.group("threads"))
        ants = int(match.group("ants"))
        if ants not in EXPECTED_ANTS:
            continue

        metrics = read_summary(path)
        if "iteration_total" not in metrics:
            continue

        data.setdefault(ants, {})[threads] = metrics

    return data


def load_mpi_data(results_dir: Path) -> MpiData:
    data: MpiData = {}
    for path in sorted(results_dir.glob("summary_*_ants*.csv")):
        match = SUMMARY_MPI_RE.match(path.name)
        if not match:
            continue

        ranks = int(match.group("ranks"))
        ants = int(match.group("ants"))
        if ants not in EXPECTED_ANTS:
            continue

        metrics = read_summary(path)
        if "iteration_total" not in metrics:
            continue

        data.setdefault(ants, {})[ranks] = metrics

    return data


def load_hybrid_data(results_dir: Path) -> HybridData:
    data: HybridData = {}
    for path in sorted(results_dir.glob("summary_ranks*_omp*_ants*.csv")):
        match = SUMMARY_HYBRID_RE.match(path.name)
        if not match:
            continue

        ranks = int(match.group("ranks"))
        omp_threads = int(match.group("omp"))
        ants = int(match.group("ants"))
        if ants not in EXPECTED_ANTS or omp_threads not in EXPECTED_HYBRID_OMP:
            continue

        metrics = read_summary(path)
        if "iteration_total" not in metrics:
            continue

        data.setdefault(ants, {}).setdefault(omp_threads, {})[ranks] = metrics

    return data


def m(metrics: Dict[str, float], key: str) -> float:
    return metrics.get(key, 0.0)


def mpi_calc_comm(metrics: Dict[str, float]) -> float:
    return m(metrics, "mpi_halo_exchange") + m(metrics, "mpi_migration") + m(metrics, "mpi_food_allreduce")


def plot_total(
    ants: int,
    hybrid_omp_threads: int,
    units: list[int],
    omp_metrics: Dict[int, Dict[str, float]],
    mpi_metrics: Dict[int, Dict[str, float]],
    hybrid_metrics: Dict[int, Dict[str, float]],
    out_path: Path,
    dpi: int,
) -> None:
    omp_calc = [m(omp_metrics[u], "advance_total") for u in units]
    mpi_calc = [m(mpi_metrics[u], "advance_total") for u in units]
    hybrid_calc = [m(hybrid_metrics[u], "advance_total") for u in units]

    omp_mpi_comm = [0.0 for _ in units]
    mpi_mpi_comm = [mpi_calc_comm(mpi_metrics[u]) for u in units]
    hybrid_mpi_comm = [mpi_calc_comm(hybrid_metrics[u]) for u in units]

    omp_total = [m(omp_metrics[u], "iteration_total") for u in units]
    mpi_total = [m(mpi_metrics[u], "iteration_total") for u in units]
    hybrid_total = [m(hybrid_metrics[u], "iteration_total") for u in units]

    omp_other = [max(t - c - mc, 0.0) for t, c, mc in zip(omp_total, omp_calc, omp_mpi_comm)]
    mpi_other = [max(t - c - mc, 0.0) for t, c, mc in zip(mpi_total, mpi_calc, mpi_mpi_comm)]
    hybrid_other = [max(t - c - mc, 0.0) for t, c, mc in zip(hybrid_total, hybrid_calc, hybrid_mpi_comm)]

    x = list(range(len(units)))
    width = 0.26
    x_omp = [v - width for v in x]
    x_mpi = x
    x_hybrid = [v + width for v in x]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_omp, omp_calc, width=width, color="#2ca02c", label="OMP calc")
    ax.bar(
        x_omp,
        omp_other,
        width=width,
        bottom=omp_calc,
        color="#a1d99b",
        label="OMP other",
    )

    ax.bar(x_mpi, mpi_calc, width=width, color="#1f77b4", label="MPI calc")
    ax.bar(
        x_mpi,
        mpi_mpi_comm,
        width=width,
        bottom=mpi_calc,
        color="#6baed6",
        label="MPI comm",
    )
    ax.bar(
        x_mpi,
        mpi_other,
        width=width,
        bottom=[c + mc for c, mc in zip(mpi_calc, mpi_mpi_comm)],
        color="#c6dbef",
        label="MPI other",
    )

    ax.bar(x_hybrid, hybrid_calc, width=width, color="#d62728", label=f"Hybrid calc (OMP={hybrid_omp_threads})")
    ax.bar(
        x_hybrid,
        hybrid_mpi_comm,
        width=width,
        bottom=hybrid_calc,
        color="#fb6a4a",
        label="Hybrid MPI comm",
    )
    ax.bar(
        x_hybrid,
        hybrid_other,
        width=width,
        bottom=[c + mc for c, mc in zip(hybrid_calc, hybrid_mpi_comm)],
        color="#fcae91",
        label="Hybrid other",
    )

    ax.set_title(f"Total iteration time comparison (ants={ants}, hybrid omp={hybrid_omp_threads})")
    ax.set_xlabel("Parallel units (OMP threads / MPI ranks)")
    ax.set_ylabel("Mean time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(u) for u in units])
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(loc="best", ncol=3, fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_calc(
    ants: int,
    hybrid_omp_threads: int,
    units: list[int],
    omp_metrics: Dict[int, Dict[str, float]],
    mpi_metrics: Dict[int, Dict[str, float]],
    hybrid_metrics: Dict[int, Dict[str, float]],
    out_path: Path,
    dpi: int,
) -> None:
    omp_calc = [m(omp_metrics[u], "advance_total") for u in units]
    mpi_local_calc = [m(mpi_metrics[u], "advance_total") for u in units]
    mpi_comm = [mpi_calc_comm(mpi_metrics[u]) for u in units]
    hybrid_local_calc = [m(hybrid_metrics[u], "advance_total") for u in units]
    hybrid_comm = [mpi_calc_comm(hybrid_metrics[u]) for u in units]

    x = list(range(len(units)))
    width = 0.26
    x_omp = [v - width for v in x]
    x_mpi = x
    x_hybrid = [v + width for v in x]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_omp, omp_calc, width=width, color="#2ca02c", label="OMP calc")

    ax.bar(x_mpi, mpi_local_calc, width=width, color="#1f77b4", label="MPI calc")
    ax.bar(
        x_mpi,
        mpi_comm,
        width=width,
        bottom=mpi_local_calc,
        color="#6baed6",
        label="MPI comm",
    )

    ax.bar(
        x_hybrid,
        hybrid_local_calc,
        width=width,
        color="#d62728",
        label=f"Hybrid calc (OMP={hybrid_omp_threads})",
    )
    ax.bar(
        x_hybrid,
        hybrid_comm,
        width=width,
        bottom=hybrid_local_calc,
        color="#fb6a4a",
        label="Hybrid MPI comm",
    )

    ax.set_title(f"Compute time comparison (ants={ants}, hybrid omp={hybrid_omp_threads})")
    ax.set_xlabel("Parallel units (OMP threads / MPI ranks)")
    ax.set_ylabel("Mean time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(u) for u in units])
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(loc="best", ncol=2, fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    default_omp = repo_root / "vectorized_omp" / "results" / "sweeps_render_food10"
    default_mpi = repo_root / "distributed_subdomain_mpi" / "results" / "sweeps_render_food10"
    default_hybrid = repo_root / "distributed_subdomain_hybrid_mpi_omp" / "results" / "sweeps_render_food10"
    default_out = repo_root / "docs" / "figures" / "compare_omp_mpi_hybrid"

    parser = argparse.ArgumentParser(description="Compare OMP-only, MPI-only and hybrid MPI+OMP sweeps.")
    parser.add_argument("--omp-dir", type=Path, default=default_omp, help=f"Default: {default_omp}")
    parser.add_argument("--mpi-dir", type=Path, default=default_mpi, help=f"Default: {default_mpi}")
    parser.add_argument("--hybrid-dir", type=Path, default=default_hybrid, help=f"Default: {default_hybrid}")
    parser.add_argument("--output-dir", type=Path, default=default_out, help=f"Default: {default_out}")
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI (default: 150)")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    omp_data = load_omp_data(args.omp_dir)
    mpi_data = load_mpi_data(args.mpi_dir)
    hybrid_data = load_hybrid_data(args.hybrid_dir)

    if not omp_data or not mpi_data or not hybrid_data:
        print("Need OMP, MPI and Hybrid datasets to compare.", file=sys.stderr)
        return 1

    common_ants = [ants for ants in EXPECTED_ANTS if ants in omp_data and ants in mpi_data and ants in hybrid_data]
    if not common_ants:
        print("No common ant sizes across OMP, MPI and Hybrid datasets.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.output_dir.glob("compare3_*_ants*.png"):
        stale.unlink(missing_ok=True)

    generated: list[Path] = []
    for ants in common_ants:
        for omp_threads in EXPECTED_HYBRID_OMP:
            hybrid_for_cfg = hybrid_data[ants].get(omp_threads)
            if not hybrid_for_cfg:
                print(f"Warning: ants={ants} missing hybrid omp={omp_threads}", file=sys.stderr)
                continue

            units = [
                u
                for u in EXPECTED_PAR_UNITS
                if u in omp_data[ants] and u in mpi_data[ants] and u in hybrid_for_cfg
            ]
            if not units:
                print(
                    f"Warning: ants={ants}, hybrid omp={omp_threads} has no common parallel units in {EXPECTED_PAR_UNITS}",
                    file=sys.stderr,
                )
                continue

            out_total = args.output_dir / f"compare3_total_omp{omp_threads}_ants{ants}.png"
            out_calc = args.output_dir / f"compare3_calc_omp{omp_threads}_ants{ants}.png"

            plot_total(ants, omp_threads, units, omp_data[ants], mpi_data[ants], hybrid_for_cfg, out_total, args.dpi)
            plot_calc(ants, omp_threads, units, omp_data[ants], mpi_data[ants], hybrid_for_cfg, out_calc, args.dpi)

            generated.extend([out_total, out_calc])

    if not generated:
        print("No comparison plots generated.", file=sys.stderr)
        return 1

    print("Generated plots:")
    for path in generated:
        print(f"- {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
