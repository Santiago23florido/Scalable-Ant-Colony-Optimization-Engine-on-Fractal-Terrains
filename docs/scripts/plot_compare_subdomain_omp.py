#!/usr/bin/env python3
"""Generate combined OMP vs subdomain(MPI) plots from sweep summaries.

Input directories (default):
- distributed_subdomain_mpi/results/sweeps_render_food10
- vectorized_omp/results/sweeps_render_food10

Output (per ant size):
- compare_total_ants{N}.png
- compare_calc_ants{N}.png
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


SUMMARY_RE = re.compile(r"summary_(?:threads|ranks)(?P<procs>\d+)_ants(?P<ants>\d+)\.csv$")
EXPECTED_ANTS = [5000, 10000, 20000, 40000, 80000, 160000]
EXPECTED_PROCS = list(range(2, 13))
SweepData = Dict[int, Dict[int, Dict[str, float]]]


def read_summary(path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    in_meta = False
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            row = [c.strip() for c in row]
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


def load_summaries(results_dir: Path) -> SweepData:
    data: SweepData = {}
    for path in sorted(results_dir.glob("summary_*_ants*.csv")):
        match = SUMMARY_RE.match(path.name)
        if not match:
            continue
        procs = int(match.group("procs"))
        ants = int(match.group("ants"))
        if ants not in EXPECTED_ANTS or procs not in EXPECTED_PROCS:
            continue
        metrics = read_summary(path)
        if "iteration_total" not in metrics:
            continue
        data.setdefault(ants, {})[procs] = metrics
    return data


def m(metrics: Dict[str, float], key: str) -> float:
    return metrics.get(key, 0.0)


def mpi_calc_comm(metrics: Dict[str, float]) -> float:
    return m(metrics, "mpi_halo_exchange") + m(metrics, "mpi_migration") + m(metrics, "mpi_food_allreduce")


def plot_total(
    ants: int,
    procs: list[int],
    mpi_metrics: Dict[int, Dict[str, float]],
    omp_metrics: Dict[int, Dict[str, float]],
    out_path: Path,
    dpi: int,
) -> None:
    width = 0.35
    x_mpi = [p - width / 2 for p in procs]
    x_omp = [p + width / 2 for p in procs]

    mpi_calc = [m(mpi_metrics[p], "advance_total") + mpi_calc_comm(mpi_metrics[p]) for p in procs]
    mpi_render = [m(mpi_metrics[p], "render") + m(mpi_metrics[p], "blit") + m(mpi_metrics[p], "mpi_render_comm") for p in procs]
    mpi_total = [m(mpi_metrics[p], "iteration_total") for p in procs]
    mpi_other = [max(t - c - r, 0.0) for t, c, r in zip(mpi_total, mpi_calc, mpi_render)]

    omp_calc = [m(omp_metrics[p], "advance_total") for p in procs]
    omp_render = [m(omp_metrics[p], "render") + m(omp_metrics[p], "blit") for p in procs]
    omp_total = [m(omp_metrics[p], "iteration_total") for p in procs]
    omp_other = [max(t - c - r, 0.0) for t, c, r in zip(omp_total, omp_calc, omp_render)]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x_mpi, mpi_calc, width=width, color="#1f77b4", label="MPI calc")
    ax.bar(x_mpi, mpi_render, width=width, bottom=mpi_calc, color="#6baed6", label="MPI render")
    ax.bar(
        x_mpi,
        mpi_other,
        width=width,
        bottom=[c + r for c, r in zip(mpi_calc, mpi_render)],
        color="#c6dbef",
        label="MPI other",
    )

    ax.bar(x_omp, omp_calc, width=width, color="#2ca02c", label="OMP calc")
    ax.bar(x_omp, omp_render, width=width, bottom=omp_calc, color="#74c476", label="OMP render")
    ax.bar(
        x_omp,
        omp_other,
        width=width,
        bottom=[c + r for c, r in zip(omp_calc, omp_render)],
        color="#c7e9c0",
        label="OMP other",
    )

    ax.set_title(f"OMP vs MPI total breakdown (ants={ants})")
    ax.set_xlabel("Processes / threads")
    ax.set_ylabel("Mean time (ms)")
    ax.set_xticks(procs)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", ncol=2, fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_calc(
    ants: int,
    procs: list[int],
    mpi_metrics: Dict[int, Dict[str, float]],
    omp_metrics: Dict[int, Dict[str, float]],
    out_path: Path,
    dpi: int,
) -> None:
    width = 0.35
    x_mpi = [p - width / 2 for p in procs]
    x_omp = [p + width / 2 for p in procs]

    mpi_local = [m(mpi_metrics[p], "advance_total") for p in procs]
    mpi_comm = [mpi_calc_comm(mpi_metrics[p]) for p in procs]

    omp_ant = [m(omp_metrics[p], "ants_advance") for p in procs]
    omp_evap = [m(omp_metrics[p], "evaporation") for p in procs]
    omp_update = [m(omp_metrics[p], "update") for p in procs]
    omp_adv = [m(omp_metrics[p], "advance_total") for p in procs]
    omp_other = [max(t - (a + e + u), 0.0) for t, a, e, u in zip(omp_adv, omp_ant, omp_evap, omp_update)]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x_mpi, mpi_local, width=width, color="#ff7f0e", label="MPI local calc")
    ax.bar(x_mpi, mpi_comm, width=width, bottom=mpi_local, color="#9467bd", label="MPI calc comm")

    ax.bar(x_omp, omp_ant, width=width, color="#8c564b", label="OMP ant movement")
    ax.bar(x_omp, omp_evap, width=width, bottom=omp_ant, color="#c49c94", label="OMP evaporation")
    ax.bar(
        x_omp,
        omp_update,
        width=width,
        bottom=[a + e for a, e in zip(omp_ant, omp_evap)],
        color="#7f7f7f",
        label="OMP update",
    )
    ax.bar(
        x_omp,
        omp_other,
        width=width,
        bottom=[a + e + u for a, e, u in zip(omp_ant, omp_evap, omp_update)],
        color="#c7c7c7",
        label="OMP other calc",
    )

    ax.set_title(f"OMP vs MPI calc-only breakdown (ants={ants})")
    ax.set_xlabel("Processes / threads")
    ax.set_ylabel("Mean time (ms)")
    ax.set_xticks(procs)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", ncol=2, fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    default_mpi = repo_root / "distributed_subdomain_mpi" / "results" / "sweeps_render_food10"
    default_omp = repo_root / "vectorized_omp" / "results" / "sweeps_render_food10"
    default_out = repo_root / "docs" / "figures" / "compare_subdomain_omp"
    parser = argparse.ArgumentParser(description="Compare OMP and subdomain(MPI) sweep results.")
    parser.add_argument("--mpi-dir", type=Path, default=default_mpi, help=f"Default: {default_mpi}")
    parser.add_argument("--omp-dir", type=Path, default=default_omp, help=f"Default: {default_omp}")
    parser.add_argument("--output-dir", type=Path, default=default_out, help=f"Default: {default_out}")
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI (default: 150)")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    mpi_data = load_summaries(args.mpi_dir)
    omp_data = load_summaries(args.omp_dir)
    if not mpi_data or not omp_data:
        print("Need both MPI and OMP summary datasets.", file=sys.stderr)
        return 1

    common_ants = [ants for ants in EXPECTED_ANTS if ants in mpi_data and ants in omp_data]
    if not common_ants:
        print("No common ant sizes between MPI and OMP results.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.output_dir.glob("compare_*_ants*.png"):
        stale.unlink(missing_ok=True)

    generated: list[Path] = []
    for ants in common_ants:
        common_procs = [p for p in EXPECTED_PROCS if p in mpi_data[ants] and p in omp_data[ants]]
        if not common_procs:
            continue
        out_total = args.output_dir / f"compare_total_ants{ants}.png"
        out_calc = args.output_dir / f"compare_calc_ants{ants}.png"

        plot_total(ants, common_procs, mpi_data[ants], omp_data[ants], out_total, args.dpi)
        plot_calc(ants, common_procs, mpi_data[ants], omp_data[ants], out_calc, args.dpi)
        generated.extend([out_total, out_calc])

    if not generated:
        print("No common process/thread counts found for shared ant sizes.", file=sys.stderr)
        return 1

    print("Generated plots:")
    for path in generated:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
