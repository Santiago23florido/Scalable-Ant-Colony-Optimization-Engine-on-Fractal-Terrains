#!/usr/bin/env python3
"""Plot timing sweep results for distributed_subdomain_mpi runs.

Expected input files:
  results/sweeps_render_food10/summary_threads{P}_ants{N}.csv

Output:
  One pair of plots per ant size:
  - mpi_total_breakdown_ants{N}.png
  - mpi_calc_only_ants{N}.png
  - mpi_calc_speedup_ants{N}.png

In this script, MPI communication related to computation is included in calc:
  mpi_halo_exchange + mpi_migration + mpi_food_allreduce
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict

HELPER_DIR = Path(__file__).resolve().parents[1] / "docs" / "scripts"
if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

from report_table_utils import ant_label, write_latex_apa_table

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


def metric(metrics: Dict[str, float], key: str) -> float:
    return metrics.get(key, 0.0)


def mpi_compute_comm(metrics: Dict[str, float]) -> float:
    return (
        metric(metrics, "mpi_halo_exchange")
        + metric(metrics, "mpi_migration")
        + metric(metrics, "mpi_food_allreduce")
    )


def calc_metric(metrics: Dict[str, float]) -> float:
    return metric(metrics, "advance_total") + mpi_compute_comm(metrics)


def ordered_ant_sizes(keys: set[int]) -> list[int]:
    return [ants for ants in EXPECTED_ANTS if ants in keys]


def plot_total_for_ant(data: SweepData, ants: int, out_path: Path, dpi: int) -> None:
    procs = sorted(data[ants].keys())

    local_calc = [metric(data[ants][p], "advance_total") for p in procs]
    mpi_calc = [mpi_compute_comm(data[ants][p]) for p in procs]
    calc_total = [lc + mc for lc, mc in zip(local_calc, mpi_calc)]
    render_total = [
        metric(data[ants][p], "render")
        + metric(data[ants][p], "blit")
        + metric(data[ants][p], "mpi_render_comm")
        for p in procs
    ]
    total = [metric(data[ants][p], "iteration_total") for p in procs]
    other = [max(tt - cc - rr, 0.0) for tt, cc, rr in zip(total, calc_total, render_total)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(procs, calc_total, width=0.65, color="#1f77b4", label="Calc (local + MPI compute comm)")
    ax.bar(procs, render_total, width=0.65, bottom=calc_total, color="#6baed6", label="Render + blit + render comm")
    ax.bar(
        procs,
        other,
        width=0.65,
        bottom=[c + r for c, r in zip(calc_total, render_total)],
        color="#c6dbef",
        label="Other overhead",
    )

    ax.set_title(f"Distributed MPI total breakdown (ants={ants})")
    ax.set_xlabel("MPI processes")
    ax.set_ylabel("Mean time (ms)")
    ax.set_xticks(procs)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_calc_for_ant(data: SweepData, ants: int, out_path: Path, dpi: int) -> None:
    procs = sorted(data[ants].keys())

    local_calc = [metric(data[ants][p], "advance_total") for p in procs]
    mpi_calc = [mpi_compute_comm(data[ants][p]) for p in procs]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(procs, local_calc, width=0.65, color="#ff7f0e", label="Local compute (advance_total)")
    ax.bar(
        procs,
        mpi_calc,
        width=0.65,
        bottom=local_calc,
        color="#9467bd",
        label="MPI compute comm",
    )

    ax.set_title(f"Distributed MPI calc-only breakdown (ants={ants})")
    ax.set_xlabel("MPI processes")
    ax.set_ylabel("Mean time (ms)")
    ax.set_xticks(procs)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_calc_speedup_for_ant(data: SweepData, ants: int, out_path: Path, dpi: int) -> None:
    procs = sorted(data[ants].keys())
    if not procs:
        return

    calc_with_comm = [metric(data[ants][p], "advance_total") + mpi_compute_comm(data[ants][p]) for p in procs]
    calc_without_comm = [metric(data[ants][p], "advance_total") for p in procs]

    base_idx = 0
    base_procs = procs[base_idx]
    base_with = calc_with_comm[base_idx]
    base_without = calc_without_comm[base_idx]

    speedup_with = [(base_with / t) if t > 0.0 else 0.0 for t in calc_with_comm]
    speedup_without = [(base_without / t) if t > 0.0 else 0.0 for t in calc_without_comm]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        procs,
        speedup_with,
        marker="o",
        linewidth=2.0,
        markersize=6,
        color="#1f77b4",
        label="Speedup calc (with MPI compute comm)",
    )
    ax.plot(
        procs,
        speedup_without,
        marker="s",
        linewidth=2.0,
        markersize=6,
        color="#2ca02c",
        label="Speedup calc (without comm)",
    )

    for p, s in zip(procs, speedup_with):
        ax.annotate(str(p), (p, s), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color="#1f77b4")
    for p, s in zip(procs, speedup_without):
        ax.annotate(str(p), (p, s), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8, color="#2ca02c")

    ax.set_title(f"Distributed MPI calc speedup (ants={ants}, baseline={base_procs} procs)")
    ax.set_xlabel("MPI processes")
    ax.set_ylabel("Speedup")
    ax.set_xticks(procs)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def build_speedup_efficiency_rows(data: SweepData) -> tuple[list[str], list[list[str]], list[list[str]]]:
    ant_sizes = ordered_ant_sizes(set(data.keys()))
    procs = sorted({proc for ants in ant_sizes for proc in data[ants].keys()})
    if not ant_sizes or not procs:
        return [], [], []

    speedup_rows: list[list[str]] = []
    efficiency_rows: list[list[str]] = []

    ant_labels = [ant_label(ants) for ants in ant_sizes]
    for proc in procs:
        speedup_row = [str(proc)]
        efficiency_row = [str(proc)]
        for ants in ant_sizes:
            metrics = data[ants].get(proc)
            if not metrics:
                speedup_row.append("--")
                efficiency_row.append("--")
                continue

            base_proc = sorted(data[ants].keys())[0]
            base_calc = calc_metric(data[ants][base_proc])
            current_calc = calc_metric(metrics)
            if current_calc <= 0.0:
                speedup_row.append("--")
                efficiency_row.append("--")
                continue

            speedup = base_calc / current_calc
            efficiency = speedup / (proc / base_proc)
            speedup_row.append(f"{speedup:.2f}")
            efficiency_row.append(f"{efficiency:.2f}")

        speedup_rows.append(speedup_row)
        efficiency_rows.append(efficiency_row)

    return ant_labels, speedup_rows, efficiency_rows


def write_speedup_efficiency_tables(data: SweepData, speedup_path: Path, efficiency_path: Path) -> list[Path]:
    headers, speedup_rows, efficiency_rows = build_speedup_efficiency_rows(data)
    if not headers:
        return []

    common_note = (
        "The compute metric is $advance\\_total + mpi\\_halo\\_exchange + mpi\\_migration + "
        "mpi\\_food\\_allreduce$, matching the MPI compute plots. Efficiency is computed with respect to the "
        "number of compute MPI processes only; rendering is excluded from the process count. For each colony "
        "size, the baseline is the smallest available MPI process count $p_0$."
    )
    write_latex_apa_table(
        speedup_path,
        caption="MPI compute speedup for all colony sizes and MPI process counts.",
        label="tab:mpi_speedup",
        headers=["Proc/Ants"] + headers,
        rows=speedup_rows,
        note=common_note + " Each cell reports speedup $S=T(p_0)/T(p)$.",
        column_spec="l" + ("c" * len(headers)),
        position="!htbp",
    )
    write_latex_apa_table(
        efficiency_path,
        caption="MPI compute efficiency for all colony sizes and MPI process counts.",
        label="tab:mpi_efficiency",
        headers=["Proc/Ants"] + headers,
        rows=efficiency_rows,
        note=common_note + " Each cell reports efficiency $E=S/(p/p_0)$.",
        column_spec="l" + ("c" * len(headers)),
        position="!htbp",
    )
    return [speedup_path, efficiency_path]


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    default_results = root / "results" / "sweeps_render_food10"
    default_out = default_results / "plots"

    parser = argparse.ArgumentParser(description="Plot distributed_subdomain_mpi sweep timing summaries.")
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
    for ants in ordered_ant_sizes(set(data.keys())):
        missing_procs = [p for p in EXPECTED_PROCS if p not in data[ants]]
        if missing_procs:
            print(f"Warning: ants={ants} missing process counts: {missing_procs}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    for ants in ordered_ant_sizes(set(data.keys())):
        out_total = args.output_dir / f"mpi_total_breakdown_ants{ants}.png"
        out_calc = args.output_dir / f"mpi_calc_only_ants{ants}.png"
        out_speedup = args.output_dir / f"mpi_calc_speedup_ants{ants}.png"

        plot_total_for_ant(data, ants, out_total, args.dpi)
        plot_calc_for_ant(data, ants, out_calc, args.dpi)
        plot_calc_speedup_for_ant(data, ants, out_speedup, args.dpi)

        generated.append(out_total)
        generated.append(out_calc)
        generated.append(out_speedup)

    generated.extend(
        write_speedup_efficiency_tables(
            data,
            args.output_dir / "mpi_speedup_table.tex",
            args.output_dir / "mpi_efficiency_table.tex",
        )
    )

    print("Generated plots:")
    for path in generated:
        print(f"- {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
