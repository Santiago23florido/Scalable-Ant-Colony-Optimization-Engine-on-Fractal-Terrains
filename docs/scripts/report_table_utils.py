"""Helpers for writing compact LaTeX table snippets for the report."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def ant_label(ants: int) -> str:
    if ants % 1000 == 0:
        return f"{ants // 1000}k"
    return str(ants)


def fmt_ms(value: float) -> str:
    return f"{value:.2f}"


def fmt_speedup(value: float) -> str:
    return f"{value:.2f}x"


def fmt_value_with_unit(value: float, unit: int) -> str:
    return f"{value:.2f}@{unit}"


def write_latex_table_snippet(
    path: Path,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    column_spec: str,
    width: str = "0.98\\linewidth",
) -> None:
    """Write a standalone LaTeX snippet containing only a compact tabular block."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "% Auto-generated. Do not edit by hand.",
        "{\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.2pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        f"\\resizebox{{{width}}}{{!}}{{%",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\hline",
        " & ".join(f"\\textbf{{{header}}}" for header in headers) + " \\\\",
        "\\hline",
    ]

    for row in rows:
        lines.append(" & ".join(row) + " \\\\")

    lines.extend(
        [
            "\\hline",
            "\\end{tabular}%",
            "}",
            "\\par}",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_latex_apa_table(
    caption: str,
    label: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    note: str,
    column_spec: str,
    star: bool = False,
    position: str = "!t",
    width: str | None = None,
    tabcolsep_pt: float = 2.4,
    font_size: str = "\\normalsize",
) -> str:
    """Return a LaTeX table string with caption, note, and APA-like rules."""
    env = "table*" if star else "table"

    lines = [
        "% Auto-generated. Do not edit by hand.",
        f"\\begin{{{env}}}[{position}]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        font_size,
        f"\\setlength{{\\tabcolsep}}{{{tabcolsep_pt}pt}}",
        "\\renewcommand{\\arraystretch}{1.08}",
    ]

    if width is not None:
        lines.append(f"\\resizebox{{{width}}}{{!}}{{%")

    lines.extend(
        [
            f"\\begin{{tabular}}{{{column_spec}}}",
            "\\toprule",
            " & ".join(f"\\textbf{{{header}}}" for header in headers) + " \\\\",
            "\\midrule",
        ]
    )

    for row in rows:
        lines.append(" & ".join(row) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )

    if width is not None:
        lines.append("}%")

    lines.extend(
        [
            "\\par\\vspace{2pt}"
            "\\parbox{\\columnwidth}{\\raggedright\\footnotesize\\textit{Note.} " + note + "}",
            f"\\end{{{env}}}",
        ]
    )

    return "\n".join(lines) + "\n"


def write_latex_apa_table(
    path: Path,
    caption: str,
    label: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    note: str,
    column_spec: str,
    star: bool = False,
    position: str = "!t",
    width: str | None = None,
    tabcolsep_pt: float = 2.4,
    font_size: str = "\\normalsize",
) -> None:
    """Write a full LaTeX table snippet with caption, note, and APA-like rules."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        format_latex_apa_table(
            caption=caption,
            label=label,
            headers=headers,
            rows=rows,
            note=note,
            column_spec=column_spec,
            star=star,
            position=position,
            width=width,
            tabcolsep_pt=tabcolsep_pt,
            font_size=font_size,
        ),
        encoding="utf-8",
    )
