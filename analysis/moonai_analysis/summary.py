"""Summary rendering for MoonAI analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .io import SkippedRun
from .plots import COMPARISON_METRICS, ConditionAggregate


@dataclass(frozen=True)
class SummaryResult:
    generation: int
    table_markdown: str
    summary_path: Path
    skipped_path: Path
    index_path: Path


def write_summary(
    aggregates: list[ConditionAggregate],
    skipped_runs: list[SkippedRun],
    output_dir: Path,
    generated_paths: list[Path],
) -> SummaryResult:
    generation = min(
        int(aggregate.summary_frame["generation"].max()) for aggregate in aggregates
    )
    summary_path = output_dir / "summary.md"
    skipped_path = output_dir / "skipped_runs.md"
    index_path = output_dir / "index.md"

    summary_text = _build_summary_table(aggregates, generation)
    skipped_text = _build_skipped_runs(skipped_runs)
    index_text = _build_index(output_dir, generated_paths)

    summary_path.write_text(summary_text, encoding="utf-8")
    skipped_path.write_text(skipped_text, encoding="utf-8")
    index_path.write_text(index_text, encoding="utf-8")

    return SummaryResult(
        generation=generation,
        table_markdown=summary_text,
        summary_path=summary_path,
        skipped_path=skipped_path,
        index_path=index_path,
    )


def _build_summary_table(aggregates: list[ConditionAggregate], generation: int) -> str:
    headers = ["condition", "runs", *COMPARISON_METRICS]
    lines = [
        f"# Summary\n",
        f"Metrics at generation {generation} (mean +/- std across seeds).\n",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for aggregate in aggregates:
        row = aggregate.summary_frame[
            aggregate.summary_frame["generation"] == generation
        ].iloc[-1]
        values = [aggregate.label, str(len(aggregate.runs))]
        for metric in COMPARISON_METRICS:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            values.append(f"{mean:.3f} +/- {std:.3f}")
        lines.append("| " + " | ".join(values) + " |")

    lines.append("")
    return "\n".join(lines)


def _build_skipped_runs(skipped_runs: list[SkippedRun]) -> str:
    lines = ["# Skipped Runs", ""]
    if not skipped_runs:
        lines.append("No runs were skipped.")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Run | Reason |")
    lines.append("| --- | --- |")
    for skipped in skipped_runs:
        lines.append(f"| `{skipped.path.name}` | {skipped.reason} |")
    lines.append("")
    return "\n".join(lines)


def _build_index(output_dir: Path, generated_paths: list[Path]) -> str:
    all_paths = list(generated_paths) + [
        output_dir / "summary.md",
        output_dir / "skipped_runs.md",
        output_dir / "index.md",
    ]
    relative_paths = sorted({path.relative_to(output_dir) for path in all_paths})
    lines = ["# Analysis Output", "", "## Files", ""]
    for path in relative_paths:
        lines.append(f"- `{path.as_posix()}`")
    lines.append("")
    return "\n".join(lines)
