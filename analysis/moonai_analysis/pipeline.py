"""Full analysis pipeline orchestration."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .genome import load_latest_genome, save_genome_plot
from .io import RunData, discover_runs
from .labels import LabelResolver
from .plots import (
    build_condition_aggregate,
    save_comparison_plots,
    save_condition_plots,
)
from .summary import write_summary


def run_analysis(input_dir: Path, output_dir: Path) -> None:
    runs, skipped_runs = discover_runs(input_dir)
    if not runs:
        raise SystemExit(f"No qualifying runs found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir = output_dir / "conditions"
    comparisons_dir = output_dir / "comparisons"

    grouped_runs = _group_runs(runs)
    aggregates = [
        build_condition_aggregate(label, grouped_runs[label])
        for label in sorted(grouped_runs)
    ]

    generated_paths: list[Path] = []
    for aggregate in aggregates:
        destination_dir = conditions_dir / aggregate.label
        generated_paths.extend(save_condition_plots(aggregate, destination_dir))
        genome = load_latest_genome(aggregate.representative_run.path)
        if genome is not None:
            genome_path = destination_dir / "genome.png"
            fitness = float(genome.get("fitness", 0.0))
            title = (
                f"{aggregate.label} - Best Genome ({aggregate.representative_run.name}, "
                f"Gen {genome.get('generation', '?')}, Fitness {fitness:.3f})"
            )
            save_genome_plot(genome, genome_path, title)
            generated_paths.append(genome_path)

    generated_paths.extend(save_comparison_plots(aggregates, comparisons_dir))
    summary = write_summary(aggregates, skipped_runs, output_dir, generated_paths)
    generated_paths.extend(
        [summary.summary_path, summary.skipped_path, summary.index_path]
    )

    print(f"Analysed {len(runs)} runs across {len(aggregates)} conditions.")
    print(f"Wrote analysis bundle to {output_dir}")
    print(f"Summary: {summary.summary_path}")
    print(f"Index:   {summary.index_path}")
    print(f"Skipped: {summary.skipped_path}")


def _group_runs(runs: list[RunData]) -> dict[str, list[RunData]]:
    resolver = LabelResolver()
    grouped: dict[str, list[RunData]] = defaultdict(list)
    for run in runs:
        grouped[resolver.resolve(run)].append(run)
    return dict(grouped)
