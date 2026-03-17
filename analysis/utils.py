"""Shared utilities for MoonAI analysis scripts.

All analysis scripts import from here to avoid duplication.
"""

import json
import sys
from pathlib import Path

import pandas as pd


# Single source of truth for experiment conditions.
# Used by setup_experiments.py to create configs, and by report.py to label runs.
CONDITIONS: dict[str, dict] = {
    "baseline":      {},
    "mut_low":       {"mutation_rate": 0.1},
    "mut_high":      {"mutation_rate": 0.5},
    "pop_small":     {"predator_count": 25, "prey_count": 75},
    "pop_large":     {"predator_count": 100, "prey_count": 300},
    "no_speciation": {"compatibility_threshold": 100.0},
    "tanh":          {"activation_function": "tanh"},
    "crossover_low": {"crossover_rate": 0.25},
}


def load_config(run_dir: Path) -> dict:
    """Load config.json from a run directory."""
    with open(run_dir / "config.json") as f:
        return json.load(f)


def load_stats(run_dir: Path) -> pd.DataFrame:
    """Load stats.csv from a run directory, skipping comment lines."""
    return pd.read_csv(run_dir / "stats.csv", comment="#")


def condition_label(cfg: dict) -> str:
    """Return the condition name for a given config dict.

    Matches against known CONDITIONS by inspecting the overridden fields.
    Falls back to 'baseline' if no condition matches.
    """
    mut   = cfg.get("mutation_rate", 0.3)
    cross = cfg.get("crossover_rate", 0.75)
    compat = cfg.get("compatibility_threshold", 3.0)
    pred  = cfg.get("predator_count", 50)
    act   = cfg.get("activation_function", "sigmoid")

    if compat >= 90:
        return "no_speciation"
    if act == "tanh":
        return "tanh"
    if abs(mut - 0.1) < 1e-4:
        return "mut_low"
    if abs(mut - 0.5) < 1e-4:
        return "mut_high"
    if pred == 25:
        return "pop_small"
    if pred == 100:
        return "pop_large"
    if abs(cross - 0.25) < 1e-4:
        return "crossover_low"
    return "baseline"


def find_runs(output_dir: Path, min_generations: int = 0) -> list[Path]:
    """Return sorted list of run dirs that have config.json + stats.csv.

    Filters to runs with at least min_generations rows in stats.csv.
    Short smoke-test runs (< min_generations) are skipped.
    """
    runs: list[Path] = []
    if not output_dir.is_dir():
        print(f"Warning: output dir not found: {output_dir}", file=sys.stderr)
        return runs

    for d in sorted(output_dir.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "config.json").exists() or not (d / "stats.csv").exists():
            continue
        if min_generations > 0:
            try:
                df = pd.read_csv(d / "stats.csv", comment="#", usecols=["generation"])
                if len(df) < min_generations:
                    continue
            except Exception:
                continue
        runs.append(d)
    return runs
