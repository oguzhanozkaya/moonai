#!/usr/bin/env python3
"""Create experiment config files in config/experiments/.

Reads the base config and writes one JSON per condition with a single
parameter change. Idempotent — safe to re-run.

Usage:
    uv run python3 analysis/setup_experiments.py
    uv run python3 analysis/setup_experiments.py --config config/default_config.json
"""

import argparse
import json
import sys
from pathlib import Path

from utils import CONDITIONS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create experiment config files in config/experiments/."
    )
    parser.add_argument(
        "--config", default="config/default_config.json",
        help="Base config file (default: config/default_config.json)"
    )
    args = parser.parse_args()

    base_path = Path(args.config)
    if not base_path.exists():
        print(f"Error: base config not found: {base_path}", file=sys.stderr)
        return 1

    with open(base_path) as f:
        base_cfg = json.load(f)

    out_dir = Path("config/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, overrides in CONDITIONS.items():
        cfg = dict(base_cfg)
        cfg.update(overrides)
        out_path = out_dir / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=4)
        changes = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "(none)"
        print(f"Created {out_path}  [{changes}]")

    print(f"\n{len(CONDITIONS)} configs written to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
