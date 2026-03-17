#!/usr/bin/env python3
"""Run multiple MoonAI experiments with different seeds for statistical comparison."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple MoonAI headless experiments with different seeds."
    )
    parser.add_argument(
        "--binary", default="./build/linux-release/moonai",
        help="Path to the MoonAI binary (default: ./build/linux-release/moonai)"
    )
    parser.add_argument(
        "--config", default="config/default_config.json",
        help="Path to config JSON (default: config/default_config.json)"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46],
        help="List of seeds to run (default: 42 43 44 45 46)"
    )
    parser.add_argument(
        "-g", "--generations", type=int, default=100,
        help="Number of generations per run (default: 100)"
    )
    args = parser.parse_args()

    binary = Path(args.binary)
    if not binary.exists():
        print(f"Binary not found: {binary}", file=sys.stderr)
        sys.exit(1)

    config = Path(args.config)
    if not config.exists():
        print(f"Config not found: {config}", file=sys.stderr)
        sys.exit(1)

    failed = []
    for seed in args.seeds:
        cmd = [
            str(binary),
            "--headless",
            "--config", str(config),
            "--seed", str(seed),
            "--generations", str(args.generations),
        ]
        print(f"Running seed {seed}: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  Seed {seed} failed with exit code {result.returncode}")
            failed.append(seed)
        else:
            print(f"  Seed {seed} completed successfully.")

    print(f"\nDone. {len(args.seeds) - len(failed)}/{len(args.seeds)} runs succeeded.")
    if failed:
        print(f"Failed seeds: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
