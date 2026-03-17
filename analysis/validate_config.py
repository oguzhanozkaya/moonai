#!/usr/bin/env python3
"""Validate one or more MoonAI config JSON files against known parameter bounds.

Usage:
    python3 validate_config.py config/default_config.json
    python3 validate_config.py exp1.json exp2.json

Exit code: 0 if all configs are valid, 1 if any violation is found.
"""
import json
import sys
import argparse

# (field, min_val, max_val)  -- None means no bound on that side
NUMERIC_RULES = [
    ("grid_width",             100,   10000),
    ("grid_height",            100,   10000),
    ("predator_count",           1,   10000),
    ("prey_count",               1,   10000),
    ("predator_speed",      1e-9,     None),
    ("prey_speed",          1e-9,     None),
    ("vision_range",        1e-9,     None),
    ("attack_range",        1e-9,     None),
    ("initial_energy",      1e-9,     None),
    ("energy_drain_per_tick",  0,     None),
    ("food_count",             0,     None),
    ("food_respawn_rate",      0,     1.0),
    ("mutation_rate",          0,     1.0),
    ("crossover_rate",         0,     1.0),
    ("add_node_rate",          0,     1.0),
    ("add_connection_rate",    0,     1.0),
    ("weight_mutation_power", 1e-9,   None),
    ("generation_ticks",      10,     None),
    ("max_generations",        0,     None),
    ("compatibility_threshold", 0.1,  100.0),
    ("stagnation_limit",       1,     1000),
    ("target_fps",             1,     1000),
    ("log_interval",           1,     None),
    ("fitness_survival_weight", 0,    None),
    ("fitness_kill_weight",     0,    None),
    ("fitness_energy_weight",   0,    None),
    ("fitness_distance_weight", 0,    None),
    ("complexity_penalty_weight", 0,  None),
]

STRING_ENUMS = {
    "activation_function": {"sigmoid", "tanh", "relu"},
    "boundary_mode":       {"wrap", "clamp"},
}


def validate(path: str) -> list[str]:
    violations: list[str] = []
    try:
        with open(path) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        return [f"File not found: {path}"]
    except json.JSONDecodeError as e:
        return [f"JSON parse error in {path}: {e}"]

    for field, lo, hi in NUMERIC_RULES:
        if field not in cfg:
            continue  # missing keys use C++ defaults, not a violation
        val = cfg[field]
        if not isinstance(val, (int, float)):
            violations.append(f"  [{field}] expected number, got {type(val).__name__}")
            continue
        if lo is not None and val < lo:
            violations.append(f"  [{field}] = {val} (must be >= {lo})")
        if hi is not None and val > hi:
            violations.append(f"  [{field}] = {val} (must be <= {hi})")

    # Cross-field rule: attack_range < vision_range
    if "attack_range" in cfg and "vision_range" in cfg:
        if cfg["attack_range"] >= cfg["vision_range"]:
            violations.append(
                f"  [attack_range] = {cfg['attack_range']} must be < "
                f"[vision_range] = {cfg['vision_range']}"
            )

    # Population size cap
    pc = cfg.get("predator_count", 0)
    pr = cfg.get("prey_count", 0)
    if pc + pr > 10000:
        violations.append(
            f"  predator_count ({pc}) + prey_count ({pr}) = {pc+pr} (must be <= 10000)"
        )

    for field, allowed in STRING_ENUMS.items():
        if field not in cfg:
            continue
        val = cfg[field]
        if val not in allowed:
            violations.append(
                f"  [{field}] = \"{val}\" (must be one of {sorted(allowed)})"
            )

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate MoonAI config JSON files against known parameter bounds."
    )
    parser.add_argument("configs", nargs="+", metavar="config.json")
    args = parser.parse_args()

    any_error = False
    for path in args.configs:
        violations = validate(path)
        if violations:
            any_error = True
            print(f"INVALID: {path}")
            for v in violations:
                print(v)
        else:
            print(f"OK: {path}")

    return 1 if any_error else 0


if __name__ == "__main__":
    sys.exit(main())
