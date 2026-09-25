#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import tempfile
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "example" / "generate_epidemic_dpomdp.py"
MODEL = ROOT / "example" / "epidemic_light.dpomdp"


def check_model_reproducible() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "epidemic_light.dpomdp"
        subprocess.run(
            ["python3", str(GEN), "--output", str(out)],
            check=True,
            capture_output=True,
            text=True,
        )
        assert out.read_bytes() == MODEL.read_bytes(), (
            "toy epidemic generator does not reproduce the supplied instance"
        )


def check_structure() -> None:
    text = MODEL.read_text()
    for token in ("agents: 2", "states:", "actions:", "observations:", "start:", "T:", "O:", "R:"):
        assert token in text, f"missing {token} in toy epidemic model"



def check_intervention_tradeoff() -> None:
    spec = importlib.util.spec_from_file_location("epidemic_generator", GEN)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    # In a safe state, closing has an explicit intervention cost.
    safe_open = module.expected_team_reward("S", "S", 1, 1, 0.8, 3.0, 0.2)
    safe_closed = module.expected_team_reward("S", "S", 0, 0, 0.8, 3.0, 0.2)
    assert safe_open > safe_closed, "closing cost is not represented in the toy objective"

    # With an infected neighbour, remaining open can instead incur infection cost.
    exposed_open = module.expected_team_reward("S", "I", 1, 1, 0.8, 3.0, 0.2)
    exposed_closed = module.expected_team_reward("S", "I", 0, 0, 0.8, 3.0, 0.2)
    assert exposed_closed > exposed_open, "toy model does not expose a health/intervention trade-off"

def check_native_result(path: Path) -> None:
    assert path.is_file() and path.stat().st_size > 0, "planner result is missing or empty"
    policy = Path(str(path) + ".policy")
    assert policy.is_file() and policy.stat().st_size > 0, "planner policy is missing or empty"

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows, "planner result has no data rows"
    for key in ("time", "iter", "value"):
        assert key in rows[0], f"missing result column: {key}"
    assert math.isfinite(float(rows[-1]["value"])), "non-finite planner value"

    # Validate the native policy semantically, not merely by file existence.
    import sys
    sys.path.insert(0, str(ROOT))
    from render_policy import parse_native_policy

    rules = parse_native_policy(policy)
    assert {step for step, _, _, _ in rules} == {0, 1}, "policy does not cover both planning steps"
    assert {agent for _, agent, _, _ in rules} == {0, 1}, "policy does not cover both regions"


def check_student_policy(path: Path) -> None:
    assert path.is_file() and path.stat().st_size > 0, "student-facing policy is missing or empty"
    text = path.read_text()
    for token in (
        "AM05 Day-One coordination policy",
        "Policy produced by oSarsa-seq:",
        "region 1",
        "region 2",
    ):
        assert token in text, f"student-facing policy is missing: {token}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path)
    ap.add_argument("--student-policy", type=Path)
    args = ap.parse_args()
    check_model_reproducible()
    check_structure()
    check_intervention_tradeoff()
    if args.result:
        check_native_result(args.result)
    if args.student_policy:
        check_student_policy(args.student_policy)
    print("TEST PASS")
