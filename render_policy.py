#!/usr/bin/env python3
"""Render the native oSarsa-seq policy in student-readable terms."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

OBS = {0: "S (susceptible)", 1: "I (infected)", 2: "R (recovered)"}
ACTION = {0: "closed", 1: "open"}
STEP_RE = re.compile(r"^==== Step (\d+)$")
RULE_RE = re.compile(r"^\s*<agent(\d+)>\s+hist:\s*(.*?)\s+action:\s*(\d+)\s*$")


def parse_native_policy(path: Path):
    step = None
    rules = []
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        m = STEP_RE.match(line)
        if m:
            step = int(m.group(1))
            continue
        m = RULE_RE.match(line)
        if m:
            if step is None:
                raise ValueError("policy rule appears before a step header")
            agent = int(m.group(1))
            hist_text = m.group(2).strip()
            hist = [] if not hist_text else [int(x) for x in hist_text.split()]
            action = int(m.group(3))
            if action not in ACTION:
                raise ValueError(f"unknown action index {action}")
            if any(z not in OBS for z in hist):
                raise ValueError(f"unknown observation index in {hist}")
            rules.append((step, agent, hist, action))
    if not rules:
        raise ValueError("native policy contains no decision rules")
    return rules


def render(rules) -> str:
    lines = [
        "AM05 Day-One coordination policy",
        "=================================",
        "",
        "Toy problem: two interacting regions; local states S/I/R;",
        "local interventions closed/open; planning horizon 2.",
        "",
        "Policy produced by oSarsa-seq:",
    ]
    for step, agent, hist, action in rules:
        if hist:
            info = " -> ".join(OBS[z] for z in hist)
            condition = f"after local observation {info}"
        else:
            condition = "before any local observation"
        lines.append(f"  t={step}, region {agent + 1}, {condition}: {ACTION[action]}")
    lines += [
        "",
        "Interpretation: each line maps the information locally available to a region",
        "to the intervention selected by the planner for this toy model.",
        "This small example illustrates the model -> planner -> coordination pipeline;",
        "it is not a claim of epidemiological realism or optimal epidemic policy design.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("native_policy", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    rules = parse_native_policy(args.native_policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(rules))


if __name__ == "__main__":
    main()
