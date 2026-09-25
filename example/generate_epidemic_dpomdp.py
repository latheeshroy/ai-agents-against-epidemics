#!/usr/bin/env python3
"""
Generate a scalar Epidemic (SIR-style) Dec-POMDP in _light.dpomdp format.

State model:
  - 2 regions/agents, local states in {S, I, R}
  - 9 joint states (s0, s1)
Action model:
  - each agent action in {0,1} where 0=closed, 1=open
Observation model:
  - deterministic perfect local observation of next local state
Transition model:
  - susceptible regions can become infected through open interaction with an infected neighbour
  - infected regions remain infected with probability p and otherwise recover
Reward model (scalar):
  - expected team health reward on next state
  - a small cost for each closed region, representing the cost of intervention
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path


LOCAL_STATES = ("S", "I", "R")
OBS_INDEX = {"S": 0, "I": 1, "R": 2}
ACTIONS = (0, 1)


def idx_joint_state(s0: str, s1: str) -> int:
    return LOCAL_STATES.index(s0) * 3 + LOCAL_STATES.index(s1)


def joint_state_from_idx(idx: int) -> tuple[str, str]:
    return LOCAL_STATES[idx // 3], LOCAL_STATES[idx % 3]


def local_next_dist(si: str, sj: str, a0: int, a1: int, p: float) -> dict[str, float]:
    if si == "S" and sj == "I" and a0 == 1 and a1 == 1:
        return {"I": 1.0}
    if si == "I":
        return {"I": p, "R": 1.0 - p}
    if si == "R":
        return {"R": 1.0}
    return {"S": 1.0}


def joint_next_dist(s0: str, s1: str, a0: int, a1: int, p: float) -> dict[tuple[str, str], float]:
    d0 = local_next_dist(s0, s1, a0, a1, p)
    d1 = local_next_dist(s1, s0, a0, a1, p)
    out: dict[tuple[str, str], float] = {}
    for n0, p0 in d0.items():
        for n1, p1 in d1.items():
            out[(n0, n1)] = out.get((n0, n1), 0.0) + p0 * p1
    return out


def local_health_reward(next_local: str, c: float) -> float:
    return -c if next_local == "I" else 1.0


def expected_team_reward(
    s0: str, s1: str, a0: int, a1: int, p: float, c: float, intervention_cost: float
) -> float:
    exp_r = 0.0
    for (n0, n1), pr in joint_next_dist(s0, s1, a0, a1, p).items():
        health = 0.5 * (local_health_reward(n0, c) + local_health_reward(n1, c))
        intervention = intervention_cost * ((a0 == 0) + (a1 == 0))
        exp_r += pr * (health - intervention)
    return exp_r


def generate_lines(
    p: float, c: float, intervention_cost: float, discount: float, start_mode: str
) -> list[str]:
    states = [joint_state_from_idx(i) for i in range(9)]

    lines: list[str] = []
    lines.append("agents: 2")
    lines.append(f"discount: {discount}")
    lines.append("states: " + " ".join(str(i) for i in range(len(states))))
    lines.append("actions:")
    lines.append("0 1")
    lines.append("0 1")
    lines.append("observations:")
    lines.append("0 1 2")
    lines.append("0 1 2")

    if start_mode == "uniform":
        b0 = [1.0 / len(states)] * len(states)
    elif start_mode == "is":
        b0 = [0.0] * len(states)
        b0[idx_joint_state("I", "S")] = 1.0
    else:
        raise ValueError(f"Unsupported start_mode: {start_mode}")
    lines.append("start: " + " ".join(f"{x:.12g}" for x in b0))

    # Transitions
    for a0, a1 in product(ACTIONS, ACTIONS):
        for x, (s0, s1) in enumerate(states):
            nd = joint_next_dist(s0, s1, a0, a1, p)
            for (n0, n1), pr in sorted(nd.items(), key=lambda kv: idx_joint_state(kv[0][0], kv[0][1])):
                y = idx_joint_state(n0, n1)
                lines.append(f"T: {a0} {a1} : {x} : {y} : {pr:.12g}")

    # Deterministic observations of next local states
    for a0, a1 in product(ACTIONS, ACTIONS):
        for y, (s0, s1) in enumerate(states):
            o0 = OBS_INDEX[s0]
            o1 = OBS_INDEX[s1]
            lines.append(f"O: {a0} {a1} : {y} : {o0} {o1} : 1")

    # Scalar rewards (expected team health)
    for a0, a1 in product(ACTIONS, ACTIONS):
        for x, (s0, s1) in enumerate(states):
            r = expected_team_reward(s0, s1, a0, a1, p, c, intervention_cost)
            lines.append(f"R: {a0} {a1} : {x} : * : * : {r:.12g}")

    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate scalar epidemic_light.dpomdp")
    ap.add_argument("--output", default=None, help="Output .dpomdp path")
    ap.add_argument("--p", type=float, default=0.8, help="Infection persistence probability")
    ap.add_argument("--c", type=float, default=3.0, help="Infection cost in health reward")
    ap.add_argument(
        "--intervention-cost",
        type=float,
        default=0.2,
        help="Cost per closed region and decision step",
    )
    ap.add_argument("--discount", type=float, default=1.0, help="Discount factor")
    ap.add_argument(
        "--start-mode",
        choices=("uniform", "is"),
        default="is",
        help="Initial epidemic-state distribution: uniform or delta_(I,S)",
    )
    args = ap.parse_args()

    if not (0.0 <= args.p <= 1.0):
        raise ValueError("--p must be in [0,1]")

    if args.intervention_cost < 0.0:
        raise ValueError("--intervention-cost must be non-negative")

    if args.output is None:
        script_dir = Path(__file__).resolve().parent
        args.output = str(script_dir / "epidemic_light.dpomdp")

    lines = generate_lines(
        p=args.p,
        c=args.c,
        intervention_cost=args.intervention_cost,
        discount=args.discount,
        start_mode=args.start_mode,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote: {out}")
    print(
        f"params: p={args.p}, c={args.c}, intervention_cost={args.intervention_cost}, "
        f"discount={args.discount}, start_mode={args.start_mode}"
    )


if __name__ == "__main__":
    main()

