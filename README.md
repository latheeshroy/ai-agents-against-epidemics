# AI Agents Against Epidemics

How can AI agents coordinate interventions against a spreading epidemic?

In this project, you will learn epidemic dynamics from data, represent them as a compact **Networked Distributed POMDP (ND-POMDP)**, and use the learned model for decentralised planning:

    epidemic data -> learned ND-POMDP -> oSarsa-seq -> coordination policies

An ND-POMDP represents a decentralised decision problem through structured local interactions rather than one monolithic joint model.

## Your task

Using suitable synthetic or real epidemic data:

1. learn and validate a probabilistic model of the epidemic dynamics;
2. formulate a compact ND-POMDP, including states, local actions, observations, dynamics and objective;
3. connect your model to the problem interface used by oSarsa-seq;
4. analyse the resulting coordination policies.

The choice of data is part of the project. Document its provenance and justify why it supports the dynamics you want to learn.

## Start here

### Requirements

- CMake 3.15 or newer;
- a C++17 compiler;
- Eigen3 headers;
- Python 3.


### Run Day One

From the project root:

    ./project setup
    ./project day-one

Day One runs a supplied two-region **SIR-style** Dec-POMDP through oSarsa-seq:

    toy epidemic model -> oSarsa-seq -> coordination policy

The toy model has local states `S`, `I`, `R`, local actions `closed`/`open`, perfect local observations of the next local state, and a scalar objective balancing health with a small cost for closing a region. Its purpose is to make the model-to-decision pipeline inspectable; it is not intended as a realistic epidemic model.

A successful run ends with `DAY-ONE PASS`. Inspect `outputs/day_one_policy.txt`: it maps the information locally available to each region to the intervention selected by oSarsa-seq.

The model is in `example/epidemic_light.dpomdp`; `example/generate_epidemic_dpomdp.py` shows how it is generated. To see the planner-side contract, start from `osarsa_seq/src/main.cpp` and `osarsa_seq/src/decPOMDP/sequential/oSarsa.cpp`.

Your research develops the data-driven version:

    epidemic data -> learned ND-POMDP -> oSarsa-seq -> coordination policies

## Research question

> How can epidemic data be used to learn a compact ND-POMDP model for decentralised coordination?

## Expected contribution

A validated data-driven ND-POMDP model of epidemic dynamics and an analysis of the coordination policies it produces.

## References

- **SIR epidemic modelling — start here.** ETH Zürich, *SIR models of epidemics*: https://tb.ethz.ch/education/learningmaterials/modelingcourse/level-1-modules/SIR.html
- **ND-POMDPs.** R. Nair, P. Varakantham, M. Tambe and M. Yokoo (2005), *Networked Distributed POMDPs: A Synthesis of Distributed Constraint Optimization and POMDPs*, AAAI 2005: https://cdn.aaai.org/AAAI/2005/AAAI05-022.pdf
- **Sequential planning.** J. Peralez, A. Delage, J. Castellini, R. F. Cunha and J. S. Dibangoye (2025), *Optimally Solving Simultaneous-Move Dec-POMDPs: The Sequential Central Planning Approach*, AAAI 2025: https://doi.org/10.1609/aaai.v39i22.34494
