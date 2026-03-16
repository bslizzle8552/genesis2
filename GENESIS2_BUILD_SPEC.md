# GENESIS‑2 BUILD SPEC (Short Version)

## Goal

Build a Python simulation where agents evolve workflows, strategies, and
reusable artifacts through mutation and natural selection in an energy
economy.

Agents must evolve **processes**, not just prompts.

------------------------------------------------------------------------

# Core Principles

1.  Agents evolve workflows, not prompts
2.  Energy economy determines fitness
3.  Genomes are structured (JSON/YAML)
4.  Agents create reusable artifacts
5.  Multiple ecological niches must exist
6.  Mutation can change behavior architecture

------------------------------------------------------------------------

# Agent Structure

Agents contain:

## Identity

-   agent_id
-   parent_id
-   lineage_id
-   generation_born

## Genome

Controls behavior.

Example:

``` yaml
genome:
  specialization:
    math: 0.6
    logic: 0.5
    code: 0.7
    verification: 0.6

  workflows:
    math: [classify, plan, solve, verify, submit]
    code: [classify, plan, solve, self_test, verify, submit]

  thresholds:
    reproduce_energy: 140
    verify_tier_gte: 2

  strategy:
    aggression: 0.45
    risk_tolerance: 0.50
    artifact_reuse_bias: 0.70
```

## Runtime State

-   energy
-   current_tasks
-   generation_age

## Artifact Store

Reusable tactics like:

-   verification checklist
-   decomposition template
-   routing rule

## Actions

Agents can:

-   scan_board
-   bid
-   claim_task
-   solve
-   verify
-   critique
-   submit
-   reproduce

------------------------------------------------------------------------

# Economy

## Rewards

  Event                       Energy
  --------------------------- --------
  correct solution            +20
  successful verification     +6
  catching incorrect answer   +10
  useful subtask              +8
  artifact reuse              +6

## Costs

  Action                  Cost
  ----------------------- ------
  solve attempt           -4
  verify attempt          -2
  reproduction            -35
  upkeep per generation   -6

Doing nothing must lose energy.

------------------------------------------------------------------------

# Reproduction

Version 1 uses **asexual reproduction**.

Conditions: - energy \>= reproduction threshold - recent contribution

Child inherits: - mutated genome - limited artifacts

------------------------------------------------------------------------

# Problem Domains

Initial domains:

-   math
-   logic
-   code interpretation
-   decomposition tasks

Tier system:

  Tier   Description
  ------ ------------------------
  T1     simple
  T2     multi-step
  T3     verification valuable
  T4     decomposition required

------------------------------------------------------------------------

# Niches

The economy should support roles:

-   solver
-   verifier
-   decomposer
-   critic
-   coordinator

------------------------------------------------------------------------

# Simulation Loop

    spawn problems
    agents read board
    agents act
    score results
    update energy
    apply upkeep
    reproduce
    remove dead agents
    log generation

------------------------------------------------------------------------

# Logging

Log per generation:

-   population size
-   births/deaths
-   energy distribution
-   problem outcomes
-   artifacts created/reused
-   lineage data

Output JSON files.

------------------------------------------------------------------------

# UI Requirements

Local web interface with:

-   simulation setup page
-   presets
-   custom parameters
-   run button
-   live dashboard
-   agent inspector
-   board viewer
-   report viewer

Charts: - population - energy - problem success - diversity

------------------------------------------------------------------------

# Launcher

Include Windows launcher:

START_GENESIS2.bat

Double‑click should: 1. start backend 2. open browser UI

------------------------------------------------------------------------

# Repo Structure

    genesis2/
      README.md
      config/
      src/
        engine/
        agents/
        world/
        analytics/
        backends/
        ui/
      runs/
      tests/

------------------------------------------------------------------------

# First Experiment

Run:

-   10 agents
-   50 generations
-   no APIs
-   full logging

Success means: - diverse workflows - specialized roles - artifact
reuse - multiple lineages
