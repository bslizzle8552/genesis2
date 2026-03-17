# DIAGNOSTICS_GUIDE.md
## Genesis-2 Diagnostics and Failure Detection Guide

This guide explains how to tell whether Genesis-2 is actually evolving meaningful behavior or whether the simulation is silently failing.

The most dangerous failure mode in evolutionary systems is not obvious crashing.
It is a simulation that *runs* but evolves nothing interesting.

---

# What This Guide Is For

Use this guide to answer questions like:

- Are the agents really evolving?
- Is the economy working?
- Are niches emerging?
- Are artifacts doing anything useful?
- Is the system collapsing into passive survival?
- Is the system collapsing into one-shot solver wrappers?
- Is mutation too weak or too strong?
- Is pressure mode helping or hurting?

This document should inform both:
- UI dashboard design
- post-run reporting
- experiment debugging

---

# The Most Important Rule

Do not judge the sim only by:
- top score
- total solves
- one “smart” agent

A system can look productive while being evolutionarily dead.

The real question is:

**Are different strategies, roles, and inherited structures affecting survival over time?**

If not, the simulation is shallow even if it solves some tasks.

---

# Core Dashboard Panels

The UI should expose these diagnostics first.

## 1. Population Over Time
Chart:
- x-axis: generation
- y-axis: population size

### Healthy signal
- some fluctuation
- no instant explosion
- no instant collapse
- population changes reflect births and deaths

### Bad signals
#### Flatline explosion
Population keeps growing with little death.
Likely causes:
- reproduction too cheap
- upkeep too low
- passive survival too easy

#### Crash to zero
Population collapses quickly.
Likely causes:
- upkeep too high
- rewards too low
- tasks too hard
- reproduction too expensive

#### Perfectly flat population with no churn
May indicate:
- no real selective pressure
- too much balancing
- weak ecological consequences

---

## 2. Average Energy and Energy Distribution
Charts:
- average energy by generation
- median energy by generation
- histogram or box plot of energy distribution

### Healthy signal
- visible spread
- some winners and losers
- reproduction tied to actual success
- average energy trends but not perfectly smooth

### Bad signals
#### Everyone clustered near the same value
Likely causes:
- rewards too homogeneous
- niches not meaningful
- mutation not affecting performance

#### Everyone always near reproduction threshold
Likely causes:
- economy tuned too generously
- insufficient scarcity

#### Everyone near zero all the time
Likely causes:
- economy too harsh
- impossible tasks
- no role is profitable

---

## 3. Births and Deaths Per Generation
Chart or table:
- births
- deaths
- net population change

### Healthy signal
- births and deaths both happen
- survival requires meaningful performance
- lineages rise and fall

### Bad signals
#### Many births, almost no deaths
Likely causes:
- passive survival
- low upkeep
- low reproduction threshold

#### Many deaths, almost no births
Likely causes:
- economy too punishing
- tasks too difficult
- no profitable role exists

#### Long periods of zero births and zero deaths
Likely causes:
- no meaningful selection pressure
- reproduction impossible
- system in frozen stasis

---

## 4. Role Distribution Over Time
Chart:
- percent or count of agents acting primarily as:
  - solver
  - verifier
  - decomposer
  - critic
  - coordinator
  - scout/retriever
  - mixed/generalist

These roles can be inferred from action proportions.

### Healthy signal
- multiple roles appear
- role mix changes over time
- some stable niche differentiation emerges

### Bad signals
#### Everyone is a solver
Classic wrapper collapse.
Likely causes:
- only final answers matter
- verification/decomposition/subtasks underpaid

#### Everyone is passive / abstaining
Likely causes:
- risk not worth taking
- inactivity too weakly punished
- reproduction too accessible

#### Extreme oscillation with no stable niches
Likely causes:
- mutation too high
- environment too noisy
- reward structure too unstable

---

## 5. Workflow Diversity
Track a diversity score over time based on:
- workflow structure
- module order
- flags
- threshold patterns

### Healthy signal
- diversity grows early
- diversity stabilizes at a nonzero level
- some lineages cluster but not total monoculture

### Bad signals
#### Diversity collapses almost immediately
Likely causes:
- one strategy overwhelmingly dominant
- founder pool too homogeneous
- mutation too weak
- no niche incentives

#### Diversity stays extremely high forever with no winners
Likely causes:
- mutation too strong
- no selection pressure
- behavior changes too random

---

## 6. Lineage Survival and Dominance
Display:
- lineage lifespans
- lineage population share
- top surviving lineages
- lineage family tree or stacked area chart

### Healthy signal
- some lineages persist
- some lineages die out
- occasional new lineages break through
- no trivial dominance from generation 1

### Bad signals
#### Single lineage dominance almost immediately
Likely causes:
- reward imbalance
- mutation too weak
- diversity controls absent
- founder set too similar

#### No lineage survives long
Likely causes:
- mutation too destructive
- economy too unstable
- artifacts not helping inheritance

---

## 7. Problem Outcome Breakdown
Track:
- total posted
- solved
- unsolved
- partially solved
- incorrect then corrected
- solved via subtask chain
- solved after verification
- time to solve

Break down by:
- tier
- domain
- lineage
- workflow family

### Healthy signal
- higher tiers solved less often than lower ones
- verification and subtasks matter on harder tiers
- different lineages succeed on different task types

### Bad signals
#### T3/T4 solved exactly like T1/T2
Likely causes:
- “hard” problems not actually harder
- decomposition not economically useful
- verification not important

#### Almost all solves are monolithic final solves
Likely causes:
- intermediate work not rewarded enough
- wrapper behavior dominates

---

## 8. Verification Effectiveness
Track:
- number of verification attempts
- number of caught wrong answers
- verifier success rate
- share of total reward earned by verifiers

### Healthy signal
- verification exists
- verifiers catch some errors
- verifier lineages survive economically

### Bad signals
#### Zero or near-zero verification attempts
Likely causes:
- verification underpaid
- verification too costly
- too few plausible mistakes

#### Verification attempts high but useless
Likely causes:
- verifier rewards misaligned
- verify action too noisy
- no real mistakes to catch

---

## 9. Artifact Effectiveness
Track:
- artifacts created
- artifacts reused
- artifact reuse success rate
- inherited artifact success rate
- artifact usage by lineage

### Healthy signal
- some artifacts reused repeatedly
- inherited artifacts improve offspring performance
- lineages develop distinct toolkits

### Bad signals
#### Artifacts created but never reused
Likely causes:
- no repeated motifs in problems
- artifacts too weak
- rewards for reuse too low

#### Artifact spam
Likely causes:
- creating artifacts too cheap
- no decay or filtering for bad artifacts
- publishing artifacts rewarded too much

#### All successful artifacts are identical
Likely causes:
- too little artifact diversity
- one dominant checklist becoming universal too soon

---

## 10. Action Mix
Track average share of actions:
- bid
- solve
- verify
- post_subtask
- critique
- abstain
- reproduce

### Healthy signal
- mixed action patterns
- action mix changes with environment
- not all agents use the same action ratios

### Bad signals
#### Abstain dominates
Likely causes:
- ecology too punishing
- inactivity not punished enough
- risk too high relative to reward

#### Solve dominates overwhelmingly
Likely causes:
- wrapper collapse
- other roles not profitable

#### Critique/subtask actions near zero forever
Likely causes:
- no reason for these behaviors to exist
- no reward for intermediate process

---

## 11. Pressure Mode Diagnostics
Track:
- pressure mode activations
- generations active
- performance before/after
- diversity before/after
- births/deaths before/after

### Healthy signal
- pressure mode sometimes revives diversity
- pressure mode helps break stagnation
- no total population wipeout

### Bad signals
#### Pressure mode causes extinction spikes
Likely causes:
- mutation increase too aggressive
- penalty increase too harsh
- baseline economy already fragile

#### Pressure mode has no measurable effect
Likely causes:
- too mild
- system not actually stagnating
- no latent niches available to exploit

---

# The Five Earliest High-Value Signals

Within the first 20 to 50 generations, check these first.

## Signal 1: Passive agents die
If passive agents survive and reproduce, stop and retune the economy.

## Signal 2: More than one role exists
If everyone behaves like the same kind of solver, stop and retune rewards.

## Signal 3: At least one artifact gets reused successfully
If no artifacts matter, inheritance depth is too shallow.

## Signal 4: Verifier behavior exists and matters
If verification never helps, the system will drift toward wrappers.

## Signal 5: Workflow variation correlates with outcomes
If different workflows do not affect success or survival, mutation substrate is too weak.

If all five fail, the sim is not evolving in a meaningful way.

---

# Silent Failure Patterns

These are the dangerous cases where the sim “looks fine” but is actually dead.

## Silent Failure A: Fancy Wrapper Syndrome
Symptoms:
- most reward comes from final solves
- almost no verifier/subtask/artifact value
- all strong agents behave similarly
- T3/T4 tasks solved monolithically

Diagnosis:
The ecology rewards one-shot solving.

Fix:
- increase verification and subtask value
- make decomposition genuinely useful
- redesign harder tasks

---

## Silent Failure B: Passive Squatter Syndrome
Symptoms:
- many agents survive by doing little
- abstain rates high
- reproduction still happening
- energy not draining enough

Diagnosis:
Economy is too soft.

Fix:
- increase upkeep
- raise reproduction threshold
- add stronger inactivity penalty
- require recent contribution to reproduce

---

## Silent Failure C: Mutation Fog
Symptoms:
- lots of variation but no stable lineages
- no clear role formation
- constant churn
- no inherited advantage

Diagnosis:
Mutation too destructive or selection too weak.

Fix:
- reduce structural mutation rate
- make inheritance more stable
- strengthen niche economics

---

## Silent Failure D: Frozen Monoculture
Symptoms:
- one lineage dominates quickly
- diversity flatlines
- no new breakthroughs
- runs feel deterministic too early

Diagnosis:
Exploration too weak or one strategy too profitable.

Fix:
- increase diversity support
- boost mutation slightly
- adjust rewards to support multiple niches
- diversify founder population

---

# Dashboard Priority Order

If time is limited, build these dashboard views first in this order:

1. population over time
2. average energy + energy histogram
3. births/deaths per generation
4. role distribution over time
5. workflow diversity
6. lineage survival chart
7. problem outcome breakdown
8. artifact reuse
9. verification effectiveness
10. pressure mode diagnostics

This order gives the fastest debugging insight.

---

# Recommended Report Sections

Every run summary should include these sections.

## Executive Summary
- run name
- generations
- final population
- dominant lineages
- whether roles emerged
- whether artifacts mattered
- whether pressure mode fired

## Economy Summary
- avg energy
- top/bottom energy
- births/deaths
- passive agent death rate

## Role Summary
- final role distribution
- role distribution trend
- verifier and decomposer presence

## Problem Summary
- solved/unsolved by tier and domain
- monolithic vs multi-step solves
- incorrect answers corrected by verification

## Artifact Summary
- artifacts created
- artifacts reused
- top artifacts by success
- inherited artifact effectiveness

## Lineage Summary
- longest surviving lineages
- biggest lineages
- breakthrough lineages
- extinct lineages of note

## Warning Flags
Automatically list major warning conditions:
- passive survival detected
- no verifier niche detected
- no artifact reuse detected
- diversity collapse detected
- wrapper behavior dominant
- mutation instability detected

---

# Suggested Threshold Alerts

The UI should flag these conditions automatically.

## Alert: Passive survival risk
If >25% of agents reproduce with very low contribution counts

## Alert: Wrapper dominance risk
If >80% of reward comes from final solves for many generations

## Alert: No verifier niche
If verification actions remain near zero past generation 20

## Alert: Artifact irrelevance
If artifact reuse remains near zero past generation 30

## Alert: Diversity collapse
If diversity falls below threshold and stays there

## Alert: Mutation chaos
If lineages fail to persist beyond a short window repeatedly

---

# Best Single Question to Ask After Every Run

Ask:

**What exactly was being selected for in this run?**

If the answer is:
- caution
- one-shot solving
- survival without contribution
- random churn

then the ecology still needs work.

If the answer is:
- useful niche behavior
- inherited tactics
- role differentiation
- profitable intermediate work

then the project is on the right track.

---

# Final Diagnostic Philosophy

Do not treat diagnostics as optional polish.
Diagnostics are part of the research instrument.

If you cannot see:
- who is winning
- why they are winning
- what traits are surviving
- whether roles are real
- whether inheritance matters

then you are not running an evolutionary system.
You are just running a noisy simulation.

Build observability early.
