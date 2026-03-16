# EXPERIMENT_CONFIGS.md
## Genesis-2 Starter Experiment Configurations

This file defines practical early experiments for Genesis-2 so you can tell whether the simulation is actually evolving meaningful behavior or just drifting.

These are not final production configs. They are diagnostic runs meant to validate specific parts of the system.

---

# How to Use This File

Each experiment below includes:

- purpose
- what to tune
- what success looks like
- common failure modes
- what to do next

These should eventually map to actual YAML config presets in the app.

Suggested naming convention:

- `balanced_baseline`
- `verifier_economy_test`
- `artifact_evolution_test`
- `decomposition_pressure_test`
- `diversity_stress_test`
- `stagnation_recovery_test`

---

# Global Baseline Assumptions

Unless otherwise noted:

- starting agents: 10
- generations: 50
- starting energy: 60
- reproduction threshold: 140
- reproduction cost: 35
- upkeep per generation: 6
- inactivity penalty: 2
- mutation rate: 0.08
- structural mutation rate: 0.03
- artifact inheritance slots: 2
- backend: `HeuristicBackend`
- asexual reproduction only

These are not sacred. They are just a sane starting point.

---

# 1. Balanced Baseline

## Purpose
This is the default sanity-check run.

It tests whether:
- agents survive at all
- passive agents die
- lineages form
- workflows diverge
- the economy is roughly balanced

## Problem Mix
- T1: 40%
- T2: 35%
- T3: 20%
- T4: 5%

## Reward Emphasis
- final solve: normal
- verification: normal
- subtasks: normal
- artifact reuse: normal

## Success Looks Like
- population does not instantly collapse
- population does not explode endlessly
- some agents reproduce
- some agents die
- workflow diversity starts increasing
- at least 2 to 4 meaningful lineages survive past generation 20

## Failure Modes
### Failure: population explosion
Likely causes:
- starting energy too high
- reproduction threshold too low
- upkeep too low

### Failure: all agents die fast
Likely causes:
- upkeep too high
- problems too hard
- rewards too low

### Failure: everyone survives by doing nothing
Likely causes:
- inactivity penalty too weak
- upkeep too weak
- reproduction too cheap

## Next Step
If this run is stable, move to the verifier and artifact tests.

---

# 2. Verifier Economy Test

## Purpose
Test whether verification can become a real niche.

This is one of the most important early experiments because if verification never becomes profitable, the system will tend to collapse toward one-shot solving.

## Problem Mix
- T1: 20%
- T2: 35%
- T3: 35%
- T4: 10%

Make sure a meaningful number of tasks can produce plausible but wrong answers.

## Reward Emphasis
- final solve: normal
- successful verification: high
- catching incorrect answer: high
- critique that changes final answer: medium

Suggested reward bump:
- successful verification: +8
- catching incorrect answer: +12

## Success Looks Like
- some agents spend much more time verifying than solving
- verifier-heavy lineages survive
- incorrect answers are caught before final payout
- verifier role share grows above zero and stays there

## Failure Modes
### Failure: no one verifies
Likely causes:
- verification reward too low
- verification costs too high
- too few plausible wrong answers
- final solve reward dominates too much

### Failure: everyone verifies and nobody solves
Likely causes:
- verification too profitable
- solve costs too high
- too many tasks depend on review

## Next Step
If successful, keep verification economics in every future run.

---

# 3. Artifact Evolution Test

## Purpose
Test whether reusable artifacts actually matter.

Without this, nothing meaningful accumulates across generations except workflow drift and thresholds.

## Problem Mix
- T1: 25%
- T2: 30%
- T3: 30%
- T4: 15%

Use repeated problem motifs so artifacts can pay off across multiple generations.

Examples:
- repeated arithmetic error patterns
- repeated code-reading traps
- repeated decomposition structures

## Reward Emphasis
- artifact creation: low
- artifact reuse success: high
- artifact inheritance: enabled
- artifact decay: slow but real

Suggested reward bump:
- successful artifact reuse: +8

## Success Looks Like
- some artifacts are reused multiple times
- inherited artifacts improve outcomes for children
- artifact stores diverge between lineages
- strong lineages develop distinct “toolkits”

## Failure Modes
### Failure: no artifacts reused
Likely causes:
- artifacts too weak
- artifacts too generic or too specific
- problem motifs do not repeat enough
- artifact reward too low

### Failure: artifact spam
Likely causes:
- creating artifacts is too cheap
- publishing artifacts gives too much reward
- no decay on useless artifacts

## Next Step
If successful, artifact reuse should become a permanent tracked metric.

---

# 4. Decomposition Pressure Test

## Purpose
Test whether agents benefit from splitting problems into useful intermediate work.

If this fails, the simulation may still drift back toward wrapper behavior.

## Problem Mix
- T1: 10%
- T2: 25%
- T3: 30%
- T4: 35%

T4 tasks must be designed so that:
- subtasks are separable
- partial results are useful
- verification can change outcomes

## Reward Emphasis
- final solve: normal
- subtask completion: high
- useful partial result: medium-high
- critique: medium

Suggested bump:
- correct subtask completion: +10
- useful partial result later used: +7

## Success Looks Like
- some problems are solved through multi-step contribution chains
- certain lineages specialize in subtasks
- board shows non-random partial-result behavior
- subtasks become economically viable

## Failure Modes
### Failure: everything still solved monolithically
Likely causes:
- T4 tasks are still too easy
- subtask rewards too low
- problem decomposition is optional but not useful

### Failure: too much fragmentation
Likely causes:
- subtask reward too high
- everyone slices everything into noise

## Next Step
If successful, board analysis becomes much more important.

---

# 5. Diversity Stress Test

## Purpose
Test whether the ecosystem collapses into one dominant species too quickly.

## Problem Mix
- T1: 30%
- T2: 30%
- T3: 25%
- T4: 15%

## Special Settings
- keep mutation on
- optionally enable small diversity bonus
- allow one immigrant injection if diversity crashes

## Success Looks Like
- no immediate monoculture
- multiple workflow families remain alive through generation 30+
- diversity score does not flatline too early
- new lineages occasionally break through

## Failure Modes
### Failure: one lineage dominates instantly
Likely causes:
- reward imbalance
- mutation too low
- diversity safeguards absent
- founder population too homogeneous

### Failure: chaos and no stable lineages
Likely causes:
- mutation too high
- problems too volatile
- rewards too noisy

## Next Step
Tune mutation and diversity mechanisms before long runs.

---

# 6. Stagnation Recovery Test

## Purpose
Test whether pressure mode helps escape local optima instead of just destroying the population.

## Problem Mix
Start with:
- T1: 35%
- T2: 35%
- T3: 20%
- T4: 10%

Then let the run proceed long enough to plateau.

## Pressure Mode Trigger
Suggested trigger:
- no meaningful improvement in average energy for 12 to 15 generations
- diversity declining
- same lineage dominating too long

## Pressure Mode Effects
Temporary:
- mutation rate slightly up
- novelty reward slightly up
- abstention penalty slightly up
- T3/T4 reward slightly up
- verification and subtask rewards slightly up

Do not massively spike mutation.

## Success Looks Like
- new lineages appear after stagnation
- workflow diversity recovers
- average performance eventually improves again
- pressure mode does not cause mass extinction

## Failure Modes
### Failure: pressure mode kills everyone
Likely causes:
- mutation spike too high
- penalties too strong
- baseline economy already too harsh

### Failure: pressure mode does nothing
Likely causes:
- changes too mild
- system not actually stuck
- no unused niches exist to explore

## Next Step
Keep pressure mode conservative and targeted.

---

# 7. Population Funnel Test

## Purpose
Test your tournament / gauntlet approach before committing to huge runs.

## Setup
- spawn 30 or 50 agents
- run 15 to 25 generations
- cull bottom half
- run again
- repeat until 10 remain

This is not the final 500-generation mode. It is just to validate whether the funnel produces robust founders rather than lucky ones.

## Success Looks Like
- survivors are not all nearly identical
- multiple niche types survive the funnel
- resulting founder pool is diverse
- lineages that survive are active, not idle squatters

## Failure Modes
### Failure: only cautious agents survive
Likely causes:
- economy still rewards inactivity
- gauntlet not hard enough
- reproduction and upkeep poorly balanced

### Failure: only one niche survives
Likely causes:
- reward structure too narrow
- problem mix too one-dimensional

## Next Step
Use this before long-run founder selection.

---

# 8. Long-Run Incubation Test

## Purpose
Test whether a hardened founder population can continue evolving over many generations.

## Setup
- 15 selected founder agents
- 200 to 500 generations
- full logging
- pressure mode enabled
- artifact inheritance enabled

## Success Looks Like
- role distribution shifts over time
- workflow mutations accumulate
- artifacts meaningfully shape later performance
- lineages rise and fall instead of locking permanently
- occasional breakthrough generations happen

## Failure Modes
### Failure: total stasis
Likely causes:
- mutation too low
- ecology too narrow
- artifacts not valuable
- no pressure shifts

### Failure: endless churn with no progress
Likely causes:
- mutation too high
- scoring too noisy
- not enough stable niches

## Next Step
Only run this after the earlier tests pass.

---

# Recommended Default Presets for the UI

These presets should appear in the UI and map to actual config files later.

## Preset: Balanced
Use the Balanced Baseline config.

## Preset: Verifier Lab
Use the Verifier Economy Test config.

## Preset: Artifact Lab
Use the Artifact Evolution Test config.

## Preset: Decomposition Lab
Use the Decomposition Pressure Test config.

## Preset: Diversity Lab
Use the Diversity Stress Test config.

## Preset: Pressure Mode Lab
Use the Stagnation Recovery Test config.

## Preset: Founder Funnel
Use the Population Funnel Test config.

## Preset: Long Run
Use the Long-Run Incubation Test config.

---

# What to Watch First

The earliest meaningful signals are:

1. do passive agents die?
2. do at least two different niches survive?
3. are some artifacts reused?
4. do verifier lineages exist?
5. do workflow differences correlate with outcomes?

If the answer to all five is “no,” the system is not yet deep enough.

---

# Suggested Next Files After This One

After Codex has this file, the next useful documents are:

- `DIAGNOSTICS_GUIDE.md`
- `SIMULATION_CONFIG_TEMPLATE.yaml`
- `ROADMAP.md`

Those three will make debugging and iteration much faster.
