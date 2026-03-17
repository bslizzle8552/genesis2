# Genesis2 Codex Task Brief
Version: 1.0  
Date: 2026-03-16  
Audience: Codex / implementation partner  
Purpose: Convert current Genesis2 research findings into a coherent engineering plan with clear instrumentation, experiment, and rebalancing tasks.

---

# 1. Executive Summary

Genesis2 is already behaving like a real evolutionary ecology rather than a toy genetic algorithm.

The current hard data supports five major conclusions:

1. **Problem throughput appears to be the main carrying-capacity control knob.**
   Low-throughput runs converge to small populations even with many generations, while high-throughput runs support much larger stable populations.

2. **Population growth is real, but diversity is not scaling well.**
   Large runs are producing more agents, but not a proportionally richer ecosystem. Diversity appears to compress as the system grows.

3. **Energy concentration is likely a core structural force.**
   The new observability run shows a strongly skewed energy distribution, suggesting that a minority of agents may be capturing a disproportionate share of rewards.

4. **Collaboration is real, but likely under-selected.**
   Collaborative solves appear highly effective, but collaboration share remains modest. That suggests a gap between behavioral usefulness and reproductive payoff.

5. **Artifacts are becoming increasingly important.**
   Artifact-assisted share rises substantially in larger/longer runs, implying that persistent structure or reusable outputs may be central to long-horizon productivity.

This means Genesis2 is not failing. It is successfully generating emergent ecology. The current problem is that the ecology may be drifting toward **winner-take-most dominance** instead of **stable specialization and pluralism**.

The next implementation phase should therefore focus on:

- making concentration visible,
- making contribution visible,
- making collaboration economically legible,
- making artifact ecology measurable,
- and building experiment infrastructure so the research loop becomes systematic instead of anecdotal.

---

# 2. Research Context

Genesis2 is an evolutionary multi-agent simulation where:

- agents solve problems to earn energy,
- energy supports survival and reproduction,
- reproduction and mutation alter the agent population over time,
- and the whole system behaves like a resource-limited ecology.

This is not just a benchmarking environment. It is a research platform for studying:

- carrying capacity under limited resources,
- emergent collaboration,
- ecological specialization,
- lineage dominance,
- artifact reuse,
- and plateau behavior under no-API constraints.

The system should not be simplified back into a generic evolutionary simulator. The whole point is to preserve the ecological framing and deepen the observability and experiment rigor around it.

---

# 3. Key Findings We Need to Build Around

## 3.1 Carrying capacity appears throughput-driven

Observed pattern from run set:

- very low problems/gen -> small stable populations,
- moderate problems/gen -> much larger stable populations,
- very high problems/gen -> very large populations become possible.

Interpretation:

- problems are the primary energy supply,
- energy supply sets ecological carrying capacity,
- generations alone do not force sustainable growth.

Engineering implication:

We need tooling that treats throughput as a first-class ecological variable and produces explicit carrying-capacity curves.

## 3.2 Overshoot dynamics may already exist

At least one run shows a pattern consistent with overshoot and correction:

- population grows above what low throughput can sustain,
- then contracts toward a lower level.

Engineering implication:

We need phase-state detection and synchronized births/deaths/population/energy time-series views.

## 3.3 Energy inequality is likely driving selection pressure

In the new observability run, energy is already strongly skewed.

Example signal:

- median energy far below mean,
- p99 massively above median,
- repeat reproducers already present.

Interpretation:

A minority of agents may be extracting outsized reward and converting that into repeated evolutionary influence.

Engineering implication:

Energy concentration must become a tracked diagnostic, not an afterthought.

## 3.4 Collaboration works, but may not be fully rewarded

Observed pattern:

- collaboration share is meaningful but not dominant,
- collaborative success can exceed solo success,
- yet collaboration still does not appear to dominate evolutionary behavior.

Interpretation:

This strongly suggests that the current reward/reproduction pipeline may favor terminal reward capture more than total ecological contribution.

Engineering implication:

We need contribution-chain accounting and collaboration ROI metrics.

## 3.5 Artifact usage is now too important to ignore

Artifact-assisted share rises substantially as runs scale.

Interpretation:

Artifacts may be functioning as one or more of the following:

- reusable public goods,
- inherited infrastructure,
- memory scaffolding,
- or concentration amplifiers that mostly help already successful lineages.

Engineering implication:

Artifacts need their own ecology metrics.

---

# 4. High-Level Development Goals

The next work should serve four goals.

## Goal A — Make internal ecology legible
We need visibility into energy, reward, collaboration, lineage, and artifact flow.

## Goal B — Distinguish healthy specialization from pathological dominance
We need the ability to tell whether support roles are becoming viable or whether a few terminal solvers are absorbing most reproductive power.

## Goal C — Turn experiments into repeatable science
We need a batch harness and standardized outputs so claims about the system can be tested, compared, and reproduced.

## Goal D — Prepare the platform for ecology rebalancing
Instrumentation must be sufficient to support future reward and reproduction changes without guessing.

---

# 5. Immediate Priority Order

Implement in this order unless blocked by code architecture:

1. Concentration diagnostics
2. Reproduction concentration diagnostics
3. Collaboration economics
4. Artifact ecology diagnostics
5. Phase-state detection
6. Comparative experiment harness

Reason for this order:

- concentration and reproduction explain why the ecology is taking its current shape,
- collaboration and artifact metrics explain how work is actually getting done,
- phase-state detection explains time-dynamics,
- experiment harness turns all of it into a reliable research pipeline.

---

# 6. Detailed Codex Tasks

## Task A — Concentration Diagnostics Layer

### Objective
Quantify whether Genesis2 is producing a broad-based ecosystem or a highly concentrated reward economy.

### Why this matters
Current evidence suggests energy inequality may be one of the main hidden drivers of both diversity collapse and reproductive dominance.

### Implement
Per generation, compute and persist:

- population size
- total system energy
- min energy
- median energy
- mean energy
- p90 energy
- p95 energy
- p99 energy
- max energy
- top 10 richest agents
- top 1% energy share
- top 10% energy share
- Gini coefficient for energy
- lineage-level total energy
- top lineage energy share
- top 5 lineage energy shares
- role-level total energy
- role-level median energy

### UI / reporting requirements
Add:

- time-series for median/mean/p90/p99/max energy,
- histogram or binned distribution for current generation energy,
- "top agents" table,
- lineage energy leaderboard,
- role energy distribution summary.

### Persistence requirements
Metrics should be exported into machine-readable run artifacts, not just rendered in the dashboard.

Recommended outputs:

- `generation_metrics.jsonl`
- `lineage_metrics.jsonl`
- `role_metrics.jsonl`

### Acceptance criteria
This task is complete when:

- every generation has stored concentration metrics,
- dashboard can show the trend of concentration over time,
- exported artifacts allow offline analysis without UI,
- top-share and Gini metrics can be compared across runs.

### Non-goals
Do not change reward logic here. This task is observability only.

---

## Task B — Reproduction Concentration Diagnostics

### Objective
Measure whether reproduction is distributed across the ecosystem or dominated by a narrow subset of agents and lineages.

### Why this matters
Raw population size is not enough. We need to know who is reproducing, how often, and whether support roles are reproductively viable.

### Implement
Per generation, compute and persist:

- births count
- deaths count
- unique reproducers this generation
- repeat reproducers this generation
- cumulative unique reproducers
- cumulative repeat reproducers
- births by role
- births by lineage
- births by agent
- share of births from top 1 reproducer
- share of births from top 5 reproducers
- share of births from top 10 reproducers
- average births per reproducing agent
- median births per reproducing agent
- reproduction success rate by role
- reproduction success rate by lineage size bucket

Per agent, track:

- lifetime births
- first reproduction generation
- last reproduction generation
- lifetime contribution score (placeholder if contribution score is not implemented yet)
- lifetime energy earned
- lifetime energy spent
- lifespan in generations

### UI / reporting requirements
Add:

- births/deaths time-series,
- top reproducers table,
- births by role chart,
- births by lineage chart,
- cumulative reproduction concentration chart.

### Acceptance criteria
This task is complete when:

- any run can answer "who reproduced most?",
- any run can answer "what fraction of births came from the top N reproducers?",
- reproduction can be broken down by role and lineage,
- reproduction concentration can be compared across runs.

### Non-goals
Do not yet alter reproduction rules. This is measurement only.

---

## Task C — Collaboration Economics and Contribution-Chain Accounting

### Objective
Measure whether collaboration is merely behaviorally useful or also economically and evolutionarily viable.

### Why this matters
Current evidence suggests collaboration can outperform solo solving, but may not be rewarded proportionally. If that is true, the system will under-select collaboration even when it is genuinely effective.

### Implement
At the problem level, persist:

- problem id
- generation
- tier
- domain
- solved / unsolved
- solo / collaborative
- participant count
- participating agent ids
- participating lineage ids
- artifact-assisted yes/no
- solve duration if available
- final reward amount
- reward split by participant
- role tag per participant
- contribution-chain sequence

Standard contribution roles:

- planner_or_decomposer
- subtask_contributor
- verifier
- critic
- integrator
- final_solver

If exact role inference is imperfect, store best-effort tags with confidence or source markers rather than omitting the data.

### Aggregate metrics to compute
Per generation and per run:

- collaboration share overall
- collaboration share by tier
- collaboration share by domain
- solo success rate
- collaborative success rate
- solo average reward
- collaborative average total reward
- collaborative average reward per participant
- final solver reward share in collaborative solves
- average collaborator count
- average contribution-chain length
- success rate by contribution role presence
- contribution-role frequency in successful vs unsuccessful solves

### UI / reporting requirements
Add:

- solo vs collaborative success comparison,
- collaboration share by tier,
- collaboration share by domain,
- reward split visualization for collaborative solves,
- problem explorer with contribution-chain drilldown.

### Acceptance criteria
This task is complete when:

- we can compare solo and collaboration performance by context,
- we can see who participated in each successful chain,
- we can tell whether final solvers capture most collaborative reward,
- exported run data supports offline collaboration ROI analysis.

### Non-goals
Do not rebalance rewards in this task. This is instrumentation first.

---

## Task D — Artifact Ecology Diagnostics

### Objective
Determine whether artifacts act as public goods, inherited infrastructure, or dominance amplifiers.

### Why this matters
Artifact-assisted share rises significantly in larger and longer runs, which means artifacts are increasingly central to how the ecosystem sustains productivity.

### Implement
At minimum, track:

- artifact id
- creation generation
- creator agent id
- creator lineage id
- creator role
- artifact type/category if available
- times reused
- reuse generations
- reuser agent ids
- reuser lineage ids
- reused in successful solve yes/no
- reused in collaborative solve yes/no

Aggregate metrics:

- artifacts created per generation
- artifacts reused per generation
- average reuse count
- reuse rate by lineage
- cross-lineage reuse rate
- within-lineage reuse rate
- successful solve rate with artifact assistance
- successful solve rate without artifact assistance
- collaboration share with artifacts vs without artifacts
- reward lift associated with artifact use
- energy earned by artifact creators from downstream reuse, if applicable

### UI / reporting requirements
Add:

- artifact creation vs reuse time-series,
- artifact creator leaderboard,
- cross-lineage reuse matrix or summary table,
- artifact-assisted vs non-artifact solve comparison.

### Acceptance criteria
This task is complete when:

- we can quantify whether artifacts are broadly reused or mostly lineage-local,
- we can test whether artifacts correlate with better solve outcomes,
- we can test whether artifacts amplify concentration or diffuse capability.

### Non-goals
Do not add artifact reward policy yet unless necessary for data capture.

---

## Task E — Phase-State Detection

### Objective
Automatically identify ecological phases in a run instead of relying on visual intuition.

### Why this matters
Some runs appear to overshoot and then correct. If that is real, the system needs explicit markers for expansion, overshoot, collapse, and stabilization.

### Implement
Derive rolling metrics:

- rolling population slope
- rolling births minus deaths
- rolling median energy
- rolling mean energy
- rolling diversity slope
- rolling collaboration slope

Define phase heuristics for:

- expansion
- overshoot
- collapse
- stabilization

Initial heuristic suggestions:

- **Expansion**: positive population slope and births > deaths over rolling window
- **Overshoot**: recent peak population followed by negative population slope plus falling median energy
- **Collapse**: sustained negative population slope and deaths > births
- **Stabilization**: low absolute population slope with births roughly balanced by deaths over rolling window

Store:

- inferred phase per generation
- first generation of each detected phase
- peak population generation
- stabilization start generation

### UI / reporting requirements
Add:

- phase annotation on time-series charts,
- one-line phase summary in run report,
- peak/stabilization markers.

### Acceptance criteria
This task is complete when:

- every run can be segmented into interpretable ecological phases,
- overshoot-like runs are machine-identifiable,
- stabilization generation can be compared across parameter settings.

### Non-goals
Do not over-engineer the classifier. Start heuristic and transparent.

---

## Task F — Comparative Experiment Harness

### Objective
Make Genesis2 capable of systematic parameter sweeps with standardized outputs and comparison summaries.

### Why this matters
Right now many conclusions are inferred from individual runs. We need repeatable experimental structure.

### Implement
Build a batch runner that can sweep parameters across combinations while preserving run metadata.

Required sweep support:

- random seed
- starting agents
- generations
- starting energy
- reproduction threshold
- mutation rate
- upkeep
- problems per generation
- tier mix
- optional reward-policy config

Required output per run:

- full config snapshot
- summary.json
- run_summary.md
- generation_metrics.jsonl
- lineage_metrics.jsonl
- role_metrics.jsonl
- problem_metrics.jsonl
- artifact_metrics.jsonl (if task D implemented)

Required comparison report fields:

- final population
- peak population
- generation of peak population
- stabilization generation
- total solved
- solve rate
- collaboration share
- artifact-assisted share
- diversity index
- energy Gini
- top 10% energy share
- unique reproducers
- repeat reproducers
- births concentration share from top N reproducers

### Acceptance criteria
This task is complete when:

- a full sweep can be launched from config without manual edits per run,
- output files are consistent across runs,
- comparison summaries can rank or group runs by major ecological outcomes,
- multiple seeds can be included so conclusions are not single-seed anecdotes.

### Non-goals
Do not build a giant orchestration platform. Keep it practical and reproducible.

---

# 7. Recommended First Experiment Set

These are the experiments that should be run immediately after the above instrumentation is available.

## Experiment 1 — Carrying Capacity Sweep

### Purpose
Map the relationship between throughput and sustainable population.

### Fixed settings
- starting agents: 100
- generations: 300
- starting energy: current baseline
- reproduction threshold: current baseline
- mutation rate: current baseline
- upkeep: current baseline
- tier mix: current baseline

### Sweep
Problems per generation:

- 8
- 15
- 25
- 50
- 100
- 250
- 500

### Required outputs
For each condition and seed:

- final population
- peak population
- stabilization generation
- total solved
- solve rate
- diversity index
- energy Gini
- top 10% energy share
- collaboration share
- artifact-assisted share
- births/deaths totals

### Primary research question
Does throughput define a smooth carrying-capacity curve, a threshold curve, or multiple ecological regimes?

---

## Experiment 2 — Contribution Reward Sensitivity

### Purpose
Test whether collaboration becomes more evolutionarily stable when support work is explicitly rewarded.

### Design
Compare at least three reward policies under identical seeds and baseline config:

- Policy A: current reward logic
- Policy B: modest contribution-weighted split
- Policy C: strong contribution-weighted split with reduced final-solver dominance

### Required outputs
- collaboration share
- collaborative success rate
- reward distribution by role
- births by role
- diversity index
- energy concentration
- lineage dominance

### Primary research question
Is collaboration underexpressed because it is difficult, or because it is underpaid?

---

## Experiment 3 — Artifact Public-Good vs Concentration Test

### Purpose
Determine whether artifacts mostly help the whole ecosystem or mostly reinforce existing winners.

### Design
Compare conditions such as:

- current artifact behavior
- increased artifact reuse visibility
- optional artifact-creator residual credit (only after instrumentation)

### Required outputs
- cross-lineage reuse rate
- artifact-assisted success lift
- artifact effect on collaboration share
- artifact effect on concentration metrics
- artifact creators' lineage dominance over time

### Primary research question
Are artifacts acting like shared infrastructure or private advantage compounding?

---

## Experiment 4 — Mutation and Upkeep Interaction Sweep

### Purpose
Test whether diversity collapse is caused partly by ecological pressure rather than only reward concentration.

### Sweep
Mutation rate:

- 0.05
- 0.10
- 0.15
- 0.25
- 0.35

Upkeep:

- 4
- 5
- 6
- 7
- 8

### Required outputs
- diversity over time
- stabilization generation
- energy concentration
- lineage turnover
- final population

### Primary research question
What pressure regime best preserves diversity without killing population viability?

---

# 8. Future Rebalancing Work (Not Yet First)

These are likely next after instrumentation confirms the mechanisms.

## Ecology Rebalancing Candidate 1 — Contribution-Chain Rewards
Split rewards across:

- decomposer
- subtask contributor
- verifier
- critic
- integrator
- final solver

## Ecology Rebalancing Candidate 2 — Tier-Specific Solve Economics
Possible policy direction:

- T1: solo-friendly
- T2: verification beneficial
- T3: verification required for full payout
- T4: full payout requires a valid collaboration chain

## Ecology Rebalancing Candidate 3 — Role-Aware Reproduction
Reproduction should eventually consider more than energy alone.

Potential future inputs:

- energy
- contribution score
- artifact creation value
- verification value
- collaboration support value

Important:

These are future changes. Do not implement blindly before the observability layer shows where the real bottleneck is.

---

# 9. Data and File Format Guidance

Keep outputs boring, explicit, and easy to diff.

Recommended file structure per run:

```text
run_<seed>_<label>/
  config.json
  summary.json
  run_summary.md
  generation_metrics.jsonl
  lineage_metrics.jsonl
  role_metrics.jsonl
  problem_metrics.jsonl
  artifact_metrics.jsonl
```

Recommended principles:

- use stable field names,
- avoid hidden dashboard-only state,
- preserve enough metadata for offline analysis,
- make every run self-describing,
- include version metadata for schema changes.

Optional but useful:

- `schema_version`
- `git_commit`
- `code_version`
- `reward_policy_id`
- `experiment_id`

---

# 10. Implementation Notes for Codex

## Keep instrumentation separate from policy changes
Do not mix observability and reward-rule edits in the same task if avoidable. We need clean before/after comparisons.

## Prefer append-only metrics files where practical
JSONL is fine for generation/problem-level metrics. It is easier to inspect and less brittle than deeply nested monolithic JSON.

## Best-effort data is better than omitted data
If a contribution role is not perfectly inferable, store best-effort attribution with provenance notes rather than dropping the signal entirely.

## Preserve performance awareness
The system has already reached large populations. Instrumentation should avoid turning the simulator into a logging bottleneck.

Suggestions:

- aggregate on generation boundaries,
- sample only where full granularity is too expensive,
- gate expensive diagnostics behind config flags if needed.

## Keep dashboard and export schema aligned
The dashboard should not invent metrics that the exported artifacts cannot reproduce.

---

# 11. Definition of Success for This Development Phase

This phase is successful if, after implementation, we can answer the following with evidence instead of intuition:

1. Is carrying capacity primarily controlled by throughput?
2. Is diversity collapse caused more by concentration, low mutation, ecological pressure, or some combination?
3. Is collaboration underexpressed because it is rare, costly, or under-rewarded?
4. Are support roles reproductively viable?
5. Are artifacts ecosystem infrastructure or lineage amplifiers?
6. Which ecological phase is a given run in, and when did it transition?
7. Are conclusions robust across seeds?

If we can answer those reliably, Genesis2 will have crossed from interesting prototype into a real research platform.

---

# 12. Short Task List for Codex Ticketing

If this needs to be broken into actionable tickets, use this order:

1. Add concentration metrics export and dashboard views
2. Add reproduction concentration metrics export and dashboard views
3. Add problem-level contribution-chain accounting
4. Add collaboration ROI summaries by tier/domain
5. Add artifact creation/reuse tracking and summaries
6. Add ecological phase detection
7. Add batch experiment harness with standardized outputs
8. Run throughput sweep across multiple seeds
9. Produce comparison report from sweep
10. Only then begin reward/reproduction rebalancing experiments

---

# 13. One-Paragraph Instruction to Give Codex

Genesis2 is already behaving like a resource-limited evolutionary ecology. Your job is not to simplify it, but to make its internal dynamics measurable and experimentally testable. Prioritize instrumentation that reveals energy concentration, reproduction concentration, collaboration economics, artifact ecology, and ecological phase changes. Export all metrics in run-local machine-readable files, align the dashboard with those exports, and build a batch experiment harness so throughput, mutation, upkeep, and reward policy can be compared systematically across seeds. Do not mix observability changes with major reward-policy changes unless explicitly requested.

