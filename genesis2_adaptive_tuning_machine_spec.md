# Genesis2 — Adaptive Tuning Machine Spec for Codex

## Mission

Build a deterministic, adaptive tuning supervisor on top of the existing Genesis2 system.

This machine is not a rewrite, not a side project, and not a one-off batch runner. It must sit on top of the current Genesis2 simulation, config, persistence, outputs, and observability stack and drive them toward a very specific ecological outcome: a repeatably healthy swarm that can reach and sustain roughly 100 agents through generation 100, then become a valid launch point for longer-horizon experiments.

The prior handoff already established the core idea: the tuner must run a simulation, observe it while it is happening, stop obviously bad runs early, learn from the result, adjust parameters, and keep iterating until it either finds reproducibly healthy swarms or hits a stop condition. It must optimize for reproducibly healthy swarms, not lucky outliers, and it must preserve successful swarms for later reuse. It must also support the later ability to promote winners into 200/300/500-generation endurance runs and future “dream team” experiments. Those requirements remain binding. See the existing adaptive handoff for the baseline intent, health model, adaptive loop, early-stop requirements, and persistence expectations. fileciteturn1file1 fileciteturn1file5 fileciteturn1file4

## The actual purpose of this machine

The machine exists to solve one practical problem:

Genesis2 currently requires too much manual babysitting to find swarm conditions that are actually useful. We need Codex to build a supervisor that can take a target definition of swarm health, run a controlled sequence of experiments, diagnose what happened, try smarter follow-up configs, and tell us whether the target is genuinely achievable under the current mechanics.

The machine should answer questions like:

- Can Genesis2 reliably produce a healthy 100-agent swarm by generation 100?
- Which configs only work once, and which ones are repeatable?
- Which failures are caused by slow growth, overshoot, dominance, stagnation, or collapse?
- When do we have a real candidate worth promoting into longer runs?
- When should the search stop because the current system design is not reaching equilibrium within the allowed budget?

This is the core unlock for the project’s next stage. The broader project direction is already defined: first achieve ecological maturity at roughly 100 agents, then use that as the foundation for later distributed intelligence. The goal is not to hand-code behavior, but to define success and search for it. The harness explores, the score judges, and the developer should not have to manually micromanage every attempt. fileciteturn1file2 fileciteturn1file3

## What success means

Codex should treat the following as the default target profile for the first release of the tuning machine:

## Goal profile: Dist. Intelligence Ready - Stable

This means the swarm is ecologically mature enough to support later distributed-intelligence layers, even if those layers are not implemented yet.

### Required lifecycle

A qualifying swarm should usually do all of the following:

1. Start in the normal low-count seed state, around 20 to 25 agents.
2. Grow in a controlled way toward 100 agents.
3. Reach the target neighborhood around generation 80.
4. Remain in a healthy band near 100 through generation 100.
5. Avoid founder lock-in, runaway dominance, late collapse, and repeated boom-bust oscillation.
6. Preserve multiple viable lineages.
7. Continue reproducing late enough that the system is clearly multi-generational rather than founder-carried.
8. Maintain acceptable throughput so that “stability” is not just a stagnant or underperforming ecosystem.

These conditions were already laid out in the swarm evolution plan and the adaptive tuning handoff: the healthy swarm is about 25 to 100 growth, then stable around 90 to 110 in the late run, with ongoing births, multiple lineages, bounded dominance, and acceptable resource/economy health. fileciteturn1file2 fileciteturn1file6

## Important correction to scope

The machine is not allowed to declare victory because one pretty run happened once.

A config only counts as successful if it can reproduce the health profile across multiple attempts. This is essential. The project documents already make that explicit: valid configs must meet the health conditions consistently across trials, not as a single-run fluke. The tuning machine therefore has two jobs, not one:

1. Find plausible candidates.
2. Prove they are repeatable.

That repeatability requirement is non-negotiable. fileciteturn1file5 fileciteturn1file8

## The machine we want Codex to build

Build a new subsystem that behaves like an adaptive experiment supervisor with five layers:

### 1. Goal profiles
A formal preset system that defines what “healthy” means in code.

### 2. Live trajectory observation
A way to watch a run while it is in progress, not just after it finishes.

### 3. Health scoring and pathology diagnosis
A deterministic evaluator that can say both “how good was this run” and “what went wrong.”

### 4. Adaptive search control
A controller that chooses the next configs based on prior observed results instead of only doing blind sweeps.

### 5. Harvesting and promotion
A registry that saves successful configs and successful swarm states for reuse, replay, and later long-run or dream-team experiments.

This should feel like a real product mode inside Genesis2, not a loose pile of scripts.

## Required operator experience

The user experience should be as close as possible to this:

- choose goal profile
- choose budget or stop mode
- click start
- watch progress if desired
- see live search decisions and run diagnoses
- review candidate winners
- automatically trigger repeatability checks for promising configs
- either harvest stable swarms or conclude the target was not reached in budget

The prior adaptive handoff explicitly called for a set-it-and-forget-it mode with stop modes such as timeout, max configs tested, max rounds, stop after N qualifying swarms, or stop after first strict success. Those requirements remain part of the spec. fileciteturn1file1 fileciteturn1file6

## Critical stop rule for this phase

For this specific phase, implement a practical stop condition with a hard budget:

### One-hour search budget

The machine should be able to run for up to one hour trying to achieve equilibrium under the selected goal profile.

If it does not find equilibrium within that budget, it must stop cleanly and produce a replan report for the user instead of grinding forever.

That replan report should say, in plain English and structured data:

- how many configs were tested
- how many runs were early-stopped
- how many full runs completed
- best candidate found
- why it still failed to qualify
- dominant recurring failure modes
- which parameter directions looked promising
- what should be changed next before running again

The point is that after one hour, the user should know whether the current mechanics appear capable of reaching the target or whether the system needs a design conversation.

## What equilibrium means for this phase

For the one-hour stopping rule, define equilibrium conservatively as follows:

A config can be considered to have reached equilibrium only if it:

1. passes the ecological gates for the selected target profile,
2. holds the late-run target band at generation 100,
3. avoids major pathology flags,
4. and then passes a repeatability check across additional validation runs.

Suggested default for release 1:

- one discovery run that looks good,
- followed by three validation runs,
- with at least two out of three validations meeting the same health thresholds,
- and no catastrophic failure in the validation set.

Codex may refine the exact policy, but the machine must have an explicit repeatability phase before calling something stable.

## What must be measured during a run

Codex should not wait until the end-of-run files to understand what is happening.

The tuning machine needs intermediate checkpoints, ideally every generation, or every few generations if overhead is a concern.

At minimum, the live observer must track:

- generation number
- current population
- rolling population slope
- births this generation
- deaths this generation
- cumulative births
- lineage count
- top lineage share
- top 3 lineage share
- inequality proxy, such as mean vs median energy and upper-tail concentration
- solve rate / productivity
- recent volatility in population
- recent volatility in dominance
- whether target band is approaching, reached, or drifting away

This aligns with the current project diagnosis that Genesis2 tends toward winner-take-all dynamics unless anti-dominance pressure is added, and that explicit tracking of top-lineage share, top-3 share, inequality, reproduction concentration, and survival/extinction is required. fileciteturn1file7

## Health model Codex should implement

The prior adaptive handoff defines three layers of evaluation: ecological gates, functional scores, and robustness across trials. That is the right framework and should be preserved. fileciteturn1file0

### Layer A: ecological gates

These are pass/fail conditions. If a run fails here, it is not healthy.

#### Population growth gate
- Start near the expected seed size.
- Reach a defined proximity to target by the required horizon.
- Reject runs that are too slow, too explosive, or never plausibly approach the target.

#### Late stability gate
- Late-run mean population must sit near target.
- Late-run volatility must remain bounded.
- No severe collapse in the final segment.

#### Diversity gate
- Minimum number of surviving lineages in the late run.
- Diversity cannot be only cosmetic early survival followed by effective monoculture.
- Later-born lineages should still appear active.

#### Dominance gate
- Top lineage share must remain below threshold.
- Top 3 lineage share must remain below threshold.
- Reproduction concentration must remain below threshold.
- Energy concentration cannot indicate that the ecosystem is alive only on paper.

#### Intergenerational vitality gate
- Births must continue into the late run.
- Descendants must matter, not just founders.
- Founder lock-in must remain bounded.
- The lineage tree must demonstrate real turnover and continuity.

#### Throughput gate
- Solve/productivity must remain above a floor.
- Stability achieved by starving the system or suppressing activity should fail.

### Layer B: functional quality scores

Only runs that pass Layer A get scored here.

Suggested components:

- artifact reuse quality
- cooperation effectiveness
- lineage complementarity
- role balance or niche health
- efficiency relative to task supply
- stability smoothness

These are ranking metrics, not substitutes for core ecological validity.

### Layer C: robustness

Rank candidate configs by:

- pass rate
- mean score
- median score
- worst-case score
- instability penalty
- pathology recurrence penalty

The tuner is optimizing for reliable health, not a cinematic one-off.

## Pathology taxonomy Codex should implement

The machine needs named failure modes. Do not just output a number.

Every run should produce one or more pathology tags such as:

- slow_growth
- never_reached_target_band
- early_collapse
- late_collapse
- runaway_overshoot
- sustained_overpopulation
- founder_lock_in
- lineage_dominance
- top3_capture
- reproduction_capture
- inequality_extreme
- stagnation_low_births
- stagnation_low_throughput
- oscillation_instability
- brittle_success
- non_repeatable_candidate

This matters because the adaptive controller must use the pathology tags to choose smarter next moves.

## Early-stop logic is mandatory

This is not optional sugar. A major reason to build this machine is to stop wasting time on hopeless runs.

The existing handoff already states that obviously bad runs should be terminated early when collapse, dominance lock-in, overshoot, stagnation, or impossible growth trajectories become clear. Codex should implement this conservatively but decisively. fileciteturn1file6

### Minimum early-stop triggers

Implement configurable early-stop rules such as:

- population falls below minimum viable threshold by generation X
- population growth is too far below target curve by generation X to recover
- dominance exceeds emergency threshold too early
- top 3 share exceeds emergency threshold and is still rising
- energy concentration becomes extreme well before stabilization phase
- births effectively stop too early
- explosion above upper control band persists for Y generations with no corrective trend
- repeated instability makes target-band capture impossible in the remaining time horizon

### Important design rule

Every early stop must emit a full reasoned diagnosis. Early stop is not “discard and move on.” It is an observed failed trajectory that should teach the controller something.

## Adaptive controller behavior

Do not let Codex build a blind parameter sweeper and call it adaptive.

The controller must maintain search memory.

At minimum it should record, over time:

- which parameter deltas improved growth speed
- which worsened overshoot
- which reduced dominance
- which preserved diversity
- which hurt throughput
- which combinations frequently failed
- which candidates looked promising but brittle

The prior handoff already calls for direct search logic informed by prior runs, learned correlations, and persistent ranking of what helped or hurt. That is the behavior we want. fileciteturn1file5 fileciteturn1file6

### Required search modes

Release 1 should include at least:

- baseline replay
- bounded random search
- directed local search around promising configs
- optional sweep mode for controlled exploration
- adaptive mode as the primary real workflow

### Recommended adaptive strategy for first implementation

Codex does not need to invent a fancy research optimizer on day one. A strong heuristic controller is enough if it is structured.

Suggested approach:

1. Seed search around known good baselines and nearby perturbations.
2. Run a small exploration batch.
3. Diagnose the dominant failure modes.
4. Apply parameter-direction heuristics based on pathology tags.
5. Track which heuristics actually helped.
6. Narrow around promising regions.
7. Promote plausible winners to repeatability testing.
8. Keep a ranked frontier of best configs.

That is enough to be meaningfully adaptive without external APIs.

## The machine must explicitly separate three stages of a search session

### Stage 1: exploration
Try a mix of replay, nearby perturbation, and wider bounded search to map the local landscape.

### Stage 2: exploitation
Focus on the most promising parameter neighborhoods, using prior diagnoses to push configs toward the target profile.

### Stage 3: validation
When a candidate looks healthy, stop spending the full budget only on discovery and instead prove repeatability.

This separation is important because otherwise the machine will spend too much time “finding” candidates and too little time proving they are real.

## Configs are not the only thing to preserve

The machine must preserve enough output to support future reuse.

The prior handoff already makes clear that successful swarms should later be promotable into endurance validation and dream-team experiments, which means preserving enough information to reload swarm state, replay winners, extract agent/lineage/role information, compare harvested swarms, and compose new starting populations from multiple successful sources. That persistence direction must be respected now even if dream-team assembly is implemented later. fileciteturn1file4

### Required harvested record for each candidate

For each harvested stable candidate, save:

- config snapshot
- seed
- run ids
- validation run ids
- health score summary
- gate pass/fail report
- pathology risk flags
- reason it qualified
- why it might still be risky
- recommended next action
- linkage to the actual saved swarm data and run artifacts

### Strongly preferred
Also save a normalized feature vector for easy comparison across harvested runs.

## Endurance promotion path

The system must support a follow-up action that takes harvested winners and promotes them into longer runs.

At minimum, Codex should create the plumbing so a validated stable swarm can be queued for:

- 200-generation run
- 300-generation run
- later 500-generation run

Even if the dream-team builder is not completed now, the promotion path and persistence model should make that next phase easy.

## UI / control surface requirements

The user should not need to hand-edit code to use this.

At minimum, expose controls for:

- goal profile selection
- search mode
- max runtime budget
- max configs tested
- max search rounds
- stop after N qualified swarms
- strict success repeatability rule
- early-stop aggressiveness
- baseline config source
- whether to auto-promote winners into validation
- whether to auto-promote validated winners into endurance queue

### Live dashboard should show

- current run config summary
- current generation and population
- whether run is on-track, drifting, or out-of-bounds
- best candidates found so far
- current top pathology trends across the session
- tested config count
- early-stop count
- validation count
- time remaining in budget

The point is not just pretty visuals. The point is to let the user understand why the machine is doing what it is doing.

## Codex should build this in phases

### Phase A: infrastructure
- goal profile definitions
- normalized metric extraction
- live checkpoints
- health scorer
- pathology tagging

### Phase B: adaptive supervision
- early-stop engine
- search session manager
- heuristic controller with search memory
- candidate ranking

### Phase C: proof and harvesting
- repeatability validator
- harvested stable swarm registry
- endurance promotion plumbing
- session summary / replan report

### Phase D: polish
- dashboard controls
- live status panels
- inspection tools for harvested winners

## Required outputs from Codex

Do not accept a vague implementation.

Codex should deliver:

1. actual code integration plan against the current Genesis2 codebase,
2. file/module map for where each subsystem will live,
3. exact data model for goal profiles, run checkpoints, health scores, pathology tags, search memory, and harvested candidates,
4. deterministic scoring rules and thresholds in code terms,
5. the one-hour session manager with stop conditions,
6. repeatability validation flow,
7. persistence format for harvested swarms,
8. UI or control entry points for running the machine,
9. a final summary report per search session.

## What Codex must not do

- do not rewrite Genesis2 from scratch
- do not replace the existing sim loop unless absolutely necessary
- do not handwave with “future enhancement” for the adaptive core
- do not build only a sweep runner and call it done
- do not optimize for a single lucky run
- do not hide failure reasons behind one composite score
- do not remove emergence to make the target easier to hit

## The practical decision rule this machine should enforce

At the end of a session, the user should get one of three answers:

### Answer 1: success
We found one or more configs that repeatedly produce a healthy swarm near 100 by generation 100. These are harvested and ready for promotion.

### Answer 2: near-success
We found promising but brittle candidates. Here is what almost worked, what failed repeatability, and what parameter directions look best next.

### Answer 3: not currently achievable in budget
After one hour of adaptive search, equilibrium was not reached. Here are the dominant failure modes and the concrete reasons we should replan before continuing.

That third answer matters. The machine should be able to tell us, honestly, that the target was not achieved under current mechanics and budget.

## Final one-line directive to Codex

Build a user-friendly, deterministic, adaptive tuning supervisor inside Genesis2 that can spend up to one hour searching for reproducibly healthy swarms, watch runs while they execute, stop bad trajectories early, diagnose exactly why they failed, adapt the next configs based on observed outcomes, validate repeatability before declaring success, harvest winners for later endurance and dream-team experiments, and stop with a replan report when equilibrium is not achieved in budget.
