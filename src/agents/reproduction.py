from __future__ import annotations

import random
from src.agents.agent import Agent


REPRODUCTION_COST = 32


def reproduce(
    parent: Agent,
    next_id: int,
    generation: int,
    mutation_rate: float = 0.15,
    reproduction_cost: float = REPRODUCTION_COST,
    child_energy_fraction: float = 0.5,
) -> Agent:
    parent.energy -= reproduction_cost
    pool = max(0.0, parent.energy)
    child_fraction = max(0.0, min(1.0, float(child_energy_fraction)))
    child_genome = parent.genome.mutate(mutation_rate=mutation_rate)
    child = Agent(
        agent_id=f"A{next_id}",
        parent_id=parent.agent_id,
        lineage_id=parent.lineage_id,
        generation_born=generation,
        genome=child_genome,
        energy=pool * child_fraction,
        artifact_store=random.sample(parent.artifact_store, k=min(2, len(parent.artifact_store))),
    )
    parent.energy = pool * (1.0 - child_fraction)
    return child
