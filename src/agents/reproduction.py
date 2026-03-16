from __future__ import annotations

import random
from src.agents.agent import Agent


REPRODUCTION_COST = 35


def reproduce(parent: Agent, next_id: int, generation: int) -> Agent:
    parent.energy -= REPRODUCTION_COST
    child_genome = parent.genome.mutate()
    child = Agent(
        agent_id=f"A{next_id}",
        parent_id=parent.agent_id,
        lineage_id=parent.lineage_id,
        generation_born=generation,
        genome=child_genome,
        energy=parent.energy * 0.5,
        artifact_store=random.sample(parent.artifact_store, k=min(2, len(parent.artifact_store))),
    )
    parent.energy *= 0.5
    return child
