"""Assignment entities and explicit, auditable stage selection."""

import numpy as np


def stage_proportions(config):
    if config.merge == "repeated":
        return (config.stages[-1],) * len(config.stages)
    if config.merge == "baseline_pair":
        return (0.0,) + config.stages
    return config.stages


def merge_groups(config):
    count = len(config.stages)
    if config.merge == "baseline_pair":
        return [(f"baseline+{p:g}", (0, i + 1)) for i, p in enumerate(config.stages)]
    if config.merge == "final_pair":
        return [("final-only", (count - 1,))] + [
            (f"{config.stages[i]:g}+{config.stages[-1]:g}", (i, count - 1)) for i in range(count - 1)]
    if config.merge == "repeated":
        return [(f"repeat-{size}", tuple(range(size))) for size in range(1, count + 1)]
    return [(f"merge-{size}", tuple(range(count - size, count))) for size in range(1, count + 1)]


def assignments(config, n_nodes, membership, rng):
    if config.level == "unit":
        entities = np.arange(n_nodes)
    else:
        membership = np.asarray(membership)
        if membership.shape != (n_nodes,) or not np.issubdtype(membership.dtype, np.integer):
            raise ValueError("cluster membership must give an integer cluster for every node")
        _, entities = np.unique(membership, return_inverse=True)
    count = int(entities.max()) + 1
    if (config.merge == "baseline_pair" and config.baseline == "one" and
            config.temporal == "staggered" and np.floor(config.stages[0] * count) < 1):
        raise ValueError("the first stage treats fewer entities than the one-entity baseline; increase stages or use baseline='zero'")
    shared = rng.random(count)
    results = []
    for index, proportion in enumerate(stage_proportions(config)):
        scores = shared if config.temporal == "staggered" else rng.random(count)
        is_baseline = config.merge == "baseline_pair" and index == 0
        if is_baseline or config.randomization == "complete":
            selected = int(np.floor(proportion * count))
            if is_baseline:
                selected = 1 if config.baseline == "one" else 0
            treatment = np.zeros(count, dtype=np.int8)
            treatment[np.argsort(scores, kind="stable")[:selected]] = 1
        else:
            treatment = (scores < proportion).astype(np.int8)
        results.append(treatment[entities])
    return np.asarray(results)
