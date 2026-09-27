"""Potential outcomes and graph-specific noise-averaged GATE."""

import numpy as np


def _parts(graph, treatment, config):
    z = np.asarray(treatment, dtype=float)
    if z.shape != (graph.n_nodes,) or not np.all((z == 0) | (z == 1)):
        raise ValueError("treatment must contain one binary value per canonical node")
    exposure = graph.walk @ z
    if config.outcome == "quadratic":
        spillover = exposure**2
    elif config.outcome == "sqrt":
        spillover = np.sqrt(exposure)
    elif config.outcome == "two_hop":
        spillover = exposure + config.r2 * (graph.walk @ exposure - graph.return_probability * z)
    else:
        spillover = exposure
    return z, spillover


def mean_outcome(graph, treatment, config):
    z, spillover = _parts(graph, treatment, config)
    if config.outcome == "multiplicative":
        return config.alpha * graph.degrees / graph.degrees.mean() * (1 + config.beta * z + config.gamma * spillover)
    return config.alpha + config.beta * z + config.gamma * spillover


def sample_outcome(graph, treatment, config, rng):
    result = mean_outcome(graph, treatment, config)
    noise = rng.normal(0, config.sigma, graph.n_nodes)
    if config.outcome == "multiplicative":
        z, spillover = _parts(graph, treatment, config)
        noise *= graph.degrees / graph.degrees.mean() * (1 + config.beta * z + config.gamma * spillover)
    return result + noise


def true_gate(graph, config):
    return float(np.mean(mean_outcome(graph, np.ones(graph.n_nodes), config) -
                         mean_outcome(graph, np.zeros(graph.n_nodes), config)))
