"""Validated, serializable scientific choices; no implicit globals."""

from dataclasses import asdict, dataclass, fields
import math


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "Dataset/socfb-Stanford3.mtx"
    outcome: str = "linear"
    level: str = "unit"
    randomization: str = "complete"
    temporal: str = "staggered"
    merge: str = "suffix"
    baseline: str = "one"
    stages: tuple[float, ...] = (0.02, 0.05, 0.10, 0.25, 0.50)
    estimator: str = "gnn"
    repetitions: int = 100
    seed: int = 2026
    graph_seed: int = 0
    cluster_seed: int = 10
    model_seed: int = 2
    resolution: float = 10.0
    epochs: int = 400
    learning_rate: float = 0.004
    threads: int = 1
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    sigma: float = 0.1
    r2: float = 0.5
    purpose: str = "simulation"

    def __post_init__(self):
        choices = {
            "outcome": {"linear", "multiplicative", "quadratic", "sqrt", "two_hop", "dynamic"},
            "level": {"unit", "cluster"}, "randomization": {"complete", "bernoulli"},
            "temporal": {"staggered", "independent"},
            "merge": {"suffix", "baseline_pair", "final_pair", "repeated"},
            "baseline": {"zero", "one"}, "estimator": {"ols", "gnn"},
            "purpose": {"simulation", "smoke"},
        }
        for name, allowed in choices.items():
            value = getattr(self, name)
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}")
        if not isinstance(self.dataset, str) or not self.dataset.strip():
            raise ValueError("dataset must be a path or 'demo'")
        for name in ("repetitions", "epochs", "threads", "seed", "graph_seed", "cluster_seed", "model_seed"):
            value = getattr(self, name)
            minimum = 1 if name in {"repetitions", "epochs", "threads"} else 0
            if type(value) is not int or value < minimum or value > 2**32 - 1:
                raise ValueError(f"{name} must be an integer in [{minimum}, 2**32-1]")
        for name in ("resolution", "learning_rate", "alpha", "beta", "gamma", "sigma", "r2"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.learning_rate <= 0 or self.resolution <= 0 or self.sigma < 0 or self.r2 < 0:
            raise ValueError("learning_rate/resolution must be positive; sigma/r2 nonnegative")
        if not isinstance(self.stages, (tuple, list)) or not self.stages:
            raise ValueError("stages must be a nonempty list of treatment proportions")
        for p in self.stages:
            if isinstance(p, bool) or not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 < p <= 1:
                raise ValueError("each stage proportion must be finite and in (0,1]")
        if any(a >= b for a, b in zip(self.stages, self.stages[1:])):
            raise ValueError("stages must be strictly increasing; use merge='repeated' to repeat the final proportion")
        object.__setattr__(self, "stages", tuple(float(p) for p in self.stages))
        if self.merge == "baseline_pair" and self.baseline == "one" and self.randomization == "bernoulli":
            raise ValueError("an exact one-entity baseline requires complete randomization; use baseline='zero' with Bernoulli")

    def to_dict(self):
        result = asdict(self)
        result["stages"] = list(self.stages)
        return result

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("configuration must be a JSON object")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        return cls(**values)
