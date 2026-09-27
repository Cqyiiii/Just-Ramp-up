# Legacy notebooks

This directory archives all 37 original notebooks, grouped below for browsing. Their filenames, code, metadata and saved outputs are unchanged; only their location has changed. The maintained experiment interface is [`python -m rampup`](../../README.md#quick-start), and the [migration guide](../../docs/migration.md#scenario-mapping) maps these notebooks to current configurations.

## Notebook index

| Family | Unit assignment | Cluster assignment |
|---|---|---|
| Complete randomization, ramp-up | [CR_unit_ramp_final](CR_unit_ramp_final.ipynb) | [CR_cluster_ramp_final](CR_cluster_ramp_final.ipynb) |
| Complete randomization, no ramp-up | [CR_unit_no_ramp_final](CR_unit_no_ramp_final.ipynb) | [CR_cluster_no_ramp_final](CR_cluster_no_ramp_final.ipynb) |
| Complete randomization, repeated | [CR_unit_repeat_final](CR_unit_repeat_final.ipynb) | [CR_cluster_repeat_final](CR_cluster_repeat_final.ipynb) |
| Bernoulli, ramp-up | [Ber_unit_ramp_final](Ber_unit_ramp_final.ipynb) | [Ber_cluster_ramp_final](Ber_cluster_ramp_final.ipynb) |
| Bernoulli, no ramp-up | [Ber_unit_no_ramp_final](Ber_unit_no_ramp_final.ipynb) | [Ber_cluster_no_ramp_final](Ber_cluster_no_ramp_final.ipynb) |
| Two-stage | [2step_CR_unit](2step_CR_unit.ipynb) | [2step_CR_cluster](2step_CR_cluster.ipynb) |
| Two-stage, incremental | [incre_2step_CR_unit](incre_2step_CR_unit.ipynb) | [incre_2step_CR_cluster](incre_2step_CR_cluster.ipynb) |
| Two-hop | [2hop_CR_unit](2hop_CR_unit.ipynb) | [2hop_CR_cluster](2hop_CR_cluster.ipynb) |
| Two-hop, incremental | [incre_2hop_CR_unit](incre_2hop_CR_unit.ipynb) | [incre_2hop_CR_cluster](incre_2hop_CR_cluster.ipynb) |
| Dynamic graph | [Dynamic_CR_unit](Dynamic_CR_unit.ipynb) | [Dynamic_CR_cluster](Dynamic_CR_cluster.ipynb) |
| Dynamic graph, incremental | [Dynamic_incre_CR_unit](Dynamic_incre_CR_unit.ipynb) | [Dynamic_incre_CR_cluster](Dynamic_incre_CR_cluster.ipynb) |
| Multiplicative | [Multiplicative_CR_unit](Multiplicative_CR_unit.ipynb) | [Multiplicative_CR_cluster](Multiplicative_CR_cluster.ipynb) |
| Multiplicative, incremental two-stage | [Multiplicative_incre_2step_CR_unit](Multiplicative_incre_2step_CR_unit.ipynb) | [Multiplicative_incre_2step_CR_cluster](Multiplicative_incre_2step_CR_cluster.ipynb) |
| Square-root | [sqrt_expo_CR_unit](sqrt_expo_CR_unit.ipynb) | [sqrt_expo_CR_cluster](sqrt_expo_CR_cluster.ipynb) |
| Square-root, incremental two-stage | [sqrt_incre_2step_CR_unit](sqrt_incre_2step_CR_unit.ipynb) | [sqrt_incre_2step_CR_cluster](sqrt_incre_2step_CR_cluster.ipynb) |
| Quadratic | [square_expo_CR_unit](square_expo_CR_unit.ipynb) | [square_expo_CR_cluster](square_expo_CR_cluster.ipynb) |
| Quadratic, incremental two-stage | [square_incre_2step_CR_unit](square_incre_2step_CR_unit.ipynb) | [square_incre_2step_CR_cluster](square_incre_2step_CR_cluster.ipynb) |
| Historical HT | [ht_unit_ramp](ht_unit_ramp.ipynb) | [ht_cluster_ramp](ht_cluster_ramp.ipynb) |

[tables.ipynb](tables.ipynb) contains the historical table workflow. The companion [`table_output.py`](../../table_output.py) remains at the repository root.

## Working directory

The archived cells use paths such as `Dataset/socfb-Stanford3.mtx`, `Result/*.pkl` and `A_2hop.pkl`, relative to the **kernel's working directory**, not the notebook file. Keep the kernel in the repository root. When Jupyter starts a kernel in `notebooks/legacy/`, run this in a temporary cell before the original cells:

```python
from pathlib import Path
import os

here = Path.cwd().resolve()
repo_root = next(
    (path for path in (here, *here.parents)
     if (path / "pyproject.toml").is_file()
     and (path / "Dataset").is_dir()
     and (path / "rampup").is_dir()),
    None,
)
if repo_root is None:
    raise RuntimeError("Start the kernel inside the Just-Ramp-up checkout.")
os.chdir(repo_root)
print(Path.cwd())
```

This also works if the kernel already starts at the repository root. Existing `Dataset/` and `Result/` files have not been moved or copied into this archive.

These notebooks retain their historical dependencies, execution order and file assumptions. Some require intermediate files absent from this snapshot, including `A_2hop.pkl`; moving them does not supply those files or certify their saved results. Their save cells can overwrite files in `Result/`, so use a separate checkout if rerunning historical experiments. Use the maintained runner for new experiments and consult the [migration boundaries](../../docs/migration.md) when comparing results.
