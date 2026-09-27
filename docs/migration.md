# Migration from the original notebooks

The original 37 notebooks, `Dataset/`, `Result/` and `table_output.py` remain at their original paths. They are a historical record; the maintained execution interface is now `python -m rampup`. Saved `.pkl` arrays are not loaded or reinterpreted by the new runner. Legacy notebooks may depend on cell order, missing intermediate files and their original package environment.

## Material scientific and reproducibility differences

| Area | Historical implementation | Maintained runner |
|---|---|---|
| Unit IDs | Some assignments use array indices 0..n−1 as labels in a graph labelled 1..n. | Canonical IDs and row indices are explicit and aligned. |
| GNN feature order | Features iterate NetworkX insertion order; `dgl.from_networkx` may relabel in sorted order. | Sparse adjacency, treatment, outcomes and GNN features share sorted original IDs. |
| Dynamic graphs | Some notebooks train every stage on the final `dgl_G`, and use old cluster labels on newly indexed graphs. | Each stage carries its own graph; original node mapping and baseline clusters are preserved. Evaluation uses the last selected stage's graph. |
| Treatment count | Quantile comparisons (`<` versus `<=`) can produce different counts. | Complete randomization treats exactly floor(p × number of entities). Realized unit and entity counts are recorded. |
| Near-zero baseline | Several notebooks use no treated entities at p=0; others use one. | `baseline="one"` follows the latest manuscript. `baseline="zero"` explicitly chooses true global control. |
| Temporal independence | Some `no_ramp` notebooks redraw randomness between merge groups but share it within a group. | `temporal="independent"` draws a new assignment at every stage. |
| Repeated experiments | Repeat notebooks reuse an identical set of treated units. | `merge="repeated"` repeats the final propensity; `temporal` explicitly chooses shared or independent assignments. |
| Stage outcomes | Noise can be redrawn as a group is reconstructed. | Each stage is generated once per repetition and shared by merge comparisons. Group comparisons are paired. |
| Two-hop interference | Dense adjacency and matrix powers; some HT notebooks require a missing `A_2hop.pkl`. | Sparse matvecs and analytic diagonal removal; true GATE derives from the actual graph. |
| Dynamic evolution | Seed may reset at each stage; node insertion order affects candidate IDs. | One explicit graph-seed stream, fixed canonical IDs, recorded edge counts. |
| Repetition count | GNN notebooks allocate 1000 rows; the current manuscript reports 100 repetitions. | Paper presets use 100; the frozen run config records overrides. |
| Completion state | Preallocated zero rows can be summarized after an interrupted run. | A checkpoint exists only after the entire repetition completes; genuine numerical zeros are retained. |
| Summary | `table_output.py` requires manual result paths and a manually selected true GATE. | Summaries use each group's graph-specific truth and only validated complete records. |

These corrections preserve the intended formulas and architecture, not the exact historical random streams or numerical arrays. **Do not treat new outputs as bitwise reproductions of the published tables.** The old outputs have not been overwritten, certified or used as newly executed results.

## Graph partition discrepancy

The checked-in Stanford file validates to **11,586 nodes and 568,309 undirected edges**, matching the manuscript. Under canonical node order and the validation environment's NetworkX 3.6.1, Louvain with resolution 10 and seed 10 yields **197 communities**; the manuscript lists **192**. The implementation records the actual `clusters.json` partition, package versions and graph counts. Matching a seed and resolution alone does not recreate an unspecified historical node ordering and library implementation. Exact historical replication requires the original partition and environment.

## Scenario mapping

| Notebook family | Configuration |
|---|---|
| `CR_*_ramp_final` | `randomization=complete`, `temporal=staggered`, `merge=suffix` |
| `Ber_*_ramp_final` | `randomization=bernoulli`, `temporal=staggered`, `merge=suffix` |
| `*_no_ramp_final` | `temporal=independent` (with the correction above) |
| `CR_*_repeat_final` | `merge=repeated`; select temporal dependence explicitly |
| `2step_CR_*` | Use stages `[0.02,0.5]` and `merge=final_pair`; the pair is reported alongside final-only |
| `incre_2step_CR_*` | `merge=baseline_pair`, `outcome=linear` |
| `2hop*` / `incre_2hop*` | `outcome=two_hop`, suffix / baseline pairs |
| `Multiplicative*`, `sqrt*`, `square*` | `outcome=multiplicative`, `sqrt`, `quadratic` |
| `Dynamic*` / `Dynamic_incre*` | `outcome=dynamic`, suffix / baseline pairs |
| `*_unit` / `*_cluster` | `level=unit` / `cluster` |

`ht_unit_ramp.ipynb`, `ht_cluster_ramp.ipynb` and `tables.ipynb` remain legacy-only. This repository snapshot has no complete MII, Hájek or Lagrange implementation, and contains Cornell5 rather than the manuscript's supplementary MIT8 dataset. The refactor does not fabricate those missing experiments.

## Resume boundaries

Resume requires identical configuration, imported dependency versions, data filename/size/mtime, graph counts, node and cluster mapping, merge definitions, source Git revision and package-source fingerprint. Each checkpoint carries the frozen execution identity, so records from different outcome models cannot be silently combined even when their seeds, groups and true GATE happen to match. Repetitions use separate deterministic streams, so recomputing a missing repetition does not change any completed result. One aggregate fingerprint of the small Python package detects source edits even without a Git repository; datasets are not exhaustively hashed. This protects against accidental evidence mixing, not deliberate tampering or in-memory monkey-patching. Commit implementation changes before starting long studies.

A report rebuilt from corrupt records is prominently marked invalid and publishes no aggregate results. Report generation and execution share the run lock. Cached engineering validation additionally requires all expected repetitions and the current implementation/runtime/data identity; a stale `complete` flag is insufficient. New runs use checkpoint schema 2. Early development outputs without execution identities cannot be resumed; preserve their historical reports and rerun into a new directory.
