# Stanford Louvain partition reproduction

The 2026-10-22 audit fixes seed 10 and resolution 10 throughout. No parameter search or full GNN experiment is needed to explain the discrepancy.

## Confirmed cause

The archived notebooks use `pd.read_table(..., skiprows=1)` followed by `nx.from_pandas_edgelist`. Nodes enter the graph on first occurrence in the file: `23, 1, 63, 193, ...`. The maintained sparse construction inserts all row indices in canonical order, corresponding to original IDs `1, 2, 3, ...`.

Louvain shuffles `list(G.nodes)` with its seed, so equal seeds do not remove differences in the input order. Its internal graph copying and greedy traversal also depend on iteration order. The audit isolates the input cause, not the separate contributions of each internal propagation mechanism.

| NetworkX 3.6.1 input | Communities | Full partition |
|---|---:|---|
| Archived edge insertion | 192 | A |
| Maintained sparse construction | 197 | B |
| Original edge stream, but sorted nodes pre-inserted | 197 | Exactly B |
| Archived construction, node labels changed to zero-based | 192 | Exactly A after ID mapping |
| Maintained construction, labels changed back to one-based | 197 | Exactly B after ID mapping |

The complete edge sets and each input node's neighbor order agree between the original and maintained graphs. The sorted-node intervention alone is sufficient to produce the exact maintained partition. Removing explicit unit weights, disturbing global Python/NumPy RNGs, and changing the process hash seed did not alter the respective partitions. NetworkX 2.8.8, 3.2.1, 3.3 and 3.6.1 all gave the same A/B memberships under Python 3.12.14. All 37 archived notebooks and the Stanford dataset match the 2024 initial commit. Notebook metadata records Python 3.10.13, but does not identify the historical NetworkX version.

## Explicit compatibility option

The default remains `canonical`. Set `"clustering": "legacy_edgelist"` in a cluster configuration, or pass `--clustering legacy_edgelist` to `run`. This rebuilds the NetworkX graph from the original edge stream, checks its full adjacency against the canonical graph, performs Louvain and preserves stable descending-size community numbering. The resulting membership vector is still aligned to sorted original node IDs, so adjacency, features and treatment remain aligned.

```bash
python -m rampup inspect Dataset/socfb-Stanford3.mtx --clusters --clustering legacy_edgelist
python scripts/validate_louvain.py --output outputs/louvain-validation
python -m rampup run --config configs/stanford-legacy-cluster-ols.json --output outputs/legacy-cluster-ols
```

The first command partitions only. The second makes two partition calls and checks all memberships using label-invariant SHA-256 expectations. The third is an explicitly labeled 3-repetition OLS smoke diagnostic. It is not a paper-sized GNN run. The committed [validation summary](../reports/louvain-validation.json) records the tested implementation hashes and runtime.

The legacy input option accepts only the repository format (`n n edge_rows`, then one-based endpoint pairs). It rejects standard Matrix Market and synthetic `demo` inputs rather than inventing a historical edge order. It does not add pandas to the maintained dependencies.

## Frozen identities and remaining scientific differences

New cluster runs save the clustering mode and a hash of `{"node_ids": [...], "membership": [...]}` serialized with JSON separators `(',', ':')`. This hash preserves community numbering. Legacy mode additionally records the data-file SHA-256. Both become part of checkpoint execution identity. Old canonical configurations retain their original serialized shape so historical reports remain readable; resuming across code revisions remains prohibited by the existing provenance rules.

The validation script uses a different, label-invariant partition hash: sort members by original ID, sort communities by `(-size, min_id)`, serialize that list using the same compact separators and hash the UTF-8 bytes. Expected hashes:

| Object | SHA-256 |
|---|---|
| Stanford file | `274f00a9938219096a017d6050fce585193fcef2dc64b7ae9efff89900b52e7b` |
| A: reconstructed 192 partition | `0ec5f5f56cee5c71257db7e639da6a3db26409d84feb055f941379944d71a9c2` |
| B: maintained 197 partition | `68b68379015a0ccbec767b8b425f97b83ce91193f6d470dbc47289117c767b17` |

The partitions' adjusted Rand index is 0.842927. Even after optimal one-to-one community matching, 2,296 of 11,586 nodes cannot be matched (19.82%). This establishes a change in the randomization design, not the direction or magnitude of any estimator bias or MSE change.

Let `p` be treatment proportion and `K` the number of communities. Maintained complete randomization selects `floor(p*K)`. The archived CR notebooks use scores below a linear-interpolated quantile:

| p | Maintained rule, K=197 | Maintained rule, K=192 | Archived rule, K=192 |
|---:|---:|---:|---:|
| 0.02 | 3 | 3 | 4 |
| 0.05 | 9 | 9 | 10 |
| 0.10 | 19 | 19 | 20 |
| 0.25 | 49 | 48 | 48 |
| 0.50 | 98 | 96 | 96 |

The option intentionally keeps the maintained assignment rule and RNG streams. It also preserves the historical stable size-only community numbering; canonical tie-breaking would relabel 79 of the recovered communities. Relabeling preserves the partition and exchangeable assignment distribution, but can change realized assignments at a fixed seed. The two HT notebooks use resolution 5, unlike the 34 GNN notebooks using 10, and remain a separate historical setting.

Do not replace an existing `clusters.json` or resume a run under another clustering mode. Use a new output directory. No old result array or report is rewritten by this change. No exact historical membership artifact tied to the paper run was found, so A is an **archived-algorithm reconstruction matching the paper's count**, not certification of the full numerical tables. The minimal remaining historical check is comparison to an original node-to-community export, or running just graph loading and Louvain in the original environment and exporting its partition and versions.
