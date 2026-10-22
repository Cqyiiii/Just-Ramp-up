"""Bounded Stanford partition regression: two Louvain calls, no model training."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
import scipy

from rampup.graphs import cluster_membership, load_graph


def partition_hash(graph, membership):
    communities = [sorted(int(n) for n in graph.node_ids[membership == c]) for c in np.unique(membership)]
    communities.sort(key=lambda nodes: (-len(nodes), min(nodes)))
    return hashlib.sha256(json.dumps(communities, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/louvain-validation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = ROOT / "Dataset/socfb-Stanford3.mtx"
    data_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if data_hash != "274f00a9938219096a017d6050fce585193fcef2dc64b7ae9efff89900b52e7b":
        raise ValueError("Stanford data bytes changed; review provenance before updating expectations")
    graph = load_graph(dataset)
    if (graph.n_nodes, graph.n_edges) != (11586, 568309):
        raise ValueError("Stanford graph counts changed")
    expected = {
        "canonical": (197, "68b68379015a0ccbec767b8b425f97b83ce91193f6d470dbc47289117c767b17"),
        "legacy_edgelist": (192, "0ec5f5f56cee5c71257db7e639da6a3db26409d84feb055f941379944d71a9c2"),
    }
    report = {
        "scope": "partition regression only; not numerical reproduction of paper tables",
        "versions": {"python": platform.python_version(), "networkx": nx.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__},
        "dataset_sha256": data_hash, "nodes": graph.n_nodes, "edges": graph.n_edges,
        "seed": 10, "resolution": 10, "gnn_training_runs": 0,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [ROOT / "rampup/graphs.py", Path(__file__)]},
        "cases": [],
    }
    for mode, (count, expected_hash) in expected.items():
        start = time.monotonic()
        membership = cluster_membership(graph, resolution=10, seed=10,
                                        legacy_edgelist=dataset if mode == "legacy_edgelist" else None)
        actual_hash = partition_hash(graph, membership)
        if len(np.unique(membership)) != count or actual_hash != expected_hash:
            raise ValueError(f"{mode} partition changed; do not tune parameters to match the expected count")
        ordered = {"node_ids": graph.node_ids.tolist(), "membership": membership.tolist()}
        (args.output / f"{mode}.json").write_text(json.dumps(ordered, separators=(",", ":")) + "\n")
        report["cases"].append({"clustering": mode, "communities": count,
                                "label_invariant_partition_sha256": actual_hash,
                                "ordered_membership_sha256": hashlib.sha256(json.dumps(ordered, separators=(",", ":")).encode()).hexdigest(),
                                "seconds": round(time.monotonic() - start, 3)})
    report["status"] = "passed"
    text = json.dumps(report, indent=2) + "\n"
    (args.output / "summary.json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
