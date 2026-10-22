"""Order-sensitive regressions without training or a full Stanford run."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import networkx as nx
import numpy as np
import pytest

from rampup.config import ExperimentConfig
from rampup.graphs import cluster_membership, load_graph
from rampup.runner import load_records, run_experiment


@pytest.fixture
def edge_file(tmp_path):
    # Reversing this fixed edge stream changes insertion order, not topology.
    edges = [(u + 1, v + 1) for u, v in reversed(list(nx.karate_club_graph().edges()))]
    path = tmp_path / "karate.mtx"
    path.write_text(f"34 34 {len(edges)}\n" + "".join(f"{u} {v}\n" for u, v in edges))
    return path


def test_legacy_partition_preserves_edge_insertion_and_cluster_label_order(edge_file):
    graph = load_graph(edge_file)
    membership = cluster_membership(graph, resolution=1, seed=10, legacy_edgelist=edge_file)
    raw = nx.Graph()
    raw.add_edges_from(tuple(map(int, line.split())) for line in edge_file.read_text().splitlines()[1:])
    communities = sorted(nx.community.louvain_communities(raw, resolution=1, seed=10), key=len, reverse=True)
    expected_sets = [
        [1, 5, 6, 7, 11, 12, 13, 17, 18, 20, 22], [2, 3, 4, 8, 10, 14],
        [9, 15, 16, 19, 21, 23, 27, 30, 31, 33, 34], [24, 25, 26, 28, 29, 32],
    ]
    assert sorted(sorted(c) for c in communities) == expected_sets
    expected = {node: i for i, c in enumerate(communities) for node in c}
    assert membership.tolist() == [expected[node] for node in graph.node_ids]
    assert not np.array_equal(membership, cluster_membership(graph, resolution=1, seed=10))
    np.testing.assert_array_equal(graph.node_ids, np.arange(1, 35))


def test_legacy_source_cannot_supply_a_different_topology(edge_file):
    graph = load_graph(edge_file)
    text = edge_file.read_text().splitlines()
    text[1] = "1 34"  # Same node and edge counts, different edge set.
    edge_file.write_text("\n".join(text) + "\n")
    with pytest.raises(ValueError, match="match"):
        cluster_membership(graph, legacy_edgelist=edge_file)


def test_legacy_mode_rejects_matrix_market_instead_of_guessing_order(tmp_path):
    path = tmp_path / "standard.mtx"
    path.write_text("%%MatrixMarket matrix coordinate real symmetric\n3 3 2\n2 1 1\n3 2 1\n")
    graph = load_graph(path)
    with pytest.raises(ValueError, match="edge-list"):
        cluster_membership(graph, legacy_edgelist=path)


def test_new_config_is_explicit_without_rewriting_old_canonical_configs():
    old = ExperimentConfig().to_dict()
    assert "clustering" not in old  # Preserve old report identity serialization.
    cfg = ExperimentConfig(level="cluster", clustering="legacy_edgelist")
    assert cfg.to_dict()["clustering"] == "legacy_edgelist"
    assert ExperimentConfig.from_dict(cfg.to_dict()) == cfg
    assert ExperimentConfig.from_dict(old).to_dict() == old
    for options in [dict(clustering="unknown"), dict(clustering="legacy_edgelist"),
                    dict(clustering="legacy_edgelist", level="cluster", dataset="demo")]:
        with pytest.raises(ValueError, match="clustering|legacy_edgelist"):
            ExperimentConfig(**options)


def test_legacy_run_freezes_partition_and_cannot_resume_as_canonical(tmp_path, edge_file):
    cfg = ExperimentConfig(dataset=str(edge_file), level="cluster", clustering="legacy_edgelist",
                           resolution=1, estimator="ols", repetitions=1, stages=(0.5,), purpose="smoke")
    output = tmp_path / "run"
    run_experiment(cfg, output)
    manifest = json.loads((output / "manifest.json").read_text())
    ids = json.loads((output / "node_ids.json").read_text())
    labels = json.loads((output / "clusters.json").read_text())
    encoded = json.dumps({"node_ids": ids, "membership": labels}, separators=(",", ":")).encode()
    assert manifest["graph"]["clustering"] == "legacy_edgelist"
    assert manifest["graph"]["partition_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert manifest["source"]["sha256"] == hashlib.sha256(edge_file.read_bytes()).hexdigest()
    before = (output / "manifest.json").read_bytes()
    records = load_records(output)
    with pytest.raises(ValueError, match="configuration"):
        run_experiment(replace(cfg, clustering="canonical"), output, resume=True)
    assert (output / "manifest.json").read_bytes() == before
    run_experiment(cfg, output, resume=True)
    assert load_records(output) == records


def test_cli_inspects_legacy_partition_without_training(edge_file):
    command = [sys.executable, "-m", "rampup", "inspect", str(edge_file), "--clusters",
               "--clustering", "legacy_edgelist"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    info = json.loads(result.stdout)
    assert info["clustering"] == "legacy_edgelist"
    assert info["nodes"] == 34 and info["clusters"] > 0


def test_existing_report_configuration_identity_remains_readable(tmp_path):
    # New optional config keys must not change the serialized default used by
    # historical execution identities. This is the pre-feature default schema.
    cfg = ExperimentConfig(dataset="demo", estimator="ols", repetitions=1,
                           stages=(0.5,), purpose="smoke")
    run_experiment(cfg, tmp_path / "old-style")
    config = json.loads((tmp_path / "old-style/config.json").read_text())
    assert "clustering" not in config
    assert len(load_records(tmp_path / "old-style")) == 1
