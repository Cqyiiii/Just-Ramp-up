import numpy as np
import networkx as nx
import pytest

from rampup.config import ExperimentConfig
from rampup.graphs import from_networkx, load_graph, evolve_graphs
from rampup.design import assignments, merge_groups
from rampup.outcomes import mean_outcome, sample_outcome, true_gate


def test_configuration_rejects_unknown_fields_and_invalid_values():
    for values in ({"epoch": 4}, {"stages": [0.5, 0.2]}, {"sigma": float("nan")},
                   {"repetitions": 0}, {"epochs": True}, {"level": "groups"},
                   {"seed": -1}, {"learning_rate": 0}):
        with pytest.raises(ValueError):
            ExperimentConfig.from_dict(values)


def test_config_round_trip():
    cfg = ExperimentConfig.from_dict({"stages": [0.2, 0.5], "merge": "baseline_pair"})
    assert ExperimentConfig.from_dict(cfg.to_dict()) == cfg


def test_graph_aligns_sparse_adjacency_to_sorted_original_ids():
    graph = from_networkx(nx.Graph([(30, 10), (10, 20)]))
    np.testing.assert_array_equal(graph.node_ids, [10, 20, 30])
    np.testing.assert_array_equal(graph.degrees, [2, 1, 1])
    np.testing.assert_array_equal(graph.walk @ [0, 1, 0], [0.5, 0, 0])


def test_graph_loader_reads_repository_header_and_validates_it(tmp_path):
    path = tmp_path / "network.mtx"
    path.write_text("3 3 2\n3 1\n1 2\n")
    graph = load_graph(path)
    assert (graph.n_nodes, graph.n_edges) == (3, 2)
    for invalid in ("4 4 2\n3 1\n1 2\n", "3 3 3\n3 1\n1 2\n",
                    "3 3 2\n4 1\n1 2\n", "3 3 2\n1 1\n1 2\n"):
        path.write_text(invalid)
        with pytest.raises(ValueError):
            load_graph(path)


def test_complete_rollout_has_exact_counts_and_nested_assignments():
    cfg = ExperimentConfig(stages=(0.2, 0.5, 1.0))
    z = assignments(cfg, 10, None, np.random.default_rng(4))
    np.testing.assert_array_equal(z.sum(axis=1), [2, 5, 10])
    assert np.all(np.diff(z, axis=0) >= 0)


def test_cluster_randomization_keeps_clusters_intact_and_counts_entities():
    cfg = ExperimentConfig(level="cluster", stages=(0.5, 1.0))
    membership = np.array([0, 0, 0, 1, 1, 2, 3])
    z = assignments(cfg, 7, membership, np.random.default_rng(1))
    assert z[0, [0, 3, 5, 6]].sum() == 2
    np.testing.assert_array_equal(z[:, 0], z[:, 1])
    np.testing.assert_array_equal(z[:, 3], z[:, 4])


def test_zero_and_one_entity_baselines_are_different_and_nested():
    for baseline, count in (("zero", 0), ("one", 1)):
        cfg = ExperimentConfig(stages=(0.5,), merge="baseline_pair", baseline=baseline)
        z = assignments(cfg, 8, None, np.random.default_rng(1))
        assert z[0].sum() == count
        assert np.all(z[0] <= z[1])


def test_independent_repeated_assignments_are_not_fixed_rollout():
    cfg = ExperimentConfig(stages=(0.2, 0.5), merge="repeated", temporal="independent")
    z = assignments(cfg, 100, None, np.random.default_rng(8))
    np.testing.assert_array_equal(z.sum(axis=1), [50, 50])
    assert not np.array_equal(z[0], z[1])


def test_one_entity_baseline_cannot_break_staggered_assignment():
    cfg = ExperimentConfig(stages=(0.02, 0.5), merge="baseline_pair")
    with pytest.raises(ValueError, match="baseline"):
        assignments(cfg, 10, None, np.random.default_rng(1))
    with pytest.raises(ValueError, match="baseline"):
        ExperimentConfig(merge="baseline_pair", randomization="bernoulli")


def test_bernoulli_rollout_is_nested_and_uses_threshold_probabilities():
    cfg = ExperimentConfig(randomization="bernoulli", stages=(0.2, 0.8))
    z = assignments(cfg, 10000, None, np.random.default_rng(20))
    assert np.all(z[0] <= z[1])
    np.testing.assert_allclose(z.mean(axis=1), [0.2, 0.8], atol=0.02)


def test_merge_groups_are_explicit_and_ordered_by_number_of_rounds():
    cfg = ExperimentConfig(stages=(0.1, 0.2, 0.5))
    assert [indices for _, indices in merge_groups(cfg)] == [(2,), (1, 2), (0, 1, 2)]
    cfg = ExperimentConfig(stages=(0.1, 0.5), merge="baseline_pair")
    assert [indices for _, indices in merge_groups(cfg)] == [(0, 1), (0, 2)]


@pytest.mark.parametrize("outcome,want", [
    ("linear", [2, 2, 1]), ("quadratic", [2, 1.5, 1]),
    ("sqrt", [2, 1 + np.sqrt(2), 1]),
    ("multiplicative", [1.5, 2.25, 0.75]), ("two_hop", [2, 1.5, 1.25]),
])
def test_outcomes_on_hand_calculated_path(outcome, want):
    graph = from_networkx(nx.path_graph(3))
    cfg = ExperimentConfig(outcome=outcome, gamma=2 if outcome in ("linear", "quadratic", "sqrt") else 1)
    np.testing.assert_allclose(mean_outcome(graph, np.array([1, 0, 0]), cfg), want)


def test_two_hop_gate_removes_closed_walks():
    graph = from_networkx(nx.path_graph(3))
    # diag(W^2) = [1/2,1,1/2]; mean second-hop spillover = 1/6.
    assert true_gate(graph, ExperimentConfig(outcome="two_hop")) == pytest.approx(13 / 6)


def test_noiseless_outcome_and_nonmutating_sampling():
    graph = from_networkx(nx.cycle_graph(4))
    z = np.array([0., 1., 0., 1.])
    cfg = ExperimentConfig(sigma=0)
    np.testing.assert_allclose(sample_outcome(graph, z, cfg, np.random.default_rng(1)), [2, 2, 2, 2])
    np.testing.assert_array_equal(z, [0, 1, 0, 1])


def test_dynamic_evolution_preserves_ids_and_is_deterministic():
    graph = from_networkx(nx.cycle_graph(12))
    first = evolve_graphs(graph, 2, seed=3)
    second = evolve_graphs(graph, 2, seed=3)
    assert first[0].n_edges >= graph.n_edges
    assert first[1].n_edges >= first[0].n_edges
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a.node_ids, graph.node_ids)
        assert (a.adjacency != b.adjacency).nnz == 0
        assert not a.adjacency.diagonal().any()
