import numpy as np
import networkx as nx
import pytest

from rampup.config import ExperimentConfig
from rampup.graphs import from_networkx
from rampup.estimators import StageData, fit_estimate
from rampup.metrics import summarize


def test_ols_pools_all_observations_and_returns_gate():
    graph = from_networkx(nx.cycle_graph(4))
    first = StageData(graph, np.array([0, 0, 0, 1]), np.array([1, 1, 1, 4]))
    second = StageData(graph, np.array([0, 1, 1, 1]), np.array([2, 5, 5, 5]))
    estimate, loss = fit_estimate([first, second], graph, ExperimentConfig(estimator="ols"))
    # E[Y|z=1]=4.75 and E[Y|z=0]=1.25, over the pooled observations.
    assert estimate == pytest.approx(3.5)
    assert loss == pytest.approx(0.1875)


def test_ols_refuses_no_treatment_variation_and_misaligned_inputs():
    graph = from_networkx(nx.cycle_graph(4))
    with pytest.raises(ValueError, match="variation"):
        fit_estimate([StageData(graph, np.zeros(4), np.ones(4))], graph, ExperimentConfig(estimator="ols"))
    with pytest.raises(ValueError):
        fit_estimate([StageData(graph, np.ones(3), np.ones(4))], graph, ExperimentConfig(estimator="ols"))
    with pytest.raises(ValueError):
        fit_estimate([StageData(graph, np.array([0, 1, 0, 1]), np.array([1, np.nan, 2, 3]))], graph, ExperimentConfig(estimator="ols"))


def test_metrics_keep_real_zeros_and_use_per_record_truth():
    rows = [{"replicate": 0, "group": "test", "estimate": 0., "truth": 1.},
            {"replicate": 1, "group": "test", "estimate": 2., "truth": 1.}]
    stats = summarize(rows)[0]
    assert stats["n"] == 2
    assert stats["bias"] == 0
    assert stats["std"] == 1
    assert stats["mse"] == 1
    assert stats["bias_mcse"] == 1
    assert stats["bias_ci95"][0] < 0 < stats["bias_ci95"][1]


def test_metrics_reject_nonfinite_or_duplicate_records():
    base = {"replicate": 0, "group": "a", "estimate": 2., "truth": 2.}
    with pytest.raises(ValueError):
        summarize([base, base])
    with pytest.raises(ValueError):
        summarize([{**base, "estimate": float("nan")}])
    assert summarize([base])[0]["bias_ci95"] is None


@pytest.mark.gnn
def test_real_gnn_is_deterministic_and_uses_each_stage_graph(monkeypatch):
    pytest.importorskip("torch")
    dgl = pytest.importorskip("dgl")
    sparse_graph = from_networkx(nx.path_graph(4))
    dense_graph = from_networkx(nx.complete_graph(4))
    stages = [StageData(sparse_graph, np.array([0, 0, 0, 1]), np.array([1., 1., 2., 2.])),
              StageData(dense_graph, np.array([0, 1, 1, 1]), np.array([2., 3., 3., 3.]))]
    cfg = ExperimentConfig(epochs=2)
    seen = []
    original = dgl.nn.ChebConv.forward

    def observed_forward(self, graph, *args, **kwargs):
        seen.append(graph.num_edges())
        return original(self, graph, *args, **kwargs)

    monkeypatch.setattr(dgl.nn.ChebConv, "forward", observed_forward)
    first = fit_estimate(stages, dense_graph, cfg)
    assert seen[:6] == [6, 6, 6, 12, 12, 12]
    assert np.isfinite(first).all()
    second = fit_estimate(stages, dense_graph, cfg)
    np.testing.assert_allclose(first, second, rtol=0, atol=0)
    np.testing.assert_array_equal(stages[0].treatment, [0, 0, 0, 1])
