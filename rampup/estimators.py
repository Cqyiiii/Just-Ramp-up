"""OLS and the paper's lightweight ChebConv regressor."""

from dataclasses import dataclass
import numpy as np

from .graphs import GraphData


@dataclass(frozen=True)
class StageData:
    graph: GraphData
    treatment: np.ndarray
    outcome: np.ndarray


def _validate(stages, evaluation_graph):
    if not stages:
        raise ValueError("at least one training stage is required")
    for stage in stages:
        z, y = np.asarray(stage.treatment), np.asarray(stage.outcome)
        if z.shape != (stage.graph.n_nodes,) or y.shape != z.shape:
            raise ValueError("stage treatment/outcome dimensions must match its graph")
        if not np.all((z == 0) | (z == 1)) or not np.isfinite(y).all():
            raise ValueError("stage treatment must be binary and outcomes finite")
        if not np.array_equal(stage.graph.node_ids, evaluation_graph.node_ids):
            raise ValueError("all stages must use the same canonical node IDs")


def fit_estimate(stages, evaluation_graph, config):
    """Return counterfactual mean difference and pooled training MSE."""
    _validate(stages, evaluation_graph)
    if config.estimator == "ols":
        z = np.concatenate([stage.treatment for stage in stages])
        y = np.concatenate([stage.outcome for stage in stages])
        if np.unique(z).size < 2:
            raise ValueError("OLS needs treatment variation across the selected stages")
        # The graph-agnostic intercept + binary treatment model has this exact solution.
        intercept = y[z == 0].mean()
        estimate = float(y[z == 1].mean() - intercept)
        loss = float(np.mean((y - intercept - estimate * z) ** 2))
    else:
        estimate, loss = _fit_gnn(stages, evaluation_graph, config)
    if not np.isfinite([estimate, loss]).all():
        raise ValueError("estimator produced nonfinite predictions or loss")
    return estimate, loss


def _fit_gnn(stages, evaluation_graph, config):
    try:
        import torch
        import dgl
        from dgl.nn import ChebConv
    except (ImportError, OSError, FileNotFoundError) as exc:
        raise RuntimeError("GNN requires compatible Torch/DGL installations; see docs/experiments.md. OLS and demo do not require them.") from exc

    torch.set_num_threads(config.threads)
    graphs = {}

    def tensor_graph(graph):
        key = id(graph)
        if key not in graphs:
            coo = graph.adjacency.tocoo()
            graphs[key] = dgl.graph((torch.from_numpy(coo.row.astype(np.int64)),
                                     torch.from_numpy(coo.col.astype(np.int64))), num_nodes=graph.n_nodes)
        return graphs[key]

    class Regressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([
                ChebConv(2, 16, 2), ChebConv(16, 16, 1), ChebConv(16, 1, 1)])

        def forward(self, graph, features):
            for layer in self.layers:
                features = layer(graph, features, lambda_max=[2.0])
            return features.squeeze(-1)

    tensors = [(tensor_graph(stage.graph),
                torch.tensor(np.column_stack((stage.treatment, stage.graph.degrees)), dtype=torch.float32),
                torch.tensor(stage.outcome, dtype=torch.float32)) for stage in stages]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.model_seed)
        model = Regressor()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
        model.train()
        for _ in range(config.epochs):
            for graph, features, target in tensors:
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(model(graph, features), target)
                if not torch.isfinite(loss):
                    raise ValueError("GNN training loss became nonfinite")
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = float(np.mean([torch.nn.functional.mse_loss(model(g, x), y).item()
                                        for g, x, y in tensors]))
            graph = tensor_graph(evaluation_graph)
            features = torch.tensor(np.column_stack((np.zeros(evaluation_graph.n_nodes),
                                                       evaluation_graph.degrees)), dtype=torch.float32)
            control = model(graph, features).mean().item()
            features[:, 0] = 1
            treatment = model(graph, features).mean().item()
        return treatment - control, train_loss
