"""One command surface for configuration, execution, inspection and reporting."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from .config import ExperimentConfig


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rampup", description="Reproducible ramp-up experiments under network interference")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a JSON configuration")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--resume", action="store_true")
    for name, kind in (("repetitions", int), ("epochs", int), ("seed", int), ("threads", int)):
        run.add_argument(f"--{name}", type=kind)
    run.add_argument("--dataset")
    run.add_argument("--estimator", choices=["ols", "gnn"])
    run.add_argument("--clustering", choices=["canonical", "legacy_edgelist"])
    demo = sub.add_parser("demo", help="small synthetic smoke run; defaults to OLS")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    demo.add_argument("--estimator", choices=["ols", "gnn"], default="ols")
    demo.add_argument("--repetitions", type=int, default=5)
    demo.add_argument("--epochs", type=int, default=8)
    demo.add_argument("--resume", action="store_true")
    report = sub.add_parser("report", help="rebuild an offline HTML report from recorded evidence")
    report.add_argument("directory", type=Path)
    inspect = sub.add_parser("inspect", help="validate graph and display counts")
    inspect.add_argument("dataset", type=Path)
    inspect.add_argument("--clusters", action="store_true")
    inspect.add_argument("--clustering", choices=["canonical", "legacy_edgelist"], default="canonical")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            if args.clustering != "canonical" and not args.clusters:
                raise ValueError("--clustering legacy_edgelist requires --clusters")
            from .graphs import load_graph, cluster_membership
            graph = load_graph(args.dataset)
            info = {"dataset": args.dataset.name, "nodes": graph.n_nodes, "edges": graph.n_edges,
                    "minimum_degree": float(graph.degrees.min()), "maximum_degree": float(graph.degrees.max())}
            if args.clusters:
                import numpy as np
                legacy_edgelist = args.dataset if args.clustering == "legacy_edgelist" else None
                info["clusters"] = len(np.unique(cluster_membership(graph, legacy_edgelist=legacy_edgelist)))
                info["clustering"] = args.clustering
            print(json.dumps(info, indent=2))
        elif args.command == "report":
            from .report import render_report
            print(render_report(args.directory).resolve())
        else:
            from .runner import run_experiment
            if args.command == "demo":
                config = ExperimentConfig(dataset="demo", estimator=args.estimator, repetitions=args.repetitions,
                                          epochs=args.epochs, stages=(0.1, 0.25, 0.5), purpose="smoke")
            else:
                config = ExperimentConfig.from_dict(json.loads(args.config.read_text()))
                overrides = {key: getattr(args, key) for key in ("repetitions", "epochs", "seed", "threads", "dataset", "estimator", "clustering")
                             if getattr(args, key) is not None}
                config = replace(config, **overrides)
            print(f"Running {config.repetitions} repetitions · {config.estimator} · {config.outcome}", flush=True)
            directory = run_experiment(config, args.output, resume=args.resume)
            print(f"Report: {(directory / 'report.html').resolve()}")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"rampup: {exc}", file=sys.stderr)
        return 1
    return 0
