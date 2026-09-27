"""Deterministic experiment orchestration and atomic, validated checkpoints."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import networkx as nx
import numpy as np
import scipy

from . import __version__
from .config import ExperimentConfig
from .design import assignments, merge_groups, stage_proportions
from .estimators import StageData, fit_estimate
from .graphs import cluster_membership, demo_graph, evolve_graphs, load_graph
from .outcomes import sample_outcome, true_gate


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_text_atomic(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(value)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    write_text_atomic(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def _writer_lock(directory):
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError("the runner currently requires POSIX file locks (Linux or macOS)") from exc
    with (directory / ".lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another process is writing this run directory") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def runtime_versions(estimator):
    versions = {"rampup": __version__, "python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "networkx": nx.__version__}
    if estimator == "gnn":
        try:
            import torch
            import dgl
        except (ImportError, OSError, FileNotFoundError) as exc:
            raise RuntimeError("install compatible Torch/DGL versions from docs/experiments.md to use GNN") from exc
        versions.update(torch=torch.__version__, dgl=dgl.__version__)
    return versions


def _git_identity():
    root = Path(__file__).resolve().parent.parent
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip())
        return {"commit": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _implementation_identity():
    """One small package fingerprint per invocation, including uncommitted edits."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return "sha256:" + digest.hexdigest()


def _execution_identity(config, manifest):
    frozen = {key: manifest[key] for key in
              ("schema_version", "versions", "implementation", "source", "graph", "groups", "git")}
    frozen["config"] = config.to_dict()
    encoded = json.dumps(frozen, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _source_identity(config):
    if config.dataset == "demo":
        return {"name": "synthetic Watts-Strogatz", "nodes": 60, "degree": 6, "rewire": 0.2,
                "graph_seed": config.graph_seed}
    path = Path(config.dataset)
    stat = path.stat()
    return {"name": path.name, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _prepare(config):
    base = demo_graph(config.graph_seed) if config.dataset == "demo" else load_graph(config.dataset)
    membership = cluster_membership(base, config.resolution, config.cluster_seed) if config.level == "cluster" else None
    graphs = evolve_graphs(base, len(config.stages), config.graph_seed) if config.outcome == "dynamic" else [base] * len(config.stages)
    if config.merge == "baseline_pair":
        graphs = [base] + graphs
    return base, graphs, membership


def _validate_diagnostics(stages, config, graph):
    proportions = stage_proportions(config)
    if not isinstance(stages, list) or len(stages) != len(proportions):
        raise ValueError("stage diagnostics are missing or incomplete")
    n = graph["nodes"]
    entities = n if config.level == "unit" else graph["clusters"]
    for index, (stage, proportion) in enumerate(zip(stages, proportions)):
        units, treated = stage["treated_units"], stage["treated_entities"]
        if (type(units) is not int or type(treated) is not int or not 0 <= units <= n or
                not 0 <= treated <= entities or stage["entity_count"] != entities):
            raise ValueError("invalid treatment counts")
        if (stage["stage"] != index or stage["nominal_proportion"] != proportion or
                stage["edges"] != graph["stage_edges"][index] or stage["unit_proportion"] != units / n):
            raise ValueError("stage identity or realized proportion mismatch")
        variance = stage["exposure_variance"]
        if not np.isfinite(variance) or not 0 <= variance <= 0.25 + 1e-12:
            raise ValueError("invalid exposure variance")
        is_baseline = config.merge == "baseline_pair" and index == 0
        if is_baseline or config.randomization == "complete":
            expected = (1 if config.baseline == "one" else 0) if is_baseline else int(np.floor(proportion * entities))
            if treated != expected:
                raise ValueError("treatment count contradicts the assignment design")
        if config.level == "unit" and treated != units:
            raise ValueError("unit assignment counts disagree")


def load_records(directory):
    """Validate full repetition files; missing files stay missing, never zero-filled."""
    directory = Path(directory)
    config = ExperimentConfig.from_dict(read_json(directory / "config.json"))
    manifest = read_json(directory / "manifest.json")
    groups = manifest.get("groups", [])
    paths = sorted((directory / "replicates").glob("*.json"))
    if manifest.get("schema_version") != 2:
        raise ValueError("unsupported run schema; checkpoints require a frozen execution identity")
    if groups or paths:
        try:
            identity = _execution_identity(config, manifest)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("frozen execution identity is missing or invalid") from exc
        if manifest.get("execution_identity") != identity:
            raise ValueError("frozen execution identity does not match configuration or provenance")
    else:
        return []  # A preparation failure has no initialized groups or outcomes.
    expected = {item["label"]: item for item in groups}
    records = []
    for path in paths:
        try:
            payload = read_json(path)
            if not isinstance(payload, dict):
                raise ValueError("checkpoint must be a JSON object")
            if payload.get("execution_identity") != identity:
                raise ValueError("execution identity mismatch; checkpoint belongs to another run definition")
            replicate = payload["replicate"]
            if (type(replicate) is not int or not 0 <= replicate < config.repetitions or
                    path.name != f"{replicate:06d}.json" or payload["seed"] != [config.seed, replicate]):
                raise ValueError("repetition identity mismatch")
            _validate_diagnostics(payload["stages"], config, manifest["graph"])
            rows = payload["records"]
            if not isinstance(rows, list) or len(rows) != len(groups) or not groups:
                raise ValueError("incomplete merge groups")
            if [row["group"] for row in rows] != list(expected):
                raise ValueError("group order or identity mismatch")
            for row in rows:
                group = expected[row["group"]]
                if row["replicate"] != replicate or row["stage_indices"] != group["indices"]:
                    raise ValueError("row identity mismatch")
                if not np.isfinite([row["estimate"], row["truth"], row["train_loss"]]).all():
                    raise ValueError("nonfinite numeric result")
                if row["truth"] != group["truth"] or row["train_loss"] < 0:
                    raise ValueError("truth or training loss mismatch")
            records.extend(rows)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid checkpoint {path.name}: {exc}") from exc
    return records


def load_complete_records(directory):
    """Require the full validated repetition/group set before a completion gate."""
    directory = Path(directory)
    records = load_records(directory)
    config = ExperimentConfig.from_dict(read_json(directory / "config.json"))
    manifest = read_json(directory / "manifest.json")
    completed = {row["replicate"] for row in records}
    if (manifest["status"] != "complete" or completed != set(range(config.repetitions)) or
            manifest["completed_replicates"] != len(completed) or
            len(records) != config.repetitions * len(manifest.get("groups", []))):
        raise ValueError("incomplete run: validated records do not satisfy the completion gate")
    return records


def _run_repetition(config, replicate, graphs, membership, groups, execution_identity):
    seed = np.random.SeedSequence([config.seed, replicate])
    assignment_seed, noise_seed = seed.spawn(2)
    z = assignments(config, graphs[0].n_nodes, membership, np.random.default_rng(assignment_seed))
    noise_streams = noise_seed.spawn(len(graphs))
    stages = [StageData(graph, treatment, sample_outcome(graph, treatment, config, np.random.default_rng(noise)))
              for graph, treatment, noise in zip(graphs, z, noise_streams)]
    diagnostics = []
    for index, stage in enumerate(stages):
        entities = stage.treatment if membership is None else stage.treatment[np.unique(membership, return_index=True)[1]]
        diagnostics.append({"stage": index, "nominal_proportion": stage_proportions(config)[index],
                            "treated_units": int(stage.treatment.sum()), "unit_proportion": float(stage.treatment.mean()),
                            "treated_entities": int(entities.sum()), "entity_count": len(entities),
                            "exposure_variance": float(np.var(stage.graph.walk @ stage.treatment)),
                            "edges": stage.graph.n_edges})
    rows = []
    for group in groups:
        selected = [stages[index] for index in group["indices"]]
        estimate, loss = fit_estimate(selected, selected[-1].graph, config)
        rows.append({"replicate": replicate, "group": group["label"], "stage_indices": group["indices"],
                     "estimate": estimate, "truth": group["truth"], "train_loss": loss})
    return {"replicate": replicate, "seed": [config.seed, replicate], "execution_identity": execution_identity,
            "stages": diagnostics, "records": rows}


def run_experiment(config, output_dir, resume=False):
    """Run serially; a failed repetition is recomputed from its own seed on resume."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with _writer_lock(output):
        existing = [p for p in output.iterdir() if p.name != ".lock"]
        if not resume and existing:
            raise FileExistsError("output directory is not empty; use --resume for an unchanged run")
        if resume:
            if not (output / "config.json").exists() or not (output / "manifest.json").exists():
                raise ValueError("resume requires an existing configuration and manifest")
            if read_json(output / "config.json") != config.to_dict():
                raise ValueError("resume configuration differs from the frozen configuration")
            manifest = read_json(output / "manifest.json")
            if manifest.get("schema_version") != 2:
                raise ValueError("unsupported run schema")
        else:
            write_json(output / "config.json", config.to_dict())
            manifest = {"schema_version": 2, "run_id": output.name, "created_at": utc_now(),
                        "status": "running", "attempts": [], "completed_replicates": 0,
                        "expected_replicates": config.repetitions, "git": _git_identity()}
        (output / "replicates").mkdir(exist_ok=True)
        started = time.monotonic()
        attempt = {"started_at": utc_now(), "resume": resume, "status": "running", "completed_this_attempt": 0,
                   "git": _git_identity()}
        manifest["attempts"].append(attempt)
        manifest["status"] = "running"
        write_json(output / "manifest.json", manifest)
        try:
            versions = runtime_versions(config.estimator)
            implementation = _implementation_identity()
            source = _source_identity(config)
            if resume:
                if "versions" in manifest and versions != manifest["versions"]:
                    raise ValueError("resume runtime versions differ from the recorded environment")
                if "source" in manifest and source != manifest["source"]:
                    raise ValueError("resume data identity changed (name, size or modification time)")
                if "implementation" in manifest and implementation != manifest["implementation"]:
                    raise ValueError("resume implementation changed; start a new run for the modified package")
                if manifest["git"]["commit"] != attempt["git"]["commit"]:
                    raise ValueError("resume Git revision changed; start a new run to keep provenance unambiguous")
            base, graphs, membership = _prepare(config)
            graph_info = {"nodes": base.n_nodes, "edges": base.n_edges,
                          "clusters": int(len(np.unique(membership))) if membership is not None else None,
                          "minimum_degree": float(base.degrees.min()), "maximum_degree": float(base.degrees.max()),
                          "stage_edges": [g.n_edges for g in graphs], "node_order": "sorted original IDs"}
            groups = [{"label": label, "indices": list(indices), "truth": true_gate(graphs[indices[-1]], config)}
                      for label, indices in merge_groups(config)]
            if resume and "graph" in manifest:
                if graph_info != manifest["graph"] or groups != manifest["groups"]:
                    raise ValueError("resume graph or experiment groups changed")
                if read_json(output / "node_ids.json") != base.node_ids.tolist():
                    raise ValueError("resume node mapping changed")
                if membership is not None and read_json(output / "clusters.json") != membership.tolist():
                    raise ValueError("resume clustering changed")
            else:
                write_json(output / "node_ids.json", base.node_ids.tolist())
                if membership is not None:
                    write_json(output / "clusters.json", membership.tolist())
            frozen = dict(manifest, versions=versions, implementation=implementation,
                          source=source, graph=graph_info, groups=groups)
            identity = _execution_identity(config, frozen)
            if resume and "execution_identity" in manifest and identity != manifest["execution_identity"]:
                raise ValueError("resume frozen execution identity changed")
            manifest.update(versions=versions, implementation=implementation, execution_identity=identity,
                            source=source, graph=graph_info, groups=groups,
                            seed_schedule="SeedSequence([seed, replicate]).spawn(assignment, noise); noise.spawn(stage)")
            write_json(output / "manifest.json", manifest)
            completed = {row["replicate"] for row in load_records(output)}
            manifest["completed_replicates"] = len(completed)
            for replicate in range(config.repetitions):
                if replicate in completed:
                    continue
                payload = _run_repetition(config, replicate, graphs, membership, groups, identity)
                write_json(output / "replicates" / f"{replicate:06d}.json", payload)
                completed.add(replicate)
                attempt["completed_this_attempt"] += 1
                manifest["completed_replicates"] = len(completed)
                write_json(output / "manifest.json", manifest)
            load_records(output)
            manifest["status"] = attempt["status"] = "complete"
        except BaseException as exc:
            manifest["status"] = attempt["status"] = "failed"
            attempt["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            attempt["finished_at"] = utc_now()
            attempt["elapsed_seconds"] = round(time.monotonic() - started, 6)
            write_json(output / "manifest.json", manifest)
            # Also report failed runs; an invalid checkpoint is displayed as invalid, not omitted.
            from .report import _render_report_locked
            _render_report_locked(output)
    return output
