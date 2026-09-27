from dataclasses import replace
import json
import subprocess
import sys

import pytest

from rampup.config import ExperimentConfig
from rampup.runner import run_experiment, load_records
from rampup.report import render_report


@pytest.fixture
def config():
    return ExperimentConfig(dataset="demo", estimator="ols", repetitions=3,
                            stages=(0.1, 0.25, 0.5), purpose="smoke")


def test_run_resume_is_identical_and_never_includes_unfinished_zeros(tmp_path, config):
    run = tmp_path / "run"
    run_experiment(config, run)
    original = load_records(run)
    assert len(original) == 9
    first_bytes = (run / "replicates/000000.json").read_bytes()
    (run / "replicates/000002.json").unlink()
    assert len(load_records(run)) == 6
    render_report(run)
    assert "incomplete" in (run / "report.html").read_text()
    run_experiment(config, run, resume=True)
    assert load_records(run) == original
    assert (run / "replicates/000000.json").read_bytes() == first_bytes
    assert json.loads((run / "manifest.json").read_text())["status"] == "complete"


def test_resume_rejects_config_changes_without_modifying_evidence(tmp_path, config):
    run_experiment(config, tmp_path / "run")
    before = (tmp_path / "run/manifest.json").read_bytes()
    with pytest.raises(ValueError, match="configuration"):
        run_experiment(replace(config, seed=1), tmp_path / "run", resume=True)
    assert (tmp_path / "run/manifest.json").read_bytes() == before
    with pytest.raises(FileExistsError):
        run_experiment(config, tmp_path / "run")


def test_corrupt_or_mixed_checkpoint_is_rejected(tmp_path, config):
    run_experiment(config, tmp_path / "run")
    path = tmp_path / "run/replicates/000000.json"
    payload = json.loads(path.read_text())
    payload["records"][0]["truth"] = 123
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checkpoint"):
        run_experiment(config, tmp_path / "run", resume=True)
    assert json.loads((tmp_path / "run/manifest.json").read_text())["status"] == "failed"
    assert "invalid" in (tmp_path / "run/report.html").read_text()


def test_missing_dataset_failure_is_recorded_and_reported(tmp_path, config):
    with pytest.raises(FileNotFoundError):
        run_experiment(replace(config, dataset=str(tmp_path / "missing.mtx")), tmp_path / "run")
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["attempts"][-1]["error"]["type"] == "FileNotFoundError"
    assert (tmp_path / "run/report.html").exists()


def test_report_is_standalone_and_sanitizes_metadata(tmp_path, config):
    run_experiment(config, tmp_path / "run")
    manifest_path = tmp_path / "run/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["attempts"][0]["note"] = "/Users/private-person/private/run <script>alert(1)</script>"
    manifest_path.write_text(json.dumps(manifest))
    render_report(tmp_path / "run")
    html = (tmp_path / "run/report.html").read_text()
    assert "/Users/" not in html
    assert "<script>alert(1)" not in html
    assert "<script src=" not in html and "<link " not in html
    assert "Machine-readable handoff" in html
    assert "not a reproduction" in html
    assert "Raw repetition records" in html


def test_concurrent_writer_is_rejected(tmp_path, config):
    fcntl = pytest.importorskip("fcntl")
    run = tmp_path / "run"
    run.mkdir()
    with (run / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another process"):
            run_experiment(config, run)


@pytest.mark.parametrize("outcome", ["linear", "multiplicative", "quadratic", "sqrt", "two_hop", "dynamic"])
@pytest.mark.parametrize("merge,level", [("suffix", "unit"), ("baseline_pair", "cluster"), ("repeated", "unit"), ("final_pair", "unit")])
def test_supported_scenarios_produce_complete_records(tmp_path, config, outcome, merge, level):
    config = replace(config, outcome=outcome, merge=merge, level=level, repetitions=1)
    run_experiment(config, tmp_path / "run")
    assert len(load_records(tmp_path / "run")) == 3


def test_checkpoint_diagnostics_are_validated_before_publication(tmp_path, config):
    run_experiment(config, tmp_path / "run")
    path = tmp_path / "run/replicates/000000.json"
    payload = json.loads(path.read_text())
    payload["stages"][0]["treated_units"] = 99999
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checkpoint"):
        load_records(tmp_path / "run")


def test_resume_rejects_changed_runtime(tmp_path, config, monkeypatch):
    from rampup import runner
    run_experiment(config, tmp_path / "run")
    original = runner.runtime_versions(config.estimator)
    monkeypatch.setattr(runner, "runtime_versions", lambda estimator: {**original, "numpy": "different"})
    with pytest.raises(ValueError, match="versions"):
        run_experiment(config, tmp_path / "run", resume=True)


def test_resume_rejects_changed_dataset(tmp_path, config):
    source = tmp_path / "graph.mtx"
    source.write_text("4 4 4\n1 2\n2 3\n3 4\n4 1\n")
    config = replace(config, dataset=str(source), stages=(0.25, 0.5), repetitions=1)
    run_experiment(config, tmp_path / "run")
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="data identity"):
        run_experiment(config, tmp_path / "run", resume=True)


def test_cli_demo_and_report_work_without_notebooks(tmp_path):
    command = [sys.executable, "-m", "rampup", "demo", "--output", str(tmp_path / "demo"), "--repetitions", "2"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "demo/report.html").is_file()
    result = subprocess.run([sys.executable, "-m", "rampup", "report", str(tmp_path / "demo")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, "-m", "rampup", "run", "--config", "missing.json"], capture_output=True, text=True)
    assert result.returncode != 0
