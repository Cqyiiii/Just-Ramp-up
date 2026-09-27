"""Run bounded engineering checks and build a shareable, offline evidence report.

Run from the repository root: python scripts/validate_refactor.py
Use --reuse to rebuild from existing run records while rerunning code checks.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rampup.config import ExperimentConfig
from rampup.metrics import summarize
from rampup.report import CSS, metric_table, pre, render_report
from rampup.runner import load_records, read_json, run_experiment, write_json


def command_check(arguments, cwd):
    started = time.monotonic()
    result = subprocess.run([sys.executable, *arguments], cwd=cwd, capture_output=True, text=True)
    return {"command": "python " + " ".join(arguments), "exit_code": result.returncode,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/refactor-validation")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/refactor-validation.html")
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = ExperimentConfig(dataset="demo", estimator="ols", stages=(0.1, 0.25, 0.5),
                            repetitions=8, purpose="smoke")
    real = str(ROOT / "Dataset/socfb-Stanford3.mtx")
    cases = {
        "synthetic-ols": base,
        "synthetic-gnn": replace(base, estimator="gnn", repetitions=3, epochs=8),
        "synthetic-dynamic-gnn": replace(base, estimator="gnn", outcome="dynamic", level="cluster", repetitions=2, epochs=4),
        "stanford-two-hop-ols": replace(base, dataset=real, outcome="two_hop", stages=(.02, .05, .1, .25, .5), repetitions=3),
        "stanford-cluster-baseline-ols": replace(base, dataset=real, level="cluster", merge="baseline_pair",
                                                  stages=(.02, .05, .1, .25, .5), repetitions=3),
    }
    print("Checking tests and wheel build", flush=True)
    checks = [command_check(["-m", "pytest", "-q"], ROOT),
              command_check(["-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--no-cache-dir",
                             "--wheel-dir", str(args.output / "wheels"), "."], ROOT)]
    summaries = []
    case_html = []
    for name, config in cases.items():
        print(f"Validating {name}", flush=True)
        directory = args.output / name
        error = None
        try:
            if args.reuse:
                if read_json(directory / "config.json") != config.to_dict():
                    raise ValueError("cached case configuration differs")
                load_records(directory)
                render_report(directory)
            else:
                run_experiment(config, directory)
            records = load_records(directory)
            manifest = read_json(directory / "manifest.json")
            metrics = summarize(records)
            summaries.append({"case": name, "status": manifest["status"], "repetitions": config.repetitions,
                              "records": len(records), "graph": manifest["graph"], "metrics": metrics,
                              "versions": manifest["versions"], "git": manifest["git"]})
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            summaries.append({"case": name, "status": "failed", "error": error})
        if (directory / "report.html").exists():
            full = (directory / "report.html").read_text()
            body = full.split("<main>", 1)[1].rsplit("</main>", 1)[0]
            body = body.replace('id="handoff"', f'id="handoff-{name}"')
            case_html.append(f'<details><summary>{html.escape(name)} — complete embedded experiment report</summary>{body}</details>')
    passed = all(c["exit_code"] == 0 for c in checks) and all(s["status"] == "complete" for s in summaries)
    count_match = re.search(r"(\d+) passed", checks[0]["stdout"])
    test_count = int(count_match.group(1)) if count_match else None
    record_count = sum(case.get("records", 0) for case in summaries)
    repetitions = sum(case.get("repetitions", 0) for case in summaries if case["status"] == "complete")
    evidence = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "status": "passed" if passed else "failed", "tests_passed": test_count,
                "checks": checks, "cases": summaries, "cache_reused": args.reuse}
    write_json(args.output / "validation.json", evidence)
    handoff = {"study_id": "just-ramp-up-refactor-validation-2026-09-27", "status": evidence["status"],
               "frozen_claim": "Engineering correctness and bounded smoke execution only; no published-table replication claim.",
               "outcome_access": "validated smoke outcomes read; historical Result arrays and full-paper outcomes not analyzed",
               "tests_passed": test_count, "completed_cases": sum(s["status"] == "complete" for s in summaries),
               "completed_repetitions": repetitions, "estimate_records": record_count,
               "major_limitations": ["Full 100-repetition/400-epoch paper experiments were not run.",
                                     "Stanford Louvain partition has 197 clusters here, versus 192 in the manuscript.",
                                     "Node alignment, temporal assignment, baseline and RNG corrections change legacy behavior.",
                                     "MII/Hájek/Lagrange/MIT8 supplementary reproductions remain outside implemented scope."],
               "next_authorized_step": "Finish independent review and push the refactor; full paper replication is a separate study."}
    result_table = '<div class="scroll"><table><tr><th>Case</th><th>Status</th><th>Repetitions</th><th>Estimates</th><th>Nodes</th><th>Initial edges</th></tr>'
    for case in summaries:
        result_table += '<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in (
            case["case"], case["status"], case.get("repetitions", "—"), case.get("records", "—"),
            case.get("graph", {}).get("nodes", "—"), case.get("graph", {}).get("edges", "—"))) + '</tr>'
    result_table += '</table></div>'
    documents = (ROOT / "docs/migration.md").read_text()
    parts = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Just Ramp-Up — Refactor validation</title><style>{CSS}</style></head><body><main>',
             '<header><div class="eyebrow">Just Ramp-Up · NeurIPS 2026 · Engineering evidence</div><h1>A reproducible experiment system,<br>with explicit scientific boundaries.</h1>',
             f'<span class="pill">VALIDATION {evidence["status"].upper()}</span><p class="lede">{test_count} tests, {len(cases)} bounded execution cases, {record_count} inspectable estimates. This report records the system refactor and its verification; it does not claim to reproduce the paper’s numerical tables.</p></header>',
             '<h2>What was changed and why</h2><p>The original repository distributed experimental logic across 37 notebooks with duplicated graph loading, randomization, outcome functions, model training and result handling. The user requested maintainability and reproducibility, a polished NeurIPS 2026 README, and a push back to the same repository. The latest supplied manuscript defines the intended experimental settings.</p>',
             '<p>The maintained <code>rampup</code> package now separates those responsibilities and exposes a configuration-driven CLI. Historical notebooks and result arrays remain available in place. The new engine corrects node/feature alignment and dynamic-graph forwarding, records realized experimental designs, and commits complete repetitions atomically. Each run includes its own offline report.</p>',
             '<h2>Frozen verification design and gates</h2><p>The confirmatory engineering checks are numerical tests on hand-computed fixtures, real GNN stage-graph/determinism tests, recovery/corruption/concurrency tests, CLI integration, all six outcome families across four merge/level combinations, and a distributable wheel build. Acceptance requires a zero test/build exit code and valid complete records for every execution case. No accuracy threshold or paper-effect confirmation is imposed on small smoke runs.</p>',
             '<p>The five execution cases were chosen before execution to cover OLS, GNN, dynamic stage graphs, sparse two-hop interference on Stanford, and the near-zero cluster baseline. Their seed is 2026; GNN initialization uses seed 2. Synthetic cases use 60 nodes and three proportions; real cases use the five paper proportions. Full frozen configurations and exact per-case evidence are embedded below.</p>',
             result_table,
             '<h2>Evidence and interpretation</h2><p>Small-run metrics are exploratory diagnostics of the configured pipeline, not confirmatory evidence for the paper. In particular, the GNN runs use only 4 or 8 epochs and 2 or 3 repetitions: poor estimates, large errors or reversals are retained rather than filtered. OLS and GNN results from different configurations are not a controlled estimator comparison.</p>']
    for case in summaries:
        if "metrics" in case:
            parts += [f'<h3>{html.escape(case["case"])}</h3>', metric_table(case["metrics"])]
    parts += ['<h2>Data provenance and scientific deviations</h2><p>The repository Stanford edge file has 11,586 nodes and 568,309 undirected edges. The current deterministic partition contains 197 communities; the manuscript lists 192. Canonical ordering and the recorded NetworkX version define this run’s partition. Historical arrays were preserved and not loaded for validation, so no claim is made about their completion status or correctness.</p>',
              '<p>The static estimand is the noise-averaged difference between global treatment and control on the evaluation graph. Assignment and noise use separate seed streams, and merge groups share their underlying stage data. No train/test split, model-selection sweep or unreported exclusion was performed. Complete configuration, imported versions, graph counts, selected repetition IDs and failure/resume records are embedded in each case.</p>',
              '<details><summary>Complete migration audit and scenario mapping</summary><pre>' + html.escape(documents) + '</pre></details>',
              '<h2>Exact verification record</h2>', pre(checks),
              '<p>The first implementation checks intentionally failed while the new modules were absent. Subsequent red/green cycles exposed and fixed an impossible one-entity baseline on tiny staggered designs and missing validation of stored treatment-count diagnostics. One square-root numerical fixture was corrected after its handwritten expectation omitted its configured factor of two; the production formula was already correct. No historical experiment outcomes were substituted into these checks.</p>',
              '<h2>Embedded experiment reports</h2><p>Each section contains the full run configuration, metrics, uncertainty formulas, stage diagnostics, execution attempts, limitations, raw estimates and a machine-readable handoff. All content is embedded; no network, server or sibling file is needed.</p>',
              *case_html,
              '<h2>Reproducibility and handoff</h2><p>Rebuild with <code>python scripts/validate_refactor.py --output outputs/refactor-validation --report reports/refactor-validation.html</code> from a compatible environment. Use a new output directory for new execution; <code>--reuse</code> reuses validated cached case records and reruns test/build checks. The evidence JSON and per-case artifacts live under the selected output directory. Source paths and local task metadata are sanitized in this shareable HTML.</p>',
              '<p>The code and README still require the recorded independent code review and remote-push verification; those repository operations are distinct from the completed local experiment evidence. Full paper-sized runs, historical partition recovery and supplementary missing-source experiments are separate follow-up work.</p>',
              '<h3>Machine-readable handoff</h3>', pre(handoff, "handoff"),
              '<details><summary>Complete validation metadata</summary>', pre(evidence), '</details>',
              '<footer>NeurIPS 2026 · <a href="https://arxiv.org/abs/2410.12740">arXiv:2410.12740</a> · Offline standalone report · No hidden or quarantined outcomes included</footer></main></body></html>']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(parts)
    if re.search(r'<(?:script|link|img)[^>]+(?:src|href)=["\'](?:https?://|file:)', rendered):
        raise ValueError("report contains an external rendering dependency")
    if "/Users/" in rendered or "/private/var/" in rendered:
        raise ValueError("report contains unsanitized local paths")
    args.report.write_text(rendered, encoding="utf-8")
    print(f"Validation {evidence['status']}: {args.report}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
