"""Offline HTML reports built only from validated, completed observations."""

import html
import json
from pathlib import Path
import re

from .metrics import summarize


CSS = """
:root{--ink:#172a36;--muted:#536976;--line:#d9e2e5;--paper:#f7f8f4;--blue:#215e78;--accent:#c85b41}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
main{max-width:1160px;margin:auto;padding:52px 36px 80px}header{padding:34px 0 40px;border-bottom:2px solid var(--ink)}
h1{font:600 clamp(32px,5vw,58px)/1.08 Georgia,serif;letter-spacing:-1.6px;margin:18px 0 24px;max-width:1000px}
h2{font:600 29px/1.2 Georgia,serif;margin:42px 0 16px}h3{font-size:17px;margin:25px 0 8px}p{max-width:950px;margin:12px 0}
.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:2px;color:var(--blue);font-weight:700}.lede{font-size:20px;color:var(--muted);max-width:860px}
.pill{display:inline-block;background:#e1ece7;color:#22513e;padding:4px 12px;border-radius:16px;font-size:12px;font-weight:700;margin-right:8px}
.caution{background:#f5e4dc;color:#853c2c}.notice{border-left:4px solid var(--accent);padding:12px 20px;background:#fff3ec;margin:22px 0}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;margin:30px 0}.metric{padding:18px 0;border-top:1px solid var(--line)}.metric strong{display:block;font-size:32px;font-weight:600;line-height:1.2}.metric span{font-size:13px;color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums;background:white}th,td{padding:12px 13px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th{background:#e7eef0;font-weight:600}th:first-child,td:first-child{text-align:left}.scroll{overflow-x:auto;margin:18px 0}tr:last-child td{border-bottom:0}
details{border:1px solid var(--line);background:white;padding:14px 18px;margin:14px 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;background:#eaf0f1;padding:18px;border-radius:5px;max-height:650px;overflow:auto}
code{font:13px ui-monospace,SFMono-Regular,Consolas,monospace}.caption,small{color:var(--muted);font-size:13px}.chart{display:block;width:100%;height:auto;background:white;border:1px solid var(--line)}a{color:var(--blue)}li{margin:7px 0}footer{margin-top:46px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}
@media(max-width:650px){main{padding:22px 18px 44px}.metrics{grid-template-columns:repeat(2,1fr)}h1{letter-spacing:-.7px}th,td{padding:9px}}@media print{body{background:white}main{padding:0}details{break-inside:avoid}pre{max-height:none}.scroll{overflow:visible}}
"""


def sanitize(value):
    """Keep source basenames and error meaning, discard local absolute paths."""
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        if value.startswith(("/", "\\")) and " " not in value:
            return value.replace("\\", "/").rsplit("/", 1)[-1]
        return re.sub(r"/(?:Users|home|private|tmp|opt|var)/[^\s<>\"']+", "[local path]", value)
    return value


def pre(value, element_id=None):
    text = json.dumps(sanitize(value), indent=2, ensure_ascii=False, allow_nan=False)
    attribute = f' id="{html.escape(element_id)}"' if element_id else ""
    return f"<pre{attribute}>{html.escape(text)}</pre>"


def number(value):
    return "—" if value is None else f"{value:.6g}"


def metric_table(stats):
    rows = []
    for row in stats:
        interval = row["bias_ci95"]
        cells = [html.escape(row["group"]), str(row["n"]), number(row["truth"]), number(row["mean_estimate"]),
                 number(row["bias"]), number(row["std"]), number(row["mse"]),
                 f"[{number(interval[0])}, {number(interval[1])}]" if interval else "not estimable"]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    headings = ["Merged stages", "MC n", "True GATE", "Mean estimate", "Bias", "Std", "MSE", "95% MC CI: bias"]
    return '<div class="scroll"><table><thead><tr>' + "".join(f"<th>{h}</th>" for h in headings) + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"


def mse_chart(stats):
    if not stats:
        return ""
    maximum = max(row["mse"] for row in stats) or 1
    height = 60 + 52 * len(stats)
    parts = [f'<svg class="chart" viewBox="0 0 1000 {height}" role="img" aria-label="MSE by selected stage group">',
             '<text x="22" y="26" fill="#536976" font-size="13">Empirical mean squared error · axis begins at zero</text>']
    for index, row in enumerate(stats):
        y = 45 + 52 * index
        width = 660 * row["mse"] / maximum
        parts.extend([f'<text x="22" y="{y + 21}" font-size="15" fill="#172a36">{html.escape(row["group"])}</text>',
                      f'<rect x="205" y="{y}" width="{width:.3f}" height="30" rx="3" fill="#215e78"/>',
                      f'<text x="{max(220, 215 + width):.3f}" y="{y + 21}" font-size="14" fill="#172a36">{number(row["mse"])}</text>'])
    parts.append("</svg>")
    return "".join(parts)


def render_report(directory):
    """Rebuild one consistent snapshot; reject writers already holding the run lock."""
    from .runner import _writer_lock

    directory = Path(directory)
    with _writer_lock(directory):
        return _render_report_locked(directory)


def _render_report_locked(directory):
    """Internal entry point for callers that already own the run lock."""
    from .runner import load_records, read_json, write_text_atomic

    directory = Path(directory)
    config = read_json(directory / "config.json")
    manifest = read_json(directory / "manifest.json")
    invalid = None
    try:
        records = load_records(directory)
        stats = summarize(records)
    except (ValueError, KeyError, TypeError) as exc:
        invalid = str(exc)
        records, stats = [], []
    completed = len({row["replicate"] for row in records})
    status = "invalid" if invalid else manifest["status"]
    if status == "complete" and completed != config["repetitions"]:
        status = "incomplete"
    smoke = config.get("purpose") == "smoke"
    claim = ("Engineering smoke test of the experiment pipeline; not a reproduction of the paper's numerical results."
             if smoke else "Monte Carlo estimates under the frozen configuration; not a reproduction certificate for published tables.")
    limitations = ["A corrected implementation changes historical node alignment, rounding and random streams; historical numerical equality is not claimed.",
                   "Uncertainty intervals describe Monte Carlo error conditional on this graph, configuration and model initialization; they are not causal confidence intervals for a real deployment.",
                   "Groups reuse stage data within each repetition and are dependent; marginal intervals do not test between-group differences.",
                   "GNN architecture is retained, but library versions can affect optimization. No full paper replication gate is evaluated automatically."]
    if smoke:
        limitations.insert(0, "Small repetition/epoch counts verify execution only; accuracy rankings and paper-level scientific conclusions are unsupported.")
    if invalid:
        limitations.insert(0, "Checkpoint validation failed. All aggregate outcomes are withheld from this report; repair the records before analysis.")
    handoff = {
        "study_id": manifest["run_id"], "status": status, "frozen_claim": claim,
        "outcome_access": "invalid_records_not_summarized" if invalid else "validated_completed_records_read",
        "expected_repetitions": config["repetitions"], "completed_repetitions": completed,
        "validated_estimate_count": len(records), "key_metrics": stats, "limitations": limitations,
        "next_authorized_step": "Inspect configuration and diagnostics before a full replication run; do not infer unrun paper outcomes.",
    }
    diagnostics = []
    if not invalid:
        for path in sorted((directory / "replicates").glob("*.json")):
            payload = read_json(path)
            diagnostics.append({"replicate": payload["replicate"], "seed": payload["seed"], "stages": payload["stages"]})
    graph = manifest.get("graph", {})
    title = "Ramp-up experiment report"
    safe_status = html.escape(status)
    blocks = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{CSS}</style></head><body><main>',
              '<header><div class="eyebrow">Just Ramp-Up · NeurIPS 2026 · Reproducible experiments</div>',
              f'<h1>{title}</h1><span class="pill {"caution" if status != "complete" else ""}">{safe_status.upper()}</span>',
              f'<span class="pill">{html.escape(config["estimator"].upper())} · {html.escape(config["outcome"])}</span>',
              f'<p class="lede">{html.escape(claim)}</p></header>',
              '<section><h2>Execution outcome</h2>',
              f'<p>Run <code>{html.escape(str(sanitize(manifest["run_id"])))}</code> contains {completed} validated complete repetitions out of {config["repetitions"]} requested, producing {len(records)} estimates. Status is derived from both the execution record and the available checkpoints.</p>',
              '<div class="metrics">' + ''.join(f'<div class="metric"><strong>{html.escape(str(v))}</strong><span>{k}</span></div>' for k, v in [
                  ("Complete repetitions", completed), ("Estimate records", len(records)), ("Graph nodes", graph.get("nodes", "—")), ("Initial edges", graph.get("edges", "—"))]) + '</div>']
    if invalid:
        blocks.append(f'<div class="notice"><strong>Invalid evidence:</strong> {html.escape(str(sanitize(invalid)))}</div>')
    if status != "complete":
        blocks.append('<div class="notice">This run has not completed successfully. Missing, failed and invalid repetitions are not substituted with zeros. Any visible summaries describe the validated subset only.</div>')
    blocks.extend(['</section><section><h2>Objective and experimental design</h2>',
                   '<p>GATE is the difference between mean potential outcomes under global treatment and global control. The study examines whether training a regression model on multiple treatment proportions improves this extrapolation under network interference. These runs quantify configured scenarios; no automatic hypothesis acceptance gate is imposed.</p>',
                   '<p>Design reference: the supplied NeurIPS 2026 manuscript, <em>Just Ramp-Up: Debiasing Regression-based Estimator for A/B Tests under Network Interference</em>, simulation study and detailed-experiment appendices. The linked preprint may retain its earlier title: <a href="https://arxiv.org/abs/2410.12740">arXiv:2410.12740</a>.</p>',
                   '<p>Each repetition has independent assignment and noise random streams derived from its root seed and repetition ID. Stages are generated once and reused across merge groups. Complete randomization selects floor(p × entity count) entities; cluster assignments preserve membership. A one-entity baseline differs from exact global control. The latest selected stage supplies the graph used for counterfactual evaluation.</p>',
                   '<h3>Frozen configuration</h3>', pre(config), '<h3>Selected stage groups and graph-specific truth</h3>', pre(manifest.get("groups", [])),
                   '</section><section><h2>Results and Monte Carlo uncertainty</h2>'])
    if stats:
        blocks.extend([metric_table(stats), mse_chart(stats),
                       '<p class="caption">Signed bias = mean(estimate − truth). Std uses ddof=0. MSE = mean((estimate − truth)²). Bias intervals use a Student-t critical value and the sample standard error of the errors. MSE intervals (in the detailed metrics) use a normal approximation, clipped at zero. With fewer than two repetitions, intervals are unavailable.</p>',
                       '<details><summary>Exact metrics and uncertainty</summary>', pre(stats), '</details>'])
    else:
        blocks.append('<p>No validated result is available for summarization.</p>')
    blocks.extend(['</section><section><h2>Methods, provenance and diagnostics</h2>',
                   '<p>Node IDs are sorted once and shared by adjacency, treatment, features and responses. Graphs must be simple, unweighted and have no isolated nodes. Two-hop interference uses Wz + r₂[W(Wz) − diag(W²)z], with W = D⁻¹A; the diagonal removes closed walks without constructing a dense squared adjacency. True GATE is computed by averaging the noise-free global-treatment and global-control outcomes on the evaluation graph.</p>',
                   '<p>OLS fits an intercept and treatment on pooled observations. GNN uses three DGL ChebConv layers (2→16, K=2), (16→16, K=1), (16→1, K=1), ReLU, degree and treatment features, and Adam with betas (0.9, 0.999). Each epoch steps the optimizer through the selected stages in order, using each stage’s own graph. No held-out predictive test set or model-selection search is used.</p>',
                   '<p>Dynamic graphs are fixed across repetitions and evolve independently of treatment: each node at or above the degree median gets five candidate neighbors drawn with replacement proportional to current degree; loops and existing edges are discarded. Initial cluster membership stays fixed. Recorded edge counts are realized values, not a promise to match historical notebook RNG output.</p>',
                   pre({"source": manifest.get("source"), "graph": graph, "versions": manifest.get("versions"), "git": manifest.get("git"), "implementation": manifest.get("implementation"), "execution_identity": manifest.get("execution_identity"), "seed_schedule": manifest.get("seed_schedule")}),
                   '<details><summary>Realized treatment counts, exposure variation and stage graphs</summary>', pre(diagnostics), '</details>',
                   '</section><section><h2>Exact execution record</h2>',
                   '<p>Each checkpoint is committed atomically only after all merge groups finish and carries the frozen execution identity. Resume validates configuration, package-source fingerprint, runtime versions, source file metadata, graph and group identities, and completed records. Reports and execution share a run lock. A failed repetition restarts from the same deterministic seed. There are no hidden retries or automatic exclusions. Dataset provenance uses filename, size and modification time, not a content hash; these checks prevent accidental mixing, not deliberate tampering.</p>',
                   pre(manifest.get("attempts", [])), '</section><section><h2>Limitations and next step</h2><ul>',
                   ''.join(f'<li>{html.escape(item)}</li>' for item in limitations), '</ul>',
                   '<p>Inspect the diagnostics and migration notes before commissioning the full paper settings. Engineering tests and smoke runs do not establish the manuscript’s scientific results. HT notebooks remain historical references; this runner does not implement the paper’s MII, Hájek, Lagrange or missing MIT8 experiments.</p>',
                   '</section><section><h2>Reproducibility and artifact index</h2>',
                   '<p>The run directory contains <code>config.json</code>, <code>manifest.json</code>, <code>node_ids.json</code>, optional <code>clusters.json</code>, atomic <code>replicates/NNNNNN.json</code> files and <code>report.html</code>. This HTML embeds the reviewed metrics, raw estimates, diagnostics and execution record; sibling files and a network connection are not required to read it. Source basenames are retained while absolute local paths are removed.</p>',
                   '<details><summary>Raw repetition records</summary>', pre(records), '</details>',
                   '<h3>Machine-readable handoff</h3>', pre(handoff, "handoff"),
                   '</section><footer>Standalone report · No external scripts, fonts, styles or images · Generated from validated experiment artifacts</footer></main></body></html>'])
    path = directory / "report.html"
    write_text_atomic(path, "".join(blocks))
    return path
