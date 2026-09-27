"""Monte Carlo summaries from completed, finite repetition records only."""

from collections import defaultdict
import numpy as np
from scipy.stats import t


def summarize(records):
    grouped = defaultdict(list)
    seen = set()
    for row in records:
        key = (row["replicate"], row["group"])
        if key in seen:
            raise ValueError(f"duplicate repetition/group: {key}")
        if not np.isfinite([row["estimate"], row["truth"]]).all():
            raise ValueError("cannot summarize nonfinite estimates/truth")
        seen.add(key)
        grouped[row["group"]].append(row)
    result = []
    for group, rows in grouped.items():
        estimates = np.array([r["estimate"] for r in rows], dtype=float)
        truth = np.array([r["truth"] for r in rows], dtype=float)
        errors = estimates - truth
        n = len(rows)
        bias, mse = float(errors.mean()), float(np.mean(errors**2))
        bias_se = float(errors.std(ddof=1) / np.sqrt(n)) if n > 1 else None
        mse_se = float(np.std(errors**2, ddof=1) / np.sqrt(n)) if n > 1 else None
        half_width = float(t.ppf(0.975, n - 1) * bias_se) if n > 1 else None
        result.append({
            "group": group, "n": n, "mean_estimate": float(estimates.mean()),
            "truth": float(truth.mean()), "bias": bias, "std": float(estimates.std(ddof=0)),
            "mse": mse, "bias_mcse": bias_se, "mse_mcse": mse_se,
            "bias_ci95": [bias - half_width, bias + half_width] if half_width is not None else None,
            "mse_ci95": [max(0., mse - 1.96 * mse_se), mse + 1.96 * mse_se] if mse_se is not None else None,
        })
    return result
