"""Paired run comparison tool with bootstrap confidence intervals and exact McNemar significance test."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.stats import binomtest


def load_report(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Report file not found: {target}")
    with open(target, encoding="utf-8") as f:
        return json.load(f)


def compare_runs(
    report_a_path: str | Path,
    report_b_path: str | Path,
    markdown: bool = False,
) -> dict[str, Any]:
    console = Console()
    rep_a = load_report(report_a_path)
    rep_b = load_report(report_b_path)

    run_name_a = rep_a.get("run_name", Path(report_a_path).parent.name)
    run_name_b = rep_b.get("run_name", Path(report_b_path).parent.name)

    records_a = {
        r["query_id"]: r for r in rep_a.get("records", []) if r.get("ground_truth_context_ids")
    }
    records_b = {
        r["query_id"]: r for r in rep_b.get("records", []) if r.get("ground_truth_context_ids")
    }

    common_ids = sorted(set(records_a.keys()).intersection(set(records_b.keys())))
    if not common_ids:
        raise ValueError("No common answerable queries found between the two reports.")

    n = len(common_ids)

    # Metrics to compare
    metric_fields = [
        ("hit_rate", "retrieval_hit_rate", True),
        ("mrr", "retrieval_mrr", False),
        ("context_precision", "retrieval_precision", True),
        ("context_recall", "retrieval_recall", True),
    ]

    rng = np.random.default_rng(0)
    results: dict[str, Any] = {
        "run_a": run_name_a,
        "run_b": run_name_b,
        "n_aligned": n,
        "metrics": {},
    }

    # McNemar test on hit_rate
    hits_a = np.array([records_a[qid]["retrieval_hit_rate"] for qid in common_ids])
    hits_b = np.array([records_b[qid]["retrieval_hit_rate"] for qid in common_ids])

    # Discordant pairs: b (A=1, B=0), c (A=0, B=1)
    b_discordant = int(np.sum((hits_a >= 1.0) & (hits_b < 1.0)))
    c_discordant = int(np.sum((hits_a < 1.0) & (hits_b >= 1.0)))
    discordant_total = b_discordant + c_discordant

    if discordant_total > 0:
        mcnemar_res = binomtest(b_discordant, n=discordant_total, p=0.5)
        mcnemar_p = float(mcnemar_res.pvalue)
    else:
        mcnemar_p = 1.0

    results["mcnemar"] = {
        "b_a_only": b_discordant,
        "c_b_only": c_discordant,
        "discordant_pairs": discordant_total,
        "p_value": mcnemar_p,
    }

    for label, field, is_pct in metric_fields:
        vals_a = np.array([records_a[qid][field] for qid in common_ids], dtype=float)
        vals_b = np.array([records_b[qid][field] for qid in common_ids], dtype=float)

        mean_a = float(np.mean(vals_a))
        mean_b = float(np.mean(vals_b))
        diff = vals_b - vals_a
        delta_mean = float(np.mean(diff))

        # Bootstrap 95% CI on delta
        boot_deltas = []
        for _ in range(2000):
            idx = rng.integers(0, n, n)
            boot_deltas.append(float(np.mean(diff[idx])))

        ci_delta = tuple(float(x) for x in np.percentile(boot_deltas, [2.5, 97.5]))

        results["metrics"][label] = {
            "mean_a": mean_a,
            "mean_b": mean_b,
            "delta": delta_mean,
            "ci95_delta": ci_delta,
            "is_pct": is_pct,
        }

    # Format output
    if markdown:
        lines = [
            f"### Paired Comparison: `{run_name_b}` vs `{run_name_a}` (n = {n} answerable)",
            "",
            f"| Metric | `{run_name_a}` | `{run_name_b}` | Δ (95% CI) | Significance |",
            "|---|:---:|:---:|:---:|:---:|",
        ]
        for label, mdata in results["metrics"].items():
            fmt = ".2%" if mdata["is_pct"] else ".3f"
            delta_fmt = "+.2%" if mdata["is_pct"] else "+.3f"
            delta_str = f"{mdata['delta']:{delta_fmt}} [{mdata['ci95_delta'][0]:{fmt}}, {mdata['ci95_delta'][1]:{fmt}}]"
            sig = ""
            if label == "hit_rate":
                sig = f"McNemar p = {mcnemar_p:.4f} (discordant = {discordant_total})"
            lines.append(
                f"| {label} | {mdata['mean_a']:{fmt}} | {mdata['mean_b']:{fmt}} | {delta_str} | {sig} |"
            )
        print("\n".join(lines))
    else:
        table = Table(
            title=f"Paired Run Comparison: {run_name_b} vs {run_name_a} (n={n})",
            header_style="bold magenta",
        )
        table.add_column("Metric", style="cyan")
        table.add_column(run_name_a, justify="right", style="dim")
        table.add_column(run_name_b, justify="right", style="green")
        table.add_column("Delta (95% CI)", justify="right", style="bold")
        table.add_column("Statistical Note", style="yellow")

        for label, mdata in results["metrics"].items():
            fmt = ".2%" if mdata["is_pct"] else ".3f"
            delta_fmt = "+.2%" if mdata["is_pct"] else "+.3f"
            delta_str = f"{mdata['delta']:{delta_fmt}} [{mdata['ci95_delta'][0]:{fmt}}, {mdata['ci95_delta'][1]:{fmt}}]"
            note = ""
            if label == "hit_rate":
                note = f"McNemar p={mcnemar_p:.4f} (discordant={discordant_total})"
            table.add_row(
                label,
                f"{mdata['mean_a']:{fmt}}",
                f"{mdata['mean_b']:{fmt}}",
                delta_str,
                note,
            )

        console.print(table)
        console.print(
            f"hit_rate discordant pairs = {discordant_total} "
            f"({run_name_a} only: {b_discordant}, {run_name_b} only: {c_discordant}), "
            f"McNemar p = {mcnemar_p:.4f}\n"
        )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired comparison of two evaluation run reports with bootstrap CIs and McNemar test"
    )
    parser.add_argument("report_a", help="Path to baseline evaluation report JSON")
    parser.add_argument("report_b", help="Path to comparison evaluation report JSON")
    parser.add_argument(
        "--markdown", action="store_true", help="Print table in GitHub markdown format"
    )

    args = parser.parse_args()
    compare_runs(args.report_a, args.report_b, markdown=args.markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
