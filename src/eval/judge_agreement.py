"""Judge calibration against human ground-truth labels using Cohen's kappa and Spearman rho."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from scipy.stats import spearmanr

from src.eval.llm_judge import LLMJudge


class DiscrepancyRecord(BaseModel):
    id: str
    query: str
    human_faithfulness: int
    judge_faithfulness: int
    human_correctness: int
    judge_correctness: int
    reason: str


class AgreementMetrics(BaseModel):
    total_samples: int
    faithfulness_agreement_pct: float
    cohen_kappa: float
    spearman_rho: float
    spearman_pvalue: float
    discrepancy_count: int


def calculate_cohen_kappa(y1: list[int], y2: list[int]) -> float:
    """Computes Cohen's kappa for two raters with binary ratings."""
    n = len(y1)
    if n == 0:
        return 1.0

    po = sum(1 for a, b in zip(y1, y2) if a == b) / n

    p1_1 = sum(1 for a in y1 if a == 1) / n
    p1_0 = 1.0 - p1_1
    p2_1 = sum(1 for b in y2 if b == 1) / n
    p2_0 = 1.0 - p2_1

    pe = (p1_1 * p2_1) + (p1_0 * p2_0)
    if pe >= 1.0:
        return 1.0

    kappa = (po - pe) / (1.0 - pe)
    return round(float(kappa), 4)


def calculate_spearman_rho(y1: list[int], y2: list[int]) -> tuple[float, float]:
    """Computes Spearman rank correlation between human and judge rubric grades."""
    if len(y1) < 2:
        return 1.0, 0.0
    res = spearmanr(y1, y2)
    rho = float(res.statistic) if not res.statistic != res.statistic else 0.0
    p_val = float(res.pvalue) if not res.pvalue != res.pvalue else 1.0
    return round(rho, 4), round(p_val, 4)


def evaluate_judge_agreement(
    labels_path: str | Path,
    mode: str = "offline",
    model_name: str = "gpt-4o-mini",
    provider: str = "openai",
) -> tuple[AgreementMetrics, list[DiscrepancyRecord]]:
    """Evaluates judge performance against ground-truth human annotations."""
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Human labels file not found: {path}")

    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records found in {path}")

    judge = LLMJudge(model_name=model_name, mode=mode, provider=provider)

    human_faith: list[int] = []
    judge_faith: list[int] = []
    human_corr: list[int] = []
    judge_corr: list[int] = []
    discrepancies: list[DiscrepancyRecord] = []

    for item in records:
        q_id = item["id"]
        query = item["query"]
        context = item["retrieved_context"]
        answer = item["generated_answer"]
        gt_answer = item["ground_truth_answer"]

        h_faith = int(item["human_faithfulness"])
        h_corr = int(item["human_correctness_rubric"])

        # Judge evaluation
        faith_eval = judge.evaluate_faithfulness(query, context, answer)
        corr_eval = judge.evaluate_correctness(query, answer, gt_answer)

        # Threshold judge faithfulness (>= 0.70 is binary 1, else 0)
        j_faith = 1 if faith_eval.score >= 0.70 else 0
        j_corr = corr_eval.rubric_grade

        human_faith.append(h_faith)
        judge_faith.append(j_faith)
        human_corr.append(h_corr)
        judge_corr.append(j_corr)

        # Track discrepancies
        if h_faith != j_faith or abs(h_corr - j_corr) >= 2:
            discrepancies.append(
                DiscrepancyRecord(
                    id=q_id,
                    query=query,
                    human_faithfulness=h_faith,
                    judge_faithfulness=j_faith,
                    human_correctness=h_corr,
                    judge_correctness=j_corr,
                    reason=f"Faithfulness diff: human={h_faith} judge={j_faith}; Correctness diff: human={h_corr} judge={j_corr}",
                )
            )

    # Agreement calculations
    n = len(records)
    agree_pct = sum(1 for a, b in zip(human_faith, judge_faith) if a == b) / n
    kappa = calculate_cohen_kappa(human_faith, judge_faith)
    rho, p_val = calculate_spearman_rho(human_corr, judge_corr)

    metrics = AgreementMetrics(
        total_samples=n,
        faithfulness_agreement_pct=round(agree_pct * 100.0, 2),
        cohen_kappa=kappa,
        spearman_rho=rho,
        spearman_pvalue=p_val,
        discrepancy_count=len(discrepancies),
    )

    return metrics, discrepancies


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute Judge vs Human Agreement (Cohen's Kappa & Spearman Rho)"
    )
    parser.add_argument(
        "--labels", default="data/human_labels.jsonl", help="Path to human labels JSONL"
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "live"],
        default="offline",
        help="Evaluation mode (offline or live)",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Judge model name")
    parser.add_argument("--provider", default="openai", help="Judge provider")
    parser.add_argument("--output", default=None, help="Optional output JSON path")

    args = parser.parse_args()
    console = Console()

    console.print(
        f"\n[bold]Evaluating Judge Agreement against Human Ground Truth ({args.mode} mode)[/bold]"
    )

    metrics, discrepancies = evaluate_judge_agreement(
        labels_path=args.labels,
        mode=args.mode,
        model_name=args.model,
        provider=args.provider,
    )

    # Display Metrics Table
    table = Table(title="Judge Calibration Metrics", header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Acceptance Target", justify="right", style="yellow")
    table.add_column("Status", justify="center")

    k_pass = metrics.cohen_kappa >= 0.60
    r_pass = metrics.spearman_rho >= 0.70

    table.add_row(
        "Total Samples Evaluated",
        str(metrics.total_samples),
        ">= 30",
        "[green]PASS[/green]" if metrics.total_samples >= 30 else "[red]FAIL[/red]",
    )
    table.add_row(
        "Faithfulness Raw Agreement",
        f"{metrics.faithfulness_agreement_pct:.1f}%",
        ">= 80.0%",
        (
            "[green]PASS[/green]"
            if metrics.faithfulness_agreement_pct >= 80.0
            else "[yellow]WARN[/yellow]"
        ),
    )
    table.add_row(
        "Faithfulness Cohen's Kappa (κ)",
        f"{metrics.cohen_kappa:.3f}",
        ">= 0.600",
        "[green]PASS[/green]" if k_pass else "[red]FAIL[/red]",
    )
    table.add_row(
        "Correctness Spearman Rho (ρ)",
        f"{metrics.spearman_rho:.3f}",
        ">= 0.700",
        "[green]PASS[/green]" if r_pass else "[red]FAIL[/red]",
    )
    table.add_row("Spearman p-value", f"{metrics.spearman_pvalue:.2e}", "< 0.05", "INFO")
    table.add_row("Discrepancies", str(metrics.discrepancy_count), "Low", "INFO")

    console.print(table)

    if discrepancies:
        d_table = Table(
            title=f"Discrepancies ({len(discrepancies)} total)", header_style="bold red"
        )
        d_table.add_column("ID", style="dim")
        d_table.add_column("Query", style="cyan")
        d_table.add_column("Faith (H / J)", justify="center")
        d_table.add_column("Corr (H / J)", justify="center")
        d_table.add_column("Reason")

        for d in discrepancies[:5]:  # print first 5
            d_table.add_row(
                d.id,
                d.query[:45] + "...",
                f"{d.human_faithfulness} / {d.judge_faithfulness}",
                f"{d.human_correctness} / {d.judge_correctness}",
                d.reason,
            )
        console.print(d_table)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "metrics": metrics.model_dump(),
            "discrepancies": [d.model_dump() for d in discrepancies],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        console.print(f"[green]Saved agreement report to {out_path}[/green]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
