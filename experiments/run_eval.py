"""CLI entrypoint to execute batch RAG evaluation benchmarks."""

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console

from src.eval.runner import EvaluationRunner
from src.utils.config import PipelineConfig

# Ensure working directory and sys.path are rooted at repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RAG Evaluation Benchmark against Golden Q&A set."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML pipeline configuration file.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing of documents into vector store.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["offline", "live"],
        default=None,
        help="Evaluation mode: 'offline' for local/mock, 'live' for real model inference.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["check-one", "check-remaining", "full"],
        default=None,
        help="Execution stage: 'check-one' (N=1), 'check-remaining' (N=1), or 'full' (all queries).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of queries to evaluate.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override generation model name (e.g. openrouter/free, nvidia/nemotron-3.5-super-vl:free).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Override judge model name (e.g. perplexity/pplx-decider-flash-v1.2).",
    )
    parser.add_argument(
        "--pacing-delay",
        type=float,
        default=None,
        help="Delay in seconds between requests in live mode (defaults to 1.5s).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        console.print(f"[bold red]Error: Config file not found at {config_path}[/bold red]")
        sys.exit(1)

    console.print(f"[bold cyan]Loading pipeline config from {config_path}...[/bold cyan]")
    config = PipelineConfig.from_yaml(config_path)

    if args.mode:
        config.mode = args.mode
        config.generation.mode = args.mode
        config.evaluation.mode = args.mode

    if args.model:
        config.generation.model_name = args.model
    if args.judge_model:
        config.evaluation.judge_model = args.judge_model
    if args.stage:
        config.evaluation.stage = args.stage
    if args.pacing_delay is not None:
        config.generation.pacing_delay_sec = args.pacing_delay

    if config.generation.mode == "live":
        import json

        from src.llm.pricing import estimate_batch_cost_eur

        golden_path = Path(config.evaluation.golden_qa_path)
        num_q = 30
        if golden_path.exists():
            try:
                with open(golden_path, encoding="utf-8") as f:
                    num_q = len(json.load(f))
            except Exception:
                pass

        effective_stage = args.stage or getattr(config.evaluation, "stage", "full")
        if effective_stage in ("check-one", "check-remaining"):
            num_q = 1
        elif args.limit is not None and args.limit > 0:
            num_q = min(num_q, args.limit)

        cost_est = estimate_batch_cost_eur(config.generation.model_name, num_q)
        console.print(
            f"[bold yellow]Projected live run cost: €{cost_est:.4f} "
            f"({num_q} queries on '{config.generation.model_name}')[/bold yellow]"
        )

    try:
        runner = EvaluationRunner(config)
        report = runner.run_benchmark(
            force_reindex=args.reindex,
            stage=args.stage,
            limit=args.limit,
            pacing_delay=args.pacing_delay,
        )
    except Exception as e:
        console.print(f"[bold red]Benchmark execution failed: {e}[/bold red]")
        sys.exit(1)

    console.print(
        f"[bold green]✔ Evaluation completed for '{report.run_name}' (mode: {report.mode}). "
        f"Evaluated {report.total_samples} test cases.[/bold green]"
    )


if __name__ == "__main__":
    main()
