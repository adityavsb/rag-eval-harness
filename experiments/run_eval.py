"""CLI entrypoint to execute batch RAG evaluation benchmarks."""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from src.eval.runner import EvaluationRunner
from src.utils.config import PipelineConfig

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
        cost_est = estimate_batch_cost_eur(config.generation.model_name, num_q)
        console.print(
            f"[bold yellow]Projected live run cost: €{cost_est:.4f} "
            f"({num_q} queries on '{config.generation.model_name}')[/bold yellow]"
        )

    try:
        runner = EvaluationRunner(config)
        report = runner.run_benchmark(force_reindex=args.reindex)
    except Exception as e:
        console.print(f"[bold red]Benchmark execution failed: {e}[/bold red]")
        sys.exit(1)

    console.print(
        f"[bold green]✔ Evaluation completed for '{report.run_name}' (mode: {report.mode}). "
        f"Evaluated {report.total_samples} test cases.[/bold green]"
    )


if __name__ == "__main__":
    main()
