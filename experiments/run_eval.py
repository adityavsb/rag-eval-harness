"""CLI entrypoint to execute batch RAG evaluation benchmarks."""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from src.eval.runner import EvaluationRunner
from src.utils.config import PipelineConfig

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG Evaluation Benchmark against Golden Q&A set.")
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
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        console.print(f"[bold red]Error: Config file not found at {config_path}[/bold red]")
        sys.exit(1)

    console.print(f"[bold cyan]Loading pipeline config from {config_path}...[/bold cyan]")
    config = PipelineConfig.from_yaml(config_path)

    runner = EvaluationRunner(config)
    report = runner.run_benchmark(force_reindex=args.reindex)

    console.print(
        f"[bold green]✔ Evaluation completed for '{report.run_name}'. "
        f"Evaluated {report.total_samples} test cases.[/bold green]"
    )


if __name__ == "__main__":
    main()
