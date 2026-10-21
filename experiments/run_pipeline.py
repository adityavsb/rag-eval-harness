"""Interactive and ad-hoc CLI query tool for the RAG pipeline."""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.pipeline.generator import RAGGenerator
from src.pipeline.indexing import VectorIndexer
from src.pipeline.ingestion import chunk_documents, load_raw_documents
from src.pipeline.retriever import RAGRetriever
from src.utils.config import PipelineConfig

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query the RAG pipeline interactively or with a prompt."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/chunk_optimized.yaml",
        help="Path to YAML pipeline configuration file.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional single query to evaluate. If omitted, starts interactive REPL.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing of documents into vector store.",
    )
    return parser.parse_args()


def setup_pipeline(config: PipelineConfig, reindex: bool = False):
    indexer = VectorIndexer(config.embedding, config.vector_store)
    if indexer.count() == 0 or reindex:
        console.print(f"[cyan]Indexing documents from {config.corpus.data_dir}...[/cyan]")
        docs = load_raw_documents(config.corpus.data_dir, config.corpus.file_pattern)
        chunks = chunk_documents(docs, config.chunking)
        indexer.index_chunks(chunks)

    retriever = RAGRetriever(indexer, config.retriever)
    generator = RAGGenerator(config.generation)
    return retriever, generator


def process_query(query: str, retriever: RAGRetriever, generator: RAGGenerator):
    console.print(f"\n[bold yellow]Query:[/bold yellow] {query}")

    # 1. Retrieve
    res = retriever.retrieve(query)
    table = Table(title=f"Retrieved Chunks ({res.stage}, {res.retrieval_latency_ms:.1f}ms)")
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Chunk ID", style="green", width=25)
    table.add_column("Score", style="magenta", width=10)
    table.add_column("Snippet", style="white")

    for c in res.chunks:
        snippet = c.content.replace("\n", " ")[:90] + "..."
        table.add_row(str(c.rank), c.chunk_id, f"{c.score:.4f}", snippet)
    console.print(table)

    # 2. Generate
    gen_res = generator.generate(query, res.chunks)
    panel_title = f"Generated Answer ({gen_res.generation_latency_ms:.1f}ms)"
    console.print(Panel(gen_res.answer, title=panel_title, border_style="cyan"))


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        console.print(f"[red]Config file {config_path} not found.[/red]")
        sys.exit(1)

    config = PipelineConfig.from_yaml(config_path)
    retriever, generator = setup_pipeline(config, reindex=args.reindex)

    if args.query:
        process_query(args.query, retriever, generator)
    else:
        console.print(
            "[bold green]Entering interactive RAG shell. Type 'exit' to quit.[/bold green]"
        )
        while True:
            try:
                user_input = console.input("\n[bold cyan]Ask a question > [/bold cyan]").strip()
                if not user_input or user_input.lower() in ("exit", "quit", "q"):
                    break
                process_query(user_input, retriever, generator)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
