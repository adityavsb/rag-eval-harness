"""Discover and inspect currently active free models from OpenRouter."""

import json
import urllib.request

from rich.console import Console
from rich.table import Table

console = Console()


def fetch_free_models():
    """Queries OpenRouter API for active free tier models."""
    url = "https://openrouter.ai/api/v1/models"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "rag-eval-harness/0.3.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        console.print(f"[bold red]Failed to fetch OpenRouter models: {e}[/bold red]")
        return []

    models = data.get("data", [])
    free_models = []
    for m in models:
        m_id = m.get("id", "")
        pricing = m.get("pricing", {})
        prompt_cost = float(pricing.get("prompt", 1.0))
        comp_cost = float(pricing.get("completion", 1.0))

        if ":free" in m_id or (prompt_cost == 0.0 and comp_cost == 0.0):
            free_models.append(
                {
                    "id": m_id,
                    "name": m.get("name", m_id),
                    "context_length": m.get("context_length", 0),
                    "description": m.get("description", "")[:80],
                }
            )

    return free_models


def main():
    console.print("[cyan]Querying OpenRouter API for active free models...[/cyan]")
    free_models = fetch_free_models()
    if not free_models:
        console.print("[yellow]No free models returned or network unavailable.[/yellow]")
        return

    table = Table(title=f"OpenRouter Active Free Models ({len(free_models)} found)")
    table.add_column("Model ID", style="bold cyan")
    table.add_column("Display Name", style="magenta")
    table.add_column("Context Window", style="green")

    for m in free_models:
        table.add_row(m["id"], m["name"], f"{m['context_length']:,}")

    console.print(table)


if __name__ == "__main__":
    main()
