#!/usr/bin/env python3
"""
Backtest: validate signal lead time on known historical deals.

For each known deal, look back N days before announcement and ask:
  - Was the composite score elevated at T-30, T-20, T-10, T-5 days?
  - What was the model probability at each point?
  - How does this compare to a random non-deal company on the same date?

This tells us:
  - Average lead time (how many days of warning we get)
  - Precision at threshold (% of high-score alerts that are real deals)
  - Recall at threshold (% of real deals we catch before announcement)

Usage:
  python scripts/backtest.py
  python scripts/backtest.py --threshold 0.5 --horizon 60
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import typer
import structlog
from rich.console import Console
from rich.table import Table

from src.db.models import KnownDeal, SignalVector
from src.db.session import init_db, SessionLocal
from src.models.predictor import get_predictor
from src.signals.scoring import SignalScorer

logger = structlog.get_logger()
console = Console()
app = typer.Typer()

LOOKBACK_WINDOWS = [60, 45, 30, 20, 10, 5]  # days before announcement


def get_score_at_date(
    scorer: SignalScorer,
    ticker: str,
    as_of: datetime,
) -> float | None:
    """Get composite score for ticker at a specific historical date."""
    try:
        result = scorer.score(ticker, as_of=as_of)
        return result["model_probability"] or result["composite_score"]
    except Exception:
        return None


@app.command()
def main(
    threshold: float = typer.Option(0.50, help="Alert threshold to evaluate"),
    horizon: int = typer.Option(60, help="Days-before-announcement to consider"),
    max_deals: int = typer.Option(50, help="Max deals to evaluate (speed)"),
):
    console.print("[bold cyan]HARBINGER[/bold cyan] — Backtest")
    console.print(f"  Threshold:  {threshold}")
    console.print(f"  Horizon:    {horizon} days")
    console.print()

    init_db()
    db = SessionLocal()
    predictor = get_predictor()
    scorer = SignalScorer(db, model=predictor if predictor.is_available else None)

    deals = db.query(KnownDeal).limit(max_deals).all()
    if not deals:
        console.print("[red]No deals in database. Run: python scripts/ingest_historical.py[/red]")
        raise typer.Exit(1)

    # Results per lookback window
    window_results: dict[int, list[float | None]] = {w: [] for w in LOOKBACK_WINDOWS}
    detected_at: list[int | None] = []  # days of first detection

    for deal in deals:
        first_detected = None
        for days_before in sorted(LOOKBACK_WINDOWS, reverse=True):
            check_date = deal.announced_at - timedelta(days=days_before)
            score = get_score_at_date(scorer, deal.target_ticker, check_date)
            window_results[days_before].append(score)

            if score is not None and score >= threshold and first_detected is None:
                first_detected = days_before

        detected_at.append(first_detected)

    # Compute stats
    console.print(f"[bold]Results for {len(deals)} known deals[/bold]")
    console.print()

    table = Table(title="Detection Rate by Lookback Window", show_header=True)
    table.add_column("Days Before Announcement", style="cyan")
    table.add_column("Avg Score", style="yellow")
    table.add_column(f"% Above {threshold}", style="green")
    table.add_column("N Scored", style="white")

    for window in LOOKBACK_WINDOWS:
        scores = [s for s in window_results[window] if s is not None]
        if not scores:
            table.add_row(f"T-{window}", "N/A", "N/A", "0")
            continue
        avg_score = np.mean(scores)
        pct_above = np.mean([s >= threshold for s in scores]) * 100
        table.add_row(
            f"T-{window}",
            f"{avg_score:.3f}",
            f"{pct_above:.1f}%",
            str(len(scores)),
        )

    console.print(table)
    console.print()

    # Lead time distribution
    detected = [d for d in detected_at if d is not None]
    if detected:
        console.print(f"[green]Deals with any detection:[/green] {len(detected)}/{len(deals)}")
        console.print(f"[green]Avg lead time:[/green] {np.mean(detected):.1f} days")
        console.print(f"[green]Median lead time:[/green] {np.median(detected):.1f} days")
        console.print(f"[green]Max lead time:[/green] {max(detected)} days")
    else:
        console.print("[yellow]No deals detected at this threshold with available signal data.[/yellow]")
        console.print("This is expected if signal vectors haven't been computed for historical dates.")
        console.print("Run: python scripts/collect.py --historical to build historical vectors.")

    db.close()


if __name__ == "__main__":
    app()
