#!/usr/bin/env python3
"""
Train the XGBoost M&A prediction model.
Requires signal vectors in DB (run collect.py first, or use synthetic data).

Usage:
  python scripts/train.py
  python scripts/train.py --horizon 30    # predict within 30 days
  python scripts/train.py --splits 3      # fewer CV folds (faster, less data)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
import structlog
from rich.console import Console
from rich.table import Table

from src.db.session import init_db, SessionLocal
from src.models.trainer import ModelTrainer

logger = structlog.get_logger()
console = Console()
app = typer.Typer()


@app.command()
def main(
    horizon: int = typer.Option(60, help="Days ahead to predict M&A event"),
    splits: int = typer.Option(5, help="Number of time-series CV splits"),
):
    console.print("[bold cyan]HARBINGER[/bold cyan] — Training M&A predictor")
    console.print(f"  Label horizon: {horizon} days")
    console.print(f"  CV splits:     {splits}")
    console.print()

    init_db()
    db = SessionLocal()

    try:
        trainer = ModelTrainer(db)
        console.print("[yellow]Building training dataset...[/yellow]")
        metrics = trainer.train(label_horizon_days=horizon, n_splits=splits)

        # Print results
        table = Table(title="Training Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for k, v in metrics.items():
            table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))

        console.print(table)
        console.print()

        # Feature importance
        try:
            importance = trainer.feature_importance()
            imp_table = Table(title="Top Feature Importances", show_header=True)
            imp_table.add_column("Feature", style="cyan")
            imp_table.add_column("Importance", style="green")

            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for feat, imp in top_features:
                imp_table.add_row(feat, f"{imp:.4f}")

            console.print(imp_table)
        except Exception:
            pass

        console.print("[green]✓ Model saved.[/green] Run the API: [bold]uvicorn src.api.main:app --reload[/bold]")

    finally:
        db.close()


if __name__ == "__main__":
    app()
