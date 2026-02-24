"""
Prediction API routes.

GET /predict/{ticker}         - Full score + signal breakdown for one ticker
GET /predict/{ticker}/raw     - Just the probability float (machine-readable)
GET /watchlist                - Top 20 highest-scoring tickers right now
GET /alerts                   - Tickers that crossed alert threshold in last 24h
"""

from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.db.session import get_db
from src.db.models import Company, SignalVector
from src.models.predictor import Predictor, get_predictor
from src.signals.scoring import SignalScorer

router = APIRouter(prefix="/predict", tags=["predictions"])


def _get_scorer(
    db: Session = Depends(get_db),
    predictor: Predictor = Depends(get_predictor),
) -> SignalScorer:
    return SignalScorer(db, model=predictor if predictor.is_available else None)


@router.get("/{ticker}")
def predict_ticker(
    ticker: str,
    as_of: str | None = Query(default=None, description="ISO date, e.g. 2024-06-01"),
    scorer: SignalScorer = Depends(_get_scorer),
) -> dict[str, Any]:
    """
    Full M&A probability score for a ticker with signal breakdown.

    Response includes:
    - composite_score: rule-based 0-1 (always available)
    - model_probability: ML-based 0-1 (requires trained model)
    - alert_level: none | watch | elevated | high
    - signal_breakdown: per-category contribution
    - features: raw normalized feature values
    """
    ticker = ticker.upper().strip()

    as_of_dt = None
    if as_of:
        try:
            as_of_dt = datetime.fromisoformat(as_of)
        except ValueError:
            raise HTTPException(status_code=400, detail="as_of must be ISO format date")

    try:
        result = scorer.score(ticker, as_of=as_of_dt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


@router.get("/{ticker}/probability")
def predict_ticker_probability(
    ticker: str,
    scorer: SignalScorer = Depends(_get_scorer),
) -> dict[str, float | str | None]:
    """
    Minimal endpoint: just the probability score.
    Designed for programmatic consumption / quant pipelines.
    """
    ticker = ticker.upper().strip()
    result = scorer.score(ticker)
    return {
        "ticker": ticker,
        "probability": result["model_probability"] or result["composite_score"],
        "source": "model" if result["model_probability"] is not None else "composite",
        "alert_level": result["alert_level"],
        "as_of": result["as_of"],
    }


router_watchlist = APIRouter(tags=["watchlist"])


@router_watchlist.get("/watchlist")
def get_watchlist(
    top_n: int = Query(default=20, ge=1, le=100),
    min_score: float = Query(default=0.0, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
    predictor: Predictor = Depends(get_predictor),
) -> dict[str, Any]:
    """
    Top N companies by current M&A probability score.
    Uses cached signal vectors for speed; recomputes composite from stored features.
    """
    scorer = SignalScorer(
        db, model=predictor if predictor.is_available else None
    )

    # Get all tracked companies
    companies = db.query(Company).all()
    tickers = [c.ticker for c in companies]

    if not tickers:
        return {"tickers": [], "count": 0, "generated_at": datetime.utcnow().isoformat()}

    results = scorer.watchlist(tickers, top_n=top_n)
    filtered = [r for r in results if r["composite_score"] >= min_score]

    return {
        "tickers": filtered,
        "count": len(filtered),
        "generated_at": datetime.utcnow().isoformat(),
        "model_active": predictor.is_available,
    }


@router_watchlist.get("/alerts")
def get_alerts(
    threshold: float = Query(default=0.50, ge=0.0, le=1.0),
    hours_back: int = Query(default=24, ge=1, le=168),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Companies whose signal score crossed the alert threshold in the past N hours.
    Useful for setting up automated notifications.
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours_back)

    # Find vectors computed recently that exceed threshold
    recent_high = (
        db.query(SignalVector)
        .filter(
            SignalVector.computed_at >= cutoff,
            SignalVector.composite_score >= threshold,
        )
        .order_by(SignalVector.composite_score.desc())
        .limit(50)
        .all()
    )

    alerts = [
        {
            "ticker": sv.ticker,
            "composite_score": sv.composite_score,
            "model_probability": sv.model_probability,
            "as_of": sv.as_of_date.isoformat(),
            "computed_at": sv.computed_at.isoformat(),
            "primary_signal": _identify_primary_signal(sv),
        }
        for sv in recent_high
    ]

    return {
        "alerts": alerts,
        "count": len(alerts),
        "threshold": threshold,
        "window_hours": hours_back,
        "generated_at": datetime.utcnow().isoformat(),
    }


def _identify_primary_signal(sv: SignalVector) -> str:
    """Identify which signal group is driving the score."""
    sec_score = sv.s4_filings_30d * 6 + sv.sc_to_t_filings_30d * 5
    flight_score = sv.flight_anomaly_score_30d * 7
    job_score = 5 if sv.job_freeze_flag else 0
    patent_score = (sv.patent_assignments_outbound_30d + sv.patent_assignments_inbound_30d)

    scores = {
        "sec_filings": sec_score,
        "flight_activity": flight_score,
        "job_freeze": job_score,
        "patent_transfers": patent_score,
    }
    return max(scores, key=scores.get)
