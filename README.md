# HARBINGER

**Pre-announcement M&A signal detection using public alternative data.**

Hedge funds pay $5–50M/year for datasets that move before M&A announcements. This is the infrastructure to build one.

---

## The Thesis

Mergers leave a trail of computable public signals weeks before announcement:

| Signal | Source | Why It Matters |
|--------|--------|----------------|
| SEC form clustering | EDGAR | S-4s, SC TO-Ts, 13-D amendments cluster before deals |
| Corporate jet co-routing | OpenSky Network / FAA | Executives meet secretly; planes don't lie |
| Job posting freeze | Indeed / LinkedIn | Targets freeze hiring 3–6 weeks before close |
| Patent cross-licensing | USPTO | IP agreements often precede full acquisitions |
| Law firm lateral hires | State bar / LinkedIn | M&A lawyers move before deals, not during |
| Lobbying disclosure spikes | Senate LBDB | Regulatory prep happens before announcement |

None of this is insider information. All of it is public. Nobody has aggregated it systematically.

---

## Architecture

```
collectors/          → Pull raw signals from public APIs
  sec_edgar.py       → EDGAR full-text search, filing parse
  flight_tracker.py  → OpenSky Network corporate jet tracking
  job_postings.py    → Job freeze detection
  patent_signals.py  → USPTO assignment API

signals/             → Feature engineering on raw data
  features.py        → Per-company signal vectors
  scoring.py         → Composite anomaly scores

models/              → ML prediction layer
  predictor.py       → XGBoost model: P(acquisition | signals)
  trainer.py         → Training on historical deal data

api/                 → Serve predictions
  routes/predictions.py  → GET /predict/{ticker}
  routes/signals.py      → GET /signals/{ticker}

scripts/
  backtest.py        → Validate signal lead time on historical deals
  ingest_historical.py → Seed DB with known deals for training
```

---

## Quickstart

```bash
cp .env.example .env
docker-compose up -d
pip install -r requirements.txt

# Seed historical deals and train
python scripts/ingest_historical.py
python scripts/train.py

# Run the API
uvicorn src.api.main:app --reload
```

**Endpoints:**
- `GET /predict/{ticker}` — Current M&A probability score
- `GET /signals/{ticker}` — Raw signal breakdown
- `GET /watchlist` — Top 20 highest-scoring companies right now
- `GET /alerts` — New high-score crossings in last 24h

---

## Revenue Model

1. **License the dataset** — Quant funds pay $1–10M/year for edge. One confirmed signal sells the product.
2. **Signal API** — SaaS, $50K–500K/year per fund. Scales to zero marginal cost.
3. **Run your own book** — Use the model yourself. No ceiling.

---

## Legal Status

All data sources are public. The legal theory is identical to how firms like Quiver Quantitative, Thinknum, and Eagle Alpha operate. MNPI (material non-public information) law applies to information from insiders — not to pattern recognition on public filings.

We are not trading on tips. We are trading on math.
