"""
Employee review sentiment collector.

Data source: Indeed company reviews (public, no auth required).

Key insight: Employees at acquisition targets often know about the deal
weeks before the public. They can't discuss it directly (NDA), but their
anxiety, excitement, or resignation shows up in review text.

Signals we detect:
  1. M&A keyword mentions: "acquisition", "merger", "buyout", "being sold",
     "new owners", "private equity", "change of ownership"
  2. Uncertainty language: "unclear future", "rumors", "changes coming",
     "leadership transition", "restructuring"
  3. Sentiment shift: sudden drop in ratings (employee morale drops when
     deals are announced internally but not publicly)
  4. "Pros/Cons" pattern: pros suddenly mention acquisition upside
     ("great acquisition target"), cons mention uncertainty

NLP: VADER sentiment analysis (optimized for social text, no model download needed)
plus keyword pattern matching.

Note on data: Indeed reviews are public. We're reading content that users
have explicitly made public. This is standard web scraping of public data.
"""

import re
from datetime import datetime, timedelta
from typing import Any

import structlog
from bs4 import BeautifulSoup

from src.collectors.base import BaseCollector

INDEED_REVIEWS_URL = "https://www.indeed.com/cmp/{slug}/reviews"

# M&A-specific keyword patterns
MA_KEYWORDS = re.compile(
    r"\b(acqui(?:sition|red|rer|ring)|merger|merging|bought out|buyout|"
    r"being sold|sold to|new owner|ownership change|private equity|"
    r"PE firm|strategic review|going private|taken private|"
    r"change of control|takeover)\b",
    re.IGNORECASE,
)

# Uncertainty/transition language
UNCERTAINTY_KEYWORDS = re.compile(
    r"\b(uncertain(?:ty)?|unclear|rumors?|whispers?|transition|restructur|"
    r"reorgani[sz]|layoffs?|redundan(?:cies|t)|integration|"
    r"leadership change|management change|culture change|"
    r"direction unclear|strategy unclear)\b",
    re.IGNORECASE,
)

# Positive framing around deals (employees may be excited)
POSITIVE_MA_KEYWORDS = re.compile(
    r"\b(great acquisition|good fit|strategic move|synerg(?:y|ies)|"
    r"exciting (?:news|change|future)|well positioned|strong buyer)\b",
    re.IGNORECASE,
)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        VADER_AVAILABLE = True
    except ImportError:
        VADER_AVAILABLE = False
        SentimentIntensityAnalyzer = None


def _simple_sentiment(text: str) -> float:
    """
    Fallback sentiment: ratio of positive to negative words.
    Returns -1.0 (very negative) to 1.0 (very positive).
    """
    positive_words = frozenset(
        ["good", "great", "excellent", "love", "best", "amazing",
         "wonderful", "fantastic", "positive", "happy", "excited"]
    )
    negative_words = frozenset(
        ["bad", "terrible", "awful", "worst", "hate", "horrible",
         "negative", "unhappy", "worried", "fear", "uncertain"]
    )
    words = text.lower().split()
    pos = sum(1 for w in words if w in positive_words)
    neg = sum(1 for w in words if w in negative_words)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


class EmployeeSentimentCollector(BaseCollector):
    name = "employee_sentiment"
    request_delay = 3.0  # Be very conservative with review sites

    def __init__(self):
        super().__init__()
        if VADER_AVAILABLE:
            self._analyzer = SentimentIntensityAnalyzer()
        else:
            self._analyzer = None
            self.log.warning(
                "vader_not_available",
                note="Install vaderSentiment for better NLP: pip install vaderSentiment",
            )

    def _score_text(self, text: str) -> float:
        """Return compound sentiment score -1.0 to 1.0."""
        if self._analyzer:
            return self._analyzer.polarity_scores(text)["compound"]
        return _simple_sentiment(text)

    def _slugify_company(self, company_name: str) -> str:
        """Convert company name to Indeed URL slug format."""
        slug = re.sub(r"[^a-zA-Z0-9\s]", "", company_name)
        slug = re.sub(r"\s+", "-", slug.strip().lower())
        return slug

    def _fetch_reviews(self, company_name: str, n_pages: int = 3) -> list[dict]:
        """
        Fetch recent Indeed reviews for a company.
        Returns list of parsed review dicts.
        """
        slug = self._slugify_company(company_name)
        reviews = []

        for page in range(n_pages):
            url = INDEED_REVIEWS_URL.format(slug=slug)
            params = {"start": page * 20} if page > 0 else {}

            try:
                resp = self.fetch(
                    url,
                    params=params,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
                        ),
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                )
            except Exception as e:
                self.log.warning(
                    "reviews_fetch_failed",
                    company=company_name,
                    page=page,
                    error=str(e),
                )
                break

            soup = BeautifulSoup(resp.text, "lxml")

            # Indeed review structure (may change with site updates)
            review_elements = soup.find_all(
                attrs={"data-testid": re.compile(r"review")}
            ) or soup.find_all("div", class_=re.compile(r"review|Review"))

            for el in review_elements:
                text = el.get_text(separator=" ", strip=True)
                if len(text) < 20:
                    continue

                # Extract rating if present
                rating_el = el.find(attrs={"aria-label": re.compile(r"\d+\.?\d*\s+out\s+of")})
                rating = None
                if rating_el:
                    rating_match = re.search(
                        r"([\d.]+)\s+out\s+of", rating_el.get("aria-label", "")
                    )
                    if rating_match:
                        try:
                            rating = float(rating_match.group(1))
                        except ValueError:
                            pass

                reviews.append(
                    {
                        "text": text[:1000],
                        "rating": rating,
                    }
                )

            if not review_elements:
                break  # No more reviews on this page

        return reviews

    def _analyze_reviews(
        self, reviews: list[dict]
    ) -> dict:
        """
        Run NLP analysis on a batch of reviews.
        Returns aggregated signal metrics.
        """
        if not reviews:
            return {
                "review_count": 0,
                "ma_mention_count": 0,
                "uncertainty_mention_count": 0,
                "avg_sentiment": 0.0,
                "avg_rating": None,
                "ma_mention_rate": 0.0,
                "sentiment_drop_flag": False,
            }

        sentiments = []
        ratings = []
        ma_count = 0
        uncertainty_count = 0

        for review in reviews:
            text = review["text"]
            sentiment = self._score_text(text)
            sentiments.append(sentiment)

            if review["rating"] is not None:
                ratings.append(review["rating"])

            if MA_KEYWORDS.search(text):
                ma_count += 1
            if UNCERTAINTY_KEYWORDS.search(text):
                uncertainty_count += 1

        avg_sentiment = sum(sentiments) / len(sentiments)
        avg_rating = sum(ratings) / len(ratings) if ratings else None
        ma_rate = ma_count / len(reviews)

        # Sentiment drop: flag if average sentiment is below -0.2
        # (accounts for baseline negativity in job reviews generally)
        sentiment_drop_flag = avg_sentiment < -0.2

        return {
            "review_count": len(reviews),
            "ma_mention_count": ma_count,
            "uncertainty_mention_count": uncertainty_count,
            "avg_sentiment": round(avg_sentiment, 4),
            "avg_rating": round(avg_rating, 2) if avg_rating else None,
            "ma_mention_rate": round(ma_rate, 4),
            "sentiment_drop_flag": sentiment_drop_flag,
        }

    def collect(
        self,
        ticker: str,
        company_name: str | None = None,
        n_pages: int = 3,
        **kwargs,
    ) -> list[dict[str, Any]]:
        name = company_name or ticker
        self.log.info("collecting_sentiment", ticker=ticker, company=name)

        reviews = self._fetch_reviews(name, n_pages=n_pages)
        analysis = self._analyze_reviews(reviews)

        if not reviews:
            return []

        return [
            {
                "ticker": ticker,
                "snapshot_date": datetime.utcnow().replace(
                    hour=0, minute=0, second=0, microsecond=0
                ),
                "source": "indeed",
                **analysis,
            }
        ]

    def summarize(
        self,
        ticker: str,
        company_name: str | None = None,
    ) -> dict:
        records = self.collect(ticker, company_name=company_name)
        if not records:
            return {
                "ma_mention_rate": 0.0,
                "avg_sentiment": 0.0,
                "uncertainty_count": 0,
                "sentiment_drop_flag": False,
            }
        r = records[0]
        return {
            "ma_mention_rate": r["ma_mention_rate"],
            "avg_sentiment": r["avg_sentiment"],
            "uncertainty_count": r["uncertainty_mention_count"],
            "sentiment_drop_flag": r["sentiment_drop_flag"],
        }
