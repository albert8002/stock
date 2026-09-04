import os
import logging
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import numpy as np
# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlphaVantageSentimentExtractor:
    """
    Extracts and aggregates news sentiment features for a specific ticker
    over a designated timeframe using the Alpha Vantage API.
    """

    BASE_URL: str = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None, min_relevance_score: float = 0.20):
        """
        :param api_key: Alpha Vantage API key. If None, fetches ALPHA_VANTAGE_API_KEY from .env
        :param min_relevance_score: Minimum relevance score threshold (0.0 to 1.0)
        """
        # Fallback to environment variable if api_key is not passed directly
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self.min_relevance_score = min_relevance_score

        if not self.api_key:
            raise ValueError(
                "API Key not found. Please pass an api_key or set ALPHA_VANTAGE_API_KEY in your .env file.")

    def timeconvert(self, time_str: str) -> str:
        "converts yyyy-mm-dd to yyyymmddT0000"
        return time_str.replace("-", "") + "T0000"

    def fetch_sentiment_features(
            self,
            ticker: str,
            start_time: str,
            end_time: str,
            limit: int = 1000
    ) -> Dict[str, Any]:
        start_time = self.timeconvert(start_time)
        end_time = self.timeconvert(end_time)
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": start_time,
            "time_to": end_time,
            "limit": limit,
            "apikey": self.api_key
        }

        features: Dict[str, Any] = {
            "ticker": ticker,
            "start_time": start_time,
            "end_time": end_time,
            "total_relevant_articles": 0,
            "positive_articles": 0,
            "negative_articles": 0,
            "neutral_articles": 0,
            "avg_relevance_score": 0.0,
            "avg_sentiment_score": 0.0,
            "net_sentiment_ratio": 0.0
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logger.error(f"HTTP request error fetching sentiment for {ticker}: {e}")
            return features

        if "Error Message" in data:
            logger.error(f"Alpha Vantage API error: {data['Error Message']}")
            return features

        if "Information" in data and "rate limit" in data["Information"].lower():
            logger.warning("Alpha Vantage API rate limit hit.")
            return features

        articles: List[Dict[str, Any]] = data.get("feed", [])
        if not articles:
            return features

        total_relevance = 0.0
        total_sentiment = 0.0

        for article in articles:
            ticker_meta = next(
                (t for t in article.get("ticker_sentiment", []) if t.get("ticker") == ticker),
                None
            )

            if not ticker_meta:
                continue

            rel_score = float(ticker_meta.get("relevance_score", 0))
            if rel_score < self.min_relevance_score:
                continue

            sent_score = float(ticker_meta.get("ticker_sentiment_score", 0))
            label = ticker_meta.get("ticker_sentiment_label", "")

            features["total_relevant_articles"] += 1
            total_relevance += rel_score
            total_sentiment += sent_score

            if "Bullish" in label:
                features["positive_articles"] += 1
            elif "Bearish" in label:
                features["negative_articles"] += 1
            else:
                features["neutral_articles"] += 1

        total = features["total_relevant_articles"]
        if total > 0:
            features["avg_relevance_score"] = round(total_relevance / total, 4)
            features["avg_sentiment_score"] = round(total_sentiment / total, 4)

            pos = features["positive_articles"]
            neg = features["negative_articles"]
            features["net_sentiment_ratio"] = round((pos - neg) / total, 4)

        output_array = np.array([
            features["total_relevant_articles"],
            features["positive_articles"],
            features["negative_articles"],
            features["neutral_articles"],
            features["avg_relevance_score"],
            features["avg_sentiment_score"],
            features["net_sentiment_ratio"]
        ])
        return output_array


# Example usage
'''
if __name__ == "__main__":
    extractor = AlphaVantageSentimentExtractor()
    
    features = extractor.fetch_sentiment_features(
        ticker="TSLA",
        start_time="20240101T0000",
        end_time="20240107T2359"
    )
    
    print(features)'''