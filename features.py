from datetime import datetime, timedelta

import numpy as np

from earnings_dividends import get_seconds_until_next_dividend, get_seconds_until_next_earnings
from general_stock_data import get_cum_return, get_daily_return, get_trading_volume, get_volatility
from macro_events import MacroEvent, get_seconds_until_macro_event
from sentiment_features import AlphaVantageSentimentExtractor
from stock_and_market import MarketReferences, StockData

def get_features():
    stock_data = StockData()
    market_references = MarketReferences()
    sentiment_extractor = AlphaVantageSentimentExtractor()
    
    date = datetime(2024, 6, 4)
    ticker = "AAPL"
    market_ticker = "SPY"
    past_volitilities = np.array([get_volatility(ticker, date, n) for n in [1, 3, 7, 15, 30]])
    past_daily_returns = np.array([get_daily_return(ticker, date - timedelta(days=n)) for n in [1, 2, 3]])
    past_timespan_returns = np.array([get_cum_return(ticker, date, n) for n in [7, 30]])
    trading_volumes = np.array([get_trading_volume(ticker, date - timedelta(days=n)) for n in [1, 2, 3]])
    
    entropy_price_datas = np.array([stock_data.get_price_data(ticker, (date - timedelta(days=1)).isoformat(), date.isoformat()) for n in [7, 30]])
    stock_entropies = np.array([stock_data.calculate_entropy(entropy_price_datas)])
    
    stock_market_comparisons = np.array([market_references.compare_to_market(ticker, (date - timedelta(days=n)).isoformat(), date.isoformat()) for n in [2, 5, 14]])
    
    market_daily_returns = np.array([get_daily_return(market_ticker, date - timedelta(days=n)) for n in [1, 2, 3]])
    market_timespan_returns = np.array([get_cum_return(market_ticker, date, n) for n in [7, 30]])
    
    entropy_price_datas = np.array([stock_data.get_price_data(market_ticker, (date - timedelta(days=1)).isoformat(), date.isoformat()) for n in [7, 30]])
    stock_entropies = np.array([stock_data.calculate_entropy(entropy_price_datas)])
    
    seconds_until_earnings = get_seconds_until_next_earnings(ticker, date)
    seconds_until_dividends = get_seconds_until_next_dividend(ticker, date)
    
    seconds_until_macro_event = [
        get_seconds_until_macro_event(event, date)
        for event in [
            MacroEvent.CPI,
            MacroEvent.EMPLOYMENT,
            MacroEvent.PPI,
            MacroEvent.JOLTS,
            MacroEvent.GDP,
            MacroEvent.PCE,
            MacroEvent.FOMC,
        ]
    ]
    
    sentiments = np.array([sentiment_extractor.fetch_sentiment_features(ticker, (date - timedelta(days=n)).isoformat(),date.isoformat()) for n in [3, 30]])
    
    