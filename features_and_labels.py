from datetime import datetime, timedelta

import numpy as np

from earnings_dividends import get_seconds_until_next_dividend, get_seconds_until_next_earnings
from general_stock_data import get_cum_return, get_daily_return, get_trading_volume, get_volatility
from macro_events import MacroEvent, get_seconds_until_macro_event
from sentiment_features import AlphaVantageSentimentExtractor
from stock_and_market import MarketReferences, StockData

def get_features(ticker: str, date: datetime):
    stock_data = StockData()
    market_references = MarketReferences()
    sentiment_extractor = AlphaVantageSentimentExtractor()


    market_ticker = "SPY"

    def get_start_date(date_input: datetime, num_days_back: int):
        start_date = date_input

        for _ in range(num_days_back):
            start_date -= timedelta(days=1)

            while start_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                start_date -= timedelta(days=1)
        return start_date

    past_volitilities = np.array([get_volatility(ticker, date, n) for n in [1, 3, 7, 15, 30]])
    past_daily_returns = np.array([get_daily_return(ticker, get_start_date(date, n)) for n in [1, 2, 3]])
    past_timespan_returns = np.array([get_cum_return(ticker, date, n) for n in [7, 30]])
    trading_volumes = np.array([get_trading_volume(ticker, get_start_date(date, n)) for n in [1, 2, 3]])
    
    entropy_price_datas = [stock_data.get_price_data(ticker, (get_start_date(date, n)).isoformat(), date.isoformat()) for n in [7, 30]]
    stock_entropies = np.array([stock_data.calculate_entropy(entropy_price_datas[n]) for n in [0, 1]])
    
    stock_market_comparisons = np.array([market_references.compare_to_market(ticker, (get_start_date(date, n)).isoformat(), date.isoformat()) for n in [2, 5, 14]])
    
    market_daily_returns = np.array([get_daily_return('SPX', get_start_date(date, n)) for n in [1, 2, 3]])
    market_timespan_returns = np.array([get_cum_return('SPX', date, n) for n in [7, 30]]) #TODO
    
    market_entropy_price_datas = [stock_data.get_price_data(market_ticker, (get_start_date(date, n)).isoformat(), date.isoformat()) for n in [7, 30]]
    market_entropies = np.array([stock_data.calculate_entropy(market_entropy_price_datas[n]) for n in [0, 1]])
    
    seconds_until_earnings = get_seconds_until_next_earnings(ticker, date)
    seconds_until_dividends = get_seconds_until_next_dividend(ticker, date)

    volatitlity_of_volatility = np.array([stock_data.get_volatility_of_volatility(ticker, date, n, 20) for n in [7, 30]])

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
    
    sentiments = np.array([sentiment_extractor.fetch_sentiment_features(ticker, (get_start_date(date, n)).isoformat(),date.isoformat()) for n in [3, 30]])

    compound_features = np.concatenate([
        np.array(past_volitilities),
        np.array(past_daily_returns),
        np.array(past_timespan_returns),
        np.array(trading_volumes),
        np.array(stock_entropies),
        np.array(stock_market_comparisons),
        np.array(market_entropies),
        np.array(market_daily_returns),
        np.array(market_timespan_returns),
        np.array(seconds_until_earnings),
        np.array(seconds_until_dividends),
        np.array(seconds_until_macro_event),
        np.array(volatitlity_of_volatility),
        sentiments.flatten()
    ])
    return compound_features
    

def get_labels(ticker: str, date: datetime):
    def get_end_date(date_input: datetime, num_days_back: int):
        end_date = date_input

        for _ in range(num_days_back):
            end_date += timedelta(days=1)

            while end_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                end_date += timedelta(days=1)
        return end_date

    future_volatilities = np.array([get_volatility(ticker, get_end_date(date, n), n) for n in [3, 7, 15, 30]])
    return future_volatilities