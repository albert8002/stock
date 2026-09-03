from datetime import date, timedelta
import math
import statistics

import requests

def get_volatility(ticker: str, date: date, num_days_back: int):
    start_date = date - timedelta(days=num_days_back)
    r = requests.get(f"https://www.hfmarketdata.io/v1/bars/stock/{ticker}?timeframe=5min&start={start_date.isoformat()}&end={date}&order=asc&format=json")
    r.raise_for_status()
    response_data = r.json()
    data = response_data["data"]
    returns = []
    for i in range(0, len(data)-1):
        r_t = math.log(data[i+1]["close"]/data[i]["close"])
        returns.append(r_t)

    if len(returns) == 0:
        return 0.0

    std_dev_5m = statistics.pstdev(returns)
    return std_dev_5m * math.sqrt(19656)

def get_daily_return(ticker: str, date: date):
    r = requests.get(f"https://www.hfmarketdata.io/v1/bars/stock/{ticker}?timeframe=1day&start={date.isoformat()}&order=asc&format=json&limit=1")
    r.raise_for_status()
    response_data = r.json()
    data = response_data["data"]
    return math.log(data[0]["close"]/data[0]["open"])

def get_cum_return(ticker: str, date: date, num_days_back: int):
    start_date = date - timedelta(days=num_days_back)
    r = requests.get(f"https://www.hfmarketdata.io/v1/bars/stock/{ticker}?timeframe=5min&start={start_date.isoformat()}&end={date}&order=asc&format=json")
    r.raise_for_status()
    response_data = r.json()
    data = response_data["data"]
    return math.log(data[-1]["close"]/data[0]["open"])

def get_trading_volume(ticker: str, date: date):
    r = requests.get(f"https://www.hfmarketdata.io/v1/bars/stock/{ticker}?timeframe=1day&start={date.isoformat()}&order=asc&format=json&limit=1")
    r.raise_for_status()
    response_data = r.json()
    data = response_data["data"]
    return data[0]["volume"]

def get_volatility_of_volatility(ticker: str, date: date, num_days_back: int, window_size_5min: int):
    start_date = date - timedelta(days=num_days_back)
    r = requests.get(f"https://www.hfmarketdata.io/v1/bars/stock/{ticker}?timeframe=5min&start={start_date.isoformat()}&end={date}&order=asc&format=json")
    r.raise_for_status()
    response_data = r.json()
    data = response_data["data"]
    returns = []
    for i in range(0, len(data)-1):
        r_t = math.log(data[i+1]["close"]/data[i]["close"])
        returns.append(r_t)

    if len(returns) == 0:
        return 0.0

    std_dev_5m = statistics.pstdev(returns)
    return std_dev_5m * math.sqrt(19656)
# print(get_past_volatility("AAPL", date.today() - timedelta(days=365 * 2), 4))
# print(get_daily_return("AAPL", date.today() - timedelta(days=31)))

print(get_trading_volume("AAPL", date.today() - timedelta(days=31)))