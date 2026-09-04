from datetime import date, datetime, timedelta
import math
import statistics

import requests

TRADING_DAYS_PER_YEAR = 252
FIVE_MINUTE_INTERVALS_PER_DAY = 78
FIVE_MINUTE_INTERVALS_PER_YEAR = (
    TRADING_DAYS_PER_YEAR * FIVE_MINUTE_INTERVALS_PER_DAY
)

def _get_regular_session_returns(data):
    session_bars = {}
    session_open = datetime.strptime("09:30", "%H:%M").time()
    session_close = datetime.strptime("16:00", "%H:%M").time()

    for bar in data:
        timestamp = datetime.fromisoformat(bar["datetime"])
        if session_open <= timestamp.time() <= session_close:
            session_bars.setdefault(timestamp.date(), []).append((timestamp, bar))

    sessions = []
    for bars in session_bars.values():
        bars.sort(key=lambda item: item[0])
        returns = []
        for (previous_time, previous_bar), (current_time, current_bar) in zip(
            bars, bars[1:]
        ):
            if current_time - previous_time != timedelta(minutes=5):
                continue
            returns.append(math.log(current_bar["close"] / previous_bar["close"]))
        if returns:
            sessions.append(returns)

    return sessions


def get_volatility(ticker: str, date: date, num_days_back: int):
    start_date = date

    for _ in range(num_days_back):
        start_date -= timedelta(days=1)

        while start_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            start_date -= timedelta(days=1)
    r = requests.get(f"https://www.hfmarketdata.io/v1/bars/stock/{ticker}?timeframe=5min&start={start_date.isoformat()}&end={date}&order=asc&format=json")
    r.raise_for_status()
    response_data = r.json()
    data = response_data["data"]
    sessions = _get_regular_session_returns(data)
    returns = [r_t for session in sessions for r_t in session]
    if len(returns) < 2:
        raise ValueError("At least two regular-session returns are required")

    std_dev_5m = statistics.stdev(returns)
    return std_dev_5m * math.sqrt(FIVE_MINUTE_INTERVALS_PER_YEAR)



def get_daily_return(ticker: str, date: date):
    r = requests.get(f"https://www.hfmarketdata.io/v1/bars/stock/{ticker}?timeframe=1day&start={date.isoformat()}&order=asc&format=json&limit=1")
    r.raise_for_status()
    response_data = r.json()
    data = response_data["data"]
    if not data:
        raise ValueError(f"No daily data returned for {ticker}")

    return math.log(data[0]["close"]/data[0]["open"])

def get_cum_return(ticker: str, date: date, num_days_back: int):
    start_date = date

    for _ in range(num_days_back):
        start_date -= timedelta(days=1)

        while start_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            start_date -= timedelta(days=1)
    r = requests.get(f"https://www.hfmarketdata.io/v1/bars/stock/{ticker}?timeframe=5min&start={start_date.isoformat()}&end={date}&order=asc&format=json")
    r.raise_for_status()
    response_data = r.json()
    data = response_data["data"]
    if not data:
        raise ValueError(f"No intraday data returned for {ticker}")

    return math.log(data[-1]["close"]/data[0]["open"])

def get_trading_volume(ticker: str, date: date):
    r = requests.get(f"https://www.hfmarketdata.io/v1/bars/stock/{ticker}?timeframe=1day&start={date.isoformat()}&order=asc&format=json&limit=1")
    r.raise_for_status()
    response_data = r.json()
    data = response_data["data"]
    if not data:
        raise ValueError(f"No daily data returned for {ticker}")

    return data[0]["volume"]

def get_volatility_of_volatility(ticker: str, date: date, num_days_back: int, num_5min_windows: int):
    if num_5min_windows < 2:
        raise ValueError("num_5min_windows must be at least 2")

    start_date = date

    for _ in range(num_days_back):
        start_date -= timedelta(days=1)

        while start_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            start_date -= timedelta(days=1)
    r = requests.get(f"https://www.hfmarketdata.io/v1/bars/stock/{ticker}?timeframe=5min&start={start_date.isoformat()}&end={date}&order=asc&format=json")
    r.raise_for_status()
    response_data = r.json()
    data = response_data["data"]
    std_devs = []

    for session in _get_regular_session_returns(data):
        for start in range(0, len(session) - num_5min_windows + 1, num_5min_windows):
            window = session[start:start + num_5min_windows]
            std_dev_5m = statistics.stdev(window)
            std_devs.append(std_dev_5m * math.sqrt(FIVE_MINUTE_INTERVALS_PER_YEAR))

    if len(std_devs) < 2:
        raise ValueError(
            "At least two complete regular-session windows are required"
        )

    return statistics.stdev(std_devs)
