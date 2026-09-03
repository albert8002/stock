import csv
from datetime import date
import json
import os

from dotenv import load_dotenv
import requests


def csv_to_dict(csv_text: str) -> list:
    rows = csv.DictReader(csv_text.splitlines())
    return list(rows)


load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

def get_days_until_next_earnings(ticker: str) -> int:
    r = requests.get(
        "https://www.alphavantage.co/query",
        params={
            "function": "EARNINGS_CALENDAR",
            "symbol": ticker,
            "horizon": "12month",
            "apikey": ALPHA_VANTAGE_API_KEY,
        },
    )
    r.raise_for_status()
    decoded_content = r.content.decode('utf-8')
    data = csv_to_dict(decoded_content)

    today = date.today()
    upcoming_earnings_dates = []
    for earnings in data:
        report_date = earnings.get("reportDate", "").strip()
        if report_date:
            upcoming_earnings_dates.append(date.fromisoformat(report_date))

    upcoming_earnings_dates = [
        report_date
        for report_date in upcoming_earnings_dates
        if report_date >= today
    ]
    if not upcoming_earnings_dates:
        raise ValueError(f"No upcoming earnings date found for {ticker}")

    return (min(upcoming_earnings_dates) - today).days

def get_days_until_next_dividend(ticker: str) -> int:
    url = f"https://www.alphavantage.co/query?function=DIVIDENDS&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    today = date.today()
    upcoming_payment_dates = [
        date.fromisoformat(dividend["payment_date"])
        for dividend in data.get("data", [])
        if dividend.get("payment_date")
        and dividend["payment_date"] != "None"
        and date.fromisoformat(dividend["payment_date"]) >= today
    ]

    if not upcoming_payment_dates:
        raise ValueError(f"No upcoming dividend payment found for {ticker}")

    return (min(upcoming_payment_dates) - today).days

