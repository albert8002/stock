import csv
from datetime import date
import json
import os

from dotenv import load_dotenv
import requests

from datetime import datetime, time
from zoneinfo import ZoneInfo

def csv_to_dict(csv_text: str) -> list:
    rows = csv.DictReader(csv_text.splitlines())
    return list(rows)


load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

UTC = ZoneInfo("UTC")
EASTERN = ZoneInfo("America/New_York")


def get_seconds_until_next_earnings(
    ticker: str,
    current_time: datetime | None = None
) -> int:

    if current_time is None:
        current_time = datetime.now(UTC)

    if current_time.tzinfo is None:
        raise ValueError("current_time must be timezone-aware")

    current_time = current_time.astimezone(UTC)

    r = requests.get(
        "https://www.alphavantage.co/query",
        params={
            "function": "EARNINGS_CALENDAR",
            "symbol": ticker,
            "horizon": "12month",
            "apikey": ALPHA_VANTAGE_API_KEY,
        },
        timeout=30,
    )

    r.raise_for_status()

    decoded_content = r.content.decode("utf-8")
    data = csv_to_dict(decoded_content)

    upcoming_earnings_times = []

    for earnings in data:
        report_date = earnings.get("reportDate", "").strip()

        if not report_date:
            continue

        event_date = datetime.fromisoformat(report_date).date()

        # Assume earnings occur at 12:00 Eastern Time.
        event_time = datetime.combine(
            event_date,
            time(12, 0),
            tzinfo=EASTERN
        ).astimezone(UTC)

        if event_time >= current_time:
            upcoming_earnings_times.append(event_time)

    if not upcoming_earnings_times:
        raise ValueError(
            f"No upcoming earnings date found for {ticker}"
        )

    next_earnings = min(upcoming_earnings_times)

    return int(
        (next_earnings - current_time).total_seconds()
    )


def get_seconds_until_next_dividend(
    ticker: str,
    current_time: datetime | None = None
) -> int:

    if current_time is None:
        current_time = datetime.now(UTC)

    if current_time.tzinfo is None:
        raise ValueError("current_time must be timezone-aware")

    current_time = current_time.astimezone(UTC)

    r = requests.get(
        "https://www.alphavantage.co/query",
        params={
            "function": "DIVIDENDS",
            "symbol": ticker,
            "apikey": ALPHA_VANTAGE_API_KEY,
        },
        timeout=30,
    )

    r.raise_for_status()
    data = r.json()

    upcoming_dividend_times = []

    for dividend in data.get("data", []):

        # Use ex-dividend date, not payment date.
        ex_dividend_date = dividend.get("ex_dividend_date")

        if (
            not ex_dividend_date
            or ex_dividend_date == "None"
        ):
            continue

        event_date = datetime.fromisoformat(
            ex_dividend_date
        ).date()

        # Ex-dividend event starts at 00:00 Eastern Time.
        event_time = datetime.combine(
            event_date,
            time(0, 0),
            tzinfo=EASTERN
        ).astimezone(UTC)

        if event_time >= current_time:
            upcoming_dividend_times.append(event_time)

    if not upcoming_dividend_times:
        raise ValueError(
            f"No upcoming dividend found for {ticker}"
        )

    next_dividend = min(upcoming_dividend_times)

    return int(
        (next_dividend - current_time).total_seconds()
    )