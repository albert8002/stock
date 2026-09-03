from enum import Enum
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import json
import re

import requests
from bs4 import BeautifulSoup


DB_PATH = "macro_events.json"
BLS_FOLDER = Path("macro_data")

UTC = ZoneInfo("UTC")
EASTERN = ZoneInfo("America/New_York")


class MacroEvent(Enum):
    CPI = "CPI"
    EMPLOYMENT = "EMPLOYMENT"
    PPI = "PPI"
    JOLTS = "JOLTS"
    GDP = "GDP"
    PCE = "PCE"
    FOMC = "FOMC"


# -------------------------------------------------------------------
# BLS
# -------------------------------------------------------------------

BLS_NAMES = {
    "Consumer Price Index": MacroEvent.CPI,
    "Employment Situation": MacroEvent.EMPLOYMENT,
    "Producer Price Index": MacroEvent.PPI,
    "Job Openings and Labor Turnover Survey": MacroEvent.JOLTS,
}


def parse_bls_file(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    events = []

    for row in soup.find_all("tr"):
        cells = [
            cell.get_text(" ", strip=True)
            for cell in row.find_all(["td", "th"])
        ]

        if len(cells) < 3:
            continue

        date_text = cells[0]
        time_text = cells[1]
        release_name = cells[2]

        event_type = None

        for official_name, macro_event in BLS_NAMES.items():
            if official_name.lower() in release_name.lower():
                event_type = macro_event
                break

        if event_type is None:
            continue

        try:
            dt = datetime.strptime(
                f"{date_text} {time_text}",
                "%A, %B %d, %Y %I:%M %p"
            )
        except ValueError:
            continue

        dt = dt.replace(
            tzinfo=EASTERN
        ).astimezone(UTC)

        events.append({
            "event": event_type.value,
            "timestamp_utc": dt.isoformat(),
            "source": "BLS"
        })

    return events


def parse_all_bls_files() -> list[dict]:
    events = []

    paths = (
        list(BLS_FOLDER.glob("*.html")) +
        list(BLS_FOLDER.glob("*.htm"))
    )

    print(f"Found {len(paths)} BLS files")

    for path in sorted(paths):
        print(f"Parsing {path}")

        parsed = parse_bls_file(path)

        print(f"Found {len(parsed)} events in {path}")

        events.extend(parsed)

    print(f"Total BLS events: {len(events)}")

    return events
# -------------------------------------------------------------------
# BEA
# -------------------------------------------------------------------

BEA_RELEASE_DATES_URL = (
    "https://apps.bea.gov/API/signup/release_dates.json"
)


def fetch_bea_events() -> list[dict]:
    """
    Fetches the dates currently exposed by BEA's official
    machine-readable release calendar.
    """

    response = requests.get(BEA_RELEASE_DATES_URL, timeout=30)
    response.raise_for_status()

    data = response.json()

    events = []

    mapping = {
        "Gross Domestic Product": MacroEvent.GDP,
        "Personal Income and Outlays": MacroEvent.PCE,
    }

    for bea_name, event_type in mapping.items():

        release = data.get(bea_name)

        if not release:
            continue

        for timestamp in release["release_dates"]:
            dt = datetime.fromisoformat(timestamp).astimezone(UTC)

            events.append({
                "event": event_type.value,
                "timestamp_utc": dt.isoformat(),
                "source": "BEA"
            })

    return events


# -------------------------------------------------------------------
# FOMC
# -------------------------------------------------------------------

FOMC_URL = (
    "https://www.federalreserve.gov/"
    "monetarypolicy/fomccalendars.htm"
)


def fetch_fomc_events(start_year: int, end_year: int) -> list[dict]:
    response = requests.get(
        FOMC_URL,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text("\n", strip=True)

    events = []

    month_names = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }

    for year in range(start_year, end_year + 1):

        # Isolate this year's section
        year_match = re.search(
            rf"{year}\s+FOMC Meetings(.*?)(?=\d{{4}}\s+FOMC Meetings|\Z)",
            text,
            re.DOTALL
        )

        if not year_match:
            continue

        section = year_match.group(1)

        # Matches things such as:
        # June
        # 11-12*
        #
        # or
        # Apr/May
        # 30-1

        pattern = re.compile(
            r"(January|February|March|April|May|June|July|August|"
            r"September|October|November|December|Apr/May)"
            r"\s+"
            r"(\d{1,2})-(\d{1,2})\*?"
        )

        for match in pattern.finditer(section):

            month_name = match.group(1)
            first_day = int(match.group(2))
            second_day = int(match.group(3))

            if month_name == "Apr/May":
                # Example: Apr 30 - May 1
                decision_month = 5
            else:
                decision_month = month_names[month_name]

                # Handle a meeting crossing into the next month
                if second_day < first_day:
                    decision_month += 1

            decision_year = year

            if decision_month == 13:
                decision_month = 1
                decision_year += 1

            # FOMC policy statement is released at 2:00 PM ET
            dt = datetime(
                decision_year,
                decision_month,
                second_day,
                14,
                0,
                tzinfo=EASTERN
            ).astimezone(UTC)

            events.append({
                "event": MacroEvent.FOMC.value,
                "timestamp_utc": dt.isoformat(),
                "source": "FED"
            })

    return events


# -------------------------------------------------------------------
# BUILD JSON DATABASE
# -------------------------------------------------------------------

def build_macro_event_database(
    start_year: int = 2021,
    end_year: int | None = None
):
    if end_year is None:
        end_year = datetime.now().year

    events = []

    print("Loading BLS events from local HTML files...")
    events.extend(parse_all_bls_files())

    print("Downloading BEA events...")
    events.extend(fetch_bea_events())

    print("Downloading FOMC events...")
    events.extend(fetch_fomc_events(start_year, end_year))

    # Remove duplicates
    unique = {}

    for event in events:
        key = (event["event"], event["timestamp_utc"])
        unique[key] = event

    events = list(unique.values())

    # Sort chronologically
    events.sort(key=lambda x: x["timestamp_utc"])

    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)

    print(f"Saved {len(events)} events to {DB_PATH}")

# -------------------------------------------------------------------
# QUERY FUNCTION
# -------------------------------------------------------------------

def get_seconds_until_macro_event(
    event: MacroEvent,
    current_time: datetime | None = None
) -> int | None:

    if current_time is None:
        current_time = datetime.now(UTC)

    if current_time.tzinfo is None:
        raise ValueError("current_time must be timezone-aware")

    current_time = current_time.astimezone(UTC)

    with open(DB_PATH, "r", encoding="utf-8") as f:
        database = json.load(f)

    next_event_time = None

    for entry in database:
        if entry["event"] != event.value:
            continue

        event_time = datetime.fromisoformat(
            entry["timestamp_utc"]
        ).astimezone(UTC)

        if event_time < current_time:
            continue

        if next_event_time is None or event_time < next_event_time:
            next_event_time = event_time

    if next_event_time is None:
        return None

    return int(
        (next_event_time - current_time).total_seconds()
    )

# -------------------------------------------------------------------
# EXAMPLE
# -------------------------------------------------------------------

if __name__ == "__main__":

    # build_macro_event_database(
    #     start_year=2024,
    #     end_year=2026
    # )

    test_time = datetime(
        2024,
        6,
        4,
        12,
        0,
        tzinfo=EASTERN
    )

    seconds = get_seconds_until_macro_event(
        MacroEvent.FOMC,
        test_time
    )
    
    """    
    CPI = "CPI"
    EMPLOYMENT = "EMPLOYMENT"
    PPI = "PPI"
    JOLTS = "JOLTS"
    GDP = "GDP"
    PCE = "PCE"
    FOMC = "FOMC"""

    print(seconds)