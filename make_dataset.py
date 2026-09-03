"""
make_dataset.py
===============

The driver that ties everything together: Alpaca -> surfaces -> feature panel.

    python make_dataset.py AAPL --from 2024-01-02 --to 2024-12-31 --out aapl.csv

Files must sit in one directory:

    option_data.py   (yours -- not called directly)
    iv_surface.py    IVSurface.build()  -> one Surface per day
    iv_features.py   FeatureBuilder     -> one feature row per Surface
    make_dataset.py  this file          -> loops days, assembles the panel

Surfaces are cached to disk per day, so a re-run after a crash or a change to
the feature code does not re-pull anything from the API.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import pandas as pd

from iv_surface import IVSurface
from iv_features import (
    FeatureBuilder,
    add_lags,
    add_targets,
    build_panel,
    clean_for_model,
)

# US equity options: 09:30-16:00 ET. In UTC that is 14:30-21:00 (EDT) or
# 15:30-22:00 (EST); the wider window below covers both without special-casing.
SESSION_START = "13:30"
SESSION_END = "21:15"


def trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """Weekdays in range. Holidays just return no trades and get skipped."""
    return pd.bdate_range(start, end, tz="UTC")


def one_day_surface(
    iv: IVSurface,
    symbol: str,
    day: pd.Timestamp,
    min_dte: int = 7,
    max_dte: int = 400,
    model: str = "american",
    r: float = 0.045,
    q: float = 0.0,
    status: str = "inactive",
):
    """
    Build a single day's surface.

    ``status="inactive"`` is right for historical backfill, where every
    contract has since expired. If your expiry window runs past today, some
    contracts are still active and you need a second pull with status="active";
    see the note at the bottom of this file.
    """
    return iv.build(
        underlying=symbol,
        start=f"{day:%Y-%m-%d}T{SESSION_START}:00+00:00",
        end=f"{day:%Y-%m-%d}T{SESSION_END}:00+00:00",
        expiration_start=(day + pd.Timedelta(days=min_dte)).date(),
        expiration_end=(day + pd.Timedelta(days=max_dte)).date(),
        status=status,
        model=model,
        r=r,
        q=q,
    )


def collect_surfaces(
    symbol: str,
    start: str,
    end: str,
    cache_dir: str | Path = "cache",
    pause: float = 0.35,
    **kwargs,
) -> list:
    """
    One surface per trading day, cached to ``cache_dir/SYMBOL_YYYY-MM-DD.pkl``.

    Days with no trades, or that raise, are skipped rather than aborting the
    run -- a single bad day should not cost you a whole backfill.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    iv = IVSurface()
    surfaces, failures = [], []

    for day in trading_days(start, end):
        path = cache / f"{symbol}_{day:%Y-%m-%d}.pkl"

        if path.exists():
            with open(path, "rb") as fh:
                surfaces.append(pickle.load(fh))
            continue

        try:
            surf = one_day_surface(iv, symbol, day, **kwargs)
        except Exception as exc:  # noqa: BLE001 - keep the loop alive
            failures.append((day.date(), str(exc)[:80]))
            continue

        with open(path, "wb") as fh:
            pickle.dump(surf, fh)
        surfaces.append(surf)
        time.sleep(pause)  # stay under the API rate limit

    if failures:
        print(f"skipped {len(failures)} days, first few: {failures[:3]}")
    return surfaces


def spot_series(symbol: str, start: str, end: str) -> pd.Series:
    """Daily closes, used only to build price-based targets."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    iv = IVSurface()
    bars = iv.stock_client.get_stock_bars(
        StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=pd.Timestamp(start).to_pydatetime(),
            end=pd.Timestamp(end).to_pydatetime(),
        )
    ).df.reset_index()

    s = bars.set_index(pd.to_datetime(bars["timestamp"], utc=True))["close"]
    s.index = s.index.normalize()
    return s


def make_dataset(
    symbol: str,
    start: str,
    end: str,
    horizons=(1, 5, 21),
    lags=(1, 5, 21),
    cache_dir: str = "cache",
    **kwargs,
) -> pd.DataFrame:
    """
    The whole pipeline. Returns a model-ready DataFrame indexed by timestamp,
    with feature columns plus ``y_*`` target columns.
    """
    surfaces = collect_surfaces(symbol, start, end, cache_dir=cache_dir, **kwargs)
    if not surfaces:
        raise RuntimeError("no surfaces built -- check credentials, dates, and status filter")

    panel = build_panel(surfaces, FeatureBuilder(), symbol=symbol)

    # Align daily closes onto the panel's session-close timestamps.
    spot = spot_series(symbol, start, end)
    spot = spot.reindex(panel.index.normalize()).to_numpy()
    spot = pd.Series(spot, index=panel.index)

    panel = add_lags(panel, lags=lags)
    panel = add_targets(panel, spot, horizons=horizons)
    return clean_for_model(panel)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build an IV-surface feature dataset.")
    p.add_argument("symbol")
    p.add_argument("--from", dest="start", required=True)
    p.add_argument("--to", dest="end", required=True)
    p.add_argument("--out", default=None, help="CSV path")
    p.add_argument("--model", default="american", choices=["bsm", "american"])
    p.add_argument("--cache-dir", default="cache")
    args = p.parse_args()

    ds = make_dataset(
        args.symbol, args.start, args.end, cache_dir=args.cache_dir, model=args.model
    )

    feat = [c for c in ds.columns if not c.startswith(("y_", "symbol"))]
    print(f"{ds.shape[0]} rows x {len(feat)} features")
    print(f"date range: {ds.index.min()} .. {ds.index.max()}")
    print(f"target coverage: {ds['y_atm_iv_30d_chg5'].notna().sum()} usable rows")

    if args.out:
        ds.to_csv(args.out)
        print(f"wrote {args.out}")


# ---------------------------------------------------------------------------
# Note on the status filter
# ---------------------------------------------------------------------------
# Alpaca's contract endpoint filters on active/inactive, and a window that
# straddles today contains both. For a purely historical backfill (every expiry
# already past) "inactive" is correct and complete. If your window reaches into
# live contracts, run get_contracts twice and concatenate:
#
#     act = iv.get_contracts(..., status="active")
#     exp = iv.get_contracts(..., status="inactive")
#     contracts = pd.concat([act, exp]).drop_duplicates("option_symbol")
#
# then feed those symbols to get_trades directly instead of using build().