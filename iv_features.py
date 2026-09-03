"""
iv_features.py
==============

Turns the ragged output of ``iv_surface.Surface`` into fixed-length, model-ready
feature vectors.

The problem this solves: a raw surface has a different number of points every
day, at whatever strikes and expiries happened to trade. That cannot go into a
model directly. The fix is to fit a smooth parametric form per expiry slice,
resample onto a *fixed* grid of standard tenors and moneyness levels, and read
off named quantities that mean the same thing on every date.

Pipeline
--------
    raw trades -> per-expiry smile fit (quadratic or SVI)
               -> total-variance interpolation onto standard tenors
               -> named features (level / skew / curvature / term structure)
               -> panel across dates, with masks and quality flags

Quick start
-----------
    from iv_surface import IVSurface
    from iv_features import FeatureBuilder, build_panel, purged_splits

    surf = IVSurface().build(...)

    fb = FeatureBuilder()
    row = fb.snapshot(surf)          # -> pd.Series, always the same index

    panel = build_panel([surf_day1, surf_day2, ...])   # -> DataFrame
    panel = add_targets(panel, spot_series, horizons=[1, 5])

Design notes
------------
* Every feature vector has an identical index regardless of input coverage.
  Gaps are NaN and carry a companion ``mask_*`` column -- they are never
  silently filled, because a fabricated ATM vol is worse than a missing one.
* Interpolation across tenors is done in total variance (w = sigma^2 * T),
  which is the arbitrage-consistent quantity, not in raw vol.
* Nothing here looks forward. Features are stamped at the close of the data
  window; targets are shifted with ``add_targets``.
"""

from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

# Standard tenors in days. Anything outside the observed range is left NaN
# rather than extrapolated -- vol surfaces do not extrapolate safely.
DEFAULT_TENORS = (7, 30, 60, 90, 180, 365)

# Fixed log-moneyness nodes, k = log(K/S).
DEFAULT_MONEYNESS = (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20)

# Deltas for risk-reversal / butterfly features.
DEFAULT_DELTAS = (0.10, 0.25)

VERBOSE = False


def set_verbose(flag: bool = True) -> None:
    """Trace feature extraction stages to stderr."""
    global VERBOSE
    VERBOSE = flag


def _log(msg: str, indent: int = 0) -> None:
    if VERBOSE:
        print(f"    {'  '*indent}{msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Per-slice smile fits
# ---------------------------------------------------------------------------

@dataclass
class SliceFit:
    """A fitted smile for one expiry."""

    tenor: float                  # years
    kind: str                     # "quadratic" | "svi" | "flat" | "failed"
    params: np.ndarray
    n_points: int
    rmse: float
    k_min: float
    k_max: float

    def iv(self, k):
        """Implied vol at log-moneyness ``k``. Scalar or array in, same out."""
        k = np.atleast_1d(np.asarray(k, dtype=float))

        if self.kind == "quadratic":
            a, b, c = self.params
            out = a + b * k + c * k * k
        elif self.kind == "svi":
            w = _svi_total_variance(k, self.params)
            out = np.sqrt(np.maximum(w, 1e-12) / self.tenor)
        elif self.kind == "flat":
            out = np.full_like(k, self.params[0])
        else:
            out = np.full_like(k, np.nan)

        out = np.where(np.isfinite(out), out, np.nan)
        return np.clip(out, 0.01, 5.0)

    def iv_in_range(self, k, pad: float = 0.02):
        """Like ``iv`` but NaN outside the fitted strike range (no extrapolation)."""
        k = np.atleast_1d(np.asarray(k, dtype=float))
        vals = self.iv(k)
        outside = (k < self.k_min - pad) | (k > self.k_max + pad)
        return np.where(outside, np.nan, vals)


def _svi_total_variance(k, params):
    """Raw SVI: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + s^2))."""
    a, b, rho, m, s = params
    km = k - m
    return a + b * (rho * km + np.sqrt(km * km + s * s))


def fit_slice(
    k: np.ndarray,
    iv: np.ndarray,
    tenor: float,
    weights: np.ndarray | None = None,
    kind: str = "quadratic",
    min_points: int = 3,
) -> SliceFit:
    """
    Fit one expiry's smile.

    ``quadratic`` is a weighted least-squares fit in log-moneyness -- cheap,
    stable, and its coefficients *are* the level/skew/curvature features.
    ``svi`` fits raw SVI in total variance, which extrapolates better and is
    closer to arbitrage-free, but needs ~5+ points and can fail to converge.
    """
    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)

    good = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    k, iv = k[good], iv[good]
    w = np.ones_like(k) if weights is None else np.asarray(weights, float)[good]
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)

    n = len(k)
    k_min = float(k.min()) if n else np.nan
    k_max = float(k.max()) if n else np.nan

    if n == 0:
        return SliceFit(tenor, "failed", np.array([]), 0, np.nan, np.nan, np.nan)

    # Not enough spread in strikes to identify a shape -> level only.
    if n < min_points or np.ptp(k) < 1e-6:
        level = float(np.average(iv, weights=w))
        return SliceFit(tenor, "flat", np.array([level]), n, 0.0, k_min, k_max)

    if kind == "svi" and n >= 5:
        fit = _fit_svi(k, iv, tenor, w, k_min, k_max)
        if fit is not None:
            return fit
        # fall through to quadratic on failure

    # Weighted quadratic in k.
    design = np.column_stack([np.ones_like(k), k, k * k])
    sqrt_w = np.sqrt(w)
    try:
        coef, *_ = np.linalg.lstsq(design * sqrt_w[:, None], iv * sqrt_w, rcond=None)
    except np.linalg.LinAlgError:
        level = float(np.average(iv, weights=w))
        return SliceFit(tenor, "flat", np.array([level]), n, 0.0, k_min, k_max)

    resid = design @ coef - iv
    rmse = float(np.sqrt(np.average(resid**2, weights=w)))
    return SliceFit(tenor, "quadratic", coef, n, rmse, k_min, k_max)


def _fit_svi(k, iv, tenor, w, k_min, k_max) -> SliceFit | None:
    total_var = (iv**2) * tenor

    a0 = float(np.min(total_var))
    b0 = 0.1
    m0 = float(k[np.argmin(total_var)])
    s0 = max(0.1, float(np.std(k)))
    x0 = np.array([a0, b0, 0.0, m0, s0])

    def residual(p):
        return np.sqrt(w) * (_svi_total_variance(k, p) - total_var)

    bounds = (
        [-np.inf, 1e-8, -0.999, -2.0, 1e-4],
        [np.inf, 10.0, 0.999, 2.0, 5.0],
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = least_squares(residual, x0, bounds=bounds, max_nfev=2000)
    except (ValueError, RuntimeError):
        return None

    if not res.success:
        return None

    # Reject fits implying negative variance anywhere in the fitted range.
    probe = np.linspace(k_min, k_max, 25)
    if np.any(_svi_total_variance(probe, res.x) <= 0):
        return None

    model_iv = np.sqrt(np.maximum(_svi_total_variance(k, res.x), 1e-12) / tenor)
    rmse = float(np.sqrt(np.average((model_iv - iv) ** 2, weights=w)))
    return SliceFit(tenor, "svi", res.x, len(k), rmse, k_min, k_max)


# ---------------------------------------------------------------------------
# Delta-space helpers
# ---------------------------------------------------------------------------

def _call_delta(k, tenor, sigma, r=0.0, q=0.0):
    """Delta of a call at log-moneyness k (spot-normalized, S = 1, K = e^k)."""
    if tenor <= 0 or sigma <= 0:
        return np.nan
    d1 = (-k + (r - q + 0.5 * sigma * sigma) * tenor) / (sigma * math.sqrt(tenor))
    return math.exp(-q * tenor) * norm.cdf(d1)


def k_at_delta(fit: SliceFit, delta: float, is_call: bool, r=0.0, q=0.0) -> float:
    """
    Invert the fitted smile for the log-moneyness at a target delta.

    Delta and vol are mutually dependent, so this roots on the whole smile
    rather than assuming a vol. Returns NaN if the target sits outside the
    fitted range or the root cannot be bracketed.
    """
    target = delta if is_call else -delta

    def f(k):
        sigma = float(fit.iv(k)[0])
        if not np.isfinite(sigma):
            return np.nan
        cd = _call_delta(k, fit.tenor, sigma, r, q)
        if not np.isfinite(cd):
            return np.nan
        d = cd if is_call else cd - math.exp(-q * fit.tenor)  # put delta
        return d - target

    lo, hi = fit.k_min - 0.05, fit.k_max + 0.05
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return np.nan

    try:
        f_lo, f_hi = f(lo), f(hi)
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
            return np.nan
        return float(brentq(f, lo, hi, xtol=1e-6, maxiter=100))
    except (ValueError, OverflowError):
        return np.nan


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@dataclass
class FeatureBuilder:
    """
    Converts a ``Surface`` into a fixed-length feature vector.

    The output index depends only on the configuration below, never on the
    input data, so rows from different dates always align.
    """

    tenors: tuple = DEFAULT_TENORS
    moneyness: tuple = DEFAULT_MONEYNESS
    deltas: tuple = DEFAULT_DELTAS
    fit_kind: str = "quadratic"
    weight_by: str | None = "volume"      # "volume" | "n_trades" | None
    min_points_per_slice: int = 2
    include_grid: bool = True
    include_masks: bool = True
    r: float = 0.0
    q: float = 0.0

    # -- slice fitting ----------------------------------------------------

    def fit_slices(self, surface) -> dict[float, SliceFit]:
        """One fit per expiry, keyed by tenor in years."""
        pts = surface.points
        if pts is None or pts.empty:
            return {}

        pts = pts.dropna(subset=["iv", "log_moneyness", "tenor"])
        fits = {}

        for tenor, grp in pts.groupby("tenor"):
            if len(grp) < self.min_points_per_slice:
                continue
            if self.weight_by and self.weight_by in grp.columns:
                w = grp[self.weight_by].to_numpy(dtype=float)
            else:
                w = None
            fits[float(tenor)] = fit_slice(
                grp["log_moneyness"].to_numpy(),
                grp["iv"].to_numpy(),
                float(tenor),
                weights=w,
                kind=self.fit_kind,
            )

        good = {t: f for t, f in fits.items() if f.kind != "failed"}
        if VERBOSE:
            _log(f"fit {len(good)}/{len(fits)} expiry slices ({self.fit_kind}):")
            for t, f in sorted(good.items()):
                _log(f"{t*365.25:6.0f}d  {f.kind:<9} n={f.n_points:<3} "
                     f"rmse={f.rmse:.4f}  k in [{f.k_min:+.3f}, {f.k_max:+.3f}]", 1)
        return good

    # -- tenor interpolation ----------------------------------------------

    def _interp_total_variance(self, fits: dict[float, SliceFit], k: float, target_tenor: float):
        """
        Vol at (k, target_tenor), interpolated linearly in total variance.

        Returns NaN outside the observed tenor range: extrapolating a vol term
        structure is how you get 300% one-week vols in a feature column.
        """
        usable = []
        for tenor, fit in sorted(fits.items()):
            sigma = float(fit.iv_in_range(k)[0])
            if np.isfinite(sigma):
                usable.append((tenor, sigma * sigma * tenor))

        if not usable:
            return np.nan
        if len(usable) == 1:
            t0, w0 = usable[0]
            # Only accept an exact-ish tenor match when there is nothing to span.
            return math.sqrt(w0 / t0) if abs(t0 - target_tenor) / t0 < 0.15 else np.nan

        ts = np.array([u[0] for u in usable])
        ws = np.array([u[1] for u in usable])

        if target_tenor < ts.min() or target_tenor > ts.max():
            return np.nan

        w = float(np.interp(target_tenor, ts, ws))
        return math.sqrt(max(w, 1e-12) / target_tenor) if target_tenor > 0 else np.nan

    # -- the vector --------------------------------------------------------

    def snapshot(self, surface, timestamp=None) -> pd.Series:
        """Feature vector for one surface. Index is fixed by configuration."""
        _log(f"snapshot: {len(surface.points)} points -> features")
        feats: dict[str, float] = {}
        fits = self.fit_slices(surface)
        tenor_years = {d: d / 365.25 for d in self.tenors}

        # --- ATM level per standard tenor
        for days, ty in tenor_years.items():
            feats[f"atm_iv_{days}d"] = self._interp_total_variance(fits, 0.0, ty)

        # --- fixed (k, tenor) grid
        if self.include_grid:
            for days, ty in tenor_years.items():
                for k in self.moneyness:
                    tag = f"{k:+.2f}".replace(".", "").replace("+", "p").replace("-", "m")
                    feats[f"iv_{days}d_k{tag}"] = self._interp_total_variance(fits, k, ty)

        # --- shape features per standard tenor
        for days, ty in tenor_years.items():
            atm = feats[f"atm_iv_{days}d"]

            up = self._interp_total_variance(fits, 0.10, ty)
            dn = self._interp_total_variance(fits, -0.10, ty)

            # Skew: negative means puts richer than calls, the usual equity sign.
            feats[f"skew_{days}d"] = (up - dn) / 0.20 if np.isfinite(up) and np.isfinite(dn) else np.nan
            # Smile curvature around ATM.
            feats[f"curv_{days}d"] = (
                (up + dn - 2 * atm) / (0.10**2)
                if np.isfinite(up) and np.isfinite(dn) and np.isfinite(atm)
                else np.nan
            )
            # Scale-free versions travel better across vol regimes.
            feats[f"skew_norm_{days}d"] = (
                feats[f"skew_{days}d"] / atm if np.isfinite(atm) and atm > 0 else np.nan
            )

        # --- delta-space risk reversal / butterfly
        for days, ty in tenor_years.items():
            nearest = _nearest_fit(fits, ty)
            for dl in self.deltas:
                tag = int(dl * 100)
                rr = bf = np.nan
                if nearest is not None:
                    k_call = k_at_delta(nearest, dl, True, self.r, self.q)
                    k_put = k_at_delta(nearest, dl, False, self.r, self.q)
                    if np.isfinite(k_call) and np.isfinite(k_put):
                        iv_c = float(nearest.iv(k_call)[0])
                        iv_p = float(nearest.iv(k_put)[0])
                        iv_atm = float(nearest.iv(0.0)[0])
                        rr = iv_c - iv_p
                        bf = 0.5 * (iv_c + iv_p) - iv_atm
                feats[f"rr{tag}_{days}d"] = rr
                feats[f"bf{tag}_{days}d"] = bf

        # --- term structure
        feats["term_slope_30_90"] = _diff(feats.get("atm_iv_90d"), feats.get("atm_iv_30d"))
        feats["term_slope_30_365"] = _diff(feats.get("atm_iv_365d"), feats.get("atm_iv_30d"))
        feats["term_slope_7_30"] = _diff(feats.get("atm_iv_30d"), feats.get("atm_iv_7d"))
        feats["term_curv"] = _curv(
            feats.get("atm_iv_7d"), feats.get("atm_iv_30d"), feats.get("atm_iv_90d")
        )
        atm30 = feats.get("atm_iv_30d")
        feats["term_slope_30_90_norm"] = (
            feats["term_slope_30_90"] / atm30 if _pos(atm30) else np.nan
        )

        # --- log levels: vol is roughly lognormal, so these are more stationary
        for days in self.tenors:
            v = feats.get(f"atm_iv_{days}d")
            feats[f"log_atm_iv_{days}d"] = math.log(v) if _pos(v) else np.nan

        # --- quality / coverage diagnostics (usable as features and as filters)
        pts = surface.points
        trades = surface.trades
        n_pts = 0 if pts is None or pts.empty else len(pts)
        feats["n_surface_points"] = float(n_pts)
        feats["n_expiries"] = float(pts["expiration"].nunique()) if n_pts else 0.0
        feats["n_slices_fit"] = float(len(fits))
        feats["total_volume"] = float(pts["volume"].sum()) if n_pts and "volume" in pts else np.nan
        feats["k_coverage_min"] = float(pts["log_moneyness"].min()) if n_pts else np.nan
        feats["k_coverage_max"] = float(pts["log_moneyness"].max()) if n_pts else np.nan
        feats["tenor_coverage_min_d"] = float(pts["tenor_days"].min()) if n_pts else np.nan
        feats["tenor_coverage_max_d"] = float(pts["tenor_days"].max()) if n_pts else np.nan
        feats["fit_rmse_mean"] = (
            float(np.nanmean([f.rmse for f in fits.values()])) if fits else np.nan
        )
        feats["solve_rate"] = float(surface.meta.get("solve_rate", np.nan))
        feats["n_trades"] = float(len(trades)) if trades is not None else np.nan
        feats["spot"] = float(surface.spot_ref)

        series = pd.Series(feats, dtype=float)

        # --- explicit missingness indicators
        if self.include_masks:
            core = [c for c in series.index if c.startswith(("atm_iv_", "skew_", "rr", "bf"))]
            masks = pd.Series(
                {f"mask_{c}": float(np.isfinite(series[c])) for c in core}, dtype=float
            )
            series = pd.concat([series, masks])

        ts = timestamp if timestamp is not None else _surface_timestamp(surface)
        series.name = ts

        if VERBOSE:
            n_ok = int(series.notna().sum())
            _log(f"{len(series)} features, {n_ok} populated, "
                 f"{len(series)-n_ok} NaN (outside coverage)")
            for d in self.tenors:
                v = series.get(f"atm_iv_{d}d")
                mark = f"{v:.4f}" if np.isfinite(v) else "-- (no coverage)"
                _log(f"atm_iv_{d}d = {mark}", 1)
        return series

    def feature_names(self) -> list[str]:
        """The exact output index, without needing a surface. Useful for schema checks."""
        dummy = _EmptySurface()
        return list(self.snapshot(dummy, timestamp=pd.Timestamp("2000-01-01")).index)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _pos(x) -> bool:
    return x is not None and np.isfinite(x) and x > 0


def _diff(a, b):
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    return a - b


def _curv(a, b, c):
    for x in (a, b, c):
        if x is None or not np.isfinite(x):
            return np.nan
    return a + c - 2 * b


def _nearest_fit(fits: dict[float, SliceFit], target: float, tol: float = 0.5):
    """Closest fitted slice to a target tenor, within a relative tolerance."""
    if not fits:
        return None
    tenor = min(fits, key=lambda t: abs(t - target))
    return fits[tenor] if abs(tenor - target) / max(target, 1e-9) <= tol else None


def _surface_timestamp(surface):
    meta_window = surface.meta.get("window") if getattr(surface, "meta", None) else None
    if meta_window:
        try:
            return pd.Timestamp(meta_window[1])
        except (ValueError, TypeError):
            pass
    trades = getattr(surface, "trades", None)
    if trades is not None and not trades.empty and "timestamp" in trades:
        return pd.Timestamp(trades["timestamp"].max())
    return pd.NaT


class _EmptySurface:
    """Minimal stand-in used by ``feature_names``."""

    points = pd.DataFrame(
        columns=["iv", "log_moneyness", "tenor", "tenor_days", "expiration", "volume"]
    )
    trades = pd.DataFrame()
    spot_ref = np.nan
    meta: dict = {}


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------

def build_panel(surfaces, builder: FeatureBuilder | None = None, symbol=None) -> pd.DataFrame:
    """
    Stack per-date feature vectors into a time-indexed DataFrame.

    Rows are sorted by timestamp and de-duplicated. Missing features stay NaN.
    """
    builder = builder or FeatureBuilder()
    rows = [builder.snapshot(s) for s in surfaces]
    if not rows:
        return pd.DataFrame(columns=builder.feature_names())

    panel = pd.DataFrame(rows)
    panel.index.name = "timestamp"
    panel = panel[~panel.index.isna()].sort_index()
    panel = panel[~panel.index.duplicated(keep="last")]

    if symbol is not None:
        panel.insert(0, "symbol", symbol)
    return panel


def add_lags(panel: pd.DataFrame, columns=None, lags=(1, 5, 21), diffs=True) -> pd.DataFrame:
    """
    Add lagged levels and changes. Uses only past rows, so no leakage.

    Vol levels are persistent and non-stationary; the changes usually carry
    more signal than the levels do.
    """
    if columns is None:
        columns = [c for c in panel.columns if c.startswith(("atm_iv_", "skew_", "term_slope_"))]

    # Accumulate then concat once; assigning column by column fragments the
    # frame and pandas warns about it on wide panels.
    new: dict[str, pd.Series] = {}
    for col in columns:
        if col not in panel.columns:
            continue
        for lag in lags:
            shifted = panel[col].shift(lag)
            new[f"{col}_lag{lag}"] = shifted
            if diffs:
                new[f"{col}_chg{lag}"] = panel[col] - shifted

    if not new:
        return panel.copy()
    return pd.concat([panel, pd.DataFrame(new, index=panel.index)], axis=1)


def add_rolling_z(panel: pd.DataFrame, columns=None, window: int = 63, min_periods: int = 20):
    """
    Rolling z-scores against each column's own trailing history.

    Deliberately not a full-sample standardization: fitting a scaler on the
    whole panel leaks future distribution information into training rows.
    """
    if columns is None:
        columns = [c for c in panel.columns if c.startswith(("atm_iv_", "skew_", "rr", "bf"))]

    new: dict[str, pd.Series] = {}
    for col in columns:
        if col not in panel.columns:
            continue
        roll = panel[col].shift(1).rolling(window, min_periods=min_periods)
        mu, sd = roll.mean(), roll.std()
        new[f"{col}_z{window}"] = (panel[col] - mu) / sd.replace(0, np.nan)

    if not new:
        return panel.copy()
    return pd.concat([panel, pd.DataFrame(new, index=panel.index)], axis=1)


def add_targets(
    panel: pd.DataFrame,
    spot: pd.Series | None = None,
    horizons=(1, 5, 21),
    target_col: str = "atm_iv_30d",
) -> pd.DataFrame:
    """
    Attach forward-looking labels.

    Everything here is shifted with a *negative* offset, i.e. strictly future
    relative to the feature row. Rows near the end of the panel get NaN targets
    and must be dropped before training, not filled.
    """
    out = panel.copy()

    for h in horizons:
        if target_col in panel.columns:
            fwd = panel[target_col].shift(-h)
            out[f"y_{target_col}_fwd{h}"] = fwd
            out[f"y_{target_col}_chg{h}"] = fwd - panel[target_col]
            out[f"y_{target_col}_up{h}"] = (fwd > panel[target_col]).astype(float)
            out.loc[fwd.isna(), f"y_{target_col}_up{h}"] = np.nan

        if spot is not None:
            s = spot.reindex(panel.index).astype(float)
            fwd_ret = np.log(s.shift(-h) / s)
            out[f"y_ret_fwd{h}"] = fwd_ret
            out[f"y_absret_fwd{h}"] = fwd_ret.abs()
            # Realized-minus-implied: the classic variance-risk-premium label.
            if target_col in panel.columns:
                rv = np.log(s / s.shift(1)).rolling(h).std().shift(-h) * math.sqrt(252)
                out[f"y_vrp_fwd{h}"] = rv - panel[target_col]

    return out


def purged_splits(index: pd.DatetimeIndex, n_splits: int = 5, embargo: int = 5, horizon: int = 1):
    """
    Walk-forward splits with a purge gap between train and test.

    Overlapping forward-looking labels mean a naive split lets a training row's
    target window overlap the test period. The gap is ``horizon + embargo`` rows.
    """
    n = len(index)
    if n < n_splits + 2:
        return []

    fold = n // (n_splits + 1)
    gap = horizon + embargo
    splits = []

    for i in range(1, n_splits + 1):
        train_end = fold * i
        test_start = train_end + gap
        test_end = min(test_start + fold, n)
        if test_start >= n or test_end <= test_start:
            continue
        splits.append(
            (np.arange(0, train_end), np.arange(test_start, test_end))
        )
    return splits


def clean_for_model(
    panel: pd.DataFrame,
    max_nan_frac: float = 0.30,
    min_slices: int = 2,
    drop_diagnostics: bool = False,
) -> pd.DataFrame:
    """
    Drop rows and columns too sparse to model, without imputing anything.

    Tree models handle NaN natively; linear models need imputation, which
    should happen inside a fold-local pipeline, never here.
    """
    out = panel.copy()

    if "n_slices_fit" in out.columns:
        out = out[out["n_slices_fit"] >= min_slices]

    feature_cols = [c for c in out.columns if not c.startswith(("y_", "mask_", "symbol"))]
    nan_frac = out[feature_cols].isna().mean()
    keep = nan_frac[nan_frac <= max_nan_frac].index.tolist()

    diagnostics = ["spot", "n_trades", "n_surface_points", "solve_rate", "fit_rmse_mean"]
    if drop_diagnostics:
        keep = [c for c in keep if c not in diagnostics]

    other = [c for c in out.columns if c.startswith(("y_", "mask_", "symbol"))]
    return out[keep + other]


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fb = FeatureBuilder()
    names = fb.feature_names()
    print(f"{len(names)} features per snapshot")
    for n in names[:15]:
        print("  ", n)
    print("   ...")