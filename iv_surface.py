"""
iv_surface.py
=============

Builds implied-volatility surfaces from Alpaca option trade data.

Extends the ``OptionsData`` class in ``option_data.py`` with:

  * spot-price attachment (as-of merge against underlying bars)
  * Black-Scholes-Merton and Bjerksund-Stensland (American) pricers
  * a robust Brent implied-vol solver
  * outlier/arbitrage filtering
  * aggregation of raw trades into a clean (tenor x strike) surface
  * 2-D grid interpolation + plotting

Quick start
-----------
    from iv_surface import IVSurface

    iv = IVSurface()
    surf = iv.build(
        underlying="AAPL",
        start="2025-03-03T13:30:00+00:00",
        end="2025-03-03T21:00:00+00:00",
        expiration_start="2025-03-07",
        expiration_end="2025-09-19",
        status="inactive",
        model="american",
    )

    surf.plot()                      # 3-D surface
    surf.plot_smile(tenor_days=30)   # single expiry slice
    surf.table().to_csv("surface.csv")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from option_data import OptionsData

SECONDS_PER_YEAR = 365.25 * 24 * 3600
MARKET_TZ = "America/New_York"


# ---------------------------------------------------------------------------
# European pricing (Black-Scholes-Merton with continuous dividend yield)
# ---------------------------------------------------------------------------

def bs_price(S, K, T, r, q, sigma, is_call) -> float:
    """European option value. ``q`` is the continuous dividend yield."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if is_call else (K - S))

    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    if is_call:
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def bs_vega(S, K, T, r, q, sigma) -> float:
    """dPrice/dSigma, per 1.00 of vol (divide by 100 for per-vol-point)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    return S * math.exp(-q * T) * norm.pdf(d1) * sqrt_t


def bs_greeks(S, K, T, r, q, sigma, is_call) -> dict:
    """Delta, gamma, vega, theta, rho for a European option."""
    if T <= 0 or sigma <= 0:
        return dict(delta=np.nan, gamma=np.nan, vega=np.nan, theta=np.nan, rho=np.nan)

    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    disc_q, disc_r = math.exp(-q * T), math.exp(-r * T)
    pdf_d1 = norm.pdf(d1)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * disc_q * pdf_d1 * sqrt_t
    common_theta = -S * disc_q * pdf_d1 * sigma / (2 * sqrt_t)

    if is_call:
        delta = disc_q * norm.cdf(d1)
        theta = common_theta - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = common_theta + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
        rho = -K * T * disc_r * norm.cdf(-d2)

    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta / 365.0, rho=rho / 100.0)


# ---------------------------------------------------------------------------
# American pricing (Bjerksund-Stensland 1993 closed-form approximation)
# ---------------------------------------------------------------------------

def _phi_hat(S, T, gamma, H, X, r, b, sigma) -> float:
    """
    Bjerksund-Stensland phi, divided by ``X**gamma``.

    gamma can reach 1e5+ at low vol, so ``S**gamma`` overflows on its own.
    Since S < X on every call site, the normalized form ``(S/X)**gamma`` stays
    bounded, and each caller multiplies the factor back in a safe order.
    """
    var = sigma * sigma
    sqrt_t = math.sqrt(T)

    lam = (-r + gamma * b + 0.5 * gamma * (gamma - 1.0) * var) * T
    d = -(math.log(S / H) + (b + (gamma - 0.5) * var) * T) / (sigma * sqrt_t)
    kappa = 2.0 * b / var + (2.0 * gamma - 1.0)

    # kappa also scales with 1/sigma**2, so (X/S)**kappa overflows while the
    # cdf it multiplies underflows. Combine that pair in log space.
    log_ratio = math.log(X / S)
    log_tail = kappa * log_ratio + norm.logcdf(d - 2.0 * log_ratio / (sigma * sqrt_t))
    tail = math.exp(log_tail) if log_tail < 700.0 else math.inf

    log_pref = lam + gamma * math.log(S / X)
    if log_pref < -700.0:
        return 0.0
    return math.exp(log_pref) * (norm.cdf(d) - tail)


def _american_call(S, K, T, r, b, sigma) -> float:
    """Bjerksund-Stensland 1993. ``b`` is the cost of carry (r - q)."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)

    # With a non-negative carry the American call is worth the European one.
    if b >= r:
        return bs_price(S, K, T, r, r - b, sigma, True)

    # At vanishing vol beta explodes (~2b/sigma**2) and the closed form
    # overflows. The two values coincide there anyway.
    if sigma < 1e-3:
        return max(bs_price(S, K, T, r, r - b, sigma, True), max(0.0, S - K))

    var = sigma * sigma
    beta = (0.5 - b / var) + math.sqrt((b / var - 0.5) ** 2 + 2.0 * r / var)

    if abs(beta - 1.0) < 1e-10:  # degenerate trigger boundary
        return bs_price(S, K, T, r, r - b, sigma, True)

    b_inf = beta / (beta - 1.0) * K
    b_zero = max(K, r / (r - b) * K)
    spread = b_inf - b_zero

    if spread <= 1e-12:  # boundary collapses onto the immediate-exercise level
        trigger = b_zero
    else:
        h = -(b * T + 2.0 * sigma * math.sqrt(T)) * b_zero / spread
        # The derivation assumes h <= 0; negative carry at low vol can push it
        # positive, which overflows exp() and inverts the boundary.
        h = min(h, 0.0)
        trigger = b_zero + spread * (1.0 - math.exp(h))

    if S >= trigger:  # immediate exercise
        return S - K

    # Every beta-power cancels against trigger**beta, so express the alpha
    # terms as (trigger - K) * <bounded>. Writing alpha explicitly would give
    # 0 * inf = NaN whenever the trigger collapses onto the strike.
    gap = trigger - K
    return (
        gap * (S / trigger) ** beta
        - gap * _phi_hat(S, T, beta, trigger, trigger, r, b, sigma)
        + trigger * _phi_hat(S, T, 1.0, trigger, trigger, r, b, sigma)
        - trigger * _phi_hat(S, T, 1.0, K, trigger, r, b, sigma)
        - K * _phi_hat(S, T, 0.0, trigger, trigger, r, b, sigma)
        + K * _phi_hat(S, T, 0.0, K, trigger, r, b, sigma)
    )


def american_price(S, K, T, r, q, sigma, is_call) -> float:
    """American option value. Puts use the standard call-put transformation."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if is_call else (K - S))
    if is_call:
        return _american_call(S, K, T, r, r - q, sigma)
    # P(S, K, r, q) == C(K, S, q, r)
    return _american_call(K, S, T, q, q - r, sigma)


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------

def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.045,
    q: float = 0.0,
    is_call: bool = True,
    model: str = "bsm",
    lo: float = 1e-4,
    hi: float = 5.0,
    min_vega: float = 1e-3,
) -> float:
    """
    Invert the pricer for sigma. Returns NaN rather than raising when the
    price sits outside no-arbitrage bounds (stale prints, crossed markets,
    deep-ITM trades with no vega).
    """
    if not (np.isfinite(price) and np.isfinite(S) and price > 0 and S > 0 and T > 0):
        return np.nan

    pricer = american_price if model == "american" else bs_price
    if model == "american":
        lo = max(lo, 1e-3)  # the closed form degenerates below this

    def objective(sigma):
        return pricer(S, K, T, r, q, sigma, is_call) - price

    # The bracket test is itself the no-arbitrage check: f(lo) > 0 means the
    # price is below the zero-vol floor, f(hi) < 0 means it is above the cap.
    # Don't compare against raw intrinsic -- for European options the floor is
    # the discounted forward bound, which sits below it.
    try:
        if objective(lo) > 0 or objective(hi) < 0:
            return np.nan
        sigma = brentq(objective, lo, hi, xtol=1e-8, rtol=1e-10, maxiter=200)
    except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError):
        return np.nan

    # Deep ITM/OTM contracts have no vega, so the price carries no information
    # about vol and the root is an artefact. Check at the root, not at a
    # fixed reference vol, or genuinely high-vol wings get thrown away.
    if bs_vega(S, K, T, r, q, sigma) < min_vega:
        return np.nan
    return sigma


# ---------------------------------------------------------------------------
# Surface container
# ---------------------------------------------------------------------------

@dataclass
class Surface:
    """Aggregated IV surface plus the trade-level data it was built from."""

    points: pd.DataFrame          # one row per (expiration, strike, type)
    trades: pd.DataFrame          # trade-level, with iv column
    underlying: str
    spot_ref: float
    r: float
    q: float
    model: str
    meta: dict = field(default_factory=dict)

    # -- reshaping ---------------------------------------------------------

    def table(self, value: str = "iv") -> pd.DataFrame:
        """Pivot to a tenor (rows) x strike (columns) matrix."""
        return self.points.pivot_table(
            index="tenor_days", columns="strike", values=value, aggfunc="mean"
        ).sort_index()

    def smile(self, tenor_days: float, tol: float = 3.0) -> pd.DataFrame:
        """All surface points within ``tol`` days of a target tenor."""
        mask = (self.points["tenor_days"] - tenor_days).abs() <= tol
        return self.points.loc[mask].sort_values("strike")

    def term_structure(self, moneyness: float = 1.0, tol: float = 0.02) -> pd.DataFrame:
        """ATM (or fixed-moneyness) vol against tenor."""
        mask = (self.points["moneyness"] - moneyness).abs() <= tol
        return (
            self.points.loc[mask]
            .groupby("tenor_days", as_index=False)["iv"]
            .mean()
            .sort_values("tenor_days")
        )

    def grid(self, n_x: int = 60, n_y: int = 40, x: str = "log_moneyness", method: str = "linear"):
        """
        Interpolate scattered points onto a regular mesh.
        Returns ``(X, Y, Z)`` suitable for ``plot_surface`` / ``pcolormesh``.
        """
        from scipy.interpolate import griddata

        pts = self.points.dropna(subset=[x, "tenor_days", "iv"])
        if len(pts) < 4:
            raise ValueError(f"need at least 4 valid points to interpolate, got {len(pts)}")

        xi = np.linspace(pts[x].min(), pts[x].max(), n_x)
        yi = np.linspace(pts["tenor_days"].min(), pts["tenor_days"].max(), n_y)
        X, Y = np.meshgrid(xi, yi)

        coords = pts[[x, "tenor_days"]].to_numpy()
        values = pts["iv"].to_numpy()

        Z = griddata(coords, values, (X, Y), method=method)
        # Fill the convex-hull gaps so the surface renders without holes.
        holes = np.isnan(Z)
        if holes.any():
            Z[holes] = griddata(coords, values, (X, Y), method="nearest")[holes]
        return X, Y, Z

    # -- plotting ----------------------------------------------------------

    def plot(self, x: str = "log_moneyness", method: str = "linear", show_points: bool = True):
        import matplotlib.pyplot as plt

        X, Y, Z = self.grid(x=x, method=method)

        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z * 100, cmap="viridis", alpha=0.85, linewidth=0, antialiased=True)

        if show_points:
            pts = self.points.dropna(subset=[x, "iv"])
            ax.scatter(pts[x], pts["tenor_days"], pts["iv"] * 100, c="k", s=4, alpha=0.35)

        ax.set_xlabel("log(K/S)" if x == "log_moneyness" else x)
        ax.set_ylabel("days to expiry")
        ax.set_zlabel("implied vol (%)")
        ax.set_title(f"{self.underlying} IV surface — spot {self.spot_ref:.2f} ({self.model})")
        ax.view_init(elev=22, azim=-125)
        fig.tight_layout()
        return fig, ax

    def plot_smile(self, tenor_days: float, tol: float = 3.0):
        import matplotlib.pyplot as plt

        sl = self.smile(tenor_days, tol)
        fig, ax = plt.subplots(figsize=(9, 5))
        for opt_type, grp in sl.groupby("type"):
            ax.plot(grp["strike"], grp["iv"] * 100, marker="o", ms=4, label=str(opt_type))
        ax.axvline(self.spot_ref, ls="--", c="grey", lw=1, label="spot")
        ax.set_xlabel("strike")
        ax.set_ylabel("implied vol (%)")
        ax.set_title(f"{self.underlying} smile — ~{tenor_days:g}d to expiry")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig, ax

    def plot_heatmap(self, x: str = "log_moneyness", method: str = "linear"):
        import matplotlib.pyplot as plt

        X, Y, Z = self.grid(x=x, method=method)
        fig, ax = plt.subplots(figsize=(10, 6))
        mesh = ax.pcolormesh(X, Y, Z * 100, cmap="viridis", shading="auto")
        ax.contour(X, Y, Z * 100, colors="white", linewidths=0.5, alpha=0.5)
        fig.colorbar(mesh, ax=ax, label="implied vol (%)")
        ax.set_xlabel("log(K/S)" if x == "log_moneyness" else x)
        ax.set_ylabel("days to expiry")
        ax.set_title(f"{self.underlying} IV surface")
        fig.tight_layout()
        return fig, ax

    def __repr__(self) -> str:
        n_exp = self.points["expiration"].nunique() if not self.points.empty else 0
        return (
            f"<Surface {self.underlying} spot={self.spot_ref:.2f} "
            f"points={len(self.points)} expiries={n_exp} "
            f"trades={len(self.trades)} model={self.model}>"
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class IVSurface(OptionsData):
    """
    Adds a stock data client and the surface pipeline on top of OptionsData.
    Inherits ``get_contracts`` and ``get_trades`` unchanged.
    """

    def __init__(self, api_key=None, secret_key=None):
        super().__init__(api_key, secret_key)
        self.stock_client = StockHistoricalDataClient(self.api_key, self.secret_key)

    # -- step 1: underlying price at each option trade ----------------------

    def attach_spot(
        self,
        trades: pd.DataFrame,
        underlying: str,
        start,
        end,
        timeframe: TimeFrame | None = None,
        tolerance: str = "5min",
    ) -> pd.DataFrame:
        """
        As-of merge underlying bar closes onto option trades. Minute bars give
        up to 60s of staleness; pass a finer timeframe if that matters.
        """
        if trades.empty:
            return trades

        bars = self.stock_client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=underlying,
                timeframe=timeframe or TimeFrame.Minute,
                start=pd.Timestamp(start).to_pydatetime(),
                end=pd.Timestamp(end).to_pydatetime(),
            )
        ).df

        if bars.empty:
            raise ValueError(f"no {underlying} bars returned for {start} .. {end}")

        bars = bars.reset_index()[["timestamp", "close"]].rename(columns={"close": "spot"})
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        trades = trades.copy()
        trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)

        return pd.merge_asof(
            trades.sort_values("timestamp"),
            bars.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            tolerance=pd.Timedelta(tolerance),
        )

    # -- step 2: per-trade implied vol --------------------------------------

    @staticmethod
    def compute_iv(
        trades: pd.DataFrame,
        r: float = 0.045,
        q: float = 0.0,
        model: str = "bsm",
        price_col: str = "price",
        with_greeks: bool = False,
    ) -> pd.DataFrame:
        if trades.empty:
            return trades

        df = trades.copy()

        exp = pd.to_datetime(df["expiration"])
        if exp.dt.tz is None:
            exp = exp.dt.tz_localize(MARKET_TZ) + pd.Timedelta(hours=16)  # 4pm ET expiry
        exp = exp.dt.tz_convert("UTC")

        ts = pd.to_datetime(df["timestamp"], utc=True)
        df["tenor"] = (exp - ts).dt.total_seconds() / SECONDS_PER_YEAR
        df["tenor_days"] = df["tenor"] * 365.25
        df["is_call"] = df["type"].astype(str).str.lower().str.contains("call")
        df["moneyness"] = df["strike"] / df["spot"]
        df["log_moneyness"] = np.log(df["moneyness"])

        df["iv"] = [
            implied_vol(p, s, k, t, r, q, c, model)
            for p, s, k, t, c in zip(
                df[price_col], df["spot"], df["strike"], df["tenor"], df["is_call"]
            )
        ]

        if with_greeks:
            greeks = [
                bs_greeks(s, k, t, r, q, v, c) if np.isfinite(v) else
                dict(delta=np.nan, gamma=np.nan, vega=np.nan, theta=np.nan, rho=np.nan)
                for s, k, t, v, c in zip(
                    df["spot"], df["strike"], df["tenor"], df["iv"], df["is_call"]
                )
            ]
            df = pd.concat([df, pd.DataFrame(greeks, index=df.index)], axis=1)

        return df

    # -- step 3: trades -> clean surface points -----------------------------

    @staticmethod
    def aggregate(
        trades: pd.DataFrame,
        min_trades: int = 1,
        iv_bounds: tuple[float, float] = (0.02, 3.0),
        min_tenor_days: float = 1.0,
        moneyness_bounds: tuple[float, float] = (0.7, 1.3),
        weighted: bool = True,
    ) -> pd.DataFrame:
        """
        Collapse trades to one point per contract using a size-weighted mean
        IV, after dropping the prints that make surfaces ugly: unsolvable
        quotes, near-expiry noise, and far wings with no vega.
        """
        if trades.empty:
            return trades

        df = trades.dropna(subset=["iv"])
        df = df[df["iv"].between(*iv_bounds)]
        df = df[df["tenor_days"] >= min_tenor_days]
        df = df[df["moneyness"].between(*moneyness_bounds)]

        if df.empty:
            return pd.DataFrame(
                columns=["expiration", "strike", "type", "iv", "n_trades", "volume"]
            )

        if weighted and "size" in df.columns:
            df = df.assign(_w=df["size"].clip(lower=1))
        else:
            df = df.assign(_w=1.0)
        df = df.assign(_wiv=df["iv"] * df["_w"])

        grouped = df.groupby(["expiration", "strike", "type"], as_index=False).agg(
            iv_sum=("_wiv", "sum"),
            w_sum=("_w", "sum"),
            iv_std=("iv", "std"),
            n_trades=("iv", "size"),
            volume=("_w", "sum"),
            tenor=("tenor", "mean"),
            tenor_days=("tenor_days", "mean"),
            moneyness=("moneyness", "mean"),
            log_moneyness=("log_moneyness", "mean"),
            spot=("spot", "mean"),
            last_price=("price", "last"),
        )
        grouped["iv"] = grouped["iv_sum"] / grouped["w_sum"]

        grouped = grouped[grouped["n_trades"] >= min_trades]
        return grouped.drop(columns=["iv_sum", "w_sum"]).sort_values(
            ["tenor_days", "strike"]
        ).reset_index(drop=True)

    # -- one-shot pipeline ---------------------------------------------------

    def build(
        self,
        underlying: str,
        start,
        end,
        expiration_start,
        expiration_end,
        option_type=None,
        strike_min: float | None = None,
        strike_max: float | None = None,
        status=None,
        r: float = 0.045,
        q: float = 0.0,
        model: str = "bsm",
        min_trades: int = 1,
        moneyness_bounds: tuple[float, float] = (0.7, 1.3),
        with_greeks: bool = False,
    ) -> Surface:
        """
        Pull contracts + trades, attach spot, solve for IV, aggregate.

        ``model="american"`` uses Bjerksund-Stensland, which matters for
        single-name equity options; ``"bsm"`` is fine for index options.
        Past dates need ``status="inactive"`` to pick up expired contracts.
        """
        trades = self.get_trades(
            underlying=underlying,
            start=start,
            end=end,
            expiration_start=expiration_start,
            expiration_end=expiration_end,
            option_type=option_type,
            strike_min=strike_min,
            strike_max=strike_max,
            status=status,
        )
        if trades.empty:
            raise ValueError("no option trades returned for the requested window")

        trades = self.attach_spot(trades, underlying, start, end)
        trades = self.compute_iv(trades, r=r, q=q, model=model, with_greeks=with_greeks)
        points = self.aggregate(
            trades, min_trades=min_trades, moneyness_bounds=moneyness_bounds
        )

        spot_ref = float(trades["spot"].dropna().iloc[-1]) if trades["spot"].notna().any() else np.nan
        solved = int(trades["iv"].notna().sum())

        return Surface(
            points=points,
            trades=trades,
            underlying=underlying,
            spot_ref=spot_ref,
            r=r,
            q=q,
            model=model,
            meta={
                "window": (str(start), str(end)),
                "n_trades": len(trades),
                "n_solved": solved,
                "solve_rate": solved / len(trades) if len(trades) else 0.0,
            },
        )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build an implied volatility surface.")
    p.add_argument("underlying")
    p.add_argument("--start", required=True, help="e.g. 2025-03-03T14:30:00+00:00")
    p.add_argument("--end", required=True)
    p.add_argument("--exp-start", required=True)
    p.add_argument("--exp-end", required=True)
    p.add_argument("--status", default="inactive", help="'inactive' for expired contracts")
    p.add_argument("--model", default="bsm", choices=["bsm", "american"])
    p.add_argument("--rate", type=float, default=0.045)
    p.add_argument("--div-yield", type=float, default=0.0)
    p.add_argument("--csv", help="write aggregated surface points here")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    surface = IVSurface().build(
        underlying=args.underlying,
        start=args.start,
        end=args.end,
        expiration_start=args.exp_start,
        expiration_end=args.exp_end,
        status=args.status,
        model=args.model,
        r=args.rate,
        q=args.div_yield,
    )

    print(surface)
    print(f"solve rate: {surface.meta['solve_rate']:.1%}")
    print(surface.points.head(20))

    if args.csv:
        surface.points.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")

    if args.plot:
        import matplotlib.pyplot as plt

        surface.plot()
        plt.show()