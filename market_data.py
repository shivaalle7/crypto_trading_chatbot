"""CoinGecko market data utilities and technical indicators."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://api.coingecko.com/api/v3"
TIMEOUT = 15
SESSION = requests.Session()
SESSION.headers.update({
    "accept": "application/json",
    "User-Agent": "CryptoMindAI/1.1 (Streamlit app)",
})


class MarketDataError(RuntimeError):
    """User-friendly error raised when market data cannot be retrieved."""


def _get(path: str, params: dict | None = None) -> Any:
    try:
        response = SESSION.get(f"{BASE_URL}{path}", params=params, timeout=TIMEOUT)
        if response.status_code == 429:
            raise MarketDataError(
                "CoinGecko rate limit reached. Wait a little and try again."
            )
        response.raise_for_status()
        return response.json()
    except MarketDataError:
        raise
    except requests.Timeout as exc:
        raise MarketDataError("CoinGecko request timed out. Please try again.") from exc
    except requests.RequestException as exc:
        raise MarketDataError(
            "Could not connect to CoinGecko. Check your internet connection."
        ) from exc
    except ValueError as exc:
        raise MarketDataError("CoinGecko returned an invalid response.") from exc


@lru_cache(maxsize=32)
def get_markets(
    vs_currency: str = "usd",
    page: int = 1,
    per_page: int = 50,
) -> pd.DataFrame:
    """Get top coins by market cap. Cached for 60 seconds at app level."""
    payload = _get(
        "/coins/markets",
        {
            "vs_currency": vs_currency.lower(),
            "order": "market_cap_desc",
            "per_page": min(max(int(per_page), 1), 100),
            "page": max(int(page), 1),
            "sparkline": "false",
            "price_change_percentage": "24h",
        },
    )
    df = pd.DataFrame(payload)
    if df.empty:
        return df
    columns = [
        "id", "symbol", "name", "image", "current_price", "market_cap",
        "market_cap_rank", "total_volume", "high_24h", "low_24h",
        "price_change_24h", "price_change_percentage_24h",
        "circulating_supply", "last_updated",
    ]
    return df[[column for column in columns if column in df.columns]]


def search_coins(query: str) -> list[dict]:
    """Search CoinGecko coin catalogue by name or ticker symbol."""
    query = (query or "").strip()
    if not query:
        return []
    payload = _get("/search", {"query": query})
    return payload.get("coins", [])[:20]


@lru_cache(maxsize=128)
def get_coin_history(
    coin_id: str,
    vs_currency: str = "usd",
    days: int = 90,
) -> pd.DataFrame:
    """Get price history and volume for a CoinGecko coin ID."""
    allowed_days = {1, 7, 14, 30, 90, 180, 365, "max"}
    if days not in allowed_days:
        days = 90
    params = {
        "vs_currency": vs_currency.lower(),
        "days": days,
    }
    if days != 1:
        params["interval"] = "daily"
    payload = _get(f"/coins/{coin_id}/market_chart", params)
    prices = payload.get("prices", [])
    volumes = payload.get("total_volumes", [])
    if not prices:
        raise MarketDataError("No historical price data was returned for this coin.")
    df = pd.DataFrame(prices, columns=["timestamp_ms", "price"])
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.tz_convert(None)
    if volumes:
        vol_df = pd.DataFrame(volumes, columns=["volume_timestamp_ms", "volume"])
        df = df.merge(vol_df, left_on="timestamp_ms", right_on="volume_timestamp_ms", how="left")
    else:
        df["volume"] = np.nan
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df[["date", "price", "volume"]]


def add_technical_indicators(history: pd.DataFrame) -> pd.DataFrame:
    """Calculate SMA20, SMA50, RSI14, MACD, signal, and histogram."""
    df = history.copy()
    close = pd.to_numeric(df["price"], errors="coerce")
    df["SMA 20"] = close.rolling(20, min_periods=1).mean()
    df["SMA 50"] = close.rolling(50, min_periods=1).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    df["RSI 14"] = rsi

    ema12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
    df["MACD"] = ema12 - ema26
    df["MACD Signal"] = df["MACD"].ewm(span=9, adjust=False, min_periods=1).mean()
    df["MACD Histogram"] = df["MACD"] - df["MACD Signal"]
    return df


def get_simple_price(
    coin_ids: list[str],
    vs_currency: str = "usd",
) -> dict:
    """Fetch simple prices for one or more CoinGecko coin IDs."""
    if not coin_ids:
        return {}
    return _get(
        "/simple/price",
        {
            "ids": ",".join(dict.fromkeys(coin_ids)),
            "vs_currencies": vs_currency.lower(),
            "include_24hr_change": "true",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
        },
    )


def find_market_coins(query: str, vs_currency: str = "usd", limit: int = 100) -> pd.DataFrame:
    """Find matching coins from current market listing using name, symbol, or ID."""
    query = (query or "").strip().lower()
    if not query:
        return pd.DataFrame()
    df = get_markets(vs_currency, per_page=limit)
    if df.empty:
        return df
    mask = (
        df["name"].astype(str).str.lower().str.contains(query, regex=False)
        | df["symbol"].astype(str).str.lower().eq(query)
        | df["id"].astype(str).str.lower().str.contains(query, regex=False)
    )
    return df.loc[mask].copy()


def forecast_coin_price(
    coin_id: str,
    target_date,
    vs_currency: str = "usd",
    history_days: int = 365,
) -> dict:
    """Create an illustrative statistical projection for any CoinGecko coin ID.

    This is a simple log-return drift/volatility baseline, not a validated
    forecasting model and not investment advice. Long horizons are highly uncertain.
    """
    target = pd.Timestamp(target_date).tz_localize(None).normalize()
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    horizon_days = int((target - today).days)
    if horizon_days < 1:
        raise MarketDataError("Choose a target date at least one day in the future.")

    history = get_coin_history(coin_id, vs_currency, history_days).copy()
    history["price"] = pd.to_numeric(history["price"], errors="coerce")
    history = history.dropna(subset=["price"]).sort_values("date")
    if len(history) < 20:
        raise MarketDataError("Not enough historical price data to create a projection.")

    prices = history["price"].astype(float)
    current_price = float(prices.iloc[-1])
    if current_price <= 0:
        raise MarketDataError("The latest price is invalid, so a projection cannot be calculated.")

    # Use recent daily log returns and limit extreme outliers' effect on drift.
    log_returns = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    recent_returns = log_returns.tail(min(90, len(log_returns)))
    if len(recent_returns) < 10:
        raise MarketDataError("Not enough recent return observations to create a projection.")
    clipped_drift = recent_returns.clip(
        lower=recent_returns.quantile(0.05), upper=recent_returns.quantile(0.95)
    )
    daily_drift = float(clipped_drift.mean())
    daily_volatility = float(recent_returns.std(ddof=1))

    projected = current_price * float(np.exp(daily_drift * horizon_days))
    uncertainty = daily_volatility * float(np.sqrt(horizon_days)) * 1.96
    lower = current_price * float(np.exp(daily_drift * horizon_days - uncertainty))
    upper = current_price * float(np.exp(daily_drift * horizon_days + uncertainty))

    # Keep chart path illustrative: linearly spaced log-price scenarios.
    future_dates = pd.date_range(start=today + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    elapsed = np.arange(1, horizon_days + 1, dtype=float)
    center_path = current_price * np.exp(daily_drift * elapsed)
    band = daily_volatility * np.sqrt(elapsed) * 1.96
    forecast_frame = pd.DataFrame({
        "date": future_dates,
        "projected_price": center_path,
        "lower_range": current_price * np.exp(daily_drift * elapsed - band),
        "upper_range": current_price * np.exp(daily_drift * elapsed + band),
    })
    return {
        "coin_id": coin_id,
        "currency": vs_currency.upper(),
        "as_of": history["date"].iloc[-1],
        "target_date": target,
        "horizon_days": horizon_days,
        "current_price": current_price,
        "projected_price": projected,
        "lower_range": lower,
        "upper_range": upper,
        "daily_volatility": daily_volatility,
        "history_points": len(history),
        "forecast_frame": forecast_frame,
    }


def get_coin_price(coin_id: str, vs_currency: str = "usd") -> dict:
    """Fetch a single coin's latest price and 24-hour metrics in the requested currency."""
    data = get_simple_price([coin_id], vs_currency)
    coin = data.get(coin_id)
    if not coin or coin.get(vs_currency.lower()) is None:
        raise MarketDataError(
            f"No {vs_currency.upper()} price is currently available for {coin_id}."
        )
    return coin
