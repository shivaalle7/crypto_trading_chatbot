"""CoinGecko market-data helpers and technical indicators for CryptoMind AI."""
from __future__ import annotations

import requests
import pandas as pd
import numpy as np
from functools import lru_cache
from typing import Any

BASE_URL = "https://api.coingecko.com/api/v3"
TIMEOUT = 15
SESSION = requests.Session()
SESSION.headers.update({"accept": "application/json", "User-Agent": "CryptoMindAI/1.0"})


class MarketDataError(RuntimeError):
    """Raised when a market-data request fails."""


def _get(path: str, params: dict | None = None) -> Any:
    url = f"{BASE_URL}{path}"
    try:
        response = SESSION.get(url, params=params, timeout=TIMEOUT)
        if response.status_code == 429:
            raise MarketDataError(
                "CoinGecko rate limit reached. Wait a minute and try again."
            )
        response.raise_for_status()
        return response.json()
    except requests.Timeout as exc:
        raise MarketDataError("CoinGecko request timed out. Please try again.") from exc
    except requests.RequestException as exc:
        raise MarketDataError(
            "Could not reach CoinGecko. Check your internet connection and try again."
        ) from exc
    except ValueError as exc:
        raise MarketDataError("CoinGecko returned an unreadable response.") from exc


@lru_cache(maxsize=32)
def get_markets(vs_currency: str = "usd", page: int = 1, per_page: int = 50) -> pd.DataFrame:
    """Fetch top coins by market cap. Results are cached for this Python process."""
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
    wanted = [
        "id", "symbol", "name", "image", "current_price", "market_cap",
        "market_cap_rank", "total_volume", "high_24h", "low_24h",
        "price_change_24h", "price_change_percentage_24h", "circulating_supply",
        "last_updated",
    ]
    return df[[col for col in wanted if col in df.columns]]


def search_coins(query: str) -> list[dict]:
    """Search CoinGecko's coin catalogue by name or symbol."""
    query = (query or "").strip()
    if not query:
        return []
    payload = _get("/search", {"query": query})
    return payload.get("coins", [])[:20]


@lru_cache(maxsize=128)
def get_coin_history(coin_id: str, vs_currency: str = "usd", days: int = 90) -> pd.DataFrame:
    """Get historical prices and volumes for a CoinGecko coin ID."""
    days = int(days)
    if days not in (1, 7, 14, 30, 90, 180, 365,  max(1, days)):
        days = 90
    payload = _get(
        f"/coins/{coin_id}/market_chart",
        {"vs_currency": vs_currency.lower(), "days": days, "interval": "daily" if days > 1 else "hourly"},
    )
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
    """Add SMA20, SMA50, RSI14, MACD, signal and histogram."""
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
    # A sustained rise can have zero average loss; RSI is then 100.
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    df["RSI 14"] = rsi

    ema12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
    df["MACD"] = ema12 - ema26
    df["MACD Signal"] = df["MACD"].ewm(span=9, adjust=False, min_periods=1).mean()
    df["MACD Histogram"] = df["MACD"] - df["MACD Signal"]
    return df


def get_simple_price(coin_ids: list[str], vs_currency: str = "usd") -> dict:
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
