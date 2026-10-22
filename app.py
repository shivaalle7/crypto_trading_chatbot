from __future__ import annotations

import os
from datetime import datetime, timezone, date, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

from market_data import (
    MarketDataError,
    add_technical_indicators,
    get_coin_history,
    get_markets,
    get_simple_price,
    search_coins,
    forecast_coin_price,
)

load_dotenv()

st.set_page_config(
    page_title="CryptoMind AI",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
      .main-title {font-size: 2.4rem; font-weight: 800; margin-bottom: 0;}
      .sub-title {opacity: .75; margin-top: .2rem; margin-bottom: 1.2rem;}
      div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22);
        border-radius: 12px; padding: 14px;}
      .small-note {font-size: .85rem; opacity: .75;}
    </style>
    """,
    unsafe_allow_html=True,
)

CURRENCIES = {
    "USD": "US Dollar",
    "INR": "Indian Rupee",
    "EUR": "Euro",
    "GBP": "British Pound",
    "JPY": "Japanese Yen",
    "CAD": "Canadian Dollar",
    "AUD": "Australian Dollar",
    "SGD": "Singapore Dollar",
}


def get_config(key: str, default: str = "") -> str:
    """Read environment variables locally and Streamlit secrets in the cloud."""
    value = os.getenv(key)
    if value:
        return value
    try:
        return str(st.secrets[key])
    except (KeyError, FileNotFoundError):
        return default


def money(value, currency: str = "USD") -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if abs(value) >= 1:
        return f"{currency} {value:,.2f}"
    return f"{currency} {value:.8f}".rstrip("0").rstrip(".")


def fmt_compact(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    if abs(value) >= 1e12:
        return f"{value / 1e12:.2f}T"
    if abs(value) >= 1e9:
        return f"{value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.2f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.2f}K"
    return f"{value:,.2f}"


@st.cache_data(ttl=300, show_spinner=False)
def currency_rates(base: str):
    url = f"https://open.er-api.com/v6/latest/{base.upper()}"
    try:
        response = requests.get(url, timeout=12)
        response.raise_for_status()
        payload = response.json()
        if payload.get("result") != "success":
            raise ValueError("Currency service returned an unsuccessful result.")
        return payload["rates"], payload.get("time_last_update_utc", "Unknown")
    except Exception as exc:
        raise RuntimeError(
            "Currency conversion is temporarily unavailable. Try again later."
        ) from exc


def get_ai_client():
    key = get_config("GROQ_API_KEY").strip()
    if not key or key == "your_groq_api_key_here":
        return None, "Configure GROQ_API_KEY in your local .env file or Streamlit Cloud secrets."
    try:
        from langchain_groq import ChatGroq

        model_name = get_config("GROQ_MODEL", "openai/gpt-oss-120b").strip()
        temperature = float(get_config("GROQ_TEMPERATURE", "0.4"))
        return ChatGroq(
            model=model_name,
            temperature=temperature,
            groq_api_key=key,
        ), None
    except Exception as exc:
        return None, f"Could not initialize the Groq client: {exc}"


def invoke_ai(prompt: str, system: str = "You are CryptoMind AI, a cautious cryptocurrency market information assistant.") -> str:
    client, error = get_ai_client()
    if error:
        raise RuntimeError(error)
    result = client.invoke([("system", system), ("human", prompt)])
    content = result.content
    if isinstance(content, str):
        return content
    return str(content)


def fetch_live_market_context(question: str, display_currency: str = "USD") -> str:
    """Retrieve current CoinGecko context in the currency selected in the sidebar."""
    q = question.lower()
    currency_code = display_currency.lower()
    market_terms = (
        "price", "bitcoin", "btc", "ethereum", "ether", "eth", "solana", "sol",
        "crypto", "market cap", "volume", "gainer", "loser", "top coin",
        "current", "today", "trading at", "market", "ada", "cardano", "doge",
    )
    if not any(term in q for term in market_terms):
        return ""

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    context_lines = []
    seen_ids = set()

    try:
        # CoinGecko returns all market values in the requested vs_currency.
        get_markets.cache_clear()
        market_df = get_markets(currency_code, per_page=100)
        if not market_df.empty:
            for _, coin in market_df.iterrows():
                name = str(coin.get("name", ""))
                symbol = str(coin.get("symbol", "")).upper()
                coin_id = str(coin.get("id", ""))
                name_match = bool(name) and name.lower() in q
                symbol_match = symbol.lower() in q.split() or symbol.lower() in q
                if name_match or symbol_match:
                    context_lines.append(_market_row_context(coin, display_currency))
                    seen_ids.add(coin_id)

            if not context_lines and any(term in q for term in ("top coin", "market", "gainer", "loser", "crypto")):
                for _, coin in market_df.head(10).iterrows():
                    context_lines.append(_market_row_context(coin, display_currency))
                    seen_ids.add(str(coin.get("id", "")))
    except MarketDataError as exc:
        context_lines.append(f"Top-market endpoint failed: {exc}")

    common_names = {
        "bitcoin": "bitcoin", "btc": "bitcoin", "ethereum": "ethereum", "ether": "ethereum", "eth": "ethereum",
        "solana": "solana", "sol": "solana", "ripple": "ripple", "xrp": "ripple",
        "dogecoin": "dogecoin", "doge": "dogecoin", "cardano": "cardano", "ada": "cardano",
        "binance coin": "binancecoin", "bnb": "binancecoin", "avalanche": "avalanche-2", "avax": "avalanche-2",
        "polkadot": "polkadot", "dot": "polkadot", "chainlink": "chainlink", "link": "chainlink",
        "litecoin": "litecoin", "ltc": "litecoin",
    }
    matched_ids = []
    for label, coin_id in common_names.items():
        if label in q and coin_id not in seen_ids and coin_id not in matched_ids:
            matched_ids.append(coin_id)

    if matched_ids:
        try:
            prices = get_simple_price(matched_ids, currency_code)
            for coin_id, details in prices.items():
                price = details.get(currency_code)
                if price is None:
                    continue
                change = details.get(f"{currency_code}_24h_change")
                cap = details.get(f"{currency_code}_market_cap")
                volume = details.get(f"{currency_code}_24h_vol")
                price_value = float(price)
                price_text = f"{price_value:,.8f}" if abs(price_value) < 1 else f"{price_value:,.2f}"
                line = f"{coin_id}: price={display_currency.upper()} {price_text}"
                if change is not None:
                    line += f", 24h change={float(change):+.2f}%"
                context_lines.append(line)
                if cap is not None:
                    context_lines.append(f"{coin_id} market cap: {display_currency.upper()} {float(cap):,.0f}")
                if volume is not None:
                    context_lines.append(f"{coin_id} 24h volume: {display_currency.upper()} {float(volume):,.0f}")
        except MarketDataError as exc:
            context_lines.append(f"Direct coin-price endpoint failed: {exc}")

    if not context_lines:
        return (
            f"Market data was requested at {timestamp}, but no matching price was retrieved. "
            "Do not invent or guess a price."
        )
    return (
        f"Market-data retrieval timestamp: {timestamp}\n"
        "Data source: CoinGecko public API.\n"
        f"All retrieved price, market cap, and volume values are denominated in {display_currency.upper()}.\n"
        "Retrieved market data:\n- " + "\n- ".join(context_lines)
    )

def _market_row_context(coin, currency_code: str = "USD") -> str:
    name = coin.get("name", "Unknown coin")
    symbol = str(coin.get("symbol", "")).upper()
    price = coin.get("current_price")
    change = coin.get("price_change_percentage_24h")
    cap = coin.get("market_cap")
    volume = coin.get("total_volume")

    price_text = "unavailable"
    if pd.notna(price):
        price_value = float(price)
        price_text = f"{price_value:,.8f}" if abs(price_value) < 1 else f"{price_value:,.2f}"

    parts = [f"{name} ({symbol}): price {currency_code.upper()} {price_text}"]
    if pd.notna(change):
        parts.append(f"24h change {float(change):+.2f}%")
    if pd.notna(cap):
        parts.append(f"market cap {currency_code.upper()} {float(cap):,.0f}")
    if pd.notna(volume):
        parts.append(f"24h volume {currency_code.upper()} {float(volume):,.0f}")
    return " | ".join(parts)


def is_forecast_question(question: str) -> bool:
    q = question.lower()
    forecast_words = ("predict", "prediction", "forecast", "project", "price in", "price after", "what will", "estimate")
    return any(word in q for word in forecast_words) and any(word in q for word in ("price", "coin", "crypto", "bitcoin", "ethereum", "ada", "btc", "eth", "sol", "doge", "cardano"))


def parse_forecast_target_date(question: str) -> date:
    """Parse common future-date wording, defaulting vague horizons to 30 days."""
    import re
    from dateutil.relativedelta import relativedelta

    q = question.lower().replace(",", " ")
    today = date.today()
    # Explicit ISO date: 2026-12-25
    iso_match = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", q)
    if iso_match:
        return date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))

    # Month-name dates, e.g. December 25, Dec 25 2026, or 25 December.
    months = {
        "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
        "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
        "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
    }
    month_pattern = "|".join(sorted(months, key=len, reverse=True))
    match = re.search(rf"\b({month_pattern})\s+(\d{{1,2}})(?:\s+(20\d{{2}}))?\b", q)
    if match:
        month = months[match.group(1)]
        day = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else today.year
        try:
            target = date(year, month, day)
        except ValueError:
            raise ValueError("That date is not valid. Please check the month and day.")
        if not match.group(3) and target <= today:
            target = date(year + 1, month, day)
        return target
    match = re.search(rf"\b(\d{{1,2}})\s+({month_pattern})(?:\s+(20\d{{2}}))?\b", q)
    if match:
        day = int(match.group(1))
        month = months[match.group(2)]
        year = int(match.group(3)) if match.group(3) else today.year
        try:
            target = date(year, month, day)
        except ValueError:
            raise ValueError("That date is not valid. Please check the month and day.")
        if not match.group(3) and target <= today:
            target = date(year + 1, month, day)
        return target

    horizon = re.search(r"\b(\d+)\s*(day|days|week|weeks|month|months|year|years)\b", q)
    if horizon:
        amount = int(horizon.group(1))
        unit = horizon.group(2)
        if unit.startswith("day"):
            return today + timedelta(days=amount)
        if unit.startswith("week"):
            return today + timedelta(weeks=amount)
        if unit.startswith("month"):
            return today + relativedelta(months=amount)
        return today + relativedelta(years=amount)
    if "next month" in q or "in a month" in q:
        return today + relativedelta(months=1)
    if "next year" in q or "in a year" in q:
        return today + relativedelta(years=1)
    return today + timedelta(days=30)


def find_coin_for_forecast(question: str) -> dict:
    """Resolve a coin name/ticker through CoinGecko search, with common aliases."""
    import re
    q = question.lower()
    aliases = {
        "btc": "bitcoin", "bitcoin": "bitcoin", "eth": "ethereum", "ether": "ethereum",
        "ethereum": "ethereum", "ada": "cardano", "cardano": "cardano", "sol": "solana",
        "solana": "solana", "doge": "dogecoin", "dogecoin": "dogecoin", "xrp": "ripple",
        "ripple": "ripple", "bnb": "binancecoin", "ltc": "litecoin", "dot": "polkadot",
        "avax": "avalanche-2", "link": "chainlink",
    }
    # Search common names and tickers first; word boundaries avoid matching e.g. "sol" inside a longer word.
    for alias, coin_id in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", q):
            results = search_coins(alias)
            exact = next((coin for coin in results if coin.get("id") == coin_id), None)
            if exact:
                return exact
            # Stable fallback IDs for common coins when search is rate-limited or misses an alias.
            return {"id": coin_id, "name": coin_id.replace("-", " ").title(), "symbol": alias}

    # Remove forecast phrasing/date words and use remaining words as the coin search query.
    cleaned = re.sub(r"\b(predict|prediction|forecast|project|price|of|for|the|on|at|in|after|next|month|months|year|years|day|days|week|weeks|will|be|to|by|what|estimate)\b", " ", q)
    cleaned = re.sub(r"\b(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|sept|october|oct|november|nov|december|dec)\b", " ", cleaned)
    cleaned = re.sub(r"\b\d{1,4}\b", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9 ]", " ", cleaned)
    query = " ".join(cleaned.split())
    if not query:
        raise ValueError("Please include a coin name or ticker, for example: predict ADA on December 25, 2026.")
    results = search_coins(query)
    if not results:
        raise ValueError(f"I couldn't find a CoinGecko coin matching '{query}'. Try its full name or ticker symbol.")
    # Prefer exact name/symbol matches over fuzzy search order.
    exact = next((coin for coin in results if coin.get("name", "").lower() == query or coin.get("symbol", "").lower() == query), None)
    return exact or results[0]


def build_forecast_answer(question: str, selected_currency: str) -> str:
    """Calculate a deterministic, data-based baseline forecast without asking the LLM to invent one."""
    target = parse_forecast_target_date(question)
    if target <= date.today():
        raise ValueError("Please choose a future date. Example: predict ADA on December 25, 2026.")
    coin = find_coin_for_forecast(question)
    result = forecast_coin_price(coin["id"], target, selected_currency.lower(), 365)
    name = coin.get("name", coin["id"])
    symbol = str(coin.get("symbol", "")).upper()
    code = selected_currency.upper()
    projected = money(result["projected_price"], code)
    low = money(result["lower_range"], code)
    high = money(result["upper_range"], code)
    current = money(result["current_price"], code)
    return (
        f"### Illustrative {name} ({symbol}) price projection\n\n"
        f"- **Latest historical price:** {current} {code}\n"
        f"- **Target date:** {result['target_date'].strftime('%B %d, %Y')}\n"
        f"- **Days ahead:** {result['horizon_days']}\n"
        f"- **Baseline projected price:** {projected} {code}\n"
        f"- **Illustrative lower scenario:** {low} {code}\n"
        f"- **Illustrative upper scenario:** {high} {code}\n\n"
        f"Historical data is from CoinGecko through {pd.Timestamp(result['as_of']).strftime('%Y-%m-%d')}. "
        f"This is a simple statistical extrapolation from recent historical returns, not a validated prediction. "
        f"The lower and upper values are illustrative scenarios, not a reliable confidence interval or guaranteed range. "
        f"It cannot account for future news, liquidity shifts, regulation, market manipulation, or sudden market changes. "
        f"Do not use it alone to make investment decisions."
    )


# Sidebar
st.sidebar.markdown("## ₿ CryptoMind AI")
st.sidebar.caption("Cryptocurrency dashboard and AI assistant")
currency = st.sidebar.selectbox("Display currency", list(CURRENCIES), index=0)
period_labels = {
    "1 day": 1, "7 days": 7, "14 days": 14, "30 days": 30,
    "90 days": 90, "180 days": 180, "1 year": 365,
}
period_label = st.sidebar.selectbox("Chart history", list(period_labels), index=4)
days = period_labels[period_label]
top_n = st.sidebar.slider("Coins in market table", 10, 100, 30, step=10)
st.sidebar.divider()
st.sidebar.caption("Market data: CoinGecko")
st.sidebar.caption("AI features require a Groq API key.")

st.markdown('<div class="main-title">₿ CryptoMind AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Live cryptocurrency markets, technical indicators, and AI-powered explanations.</div>',
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "📊 Market Dashboard",
    "🔎 Coin Explorer",
    "📈 Price Forecast",
    "🤖 AI Assistant",
    "💱 Currency Converter",
])

# Load dashboard data
try:
    with st.spinner("Loading market data from CoinGecko…"):
        markets = get_markets(currency.lower(), per_page=top_n)
except MarketDataError as exc:
    markets = pd.DataFrame()
    st.error(str(exc))

with tabs[0]:
    if not markets.empty:
        c1, c2, c3, c4 = st.columns(4)
        first = markets.iloc[0]
        rank = first.get("market_cap_rank")
        c1.metric(
            "Top coin",
            str(first["name"]),
            f"Rank #{int(rank)}" if pd.notna(rank) else None,
        )
        total_cap = markets["market_cap"].fillna(0).sum()
        total_vol = markets["total_volume"].fillna(0).sum()
        c2.metric(f"Combined cap (top {len(markets)})", f"{currency.upper()} {fmt_compact(total_cap)}")
        c3.metric("Combined 24h volume", f"{currency.upper()} {fmt_compact(total_vol)}")
        changes = markets["price_change_percentage_24h"].dropna()
        c4.metric("Gainers in loaded list", f"{int((changes > 0).sum())}/{len(changes)}")

        st.subheader("Top cryptocurrencies")
        show = markets.copy()
        show["Price"] = show["current_price"].apply(lambda x: money(x, currency.upper()))
        show["24h change"] = show["price_change_percentage_24h"].apply(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"
        )
        show["Market cap"] = show["market_cap"].apply(lambda x: f"{currency.upper()} {fmt_compact(x)}")
        show["24h volume"] = show["total_volume"].apply(lambda x: f"{currency.upper()} {fmt_compact(x)}")
        display_cols = {
            "market_cap_rank": "Rank", "name": "Coin", "symbol": "Symbol",
            "Price": "Price", "24h change": "24h change",
            "Market cap": "Market cap", "24h volume": "24h volume",
        }
        table = show[[col for col in display_cols if col in show.columns]].rename(columns=display_cols)
        st.dataframe(table, use_container_width=True, hide_index=True)

        st.subheader("24-hour performance")
        chart_df = markets.dropna(subset=["price_change_percentage_24h"]).head(15).copy()
        if not chart_df.empty:
            chart_df = chart_df.sort_values("price_change_percentage_24h")
            fig = go.Figure(go.Bar(
                x=chart_df["price_change_percentage_24h"],
                y=chart_df["name"],
                orientation="h",
                text=chart_df["price_change_percentage_24h"].map(lambda x: f"{x:+.2f}%"),
                textposition="outside",
                hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
            ))
            fig.update_layout(
                height=470, xaxis_title="24-hour change (%)", yaxis_title="",
                margin=dict(l=5, r=40, t=15, b=5),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Market dashboard is unavailable until CoinGecko returns data.")

with tabs[1]:
    st.subheader("Explore a cryptocurrency")
    search_query = st.text_input("Search by coin name or symbol", placeholder="e.g. bitcoin, ethereum, sol")
    search_results = []
    if search_query.strip():
        try:
            with st.spinner("Searching coins…"):
                search_results = search_coins(search_query)
        except MarketDataError as exc:
            st.warning(str(exc))

    if search_results:
        options = {
            f"{coin['name']} ({coin['symbol'].upper()}) · {coin['id']}": coin
            for coin in search_results
        }
        selected_label = st.selectbox("Select a result", list(options))
        selected_coin = options[selected_label]
        coin_id = selected_coin["id"]
        coin_name = selected_coin["name"]
        coin_symbol = selected_coin["symbol"].upper()
    elif search_query.strip():
        st.info("No results found. Try a different name or symbol.")
        coin_id, coin_name, coin_symbol = "bitcoin", "Bitcoin", "BTC"
    elif not markets.empty:
        market_options = {
            f"{row['name']} ({str(row['symbol']).upper()})": row
            for _, row in markets.iterrows()
        }
        chosen = st.selectbox("Or choose from popular coins", list(market_options))
        row = market_options[chosen]
        coin_id, coin_name, coin_symbol = row["id"], row["name"], str(row["symbol"]).upper()
    else:
        coin_id, coin_name, coin_symbol = "bitcoin", "Bitcoin", "BTC"

    try:
        with st.spinner(f"Loading {coin_name} history…"):
            history = add_technical_indicators(get_coin_history(coin_id, currency.lower(), days))
        latest = history.iloc[-1]
        previous = history.iloc[-2] if len(history) > 1 else latest
        change = ((latest["price"] / previous["price"]) - 1) * 100 if previous["price"] else 0
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latest price", money(latest["price"], currency.upper()), f"{change:+.2f}% vs previous data point")
        m2.metric("RSI (14)", f"{latest['RSI 14']:.2f}" if pd.notna(latest["RSI 14"]) else "Not enough data")
        m3.metric("SMA (20)", money(latest["SMA 20"], currency.upper()))
        m4.metric("SMA (50)", money(latest["SMA 50"], currency.upper()))

        st.subheader(f"{coin_name} ({coin_symbol}) price history")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history["date"], y=history["price"], name="Price", mode="lines"))
        fig.add_trace(go.Scatter(x=history["date"], y=history["SMA 20"], name="SMA 20", mode="lines"))
        fig.add_trace(go.Scatter(x=history["date"], y=history["SMA 50"], name="SMA 50", mode="lines"))
        fig.update_layout(
            height=470, xaxis_title="Date (UTC)", yaxis_title=f"Price ({currency.upper()})",
            hovermode="x unified", legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.subheader("RSI (14)")
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=history["date"], y=history["RSI 14"], name="RSI", mode="lines"))
            rsi_fig.add_hline(y=70, line_dash="dash")
            rsi_fig.add_hline(y=30, line_dash="dash")
            rsi_fig.update_layout(height=300, yaxis=dict(range=[0, 100]), xaxis_title="Date (UTC)")
            st.plotly_chart(rsi_fig, use_container_width=True)
        with right:
            st.subheader("MACD")
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=history["date"], y=history["MACD"], name="MACD"))
            macd_fig.add_trace(go.Scatter(x=history["date"], y=history["MACD Signal"], name="Signal"))
            macd_fig.add_trace(go.Bar(x=history["date"], y=history["MACD Histogram"], name="Histogram"))
            macd_fig.update_layout(height=300, xaxis_title="Date (UTC)")
            st.plotly_chart(macd_fig, use_container_width=True)

        with st.expander("View technical indicator data"):
            st.dataframe(history.tail(30).sort_values("date", ascending=False), use_container_width=True, hide_index=True)

        st.markdown("#### AI interpretation")
        if st.button("Explain current indicators with AI", key="explain_indicators"):
            prompt = f"""
Explain these cryptocurrency indicators in plain English. Do not claim certainty or give guaranteed predictions.
Coin: {coin_name} ({coin_symbol})
Currency: {currency.upper()}
Latest price: {latest['price']}
RSI(14): {latest['RSI 14']}
SMA20: {latest['SMA 20']}
SMA50: {latest['SMA 50']}
MACD: {latest['MACD']}
MACD signal: {latest['MACD Signal']}
MACD histogram: {latest['MACD Histogram']}
Explain what the values generally suggest, possible signals, uncertainty, and risks.
"""
            try:
                with st.spinner("Generating explanation…"):
                    st.write(invoke_ai(prompt))
            except Exception as exc:
                st.error(str(exc))
    except MarketDataError as exc:
        st.error(str(exc))

with tabs[2]:
    st.subheader("Cryptocurrency Price Forecast Lab")
    st.caption("Explore an illustrative projection for any coin listed by CoinGecko and any future target date.")
    forecast_query = st.text_input("Search any cryptocurrency", placeholder="e.g. bitcoin, ethereum, solana, or a token name", key="forecast_coin_search")
    forecast_results = []
    if forecast_query.strip():
        try:
            forecast_results = search_coins(forecast_query)
        except MarketDataError as exc:
            st.warning(str(exc))

    if forecast_results:
        forecast_options = {
            f"{coin['name']} ({coin['symbol'].upper()}) · {coin['id']}": coin
            for coin in forecast_results
        }
        forecast_choice = st.selectbox("Choose the exact coin", list(forecast_options), key="forecast_coin_choice")
        forecast_coin = forecast_options[forecast_choice]
        forecast_coin_id = forecast_coin["id"]
        forecast_coin_name = forecast_coin["name"]
        forecast_coin_symbol = forecast_coin["symbol"].upper()
        min_target = date.today() + timedelta(days=1)
        target_date = st.date_input("Forecast target date", value=date.today() + timedelta(days=30), min_value=min_target, key="forecast_target_date")
        st.caption(f"Projection horizon: {(target_date - date.today()).days} days")
        if st.button("Generate projection", type="primary", key="generate_coin_forecast"):
            try:
                with st.spinner(f"Analyzing historical {forecast_coin_name} prices…"):
                    result = forecast_coin_price(forecast_coin_id, target_date, currency.lower(), 365)
                st.warning("This is a simple historical-statistics scenario, not a reliable or validated price prediction. Crypto prices can move sharply; the range is not a guarantee or a true confidence interval.")
                fc1, fc2, fc3, fc4 = st.columns(4)
                fc1.metric("Latest historical price", money(result["current_price"], currency.upper()))
                fc2.metric("Illustrative target estimate", money(result["projected_price"], currency.upper()))
                fc3.metric("Lower scenario", money(result["lower_range"], currency.upper()))
                fc4.metric("Upper scenario", money(result["upper_range"], currency.upper()))
                st.caption(f"CoinGecko history as of {pd.Timestamp(result['as_of']).strftime('%Y-%m-%d')} · target {result['target_date'].strftime('%Y-%m-%d')} · {result['horizon_days']} days · based on {result['history_points']} historical observations.")

                historical = get_coin_history(forecast_coin_id, currency.lower(), 90).copy()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=historical["date"], y=historical["price"], name="Historical price", mode="lines"))
                forecast_frame = result["forecast_frame"]
                fig.add_trace(go.Scatter(x=forecast_frame["date"], y=forecast_frame["upper_range"], name="Upper scenario", line=dict(width=0, dash="dot"), showlegend=True))
                fig.add_trace(go.Scatter(x=forecast_frame["date"], y=forecast_frame["lower_range"], name="Illustrative range", line=dict(width=0), fill="tonexty", fillcolor="rgba(100,149,237,0.18)"))
                fig.add_trace(go.Scatter(x=forecast_frame["date"], y=forecast_frame["projected_price"], name="Drift projection", mode="lines", line=dict(dash="dash")))
                fig.update_layout(height=480, xaxis_title="Date", yaxis_title=f"Price ({currency.upper()})", hovermode="x unified", legend=dict(orientation="h", y=1.08))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("**How this is calculated:** recent historical daily log returns are used to estimate average drift and volatility. The central line extrapolates the drift; the shaded scenario expands with the horizon. It does not account for future news, liquidity, regulation, market manipulation, or sudden regime changes.")
            except MarketDataError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not generate this projection: {exc}")
    elif forecast_query.strip():
        st.info("No matching coin was found. Try its full name or ticker symbol.")
    else:
        st.info("Search for a coin to begin. Select the exact CoinGecko result to avoid ticker-symbol ambiguity.")

with tabs[3]:
    st.subheader("CryptoMind AI Assistant")
    st.caption("Ask about crypto markets, live prices, coin metrics, risk, and technical indicators.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi! I'm CryptoMind AI. I can explain crypto concepts, "
                    "retrieve CoinGecko market data, explain indicators, and calculate illustrative statistical price projections for supported coins and future dates. "
                    "Projections are uncertain scenarios, not guarantees or financial advice."
                ),
            }
        ]

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask a crypto question…")

    if user_question:
        st.session_state.chat_messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Checking market data and preparing your answer…"):
                    # Forecast questions are handled by the deterministic market-data model first.
                    # This prevents the LLM from refusing the request or fabricating a number.
                    if is_forecast_question(user_question):
                        answer = build_forecast_answer(user_question, currency)
                    else:
                        live_context = fetch_live_market_context(user_question, currency)
                        # Send only recent conversation context, plus retrieved data.
                        recent_history = st.session_state.chat_messages[-9:]
                        transcript = "\n".join(
                            f"{message['role'].capitalize()}: {message['content']}"
                            for message in recent_history
                        )
                        prompt = f"""
You are CryptoMind AI, a cryptocurrency information assistant.

Follow these rules:
- Use the supplied retrieved market data to answer questions about current prices.
- Never invent a current price or claim a coin price was retrieved when it was not.
- If a requested coin is missing from the retrieved data, say so clearly.
- The selected display currency is {currency.upper()}. Use that currency consistently for all retrieved price values. State the retrieval timestamp.
- Explain that cryptocurrency prices fluctuate and data may be delayed.
- For forecast requests, the app calculates a separate illustrative statistical projection; do not claim that all forecasting is prohibited.
- Be clear, helpful, and concise. Do not present projections as guarantees or personalized financial advice.

RETRIEVED MARKET CONTEXT:
{live_context}

RECENT CONVERSATION:
{transcript}
"""
                        answer = invoke_ai(prompt)
                st.markdown(answer)
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            except Exception as exc:
                error_message = str(exc)
                st.error(error_message)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": f"Sorry, I couldn't answer that just now. {error_message}",
                })

    if st.button("Clear chat history"):
        st.session_state.chat_messages = []
        st.rerun()

with tabs[4]:
    st.subheader("Currency converter")
    st.caption("Exchange rates are provided by a public latest-rate endpoint and may be delayed.")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        amount = st.number_input("Amount", min_value=0.0, value=100.0, step=10.0)
        base = st.selectbox("From", list(CURRENCIES), index=list(CURRENCIES).index("USD"))
    with col2:
        st.markdown("<div style='text-align:center;padding-top:3rem;font-size:1.5rem'>⇄</div>", unsafe_allow_html=True)
    with col3:
        target = st.selectbox("To", list(CURRENCIES), index=list(CURRENCIES).index("INR"))

    if st.button("Convert currency", type="primary"):
        try:
            with st.spinner("Fetching exchange rates…"):
                rates, updated = currency_rates(base)
            rate = rates.get(target)
            if rate is None:
                st.error(f"No exchange rate was returned for {target}.")
            else:
                st.success(f"{amount:,.2f} {base} = {amount * float(rate):,.2f} {target}")
                st.write(f"Exchange rate: 1 {base} = {float(rate):,.6f} {target}")
                st.caption(f"Last updated by provider: {updated}")
        except RuntimeError as exc:
            st.error(str(exc))

st.markdown("---")
st.markdown(
    '<div class="small-note">Market data is provided by CoinGecko. Data may be delayed or incomplete. '
    'Technical indicators are educational tools, not reliable standalone trading signals. '
    'Cryptocurrency is volatile; this app is not financial advice.</div>',
    unsafe_allow_html=True,
)
