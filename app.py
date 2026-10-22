from __future__ import annotations

import os
from datetime import datetime
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from market_data import (
    MarketDataError,
    add_technical_indicators,
    get_coin_history,
    get_markets,
    search_coins,
)

load_dotenv()

st.set_page_config(
    page_title="CryptoMind AI",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .main-title {font-size: 2.4rem; font-weight: 800; margin-bottom: 0;}
    .sub-title {opacity: .75; margin-top: .2rem; margin-bottom: 1.2rem;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22);
      border-radius: 12px; padding: 14px;}
    .small-note {font-size: .85rem; opacity: .75;}
</style>
""", unsafe_allow_html=True)

CURRENCIES = {
    "USD": "US Dollar", "INR": "Indian Rupee", "EUR": "Euro",
    "GBP": "British Pound", "JPY": "Japanese Yen", "CAD": "Canadian Dollar",
    "AUD": "Australian Dollar", "SGD": "Singapore Dollar",
}

def money(value, currency="USD"):
    if value is None or pd.isna(value):
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if abs(value) >= 1:
        return f"{currency} {value:,.2f}"
    return f"{currency} {value:.8f}".rstrip("0").rstrip(".")


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
            "Currency conversion is temporarily unavailable. Check your connection and try again."
        ) from exc


def get_ai_client():
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key == "your_groq_api_key_here":
        return None, "Add your GROQ_API_KEY to the .env file to enable AI features."
    try:
        from langchain_groq import ChatGroq
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
        temperature = float(os.getenv("GROQ_TEMPERATURE", "0.4"))
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
    result = client.invoke([
        ("system", system),
        ("human", prompt),
    ])
    return str(result.content)


def fmt_compact(value):
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


# Sidebar controls
st.sidebar.markdown("## ₿ CryptoMind AI")
st.sidebar.caption("Crypto market dashboard and AI assistant")
currency = st.sidebar.selectbox("Display currency", list(CURRENCIES), index=0)
period_labels = {"1 day": 1, "7 days": 7, "14 days": 14, "30 days": 30, "90 days": 90, "180 days": 180, "1 year": 365}
period_label = st.sidebar.selectbox("Chart history", list(period_labels), index=4)
days = period_labels[period_label]
top_n = st.sidebar.slider("Coins in market table", 10, 100, 30, step=10)
st.sidebar.markdown("---")
st.sidebar.caption("Market data: CoinGecko public API")
st.sidebar.caption("AI features require a Groq API key.")

st.markdown('<div class="main-title">₿ CryptoMind AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Live cryptocurrency markets, technical indicators, and AI-powered explanations.</div>', unsafe_allow_html=True)

tabs = st.tabs(["📊 Market Dashboard", "🔎 Coin Explorer", "🤖 AI Assistant", "💱 Currency Converter"])

# Shared market snapshot
try:
    with st.spinner("Loading market data from CoinGecko…"):
        markets = get_markets(currency.lower(), per_page=top_n)
    if markets.empty:
        st.warning("CoinGecko returned no market data. Try again shortly.")
except MarketDataError as exc:
    markets = pd.DataFrame()
    st.error(str(exc))

with tabs[0]:
    if not markets.empty:
        c1, c2, c3, c4 = st.columns(4)
        first = markets.iloc[0]
        c1.metric("Top coin", str(first["name"]), f"Rank #{int(first['market_cap_rank'])}" if pd.notna(first.get("market_cap_rank")) else None)
        total_cap = markets["market_cap"].fillna(0).sum() if "market_cap" in markets else 0
        total_vol = markets["total_volume"].fillna(0).sum() if "total_volume" in markets else 0
        c2.metric(f"Combined cap (top {len(markets)})", f"{currency.upper()} {fmt_compact(total_cap)}")
        c3.metric(f"Combined 24h volume", f"{currency.upper()} {fmt_compact(total_vol)}")
        changes = markets["price_change_percentage_24h"].dropna() if "price_change_percentage_24h" in markets else pd.Series(dtype=float)
        advancing = int((changes > 0).sum())
        c4.metric("Gainers in loaded list", f"{advancing}/{len(changes)}")

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
        display_df = show[[col for col in display_cols if col in show.columns]].rename(columns=display_cols)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.subheader("24-hour performance")
        plot_df = markets.dropna(subset=["price_change_percentage_24h"]).head(15).copy()
        if not plot_df.empty:
            plot_df = plot_df.sort_values("price_change_percentage_24h")
            fig = go.Figure(go.Bar(
                x=plot_df["price_change_percentage_24h"],
                y=plot_df["name"],
                orientation="h",
                text=plot_df["price_change_percentage_24h"].map(lambda x: f"{x:+.2f}%"),
                textposition="outside",
                hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
            ))
            fig.update_layout(height=470, xaxis_title="24-hour change (%)", yaxis_title="", margin=dict(l=5, r=40, t=15, b=5))
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
        options = {f"{coin['name']} ({coin['symbol'].upper()}) · {coin['id']}": coin for coin in search_results}
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
        chosen_row = market_options[chosen]
        coin_id, coin_name, coin_symbol = chosen_row["id"], chosen_row["name"], str(chosen_row["symbol"]).upper()
    else:
        coin_id, coin_name, coin_symbol = "bitcoin", "Bitcoin", "BTC"

    try:
        with st.spinner(f"Loading {coin_name} history…"):
            history = add_technical_indicators(get_coin_history(coin_id, currency.lower(), days))
        latest = history.iloc[-1]
        prev = history.iloc[-2] if len(history) > 1 else latest
        change = ((latest["price"] / prev["price"]) - 1) * 100 if prev["price"] else 0
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
        fig.update_layout(height=470, xaxis_title="Date (UTC)", yaxis_title=f"Price ({currency.upper()})", hovermode="x unified", legend=dict(orientation="h", y=1.08))
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
Explain these crypto indicators in plain English. Do not claim certainty or provide guaranteed predictions.
Coin: {coin_name} ({coin_symbol})
Currency: {currency.upper()}
Latest price: {latest['price']}
RSI(14): {latest['RSI 14']}
SMA20: {latest['SMA 20']}
SMA50: {latest['SMA 50']}
MACD: {latest['MACD']}
MACD signal: {latest['MACD Signal']}
MACD histogram: {latest['MACD Histogram']}
Give: (1) what each value generally suggests, (2) possible bullish/bearish signals, (3) uncertainty and risks.
"""
            try:
                with st.spinner("Generating explanation…"):
                    st.write(invoke_ai(prompt))
            except Exception as exc:
                st.error(str(exc))
    except MarketDataError as exc:
        st.error(str(exc))

with tabs[2]:
    st.subheader("CryptoMind AI Assistant")
    st.caption("Ask about market concepts, coin metrics, risk, or the indicators shown in the Explorer.")
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hi! I'm CryptoMind AI. I can explain crypto concepts, market metrics, and technical indicators. I don't provide guaranteed price predictions."}
        ]
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask a crypto question…")
    if user_question:
        st.session_state.chat_messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        recent_history = st.session_state.chat_messages[-9:]
        transcript = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in recent_history
        )
        prompt = (
            "Answer the user's cryptocurrency question clearly and conversationally. "
            "Distinguish facts from uncertainty. Do not claim access to live prices unless "
            "they are explicitly included in the prompt. Do not give guaranteed predictions. "
            "Mention risk and avoid presenting this as personalized financial advice.\n\n"
            + transcript
        )
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking…"):
                    answer = invoke_ai(prompt)
                st.markdown(answer)
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            except Exception as exc:
                error_text = str(exc)
                st.error(error_text)
                st.session_state.chat_messages.append({"role": "assistant", "content": f"Sorry, I couldn't answer that just now. {error_text}"})

    if st.button("Clear chat history"):
        st.session_state.chat_messages = []
        st.rerun()

with tabs[3]:
    st.subheader("Currency converter")
    st.caption("Exchange rates are from ExchangeRate-API's public latest-rate endpoint and may be delayed.")
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
                result = amount * float(rate)
                st.success(f"{amount:,.2f} {base} = {result:,.2f} {target}")
                st.write(f"Exchange rate: 1 {base} = {float(rate):,.6f} {target}")
                st.caption(f"Last updated by provider: {updated}")
        except RuntimeError as exc:
            st.error(str(exc))

st.markdown("---")
st.markdown(
    '<div class="small-note">Market data is provided by CoinGecko. Data may be delayed or incomplete. '
    'Technical indicators are educational tools, not reliable standalone trading signals. '
    'Crypto assets are volatile; this app is not financial advice.</div>',
    unsafe_allow_html=True,
)
