# CryptoMind AI

Streamlit cryptocurrency dashboard using CoinGecko market data, technical indicators, Groq AI chatbot and currency conversion.

## Run locally (Windows PowerShell)

```powershell
cd C:\Projects\Crypto_Trading_Chatbot
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create `.env` in the project root:

```dotenv
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
GROQ_TEMPERATURE=0.4
```

Run:

```powershell
streamlit run app.py
```

## Streamlit Community Cloud

Push `app.py`, `market_data.py`, and `requirements.txt` to GitHub. In the app's Settings > Secrets, add:

```toml
GROQ_API_KEY = "your_actual_groq_api_key"
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_TEMPERATURE = "0.4"
```

The application reads environment variables locally and Streamlit secrets in the cloud. Do not commit `.env` or secrets to GitHub.

CoinGecko public API rate limits can apply. Current market data is fetched for relevant chat questions; if the service fails, the assistant is instructed not to invent prices.
