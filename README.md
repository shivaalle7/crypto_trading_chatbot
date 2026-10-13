# 💹 CryptoMind AI — Advanced Crypto Trading & Investment Assistant

CryptoMind AI is a Generative AI-powered chatbot designed to help users understand cryptocurrency markets, trading strategies, technical indicators, portfolio management, blockchain technology, and crypto cybersecurity.

Built using Python, Streamlit, LangChain, and Groq LLMs, the application provides an interactive conversational experience with context-aware responses and structured crypto education.

> ⚠️ **Disclaimer:** CryptoMind AI is an educational assistant, not a financial adviser. It does not provide guaranteed trading signals or automatically access live market data. Always verify market information independently before making financial decisions.

## ✨ Features

- **AI-Powered Crypto Assistant:** Ask questions about cryptocurrency, blockchain, and trading.
- **Technical Analysis Education:** Learn about RSI, MACD, moving averages, Bollinger Bands, support and resistance, and volume analysis.
- **Investment Analysis:** Explore cryptocurrency utility, tokenomics, market risks, and investment considerations.
- **Bullish and Bearish Scenarios:** Understand potential market opportunities and downside risks.
- **Portfolio Risk Management:** Learn about diversification, position sizing, volatility, and capital preservation.
- **Crypto Scam Awareness:** Understand phishing, rug pulls, fake exchanges, pump-and-dump schemes, and wallet security.
- **Conversational Memory:** Maintain context across messages during the current application session.
- **Custom System Prompt:** Configure the assistant's behavior using `template.txt`.
- **Interactive Web Interface:** Chat with CryptoMind AI through a Streamlit application.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web-based chatbot interface |
| LangChain | Prompt management and conversation workflows |
| Groq API | Access to hosted large language models |
| Llama 3.3 70B Versatile | Configured default language model |
| python-dotenv | Environment variable management |
| Git & GitHub | Version control and project hosting |

## 📁 Project Structure

```text
cryptomind-ai/
│
├── app.py
├── template.txt
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or newer
- Git
- A Groq API key
- Internet connection

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/cryptomind-ai.git
cd cryptomind-ai
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Get your API key from the [Groq Console](https://console.groq.com/keys).

Create your local environment file by copying `.env.example`.

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```

**macOS / Linux**

```bash
cp .env.example .env
```

Open `.env` and configure your values:

```env
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.4
```

Replace the placeholder with your actual API key. Confirm that the selected model is available to your Groq account.

**Security:** Never upload your `.env` file or share your API key publicly.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, typically:

```text
http://localhost:8501
```

## 💬 Example Questions

Try asking CryptoMind AI:

1. What is cryptocurrency, and how does blockchain work?
2. Explain RSI and MACD with practical examples.
3. What are the differences between Bitcoin and Ethereum?
4. How do I identify a potential crypto rug pull?
5. Explain bullish and bearish market scenarios.
6. What is the importance of position sizing in trading?
7. How do staking and DeFi lending work?
8. What are the major risks of investing in altcoins?
9. Explain support and resistance levels.
10. How can I protect my cryptocurrency wallet from phishing attacks?

## 🧠 How It Works

1. The user submits a question through the Streamlit chat interface.
2. The application combines the system prompt, conversation history, and current question.
3. LangChain sends the prompt to the configured Groq language model.
4. The model generates a response based on the provided context and its learned knowledge.
5. The application displays the response and retains the conversation history for subsequent questions in the current session.

## 🔐 Security and Responsible Use

- Never enter wallet seed phrases, private keys, exchange passwords, or API secrets into the chatbot.
- Verify cryptocurrency prices and market news through reliable, current sources.
- Treat AI-generated analysis as informational rather than guaranteed predictions.
- Be cautious of projects promising fixed or guaranteed returns.
- Understand exchange, liquidity, volatility, smart-contract, and regulatory risks before participating in cryptocurrency markets.

## ⚠️ Limitations

- Live cryptocurrency prices and real-time exchange data are not integrated by default.
- The model may generate inaccurate or outdated information.
- Technical analysis requires current price and volume data, which must be supplied or obtained from a connected data source.
- Conversation history is stored in application memory rather than a permanent database.
- The application does not execute trades or connect to cryptocurrency exchanges.

## 🚀 Future Enhancements

Potential improvements include:

- Integration with cryptocurrency market data APIs.
- Interactive price charts and technical indicator visualizations.
- Retrieval-Augmented Generation (RAG) for crypto documentation and research.
- News sentiment analysis using verified current sources.
- Portfolio tracking and risk metrics.
- Watchlists and price alerts.
- Backtesting tools for educational strategy evaluation.
- Automated testing and cloud deployment.

## 👨‍💻 Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement and test your changes.
4. Commit your changes with a descriptive message.
5. Open a pull request describing the improvement.

## 📄 License

No license has been specified yet. Add a `LICENSE` file before granting others permission to reuse, modify, or distribute this project.

## 🙌 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Groq](https://groq.com/)
- [GitHub](https://github.com/)

---

**CryptoMind AI — Analyze smarter. Understand risk. Trade responsibly.** 📈
