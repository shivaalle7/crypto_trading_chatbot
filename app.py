import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

st.set_page_config(
    page_title="CryptoMind AI",
    page_icon="💹",
    layout="centered",
)

st.title("💹 CryptoMind AI")
st.subheader("AI Crypto Trading & Investment Assistant")
st.caption(
    "Ask educational questions about crypto markets, blockchain, risk management, "
    "and investment concepts."
)

# Basic safety/configuration checks
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY is missing. Add it to your local .env file.")
    st.info("Copy .env.example to .env and add your Groq API key.")
    st.stop()

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.4"))

try:
    with open("template.txt", "r", encoding="utf-8") as file:
        system_prompt = file.read().strip()
except FileNotFoundError:
    st.error("template.txt was not found. Keep it in the same folder as app.py.")
    st.stop()

# Store conversation history per Streamlit session.
if "chat_history_store" not in st.session_state:
    st.session_state.chat_history_store = {}

if "session_id" not in st.session_state:
    st.session_state.session_id = "cryptomind-session"

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def build_chat_chain(model_name: str, temperature: float):
    """Build and cache the chat chain for the selected model settings."""
    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    base_chain = prompt | llm

    def get_session_history(session_id: str):
        store = st.session_state.chat_history_store
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

try:
    chat_chain = build_chat_chain(MODEL_NAME, TEMPERATURE)
except Exception as exc:
    st.error("Could not initialize the Groq chat model. Check your configuration.")
    st.caption(f"Technical detail: {exc}")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    st.write(f"**Model:** `{MODEL_NAME}`")
    st.caption("Your API key is read from the local environment; do not commit it.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history_store.pop(st.session_state.session_id, None)
        st.rerun()

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about BTC, ETH, trading concepts, or risk...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = chat_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": st.session_state.session_id}},
                )
                response = result.content if hasattr(result, "content") else str(result)
            except Exception as exc:
                response = (
                    "Sorry, I couldn't generate a response. Check your internet connection, "
                    "Groq API key, model availability, and account limits."
                )
                st.error(f"Request failed: {exc}")

            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
