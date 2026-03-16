# 💬 ETF RAG Chatbot (RAG + Tools + MCP) — Deployed on Google Cloud Run

An **ETF portfolio assistant** that helps you interpret the **current market regime** and build **ETF allocation + rebalancing plans** using:

- **RAG (Retrieval-Augmented Generation)** with a local PDF knowledge base (**FAISS**)
- **Tool calling (5+ tools)** for news, prices, risk metrics, and rebalancing math
- **Remote MCP tools (Alpha Vantage MCP server)** for standalone single-stock data
- **Streamlit chat UI** with continuous multi-turn conversation + progress indicators
- **Global rate limiting** across Cloud Run instances using **Firestore**
- **Logging + monitoring-ready** for production (stdout/stderr → Cloud Logging)

✅ **Hard task implemented:** Deploy to cloud with scaling (**Cloud Run**)  
✅ **Medium tasks implemented:** advanced caching, remote MCP server integration  
✅ All **core requirements** covered (RAG, chunking, similarity search, tools, validation, logging, rate limiting, UI)

---

## 🌐 Live App
**Cloud Run URL:** [APP LINK](https://etf-rag-bot-736181277830.europe-central2.run.app)

---

## 📌 Overview

This chatbot is a **domain-specialized financial assistant** that follows a strict workflow for analysis:

1. **Fetch fresh news** (RSS tool -> fallback to Alpha Vantage news sentiment)
2. **Retrieve rules/frameworks from the knowledge base** (FAISS similarity search)
3. **Validate data using tools** (prices, ETF risk metrics)
4. **Build an allocation + rebalancing plan** (tool-backed)
5. **Show sources + tool traces** in the UI

The app is built with **LangChain + LangGraph-based agents**, deployed on **Google Cloud Run** (Docker container), and supports **remote tool integration** via MCP.

---

## ✨ Features

### 🧠 RAG Knowledge Base (PDF -> FAISS)
- Creates a knowledge base of **portfolio construction frameworks and financial/economic theory**
- Uses embeddings + chunking + similarity search
- Stores/reuses FAISS index from `kb_index/` to avoid rebuilding every run

**Core components**
- `RecursiveCharacterTextSplitter` for chunking
- `FAISS` vector store for similarity search
- `OpenAIEmbeddings` for retrieval embeddings

References:
- LangChain text splitters: https://python.langchain.com/docs/concepts/text_splitters/
- LangChain FAISS: https://python.langchain.com/docs/integrations/vectorstores/faiss/
- OpenAI embeddings guide: https://platform.openai.com/docs/guides/embeddings

---

### 🔧 Tool Calling (Custom ETF Tools)
At least **3 tools required** — this project includes **5 primary tools**, all relevant to ETF portfolio workflows:

✅ `fetch_rss_news` — latest macro/news headlines using RSS streams from credible sources
✅ `get_ticker_metadata` — gets financial instrument information from Yahoo Finance, normalizes symbols and detects variants (`.us → .de → ^ (for index instruments) → raw`) 
✅ `fetch_prices` — latest prices from Stooq (with symbol fallbacks)  
✅ `etf_risk_report` — annualized return, volatility, Sharpe ratio, max drawdown  
✅ `rebalance_plan` — portfolio rebalancing plan based on constraints user prompts

**Tool trace UI** shows tool start/end/output per turn.

LangChain tools concept reference:
- https://python.langchain.com/docs/concepts/tools/

---

### 🌐 Medium Level Task: Remote Tools via MCP (Alpha Vantage)
Single-stock requests (AAPL, TSLA, NVDA, etc.) are routed to the **Alpha Vantage MCP server**.

Examples of MCP-provided tools:
- `GLOBAL_QUOTE`
- `COMPANY_OVERVIEW`
- `NEWS_SENTIMENT`

References:
- MCP adapters (LangChain): https://github.com/langchain-ai/langchain-mcp-adapters
- Alpha Vantage MCP server: https://mcp.alphavantage.co/

---

### ⚡ Medium Level Task: Advanced Caching
Caching reduces repeated calls and improves UX:

- `@st.cache_resource` for **agent**, **knowledge base**, **MCP tools**
- `@st.cache_data(ttl=...)` for **prices/news** (short TTL due to ever changing information)

Streamlit caching references:
- `st.cache_resource`: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource
- `st.cache_data`: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data

---

### 🧵 Streaming Chat UI (Thread-safe)
Streamlit does not support updating UI from background worker threads.
To keep the "streaming-like" experience:

- LLM tokens are captured in a **thread-safe queue**
- The Streamlit main thread reads from the queue and updates the UI safely

Streamlit multithreading reference:
- https://docs.streamlit.io/develop/concepts/design/multithreading

---

### 🔒 Security + Validation
Security controls included (finance-domain appropriate):

✅ User input validation (`validate.py`)  
✅ Prompt-injection safety rules (system prompt)  
✅ Langchain`s built-in middleware hooks (PII detection, retries, tool call limits, summarization, todo)  
✅ Rate limiting across Cloud Run instances (Firestore)

LangChain middleware documentation:
- https://docs.langchain.com/oss/python/langchain/middleware/built-in

---

### 🧯 Global Rate Limiting (Firestore)
Rate limiting is enforced globally across **Cloud Run scaling instances** using Firestore:

- Each session has a `client_id`
- Firestore document stores:
  - `count` (atomic increment)
  - `expireAt` (window reset time)

Firestore references:
- Firestore client setup: https://cloud.google.com/firestore/docs/samples/firestore-setup-client-create
- `google-cloud-firestore`: https://pypi.org/project/google-cloud-firestore/

---

### 📈 Hard Level Task: Cloud Deployment With Scaling (Google Cloud Run)
Cloud Run configuration supports autoscaling:

- Stateless container deployment
- `--max-instances` controls scale-out
- Concurrency set to 1 for cleaner chat/streaming UX

Cloud Run references:
- Autoscaling: https://cloud.google.com/run/docs/about-instance-autoscaling
- Concurrency: https://cloud.google.com/run/docs/about-concurrency
- Max instances: https://cloud.google.com/run/docs/configuring/max-instances
- Container contract: https://cloud.google.com/run/docs/container-contract

---

## 🗂️ Project Structure

```bash
.
├── app.py                 # Streamlit chat UI + KB retrieval + streaming wrapper
├── create_agent.py         # Agent + tools + MCP + middleware configuration
├── knowledge_base.py       # KB build/load + FAISS similarity search
├── tools.py                # Custom ETF tools (prices/news/risk/rebalance)
├── validate.py             # User input validation
├── global_limit.py         # Firestore-based global rate limiter
├── logging_utils.py        # Logging setup (stdout/stderr split)
├── requirements.txt
├── Dockerfile
├── .streamlit/
│   └── config.toml
└── README.md
```

---

## ✅ Requirements

- Python **3.12** (project tested with 3.12)
- OpenAI API key
- (Optional) Alpha Vantage API key for MCP tools
- Google Cloud project (Firestore + deployment)

---

## 🧪 Run Locally

### 1) Clone the repo
```bash
git clone https://github.com/TuringCollegeSubmissions/mzukas-AE.2.5/
cd mzukas-AE.2.5
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Set environment variables
Create `.env` or `secrets.toml` inside `.streamlit` directory or export variables:

```bash
OPENAI_API_KEY=your_openai_key
ALPHAVANTAGE_API_KEY=your_alphavantage_key
```

### 4) Run Streamlit
```bash
streamlit run app.py
```

---

## 🧱 Knowledge Base Setup 

On first run:
- if `kb_index/index.faiss` and `kb_index/index.pkl` exist -> **loads instantly**
- otherwise -> it scrapes PDFs and builds the index

**Decision for Cloud Run startup speed:**  
Build the index once locally and commit `kb_index/` into the repo.

---

## 🐳 Docker (Local Test)

### Build
```bash
docker build -t etf-rag-bot .
```

### Run
```bash
docker run -p 8080:8080 \
  -e OPENAI_API_KEY="..." \
  -e ALPHAVANTAGE_API_KEY="..." \
  etf-rag-bot
```

Then open: http://localhost:8080

Docker exec-form CMD reference:
- https://docs.docker.com/reference/dockerfile/#cmd

---

## ☁️ Deploy to Google Cloud Run (Step-by-step)

### 0) Auth + select project
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 1) Enable required APIs
```bash
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  firestore.googleapis.com
```

### 2) Create secrets (Secret Manager)
```bash
echo -n "YOUR_OPENAI_KEY" | gcloud secrets create OPENAI_API_KEY --data-file=-
echo -n "YOUR_ALPHA_KEY" | gcloud secrets create ALPHAVANTAGE_API_KEY --data-file=-
```

Cloud Run secrets reference:
- https://cloud.google.com/run/docs/configuring/services/secrets

### 3) Create Cloud Run service account
```bash
gcloud iam service-accounts create etf-rag-runner \
  --display-name="ETF RAG Cloud Run SA"
```

### 4) Grant roles to the service account
```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:etf-rag-runner@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:etf-rag-runner@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/datastore.user"
```

IAM roles reference:
- https://docs.cloud.google.com/sdk/gcloud/reference/projects/add-iam-policy-binding

### 5) Deploy from local source directory
```bash
gcloud run deploy etf-rag-bot \
  --source . \
  --region europe-central2 \
  --allow-unauthenticated \
  --service-account etf-rag-runner@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --set-secrets OPENAI_API_KEY=OPENAI_API_KEY:latest,ALPHAVANTAGE_API_KEY=ALPHAVANTAGE_API_KEY:latest \
  --memory 2Gi \
  --cpu 1 \
  --timeout 900 \
  --concurrency 1 \
  --max-instances 5 \
  --min-instances 0
```

gcloud deploy reference:
- https://cloud.google.com/sdk/gcloud/reference/run/deploy

---

## 📊 Logging + Monitoring

- Logs written to **stdout/stderr**
- Cloud Run collects them automatically in **Cloud Logging**
- View logs:
```bash
gcloud run services logs read etf-rag-bot --region europe-central2
```

Cloud Run troubleshooting:
- https://cloud.google.com/run/docs/troubleshooting

---

## 🧩 Common Issues

### `NoSessionContext()` / `missing ScriptRunContext`
This occurs when callbacks attempt to update Streamlit UI from background threads.  
Solution used here: capture tokens in isolated manner using worker threads and queueing, and update UI in Streamlit main thread.

Reference:
- https://docs.streamlit.io/develop/concepts/design/multithreading

---

## 📚 References & Resources

Streamlit:
- https://docs.streamlit.io/
- Chat UI (`st.chat_input`, `st.chat_message`): https://docs.streamlit.io/develop/api-reference/chat
- Caching: https://docs.streamlit.io/develop/concepts/architecture/caching
- Multithreading: https://docs.streamlit.io/develop/concepts/design/multithreading

LangChain / LangGraph:
- LangChain docs: https://python.langchain.com/docs/
- Tools: https://python.langchain.com/docs/concepts/tools/
- FAISS: https://python.langchain.com/docs/integrations/vectorstores/faiss/
- Middleware: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Google Cloud:
- Cloud Run docs: https://cloud.google.com/run/docs
- Secrets in Cloud Run: https://cloud.google.com/run/docs/configuring/services/secrets
- Autoscaling: https://cloud.google.com/run/docs/about-instance-autoscaling
- Firestore setup: https://cloud.google.com/firestore/docs/samples/firestore-setup-client-create
- ADC auth guide: https://cloud.google.com/docs/authentication/external/set-up-adc

OpenAI:
- API docs: https://platform.openai.com/docs
- Embeddings: https://platform.openai.com/docs/guides/embeddings

---

## ✅ Project Requirement Checklist

### Core
- [x] Knowledge base (domain PDFs)
- [x] Embeddings + chunking + similarity search (FAISS)
- [x] 3+ tool calls (custom ETF tools + MCP tools)
- [x] Domain specialization (ETF allocation + macro regime)
- [x] Security measures (validation + middleware + injection defenses)
- [x] LangChain integration
- [x] Error handling + logging
- [x] Rate limiting + API key management
- [x] Streamlit UI with tool traces + KB context + progress indicators

### Optional
- [x] Advanced caching (Medium)
- [x] Remote MCP tools integration (Alpha Vantage) (Medium)
- [x] Deploy with scaling (Cloud Run) (Hard)
