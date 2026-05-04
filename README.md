# G42 Financial Intelligence Agent

An enterprise-grade AI agent that ingests financial documents (10-Ks, invoices, reports), extracts structured KPIs, detects year-over-year anomalies, and answers natural language queries grounded in source data — with zero hallucination tolerance.

Built to meet G42's AI Agent recruitment framework: enterprise reliability, governance alignment, measurable outcome-based performance, and human-in-the-loop escalation.

## Architecture

```
PDF / CSV input → pdfplumber extraction → section chunking + embedding → pgvector
                                                                          ↓
Streamlit UI ← Pydantic JSON output ← LangGraph agent ← hybrid retrieval (BM25 + dense)
                                            ↓
                              ratio calculator tool
                              anomaly detector tool
                              trend forecaster tool
```

## Project Structure

```
financial-agent/
├── agent/
│   ├── __init__.py
│   ├── graph.py              # LangGraph agent with tool-calling & state
│   └── prompts.py            # System prompts and governance rules
├── ingestion/
│   ├── __init__.py
│   ├── pdf_extractor.py      # pdfplumber + OCR fallback pipeline
│   └── csv_loader.py         # Structured data ingestion
├── rag/
│   ├── __init__.py
│   ├── chunker.py            # Section-based chunking (not fixed tokens)
│   ├── embedder.py           # text-embedding-3-small via OpenAI
│   ├── store.py              # pgvector / in-memory FAISS store
│   └── retriever.py          # Hybrid BM25 + dense retrieval
├── tools/
│   ├── __init__.py
│   ├── ratio_calculator.py   # P/E, EBITDA margin, D/E, ROE, etc.
│   ├── anomaly_detector.py   # Z-score based YoY anomaly detection
│   └── forecaster.py         # Prophet / statsmodels trend forecaster
├── models/
│   ├── __init__.py
│   └── schemas.py            # Pydantic models for structured output
├── ui/
│   └── app.py                # Streamlit UI (uploader + dashboard + chat)
├── data/                     # Place PDFs and CSVs here
├── tests/
│   ├── test_tools.py
│   ├── test_ingestion.py
│   └── test_rag.py
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### Option A: One-command setup (recommended)

```bash
git clone https://github.com/Sausy21/G42-Financial-Intelligence-Agent.git
cd G42-Financial-Intelligence-Agent
chmod +x setup.sh
./setup.sh
```

This creates a virtual environment, installs all dependencies, generates sample data,
and runs the test suite to verify everything works.

### Option B: Manual setup

```bash
git clone https://github.com/Sausy21/G42-Financial-Intelligence-Agent.git
cd G42-Financial-Intelligence-Agent

# Create and activate virtual environment
python3 -m venv finAgent
source finAgent/bin/activate        # macOS / Linux
# finAgent\Scripts\activate         # Windows PowerShell
# finAgent\Scripts\activate.bat     # Windows CMD

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# Generate sample data
python data/generate_sample.py

# Verify installation
python -m pytest tests/ -v
```

### Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `GROQ_API_KEY` | Yes | Groq cloud LLM — free key at [console.groq.com](https://console.groq.com) |
| `AGENT_MODEL` | No | Override default model (default: `llama-3.1-8b-instant`) |
| `PGVECTOR_URL` | No | PostgreSQL with pgvector (defaults to in-memory FAISS) |

### Run the agent

Always activate the virtual environment first:

```bash
source finAgent/bin/activate

# Streamlit UI (recommended)
streamlit run ui/app.py

# CLI mode
python -m agent.graph --file data/sample_financials.csv \
  --query "What was revenue in FY2024?"

# When done
deactivate
```

## Deploying to Streamlit Cloud

Anyone with the URL can use the deployed agent — no local setup required.

### One-time setup

**1. Push to GitHub**

```bash
cd financial-agent
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/Sausy21/G42-Financial-Intelligence-Agent.git
git push -u origin main
```

**2. Deploy on Streamlit Cloud**

1. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app**
2. Connect your GitHub repo
3. Set **Main file path** to `ui/app.py`
4. Click **Advanced settings** → **Secrets** and paste:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy**

That's it. Streamlit builds the environment from `requirements.txt` automatically.

**3. Share the URL**

Streamlit gives you a public URL like `https://your-app.streamlit.app`. Share it — anyone can upload a financial document and use the agent immediately, no account or install needed.

### Updating the deployed app

```bash
git add .
git commit -m "Update description"
git push origin main
# Streamlit Cloud redeploys automatically on every push
```

### Streamlit Cloud limits (free tier)

| Resource | Limit |
|----------|-------|
| RAM | 1 GB |
| Storage | ephemeral (files reset on restart) |
| Sleep | after 7 days inactivity |
| Apps | unlimited public apps |

The agent fits comfortably within 1 GB RAM. Uploaded documents are held in memory during the session and cleared on restart — this is fine for demos.

## Evaluation Criteria

| Criterion | Standard | How We Measure |
|-----------|----------|---------------|
| Extraction fidelity | KPIs match source exactly | Unit tests against known 10-K values |
| Citation grounding | Every claim cites source paragraph | All responses include `citations` field |
| Anomaly precision | Flag only z > 2.5 outliers | Precision/recall on synthetic anomaly dataset |
| Structured output | All responses are parseable JSON | Pydantic validation on every response |

## Governance Rules

- Transactions > $5M: flagged for human review
- Sovereign entity + $10M+: escalated to CFO
- Agent NEVER approves autonomously — recommends only
- All actions logged to immutable audit trail
- PII never stored or exposed
- Operates under UAE CBUAE + FATF guidelines

## Demo Scenario

1. Upload an NVIDIA 10-K PDF
2. Agent extracts revenue, EBITDA, EPS for 3 fiscal years
3. Flags 40% revenue spike in FY24 as anomaly (z-score > 2.5)
4. Ask: "What drove the Q3 revenue jump?"
5. Agent retrieves and cites the Data Center segment commentary

## License

MIT
