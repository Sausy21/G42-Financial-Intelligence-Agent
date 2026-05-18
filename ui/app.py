"""
G42 Financial Intelligence Agent — Streamlit UI
"""
import sys, re, logging, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
from agent.graph import FinancialIntelligenceAgent
from models.schemas import AgentResponse

import concurrent.futures

import threading
import time

logging.basicConfig(level=logging.INFO)

_ingest_results: dict = {} # thread-safe result store

st.set_page_config(page_title="G42 Financial Intelligence Agent",
                   page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.stApp{background-color:#0B0F14}
.main .block-container{padding-top:2rem;max-width:1200px}
h1,h2,h3{color:#E8ECF1!important}
.gov-banner{background:linear-gradient(90deg,rgba(16,185,129,.1),rgba(59,130,246,.1));border:1px solid rgba(16,185,129,.3);border-radius:8px;padding:8px 16px;font-size:13px;color:#10B981;margin-bottom:16px}
.audit-entry{border-left:3px solid #3B82F6;padding:8px 12px;margin-bottom:8px;background:rgba(59,130,246,.05);border-radius:0 6px 6px 0;font-size:13px}
.citation-box{background:rgba(59,130,246,.08);border:1px solid rgba(59,130,246,.2);border-radius:8px;padding:10px 14px;margin:4px 0;font-size:12px}
.gov-flag{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);border-radius:8px;padding:10px 14px;color:#EF4444;margin:4px 0;font-size:13px}
.fn-dollar{color:#60A5FA;font-weight:700;font-family:monospace}
.fn-pos{color:#34D399;font-weight:700}
.fn-neg{color:#F87171;font-weight:700}
.fn-pct{color:#FBBF24;font-weight:600}
.fn-big{color:#A78BFA;font-weight:600;font-family:monospace}
.fn-eps{color:#22D3EE;font-weight:600;font-family:monospace}
</style>""", unsafe_allow_html=True)


def highlight_numbers(text: str) -> str:
    if not text:
        return text
    # Dollar amounts
    text = re.sub(r'(\$\s?\d[\d,]*(?:\.\d+)?\s?(?:trillion|billion|million|thousand|T|B|M|K)?)',
                  r'<span class="fn-dollar">\1</span>', text, flags=re.IGNORECASE)
    # Per share
    text = re.sub(r'(\d+\.\d+)\s?(per share)',
                  r'<span class="fn-eps">\1</span> \2', text, flags=re.IGNORECASE)
    # Positive %
    text = re.sub(r'(\+\s?\d[\d,.]*\s?%)', r'<span class="fn-pos">\1</span>', text)
    # Negative %
    text = re.sub(r'(\-\s?\d[\d,.]*\s?%)', r'<span class="fn-neg">\1</span>', text)
    # Neutral %
    def color_pct(m):
        before = text[max(0, m.start()-15):m.start()]
        return m.group(0) if 'class="fn-' in before else f'<span class="fn-pct">{m.group(0)}</span>'
    text = re.sub(r'(?<!\d)(\d[\d,.]*\s?%)', color_pct, text)
    # Large numbers
    def color_big(m):
        before = text[max(0, m.start()-20):m.start()]
        return m.group(0) if ('class="fn-' in before or '"' in before[-3:]) else f'<span class="fn-big">{m.group(0)}</span>'
    text = re.sub(r'(?<!["\w$])(\d{1,3}(?:,\d{3})+(?:\.\d+)?)', color_big, text)
    return text


def render_panels(resp: AgentResponse, resp_idx: int = 0):
    if resp.governance_flags:
        for flag in resp.governance_flags:
            st.markdown(f'<div class="gov-flag">⚠ {flag}</div>', unsafe_allow_html=True)
    if resp.citations:
        with st.expander(f"📎 {len(resp.citations)} source citations"):
            for c in resp.citations:
                page_str = f"p.{c.page}" if c.page else "p.—"
                st.markdown(
                    f'<div class="citation-box">'
                    f'<strong>{c.source_document}</strong>'
                    f'<span style="color:#5A6B80"> · </span>'
                    f'§ {c.section}'
                    f'<span style="color:#5A6B80"> · </span>'
                    f'{page_str}'
                    f'<span style="color:#5A6B80"> · </span>'
                    f'confidence: {c.confidence:.0%}'
                    f'</div>', unsafe_allow_html=True,
                )
    if resp.ratios:
        with st.expander("📊 Financial Ratios"):
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[r.name for r in resp.ratios], y=[r.value for r in resp.ratios], name="Actual", marker_color="#3B82F6"))
            benchmarks = [(r.name, r.benchmark) for r in resp.ratios if r.benchmark]
            if benchmarks:
                fig.add_trace(go.Bar(x=[b[0] for b in benchmarks], y=[b[1] for b in benchmarks], name="Benchmark", marker_color="#5A6B80"))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300, barmode="group")
            st.plotly_chart(fig, use_container_width=True, key=f"ratios_{resp_idx}")
    if resp.anomalies:
        with st.expander(f"⚠ {len(resp.anomalies)} Anomalies Detected"):
            for a in resp.anomalies:
                sev = {"low":"#10B981","medium":"#F59E0B","high":"#EF4444","critical":"#DC2626"}.get(a.severity.value,"#8899AE")
                cls = "fn-pos" if a.change_pct > 0 else "fn-neg"
                st.markdown(f'<span style="color:{sev};font-weight:700">▌ {a.metric}</span> — {a.direction.value} of <span class="{cls}">{a.change_pct:+.1f}%</span> (z-score: {a.z_score:.1f})', unsafe_allow_html=True)
                if a.explanation:
                    st.caption(a.explanation)
    if resp.forecasts:
        with st.expander("🔮 Forecasts"):
            for fi, f in enumerate(resp.forecasts):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[p.period for p in f.points], y=[p.predicted_value for p in f.points], mode="lines+markers", name="Forecast", line=dict(color="#3B82F6", width=2)))
                fig.add_trace(go.Scatter(x=[p.period for p in f.points]+[p.period for p in reversed(f.points)], y=[p.upper_bound for p in f.points]+[p.lower_bound for p in reversed(f.points)], fill="toself", fillcolor="rgba(59,130,246,0.1)", line=dict(width=0), name="90% CI"))
                fig.update_layout(title=f"{f.metric} — {f.model_used}", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
                st.plotly_chart(fig, use_container_width=True, key=f"forecast_{resp_idx}_{fi}")
                if f.mape: st.caption(f"MAPE: {f.mape:.1f}%")
    st.caption(f"Confidence: {resp.confidence:.0%} · Tools: {', '.join(resp.tools_invoked)} · Model: {resp.model_used}")


# ── Session state ──────────────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = FinancialIntelligenceAgent()
    st.session_state.chat_history = []
    st.session_state.responses = []
    # If pgvector is active, restore the document list from the DB
    from rag.retriever import PgVectorStore
    if isinstance(st.session_state.agent.retriever.vector_store, PgVectorStore):
        persisted = st.session_state.agent.retriever.vector_store.list_documents()
        st.session_state.documents = persisted
        if persisted:
            # Rebuild BM25 from persisted chunks (pgvector has the vectors already)
            all_chunks = st.session_state.agent.retriever.vector_store.chunks
            if all_chunks:
                st.session_state.agent.retriever.bm25_index.add(all_chunks)
    else:
        st.session_state.documents = []

agent: FinancialIntelligenceAgent = st.session_state.agent

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 G42 Financial Agent")
    st.markdown('<div class="gov-banner">🛡 CBUAE + FATF Compliant</div>', unsafe_allow_html=True)

    st.markdown("#### Upload Documents")
    uploaded_files = st.file_uploader("PDF or CSV", type=["pdf","csv"], accept_multiple_files=True)
    if uploaded_files:
        for up in uploaded_files:
            if up.name not in st.session_state.documents:
                key = f"_ingest_{up.name}"

                # First visit — kick off background thread
                if key not in st.session_state:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix) as tmp:
                        tmp.write(up.read())
                        tmp_path = tmp.name
                    st.session_state[key] = {"status": "running", "chunks": None, "error": None}

                    def _worker(path, name, k):
                        try:
                            chunks = agent.ingest_document(path, original_name=name)
                            _ingest_results[k] = {"status": "done", "chunks": chunks}
                        except Exception as e:
                            _ingest_results[k] = {"status": "error", "error": str(e)}

                    threading.Thread(target=_worker, args=(tmp_path, up.name, key), daemon=True).start()

                # Step 3 — Check result and update UI  ← REPLACES the old status check
                if key in _ingest_results:
                    result = _ingest_results.pop(key)
                    del st.session_state[key]
                    if result["status"] == "done":
                        st.session_state.documents.append(up.name)
                        st.success(f"✓ {up.name} — {len(result['chunks'])} sections")
                    else:
                        st.error(f"Failed: {result['error']}")
                elif st.session_state.get(key, {}).get("status") == "running":
                    with st.spinner(f"⏳ Indexing {up.name} — please wait…"):
                        time.sleep(3)
                    st.rerun()

    # ── Indexed documents + remove buttons ────────────────────────────
    if st.session_state.documents:
        st.markdown("#### Indexed Documents")
        to_remove = []
        for doc in st.session_state.documents:
            c1, c2 = st.columns([5, 1])
            c1.caption(f"📄 {doc}")
            if c2.button("✕", key=f"rm_{doc}", help=f"Remove {doc}"):
                to_remove.append(doc)
        for doc in to_remove:
            with st.spinner(f"Removing {doc}…"):
                agent.remove_document(doc)
                st.session_state.documents.remove(doc)
            st.toast(f"Removed: {doc}")
            st.rerun()
        if len(st.session_state.documents) > 1:
            st.markdown("")
            if st.button("🗑 Clear All", use_container_width=True):
                agent.clear_all_documents()
                st.session_state.documents.clear()
                st.session_state.chat_history.clear()
                st.session_state.responses.clear()
                st.toast("All documents cleared")
                st.rerun()

    st.markdown("---")
    st.markdown("#### Agent Status")
    ca, cb = st.columns(2)
    ca.metric("Docs", len(st.session_state.documents))
    cb.metric("Queries", len(st.session_state.responses))
    # Show which vector backend is active
    backend = getattr(agent.retriever, 'backend', 'hybrid')
    if backend == "pgvector":
        st.markdown("🟢 **Vector store: pgvector** (persistent)")
    else:
        st.markdown("🟡 **Vector store: FAISS** (in-memory)")
    if agent.audit_log:
        with st.expander("Audit Trail"):
            for entry in reversed(agent.audit_log[-10:]):
                st.markdown(f'<div class="audit-entry"><strong>{entry["action"]}</strong><br><span style="color:#8899AE">{entry["detail"]}</span><br><span style="color:#5A6B80;font-size:11px">{entry["timestamp"]}</span></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Model selector ────────────────────────────────────────────────
    st.markdown("#### 🤖 Model")

    GROQ_MODELS = {
        "llama-3.1-8b-instant": {
            "label": "Llama 3.1 8B",
            "speed": "⚡⚡⚡⚡",
            "quality": "★★★★",
            "note": "Default · best balance",
        },
        "llama-3.3-70b-versatile": {
            "label": "Llama 3.3 70B",
            "speed": "⚡⚡⚡",
            "quality": "★★★★★",
            "note": "Highest quality",
        },
        "llama-3.1-70b-versatile": {
            "label": "Llama 3.1 70B",
            "speed": "⚡⚡⚡",
            "quality": "★★★★★",
            "note": "Large context",
        },
        "mixtral-8x7b-32768": {
            "label": "Mixtral 8×7B",
            "speed": "⚡⚡⚡",
            "quality": "★★★★",
            "note": "Strong reasoning",
        },
        "gemma2-9b-it": {
            "label": "Gemma 2 9B",
            "speed": "⚡⚡⚡",
            "quality": "★★★★",
            "note": "Concise answers",
        },
    }

    options     = list(GROQ_MODELS.keys())
    current_idx = options.index(agent.model) if agent.model in options else 0

    def fmt_model(name):
        m = GROQ_MODELS[name]
        return f"{m['label']}  {m['speed']}  {m['note']}"

    selected = st.selectbox(
        "Active model",
        options,
        index=current_idx,
        format_func=fmt_model,
        help="All models are free on Groq · no download required",
    )
    if selected != agent.model:
        agent.switch_model(selected)
        st.toast(f"Switched to {GROQ_MODELS[selected]['label']}")

    info = GROQ_MODELS.get(agent.model, {})
    if info:
        c1, c2 = st.columns(2)
        c1.caption(f"Speed {info.get('speed','')}")
        c2.caption(f"Quality {info.get('quality','')}")

    st.markdown("---")
    with st.expander("⚡ Speed tips"):
        st.markdown(
            "- Groq is already fast (~1–3s) — no warmup needed\n"
            "- 8B models are faster; 70B models give richer analysis\n"
            "- Shorter, specific questions run faster\n"
            "- Free tier: ~14,400 tokens/min on 8B, ~6,000 on 70B"
        )

# ── Main ───────────────────────────────────────────────────────────────
st.markdown("# 📊 Financial Intelligence Agent")
st.markdown(
    "Numbers are color-coded: "
    '<span class="fn-dollar">$amounts</span> · '
    '<span class="fn-pos">+gains</span> · '
    '<span class="fn-neg">-losses</span> · '
    '<span class="fn-pct">%margins</span> · '
    '<span class="fn-big">large&nbsp;figures</span>',
    unsafe_allow_html=True)

if st.session_state.documents:
    st.markdown("#### Quick Actions")
    qc = st.columns(4)
    quick = [
        ("📈 Extract KPIs", "Extract all key financial KPIs — revenue, net income, EPS, gross margin, operating income — with year-over-year comparison."),
        ("⚠ Flag Anomalies", "Detect statistically unusual year-over-year changes (z-score > 2.5) in the financial data."),
        ("📊 Compute Ratios", "Calculate EBITDA margin, net profit margin, debt-to-equity, ROE, and current ratio."),
        ("🔮 Forecast Revenue", "Forecast revenue for the next 4 periods using historical trend data."),
    ]
    for i, (label, query) in enumerate(quick):
        if qc[i].button(label, use_container_width=True):
            st.session_state.chat_history.append({"role":"user","content":label})
            with st.spinner("🔍 Retrieving context… 🤖 Analyzing…"):
                resp = agent.query(query)
            st.session_state.responses.append(resp)
            st.session_state.chat_history.append({"role":"assistant","content":resp.answer})
            st.rerun()

st.markdown("---")
st.markdown("#### 💬 Ask the Agent")

for i, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(highlight_numbers(msg["content"]), unsafe_allow_html=True)
            idx = i // 2
            if idx < len(st.session_state.responses):
                render_panels(st.session_state.responses[idx], resp_idx=idx)
        else:
            st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your financial documents…"):
    st.session_state.chat_history.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        if not st.session_state.documents:
            answer = "No documents indexed yet. Upload a PDF or CSV using the sidebar."
            st.markdown(answer)
            st.session_state.chat_history.append({"role":"assistant","content":answer})
        else:
            with st.spinner("🔍 Retrieving → 🤖 Analyzing… (10–30s on first query)"):
                resp = agent.query(prompt)
            st.session_state.responses.append(resp)
            st.markdown(highlight_numbers(resp.answer), unsafe_allow_html=True)
            render_panels(resp, resp_idx=len(st.session_state.responses) - 1)
            st.session_state.chat_history.append({"role":"assistant","content":resp.answer})

if not st.session_state.documents and not st.session_state.chat_history:
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.markdown("#### 1. Upload\nDrop a 10-K PDF or CSV in the sidebar.")
    c2.markdown("#### 2. Ask\n*'What drove the FY25 revenue increase?'*")
    c3.markdown("#### 3. Analyze\nCited answers, ratios, anomalies, and forecasts.")
    st.info("💡 Get a 10-K from [SEC EDGAR](https://www.sec.gov/search-filings) by ticker (AAPL, NVDA) or from a company's investor relations page.")
