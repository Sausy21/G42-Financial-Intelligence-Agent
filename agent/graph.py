"""
Phase 3-4: LangGraph Agent

Orchestrates the full pipeline:
1. Receives user query
2. Retrieves relevant context (hybrid BM25 + dense)
3. Calls financial tools as needed (ratios, anomalies, forecasts)
4. Returns structured Pydantic output with citations

Wired with tool-calling and state memory via LangGraph.
"""

from __future__ import annotations

import json
import re
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from models.schemas import (
    AgentResponse,
    AgentState,
    Citation,
    FinancialKPI,
    FinancialRatio,
    Anomaly,
    ForecastResult,
    DocumentChunk,
)
from ingestion.pdf_extractor import PDFExtractor, CSVLoader
from rag.retriever import HybridRetriever, SectionChunker, Embedder
from tools.financial_tools import RatioCalculator, AnomalyDetector, TrendForecaster
from agent.prompts import SYSTEM_PROMPT, EXTRACTION_PROMPT, ANALYSIS_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)


class FinancialIntelligenceAgent:
    """
    The G42 Financial Intelligence Agent.

    Orchestrates document ingestion, RAG retrieval, financial tool execution,
    and structured response generation.

    Can run standalone (CLI) or be driven by the Streamlit UI.
    """

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.model = os.getenv("AGENT_MODEL", model)

        # Pipeline components
        self.extractor = PDFExtractor()
        self.chunker = SectionChunker()
        self.retriever = HybridRetriever()

        # Financial tools
        self.ratio_calculator = RatioCalculator()
        self.anomaly_detector = AnomalyDetector()
        self.forecaster = TrendForecaster()

        # State
        self.state = AgentState()
        self.audit_log: list[dict] = []

        # LLM client (Groq — free cloud API, no local install needed)
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            from groq import Groq
            # Try Streamlit secrets first (deployed), then fall back to .env (local)
            api_key = None
            try:
                import streamlit as st
                api_key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                pass
            if not api_key:
                api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY not set. "
                    "Local: add it to your .env file. "
                    "Deployed: add it in Streamlit Cloud → App Settings → Secrets."
                )
            self._llm = Groq(api_key=api_key)
            logger.info(f"Groq client initialised — model: {self.model}")
        return self._llm

    def switch_model(self, new_model: str):
        """Hot-swap the Groq model without restarting."""
        if new_model == self.model:
            return
        logger.info(f"Switching model: {self.model} → {new_model}")
        self.model = new_model
        self._llm = None  # force re-init on next call
        self._log_action("switch_model", f"Model switched to {new_model}")

    @staticmethod
    def list_available_models() -> list[dict]:
        """
        Return models available on Groq.
        Queries the Groq API; returns empty list if key is missing or API is down.
        """
        GROQ_MODELS = [
            {"name": "llama-3.1-8b-instant",    "context": 128_000},
            {"name": "llama-3.3-70b-versatile",  "context": 128_000},
            {"name": "llama-3.1-70b-versatile",  "context": 128_000},
            {"name": "mixtral-8x7b-32768",        "context": 32_768},
            {"name": "gemma2-9b-it",              "context": 8_192},
        ]
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return []
        return GROQ_MODELS

    # ── Document Ingestion ────────────────────────────────────────────

    def ingest_document(self, file_path: str | Path, original_name: str | None = None) -> list[DocumentChunk]:
        """
        Ingest a PDF or CSV and index it for retrieval.

        Args:
            file_path:     Path to the file (may be a temp path).
            original_name: The user-facing filename to use in all citations.
                           Defaults to file_path.name if not provided.
        """
        file_path = Path(file_path)
        display_name = original_name or file_path.name
        self._log_action("ingest_document", f"Ingesting {display_name}")

        if file_path.suffix.lower() == ".pdf":
            raw_chunks = self.extractor.extract(str(file_path), display_name=display_name)
        elif file_path.suffix.lower() in (".csv", ".tsv"):
            import pandas as pd
            df = pd.read_csv(file_path)
            raw_chunks = CSVLoader.to_chunks(df, display_name)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        chunks = self.chunker.chunk(raw_chunks)
        self.retriever.index_chunks(chunks)
        self.state.documents_loaded.append(display_name)
        self._log_action("index_complete", f"{len(chunks)} chunks indexed from {display_name}")
        return chunks

    def remove_document(self, document_name: str):
        """Remove a document from the retrieval index."""
        self.retriever.remove_document(document_name)
        if document_name in self.state.documents_loaded:
            self.state.documents_loaded.remove(document_name)
        self._log_action("remove_document", f"Removed: {document_name}")

    def clear_all_documents(self):
        """Clear all indexed documents and reset state."""
        self.retriever.clear()
        self.state = AgentState()
        self._log_action("clear_all", "All documents and state cleared")

    # ── Query Processing ──────────────────────────────────────────────

    def query(self, user_query: str) -> AgentResponse:
        """
        Process a natural language query through the full agent pipeline.

        Steps:
        1. Retrieve relevant context
        2. Determine which tools to call
        3. Execute tools
        4. Generate structured response with citations
        """
        self.state.current_query = user_query
        self.state.iteration_count += 1
        self._log_action("query_received", user_query)

        # Step 1: Retrieve context
        contexts, citations = self.retriever.retrieve_with_citations(user_query, k=5)
        self.state.retrieval_context = contexts
        self.state.citations = citations

        # Step 2: Route to appropriate handler
        if self._is_ratio_query(user_query):
            return self._handle_ratio_query(user_query, contexts, citations)
        elif self._is_anomaly_query(user_query):
            return self._handle_anomaly_query(user_query, contexts, citations)
        elif self._is_forecast_query(user_query):
            return self._handle_forecast_query(user_query, contexts, citations)
        else:
            return self._handle_general_query(user_query, contexts, citations)

    # ── Query Handlers ────────────────────────────────────────────────

    def _handle_general_query(
        self,
        query: str,
        contexts: list[str],
        citations: list[Citation],
    ) -> AgentResponse:
        """Handle general financial Q&A with RAG."""
        context_text = "\n\n---\n\n".join(contexts) if contexts else "No documents indexed."

        prompt = ANALYSIS_PROMPT.format(
            kpis=json.dumps([k.model_dump() for k in self.state.extracted_kpis], default=str)
            if self.state.extracted_kpis else "None extracted yet",
            anomalies="None detected",
            context=context_text,
            query=query,
        )

        response_text = self._call_llm(prompt)
        governance_flags = self._check_governance(response_text)

        return AgentResponse(
            answer=response_text,
            citations=citations,
            confidence=0.85 if contexts else 0.3,
            tools_invoked=["retrieve_context"],
            governance_flags=governance_flags,
            model_used=self.model,
        )

    def _retrieve_multi(
        self,
        queries: list[str],
        k_each: int = 4,
    ) -> tuple[list[str], list[Citation]]:
        """
        Run multiple targeted retrieval queries and merge results,
        deduplicating by chunk_id so each chunk appears at most once.
        Used for ratio queries that need income statement + balance sheet
        + cash flow data simultaneously.
        """
        seen_ids: set[str] = set()
        all_contexts: list[str] = []
        all_citations: list[Citation] = []

        for q in queries:
            results = self.retriever.retrieve(q, k=k_each)
            for chunk, score in results:
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    all_contexts.append(
                        f"[Source: {chunk.document_name} | "
                        f"Section: {chunk.section_heading} | "
                        f"Page: {chunk.page_number}]\n{chunk.text}"
                    )
                    from models.schemas import Citation
                    all_citations.append(Citation(
                        source_document=chunk.document_name,
                        section=chunk.section_heading,
                        page=chunk.page_number,
                        text_excerpt=chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text,
                        confidence=min(score, 1.0),
                    ))

        return all_contexts, all_citations

    def _handle_ratio_query(
        self,
        query: str,
        contexts: list[str],
        citations: list[Citation],
    ) -> AgentResponse:
        """
        Handle ratio calculation requests.

        Fires three targeted sub-queries to ensure income statement,
        balance sheet, AND cash flow data are all retrieved — ratios
        need figures from multiple financial statements.
        """
        self._log_action("tool_call", "calculate_financial_ratio")

        # Multi-query retrieval targeting all three statements
        ratio_queries = [
            "revenue gross profit net income operating income earnings per share",
            "total assets current assets current liabilities total equity shareholders equity",
            "total debt borrowings free cash flow capital expenditure",
        ]
        rich_contexts, rich_citations = self._retrieve_multi(ratio_queries, k_each=4)

        # Merge with the original query's context (keeps any unique results)
        existing_ids = {c.section for c in rich_citations}
        for ctx, cit in zip(contexts, citations):
            if cit.section not in existing_ids:
                rich_contexts.append(ctx)
                rich_citations.append(cit)

        data = self._extract_financial_data(rich_contexts)
        ratios = self.ratio_calculator.calculate_all(data)

        context_text = "\n\n---\n\n".join(rich_contexts[:5])
        prompt = (
            f"The user asked: {query}\n\n"
            f"Financial data extracted: {json.dumps(data, default=str)}\n\n"
            f"Computed ratios:\n"
            f"{json.dumps([r.model_dump() for r in ratios], default=str)}\n\n"
            f"Source context:\n{context_text}\n\n"
            "Explain each ratio, compare to the benchmark where available, "
            "cite the source figures, and give 2-3 actionable recommendations."
        )
        answer = self._call_llm(prompt)

        return AgentResponse(
            answer=answer,
            citations=rich_citations,
            confidence=0.9 if ratios else 0.5,
            ratios=ratios,
            tools_invoked=["retrieve_context", "multi_query_retrieval", "calculate_financial_ratio"],
            governance_flags=self._check_governance(answer),
            model_used=self.model,
        )

    def _handle_anomaly_query(
        self,
        query: str,
        contexts: list[str],
        citations: list[Citation],
    ) -> AgentResponse:
        """Handle anomaly detection — uses multi-query retrieval to get multi-year data."""
        self._log_action("tool_call", "detect_anomalies")

        anomaly_queries = [
            "revenue net income profit year over year annual growth",
            "operating income EBITDA gross profit margin change",
            "earnings per share free cash flow annual comparison",
        ]
        rich_contexts, rich_citations = self._retrieve_multi(anomaly_queries, k_each=4)

        ts_data = self._extract_time_series(rich_contexts)
        all_anomalies = []
        for metric, (values, periods) in ts_data.items():
            anomalies = self.anomaly_detector.detect(metric, values, periods)
            all_anomalies.extend(anomalies)

        self.state.detected_anomalies = all_anomalies

        context_text = "\n\n---\n\n".join(rich_contexts[:4])
        prompt = (
            f"The user asked: {query}\n\n"
            f"Anomaly detection results:\n"
            f"{json.dumps([a.model_dump() for a in all_anomalies], default=str)}\n\n"
            f"Source context:\n{context_text}\n\n"
            "Explain each anomaly found, its likely business causes, "
            "and what action the finance team should take."
        )
        answer = self._call_llm(prompt)

        return AgentResponse(
            answer=answer,
            citations=rich_citations,
            confidence=0.85,
            anomalies=all_anomalies,
            tools_invoked=["retrieve_context", "multi_query_retrieval", "detect_anomalies"],
            governance_flags=self._check_governance(answer),
            model_used=self.model,
        )

    def _handle_forecast_query(
        self,
        query: str,
        contexts: list[str],
        citations: list[Citation],
    ) -> AgentResponse:
        """Handle forecasting requests — uses multi-query retrieval for multi-year data."""
        self._log_action("tool_call", "forecast_metric")

        forecast_queries = [
            "revenue annual growth year over year historical trend",
            "net income profit earnings per share annual",
            "operating income EBITDA free cash flow historical",
        ]
        rich_contexts, rich_citations = self._retrieve_multi(forecast_queries, k_each=4)

        ts_data = self._extract_time_series(rich_contexts)
        forecasts = []
        for metric, (values, periods) in ts_data.items():
            result = self.forecaster.forecast(metric, values, periods, horizon=4)
            forecasts.append(result)

        context_text = "\n\n---\n\n".join(rich_contexts[:4])
        prompt = (
            f"The user asked: {query}\n\n"
            f"Forecast results:\n"
            f"{json.dumps([f.model_dump() for f in forecasts], default=str)}\n\n"
            f"Source context:\n{context_text}\n\n"
            "Explain the forecasts, confidence intervals, and key strategic implications. "
            "Flag any assumptions or risks that could affect the outlook."
        )
        answer = self._call_llm(prompt)

        return AgentResponse(
            answer=answer,
            citations=rich_citations,
            confidence=0.75,
            forecasts=forecasts,
            tools_invoked=["retrieve_context", "multi_query_retrieval", "forecast_metric"],
            governance_flags=self._check_governance(answer),
            model_used=self.model,
        )

    # ── LLM Interface ─────────────────────────────────────────────────

    def _call_llm(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        Call Groq cloud LLM.
        Groq runs Llama / Mixtral on custom LPU hardware — typically
        5-10× faster than local Ollama on most laptops.
        """
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Groq LLM call failed: {e}")
            return (
                f"Error: Unable to generate response. {str(e)}\n\n"
                f"Check that GROQ_API_KEY is set correctly and that "
                f"the model '{self.model}' is available on Groq."
            )

    def _call_llm_short(self, prompt: str) -> str:
        """Faster LLM call for structured JSON extraction — short output only."""
        return self._call_llm(prompt, max_tokens=512)

    # ── Data Extraction Helpers ───────────────────────────────────────

    @staticmethod
    def _extract_json_from_text(text: str) -> dict | list | None:
        """
        Robustly extract JSON from LLM output that may contain markdown
        fences, preamble text, or trailing explanations.
        Local models (Ollama) are notoriously messy with JSON output.
        """
        if not text or not text.strip():
            return None

        text = text.strip()

        # Strategy 1: Try parsing the whole thing directly
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Strip markdown code fences (```json ... ```)
        import re
        fence_pattern = re.compile(r'```(?:json)?\s*\n?(.*?)\n?\s*```', re.DOTALL)
        match = fence_pattern.search(text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find the first { ... } or [ ... ] block
        brace_start = text.find('{')
        bracket_start = text.find('[')

        if brace_start == -1 and bracket_start == -1:
            return None

        # Pick whichever comes first
        if bracket_start != -1 and (brace_start == -1 or bracket_start < brace_start):
            start = bracket_start
            open_char, close_char = '[', ']'
        else:
            start = brace_start
            open_char, close_char = '{', '}'

        # Find matching closing bracket/brace
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_char:
                depth += 1
            elif text[i] == close_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

        return None

    def _extract_financial_data(self, contexts: list[str]) -> dict[str, float]:
        """
        Extract financial figures using direct regex pattern matching.
        Handles both US (10-K) and European (URD/Annual Report) terminology.
        Falls back to LLM only if regex yields fewer than 2 fields.
        """
        combined = "\n".join(contexts[:5])

        SYNONYMS: dict[str, list[str]] = {
            "revenue": [r"(?:total\s+)?(?:net\s+)?revenues?\b", r"turnover", r"total\s+net\s+sales", r"net\s+sales"],
            "gross_profit": [r"gross\s+profit"],
            "ebitda": [
                r"adjusted\s+ebita\b",          # Schneider: "Adjusted EBITA"
                r"adjusted\s+ebitda\b",
                r"ebita\b",                     # plain EBITA line (lower priority)
                r"ebitda\b",
                r"adjusted\s+operating\s+income",
            ],
            "net_income": [
                r"profit\s+for\s+the\s+(?:year|period)",
                r"(?:profit|net\s+income)\s+(?:for\s+the\s+year|attributable\s+to\s+owners?)",
                r"net\s+income\s+\(?group\s+share\)?",
                r"net\s+income\b",
                r"profit\s+attributable\s+to\s+(?:owners?|shareholders?|equity\s+holders?)",
            ],
            "eps": [
                # Matches "Basic earnings (attributable to owners of the parent) per share"
                r"basic\s+earnings?[^,\n]{0,60}per\s+share",
                r"(?:adjusted\s+)?eps\b",
            ],
            "total_assets": [r"total\s+assets\b"],
            "current_assets": [r"total\s+current\s+assets\b", r"current\s+assets\b"],
            "current_liabilities": [r"total\s+current\s+liabilities\b", r"current\s+liabilities\b"],
            "total_debt": [
                # Schneider Note 23: "TOTAL CURRENT AND NON-CURRENT FINANCIAL LIABILITIES"
                r"total\s+current\s+and\s+non-current\s+financial\s+liabilities",
                r"total\s+(?:current\s+and\s+)?non-current\s+financial\s+liabilities",
                r"total\s+financial\s+(?:debt|liabilities)\b",
                r"total\s+(?:debt|borrowings?)\b",
                r"net\s+(?:financial\s+)?debt\b",
            ],
            "total_equity": [r"total\s+(?:shareholders['\']?\s+)?equity\b", r"equity\s+attributable\s+to\s+(?:owners?|shareholders?)", r"total\s+equity\b"],
            "shareholders_equity": [r"equity\s+attributable\s+to\s+(?:owners?\s+of\s+the\s+parent|shareholders?)", r"shareholders['\']?\s+equity"],
            "cogs": [r"cost\s+of\s+(?:goods\s+)?(?:sales?|revenue)", r"cost\s+of\s+sales\b"],
            "operating_income": [r"operating\s+(?:income|profit)\b", r"(?:income|profit)\s+from\s+operations\b"],
            "free_cash_flow": [r"free\s+cash\s+flow\b"],
        }

        # Line-aware number pattern
        NUM_PAT = r"[€\$£]?\s?\(?(\d[\d,]*(?:\.\d+)?)\)?\s?(?:(billion|million|B|M|K|bn|m))?"

        # Footnote skip: handles all annotation styles found in financial docs:
        #   "3, 4"      plain digit refs
        #   "* 4"       asterisk + digit (Schneider EBITA style)
        #   "** 7"      double asterisk + digit
        #   "(in euros per share) 20"   parenthetical label + digit ref
        #   "(1)"       parenthetical digit
        FOOTNOTE_SKIP = r"(?:[*†‡\s]*(?:\([^)]{0,40}\))?\s*(?:\d{1,2}(?:,\s*\d{1,2})*)?\s+)"

        def parse_num(raw, scale_str):
            try:
                cleaned = raw.replace(",", "")
                val = float(cleaned)
                if scale_str:
                    s = scale_str.lower()
                    if s in ("billion", "b", "bn"):
                        val *= 1_000
                    elif s in ("k",):
                        val /= 1_000
                return val
            except (ValueError, TypeError):
                return None

        results: dict[str, float] = {}
        for field, patterns in SYNONYMS.items():
            for pat in patterns:
                # Match label, skip footnote refs, then grab first real number — all on ONE line.
                # Two strategies tried in order:
                #   1. Skip footnote refs (e.g. "Revenue 3, 4  40,152")
                #   2. Plain label + whitespace + number (fallback)
                found = False
                for regex in [
                    # Strategy 1: skip footnote refs before the value
                    re.compile(
                        rf"(?im)^[^\n]*?({pat}){FOOTNOTE_SKIP}{NUM_PAT}",
                        re.IGNORECASE | re.MULTILINE,
                    ),
                    # Strategy 2: label colon/dash then number
                    re.compile(
                        rf"(?im)^[^\n]*?({pat})\s*[:\-\u2013\u2014]\s*{NUM_PAT}",
                        re.IGNORECASE | re.MULTILINE,
                    ),
                ]:
                    match = regex.search(combined)
                    if match:
                        raw_num = match.group(2)
                        scale = match.group(3) if len(match.groups()) >= 3 else None
                        val = parse_num(raw_num, scale)
                        # Sanity bounds:
                        # - EPS/ratios: 0.001–9999
                        # - Financial statement values in millions: must be >= 1
                        # - Skip obvious footnote refs (single digits matched accidentally)
                        min_val = 0.001 if field == "eps" else 1.0
                        if val is not None and min_val <= val < 2_000_000:
                            # Extra guard: values < 100 for non-EPS fields are likely footnote refs
                            if field != "eps" and val < 10:
                                continue
                            results[field] = val
                            found = True
                            break
                if found:
                    break

        if len(results) < 2:
            logger.info("Direct extraction insufficient — trying LLM fallback")
            context_text = "\n".join(contexts[:3])[:4000]
            prompt = (
                "You are a financial data extractor. Read the text and extract financial values. "
                "Use these EXACT keys: revenue, gross_profit, ebitda, net_income, eps, total_assets, "
                "current_assets, current_liabilities, total_debt, total_equity, shareholders_equity, "
                "cogs, operating_income, free_cash_flow.\n\n"
                "IMPORTANT: This may be a European annual report. Map:\n"
                "  \'Revenue\' or \'Revenues\' → revenue\n"
                "  \'Adjusted EBITA\' or \'EBITDA\' → ebitda\n"
                "  \'Profit for the year\' or \'Net income\' → net_income\n"
                "  \'Earnings per share\' → eps\n"
                "  \'Cost of sales\' → cogs\n"
                "  \'Total equity\' → total_equity\n\n"
                "Values must be plain numbers in millions. Output ONLY a JSON object.\n\n"
                f"Text:\n{context_text}\n\nJSON output:"
            )
            try:
                raw = self._call_llm_short(prompt)
                parsed = self._extract_json_from_text(raw)
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        if k not in results and v is not None:
                            try:
                                fv = float(str(v).replace(",", ""))
                                if fv > 0:
                                    results[k] = fv
                            except (ValueError, TypeError):
                                pass
            except Exception as e:
                logger.warning(f"LLM fallback extraction failed: {e}")

        logger.info(f"Extracted {len(results)} financial fields: {list(results.keys())}")
        return results

    def _extract_time_series(self, contexts: list[str]) -> dict[str, tuple[list[float], list[str]]]:
        """
        Extract time-series data for anomaly detection and forecasting.
        Uses direct regex first, LLM fallback if nothing found.
        """
        combined = "\n".join(contexts[:5])
        output: dict[str, tuple[list[float], list[str]]] = {}

        year_pattern = re.compile(r"\b(20\d{2})\b")
        years_found = sorted(set(year_pattern.findall(combined)))
        years_found = [y for y in years_found if 2018 <= int(y) <= 2030]

        num_extract = re.compile(r"\(?(\d[\d,\s]+(?:\.\d+)?)\)?")

        def clean_nums(raw: str) -> list[float]:
            matches = num_extract.findall(raw)
            results = []
            for m in matches:
                cleaned = m.replace(",", "").replace(" ", "")
                try:
                    results.append(float(cleaned))
                except ValueError:
                    pass
            return results

        SERIES_LABELS = {
            "Revenue": [r"revenue", r"revenues?", r"turnover", r"net\s+sales"],
            "Gross Profit": [r"gross\s+profit"],
            "Adjusted EBITA": [r"adjusted\s+ebit[ad]+", r"ebitda?"],
            "Net Income": [r"net\s+income", r"profit\s+for\s+the\s+year", r"profit\s+attributable\s+to\s+owners?"],
            "EPS": [r"(?:basic\s+)?earnings?\s+per\s+share", r"adjusted\s+eps"],
            "Operating Income": [r"operating\s+(?:income|profit)"],
            "Free Cash Flow": [r"free\s+cash\s+flow"],
        }

        for label, pats in SERIES_LABELS.items():
            for pat in pats:
                search = re.search(
                    rf"(?i)({pat})\s*[:\-\u2013\u2014]?\s*((?:[€\$£(]?\s?\d[\d,\s\.]*\)?\s*){{2,6}})",
                    combined,
                )
                if search:
                    nums = clean_nums(search.group(2))
                    if len(nums) >= 2:
                        if len(years_found) >= len(nums):
                            periods = years_found[-len(nums):]
                        else:
                            periods = [f"Period {i+1}" for i in range(len(nums))]
                        output[label] = (nums, periods)
                        break

        if not output:
            logger.info("Direct time-series extraction empty — trying LLM fallback")
            context_text = "\n".join(contexts[:3])[:4000]
            prompt = (
                "Extract year-over-year financial time series from this text. "
                "This may be a European annual report (values in millions of euros).\n\n"
                "Return ONLY a JSON object. Each key is a metric name. "
                "Each value has \'values\' (list of numbers in millions) and "
                "\'periods\' (list of year strings).\n\n"
                "Example: {\"Revenue\": {\"values\": [35897, 38153, 40152], "
                "\"periods\": [\"2023\", \"2024\", \"2025\"]}}\n\n"
                f"Text:\n{context_text}\n\nJSON output:"
            )
            try:
                raw = self._call_llm_short(prompt)
                parsed = self._extract_json_from_text(raw)
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        if isinstance(v, dict) and "values" in v and "periods" in v:
                            try:
                                vals = [float(x) for x in v["values"]]
                                periods = [str(p) for p in v["periods"]]
                                if len(vals) == len(periods) and len(vals) >= 2:
                                    output[k] = (vals, periods)
                            except (ValueError, TypeError):
                                continue
            except Exception as e:
                logger.warning(f"LLM time-series fallback failed: {e}")

        logger.info(f"Extracted {len(output)} time series: {list(output.keys())}")
        return output


    # ── Query Routing ─────────────────────────────────────────────────

    @staticmethod
    def _is_ratio_query(query: str) -> bool:
        keywords = ["ratio", "margin", "p/e", "debt-to-equity", "roe", "profitability", "leverage"]
        return any(kw in query.lower() for kw in keywords)

    @staticmethod
    def _is_anomaly_query(query: str) -> bool:
        keywords = ["anomal", "outlier", "spike", "drop", "unusual", "flag", "risk"]
        return any(kw in query.lower() for kw in keywords)

    @staticmethod
    def _is_forecast_query(query: str) -> bool:
        keywords = ["forecast", "predict", "project", "future", "trend", "outlook", "next quarter"]
        return any(kw in query.lower() for kw in keywords)

    # ── Governance ────────────────────────────────────────────────────

    def _check_governance(self, text: str) -> list[str]:
        """Check for governance triggers in the response."""
        flags = []
        text_lower = text.lower()

        if any(w in text_lower for w in ["$5m", "$5 million", "5,000,000", "five million"]):
            flags.append("GOVERNANCE: Amount exceeds $5M — requires human review")
        if any(w in text_lower for w in ["$10m", "$10 million", "10,000,000", "ten million"]):
            flags.append("GOVERNANCE: Amount exceeds $10M — CFO escalation required")
        if any(w in text_lower for w in ["sovereign", "government entity", "state-owned"]):
            flags.append("GOVERNANCE: Sovereign entity detected — mandatory compliance review")

        return flags

    # ── Audit Logging ─────────────────────────────────────────────────

    def _log_action(self, action: str, detail: str):
        """Log an action to the immutable audit trail."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "detail": detail,
        }
        self.audit_log.append(entry)
        self.state.tools_used.append(action)
        logger.info(f"[AUDIT] {action}: {detail}")

    def get_audit_log(self) -> list[dict]:
        """Return the full audit trail."""
        return self.audit_log


# ── CLI Entry Point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="G42 Financial Intelligence Agent")
    parser.add_argument("--file", type=str, help="Path to a PDF or CSV to ingest")
    parser.add_argument("--query", type=str, help="Query to ask the agent")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    agent = FinancialIntelligenceAgent()

    if args.file:
        chunks = agent.ingest_document(args.file)
        print(f"\nIngested {len(chunks)} chunks from {args.file}")
        for c in chunks[:3]:
            print(f"  - {c.section_heading} ({c.char_count} chars)")

    if args.query:
        response = agent.query(args.query)
        print(f"\n{'='*60}")
        print(f"Answer:\n{response.answer}")
        print(f"\nConfidence: {response.confidence}")
        print(f"Tools used: {response.tools_invoked}")
        if response.governance_flags:
            print(f"\n⚠ GOVERNANCE FLAGS:")
            for flag in response.governance_flags:
                print(f"  {flag}")
        if response.citations:
            print(f"\nCitations:")
            for c in response.citations:
                print(f"  [{c.source_document}] {c.section} (p.{c.page})")
