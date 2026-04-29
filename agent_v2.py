"""
agent_v2.py — SEC 10-K Research Advisor (Improved)
====================================================
Improvement over agent.py:
  1. Hybrid retrieval: BM25 keyword search + FAISS dense search, merged with
     Reciprocal Rank Fusion (RRF). Fixes the system's weakness on proper-noun
     and section-title queries where dense-only retrieval underperforms.
  2. Citation injection: every answer includes bracketed source references
     (e.g. [Apple 2024 10-K, Item 1A]) so responses are auditable.

Install additional dependency:
    pip install rank-bm25

Usage (drop-in replacement for agent.py):
    python agent_v2.py --query "Compare Apple and Tesla cybersecurity risks."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    _FAISS_AVAILABLE = False
    print("[WARNING] faiss not installed. Falling back to numpy cosine similarity search.")
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    print("[WARNING] rank-bm25 not installed. Falling back to dense-only search.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.1:8b"
TOP_K = 5
RRF_K = 60  # standard RRF constant

COMPANY_ALIASES: dict[str, str] = {
    "apple": "apple",
    "tesla": "tesla",
    "delta": "delta",
    "delta air lines": "delta",
    "coca-cola": "cocacola",
    "coca cola": "cocacola",
    "coke": "cocacola",
    "home depot": "homedepot",
    "the home depot": "homedepot",
}

KNOWN_COMPANIES = list(COMPANY_ALIASES)
KNOWN_YEARS = ("2023", "2024")
CANDIDATE_KEYWORDS = (
    "cybersecurity", "risk factors", "risk", "business segments", "segments",
    "products", "strategy", "business", "supply chain", "operations",
)

# ---------------------------------------------------------------------------
# Filing store — with hybrid search
# ---------------------------------------------------------------------------


class FilingStore:
    """Holds chunk metadata, FAISS index, and BM25 index for hybrid retrieval."""

    def __init__(self, data_dir: Path, embed_model: str = EMBED_MODEL) -> None:
        self.data_dir = Path(data_dir)
        self.embedder = SentenceTransformer(embed_model)
        self.chunks = self._load_chunks(self.data_dir / "chunks.jsonl")
        self._embeddings: np.ndarray | None = None  # loaded lazily for numpy fallback
        if _FAISS_AVAILABLE:
            self.index = faiss.read_index(str(self.data_dir / "faiss.index"))
        else:
            self.index = None
            self._embeddings = self._load_embeddings_numpy()
        self.available_filings = self._build_filing_list()
        self._bm25 = self._build_bm25() if _BM25_AVAILABLE else None

    @staticmethod
    def _load_chunks(path: Path) -> list[dict[str, Any]]:
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh]

    def _build_filing_list(self) -> list[str]:
        combos = sorted(
            {(str(c.get("company", "?")), str(c.get("year", "?"))) for c in self.chunks}
        )
        return [f"{company} ({year})" for company, year in combos]

    def _build_bm25(self):
        """Tokenise all chunk texts and build a BM25 index."""
        tokenised = [str(c.get("text", "")).lower().split() for c in self.chunks]
        return BM25Okapi(tokenised)

    def _load_embeddings_numpy(self) -> np.ndarray:
        """
        Fallback when faiss is unavailable: embed all chunks with the
        sentence transformer and cache as a (N, D) float32 array.
        """
        texts = [str(c.get("text", "")) for c in self.chunks]
        embs = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embs.astype("float32")

    def _numpy_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Cosine similarity search via numpy dot product (embeddings are L2-normalised)."""
        if self._embeddings is None:
            self._embeddings = self._load_embeddings_numpy()
        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")[0]
        scores = self._embeddings @ q_emb  # cosine similarity
        top_idxs = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_idxs]

    # ------------------------------------------------------------------
    # Hybrid search (BM25 + FAISS via RRF)
    # ------------------------------------------------------------------

    def hybrid_search(self, query: str, k: int = TOP_K) -> list[dict[str, Any]]:
        """
        Merge BM25 and FAISS rankings with Reciprocal Rank Fusion, then return top-k.
        Falls back to dense-only if BM25 is unavailable.
        """
        if self._bm25 is None or not _BM25_AVAILABLE:
            return self.semantic_search(query, k)

        n = len(self.chunks)

        # --- Dense ranking ---
        if _FAISS_AVAILABLE and self.index is not None:
            emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
            _, dense_idxs = self.index.search(emb, n)
            dense_rank: dict[int, int] = {int(idx): rank for rank, idx in enumerate(dense_idxs[0])}
        else:
            pairs = self._numpy_search(query, n)
            dense_rank = {idx: rank for rank, (idx, _) in enumerate(pairs)}

        # --- BM25 ranking ---
        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_order = np.argsort(bm25_scores)[::-1]
        bm25_rank: dict[int, int] = {int(idx): rank for rank, idx in enumerate(bm25_order)}

        # --- RRF fusion ---
        rrf_scores: dict[int, float] = {}
        for idx in range(n):
            dr = dense_rank.get(idx, n)
            br = bm25_rank.get(idx, n)
            rrf_scores[idx] = 1.0 / (RRF_K + dr) + 1.0 / (RRF_K + br)

        top_idxs = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]  # type: ignore[arg-type]

        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(top_idxs, start=1):
            chunk = self.chunks[idx]
            results.append(
                {
                    "rank": rank,
                    "score": round(rrf_scores[idx], 6),
                    "company": str(chunk.get("company", "?")),
                    "year": str(chunk.get("year", "?")),
                    "section": str(chunk.get("section", "?")),
                    "chunk_index": int(chunk.get("chunk_index", -1)),
                    "text": str(chunk.get("text", "")),
                }
            )
        return results

    def semantic_search(self, query: str, k: int = TOP_K) -> list[dict[str, Any]]:
        """Dense-only search: FAISS if available, otherwise numpy cosine similarity."""
        if _FAISS_AVAILABLE and self.index is not None:
            emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
            scores, idxs = self.index.search(emb, k)
            pairs = list(zip(idxs[0], scores[0]))
        else:
            pairs = self._numpy_search(query, k)

        results: list[dict[str, Any]] = []
        for rank, (idx, score) in enumerate(pairs, start=1):
            if int(idx) == -1:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "company": str(chunk.get("company", "?")),
                    "year": str(chunk.get("year", "?")),
                    "section": str(chunk.get("section", "?")),
                    "chunk_index": int(chunk.get("chunk_index", -1)),
                    "text": str(chunk.get("text", "")),
                }
            )
        return results

    def filter_chunks(
        self,
        company: str,
        year: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Filter chunks by company, optional year, and optional keyword."""
        company_lower = company.lower().strip()
        matches: list[dict[str, Any]] = []

        for chunk in self.chunks:
            if company_lower not in str(chunk.get("company", "")).lower():
                continue
            if year and str(year).strip() != str(chunk.get("year", "")).strip():
                continue
            matches.append(
                {
                    "company": str(chunk.get("company", "?")),
                    "year": str(chunk.get("year", "?")),
                    "section": str(chunk.get("section", "?")),
                    "chunk_index": int(chunk.get("chunk_index", -1)),
                    "text": str(chunk.get("text", "")),
                }
            )
            if len(matches) >= limit:
                break

        if keyword:
            kw = keyword.lower()
            prioritized = [m for m in matches if kw in m["text"].lower()]
            if prioritized:
                return prioritized[:limit]

        return matches


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

STORE = FilingStore(DEFAULT_DATA_DIR)
chat_model = ChatOllama(model=OLLAMA_MODEL, temperature=0)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _format_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No relevant filing excerpts found."
    blocks = []
    for r in results:
        blocks.append(
            f"[Result {r['rank']}]\n"
            f"Company: {r['company']} | Year: {r['year']} | Section: {r['section']}\n"
            f"Score: {r['score']:.6f}\n"
            f"Excerpt: {r['text'][:800]}"
        )
    return "\n\n---\n\n".join(blocks)


@tool
def search_sec_filings(query: str, k: int = TOP_K) -> str:
    """
    Hybrid BM25+FAISS search over the SEC 10-K corpus.

    Use for general, analytical, or cross-company questions.

    Args:
        query: Natural-language question or search phrase.
        k:     Number of top chunks to return (default 5).

    Returns:
        Ranked filing excerpts with company, year, section, and score.
    """
    return _format_results(STORE.hybrid_search(query=query, k=k))


@tool
def lookup_company_filing(
    company: str,
    year: Optional[str] = None,
    keyword: Optional[str] = None,
) -> str:
    """
    Targeted lookup for a specific company's 10-K filing.

    Use when the user explicitly names a company.

    Args:
        company: Company name (apple, tesla, delta, cocacola, homedepot).
        year:    Optional filing year (2023 or 2024).
        keyword: Optional topic filter (e.g. cybersecurity, segments, strategy).

    Returns:
        Matching filing excerpts for the requested company.
    """
    matches = STORE.filter_chunks(company=company, year=year, keyword=keyword, limit=5)
    if not matches:
        year_str = f" for {year}" if year else ""
        kw_str = f" with keyword '{keyword}'" if keyword else ""
        return f"No excerpts found for '{company}'{year_str}{kw_str}."

    blocks = []
    for i, m in enumerate(matches, start=1):
        blocks.append(
            f"[Match {i}]\n"
            f"Company: {m['company']} | Year: {m['year']} | Section: {m['section']}\n"
            f"Excerpt: {m['text'][:800]}"
        )
    return "\n\n---\n\n".join(blocks)


@tool
def list_available_filings() -> str:
    """
    List every company-year filing currently loaded in the local corpus.

    Use when the user asks what filings or companies are available.

    Returns:
        Newline-separated list of available filings.
    """
    return "Available filings:\n" + "\n".join(STORE.available_filings)


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

_AVAILABILITY_PHRASES = frozenset([
    "what filings are available", "available filings",
    "which filings are available", "what companies do you have",
    "what filings do you have", "list available filings",
])

# Improved prompt: requests inline citations from the LLM
_SYNTHESIS_TEMPLATE = """
You are a financial research assistant analyzing SEC 10-K filings.

Answer the user's question using ONLY the context below.
Rules:
- Do not use outside knowledge.
- Do not mention tools, JSON, or internal reasoning.
- After each claim, cite its source in brackets using the format: [Company Year 10-K, Section].
  Example: "Apple discloses three primary risks [Apple 2024 10-K, Item 1A]."
- If evidence is partial, infer cautiously and say so.
- If evidence is missing, state clearly that the information was not found.
- Use bullet points for lists.

Question:
{question}

Context:
{context}

Answer (with inline citations):
""".strip()


def _normalize_company(raw: str) -> str:
    return COMPANY_ALIASES.get(raw.lower().strip(), raw.lower().strip())


def _detect(q: str, candidates: tuple | list) -> Optional[str]:
    for c in candidates:
        if c in q:
            return c
    return None


def ask_agent(user_query: str) -> str:
    """Route the question, retrieve evidence with hybrid search, synthesise a cited answer."""
    q = user_query.lower().strip()

    if any(phrase in q for phrase in _AVAILABILITY_PHRASES):
        return list_available_filings.invoke({})

    detected_company = _detect(q, KNOWN_COMPANIES)
    detected_year = _detect(q, KNOWN_YEARS)
    detected_keyword = _detect(q, CANDIDATE_KEYWORDS)

    if detected_company:
        context = lookup_company_filing.invoke(
            {
                "company": _normalize_company(detected_company),
                "year": detected_year,
                "keyword": detected_keyword,
            }
        )
    else:
        context = search_sec_filings.invoke({"query": user_query, "k": TOP_K})

    if "No excerpts found" in context or len(context.strip()) < 200:
        fallback = f"{user_query} SEC 10-K business segments products strategy risk"
        context = search_sec_filings.invoke({"query": fallback, "k": TOP_K})

    prompt = _SYNTHESIS_TEMPLATE.format(question=user_query, context=context)
    return chat_model.invoke(prompt).content.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SEC 10-K Research Advisor v2 (Hybrid Search)")
    parser.add_argument("--query", type=str, help="Single question (non-interactive mode)")
    args = parser.parse_args()

    if args.query:
        print(ask_agent(args.query))
        return

    print("SEC 10-K Research Advisor v2  |  Ctrl-C to exit\n")
    while True:
        try:
            user_query = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        if not user_query:
            continue
        try:
            print(f"\nAnswer:\n{ask_agent(user_query)}\n")
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    main()
