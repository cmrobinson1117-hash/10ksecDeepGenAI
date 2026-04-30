"""
agent_v2.py — SEC 10-K Research Advisor (Improved)
====================================================
Improvement over agent.py:
  1. Hybrid retrieval: BM25 keyword search + FAISS dense search, merged with
     Reciprocal Rank Fusion (RRF). Fixes the system's weakness on proper-noun
     and section-title queries where dense-only retrieval underperforms.
  2. Citation injection: every answer includes bracketed source references
     (e.g. [Apple 2024 10-K, Item 1A]) so responses are auditable.
  3. Auto index builder: if chunks.jsonl is missing, PDFs in the repo root
     are parsed, chunked, and embedded automatically at startup.

Usage (drop-in replacement for agent.py):
    python agent_v2.py --query "Compare Apple and Tesla cybersecurity risks."
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
from sentence_transformers import SentenceTransformer

try:
    from langchain_openai import ChatOpenAI as _ChatModel
    _LLM_BACKEND = "openai"
except ImportError:
    try:
        from langchain_anthropic import ChatAnthropic as _ChatModel
        _LLM_BACKEND = "anthropic"
    except ImportError:
        try:
            from langchain_ollama import ChatOllama as _ChatModel
            _LLM_BACKEND = "ollama"
        except ImportError:
            _ChatModel = None
            _LLM_BACKEND = None
            print("[WARNING] No LLM backend found. Install langchain-openai, langchain-anthropic, or langchain-ollama.")

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    print("[WARNING] rank-bm25 not installed. Falling back to dense-only search.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT     = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed"
PDF_DIR       = REPO_ROOT   # PDFs live in the repo root
EMBED_MODEL   = "all-MiniLM-L6-v2"
OLLAMA_MODEL  = "llama3.1:8b"
TOP_K         = 5
RRF_K         = 60
CHUNK_SIZE    = 500   # words per chunk
CHUNK_OVERLAP = 50    # word overlap between chunks

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
# Index builder — runs at startup if chunks.jsonl is missing
# ---------------------------------------------------------------------------

def _parse_pdf_filename(pdf_path: Path) -> tuple[str, str] | None:
    """
    Extract (company, year) from filenames like:
        sec_filings_apple_2024.pdf
        sec_filings_homedepot_2023.pdf
    Returns None if the filename does not match the expected pattern.
    """
    stem = pdf_path.stem.lower()  # e.g. "sec_filings_apple_2024"
    match = re.search(r"sec_filings_([a-z]+)_(\d{4})", stem)
    if match:
        return match.group(1), match.group(2)
    # Fallback: last two underscore-separated tokens that are word + year
    parts = stem.split("_")
    for i in range(len(parts) - 1, 0, -1):
        if re.fullmatch(r"\d{4}", parts[i]):
            company = "_".join(parts[:i]).lstrip("sec_filings_")
            return company, parts[i]
    return None


def _extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract raw text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required to build the index. Add 'pypdf' to requirements.txt.")
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def _chunk_text(text: str, company: str, year: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict[str, Any]]:
    """Split text into overlapping word-level chunks with metadata."""
    words = text.split()
    chunks = []
    i = 0
    chunk_index = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "company": company,
            "year": year,
            "section": "10-K",
            "chunk_index": chunk_index,
            "text": chunk_text,
        })
        chunk_index += 1
        i += chunk_size - overlap
    return chunks


def build_index(pdf_dir: Path, data_dir: Path, embed_model: str = EMBED_MODEL) -> None:
    """
    Scan pdf_dir for SEC filing PDFs, extract text, chunk, embed,
    and write chunks.jsonl (and optionally faiss.index) to data_dir.
    """
    pdfs = sorted(pdf_dir.glob("sec_filings_*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDF files matching 'sec_filings_*.pdf' found in {pdf_dir}.\n"
            "Ensure your filings are named like: sec_filings_apple_2024.pdf"
        )

    print(f"[INFO] Found {len(pdfs)} PDF(s). Building index...")
    data_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict[str, Any]] = []
    for pdf_path in pdfs:
        parsed = _parse_pdf_filename(pdf_path)
        if parsed is None:
            print(f"[SKIP] Could not parse company/year from {pdf_path.name}")
            continue
        company, year = parsed
        print(f"[INFO] Processing {pdf_path.name} → company={company}, year={year}")
        try:
            text = _extract_text_from_pdf(pdf_path)
        except Exception as e:
            print(f"[ERROR] Failed to extract text from {pdf_path.name}: {e}")
            continue
        chunks = _chunk_text(text, company, year)
        all_chunks.extend(chunks)
        print(f"[INFO] {pdf_path.name}: {len(chunks)} chunks")

    if not all_chunks:
        raise RuntimeError("No chunks were produced. Check that your PDFs contain extractable text.")

    # Write chunks.jsonl
    chunks_path = data_dir / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            fh.write(json.dumps(chunk) + "\n")
    print(f"[INFO] Wrote {len(all_chunks)} chunks to {chunks_path}")

    # Embed all chunks
    print("[INFO] Embedding chunks (this may take a few minutes)...")
    embedder = SentenceTransformer(embed_model)
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True).astype("float32")

    # Save FAISS index if available, otherwise save numpy embeddings
    if _FAISS_AVAILABLE:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, str(data_dir / "faiss.index"))
        print(f"[INFO] Saved FAISS index to {data_dir / 'faiss.index'}")
    else:
        np.save(str(data_dir / "embeddings.npy"), embeddings)
        print(f"[INFO] Saved numpy embeddings to {data_dir / 'embeddings.npy'}")

    print("[INFO] Index build complete.")


# ---------------------------------------------------------------------------
# Filing store — with hybrid search
# ---------------------------------------------------------------------------


class FilingStore:
    """Holds chunk metadata, FAISS index, and BM25 index for hybrid retrieval."""

    def __init__(self, data_dir: Path, embed_model: str = EMBED_MODEL) -> None:
        self.data_dir = Path(data_dir)
        chunks_path = self.data_dir / "chunks.jsonl"
        index_path  = self.data_dir / "faiss.index"
        numpy_path  = self.data_dir / "embeddings.npy"

        # Auto-build index from PDFs if chunks.jsonl is missing
        if not chunks_path.exists():
            print("[INFO] chunks.jsonl not found — building index from PDFs...")
            build_index(PDF_DIR, self.data_dir, embed_model)

        self.embedder = SentenceTransformer(embed_model)
        self.chunks   = self._load_chunks(chunks_path)
        self._embeddings: np.ndarray | None = None

        if _FAISS_AVAILABLE and index_path.exists():
            self.index = faiss.read_index(str(index_path))
        else:
            self.index = None
            if numpy_path.exists():
                self._embeddings = np.load(str(numpy_path))
            else:
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
        tokenised = [str(c.get("text", "")).lower().split() for c in self.chunks]
        return BM25Okapi(tokenised)

    def _load_embeddings_numpy(self) -> np.ndarray:
        texts = [str(c.get("text", "")) for c in self.chunks]
        embs  = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embs.astype("float32")

    def _numpy_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self._embeddings is None:
            self._embeddings = self._load_embeddings_numpy()
        q_emb  = self.embedder.encode([query], normalize_embeddings=True).astype("float32")[0]
        scores = self._embeddings @ q_emb
        top_idxs = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_idxs]

    def hybrid_search(self, query: str, k: int = TOP_K) -> list[dict[str, Any]]:
        if self._bm25 is None or not _BM25_AVAILABLE:
            return self.semantic_search(query, k)

        n = len(self.chunks)

        if _FAISS_AVAILABLE and self.index is not None:
            emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
            _, dense_idxs = self.index.search(emb, n)
            dense_rank: dict[int, int] = {int(idx): rank for rank, idx in enumerate(dense_idxs[0])}
        else:
            pairs = self._numpy_search(query, n)
            dense_rank = {idx: rank for rank, (idx, _) in enumerate(pairs)}

        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_order  = np.argsort(bm25_scores)[::-1]
        bm25_rank: dict[int, int] = {int(idx): rank for rank, idx in enumerate(bm25_order)}

        rrf_scores: dict[int, float] = {}
        for idx in range(n):
            dr = dense_rank.get(idx, n)
            br = bm25_rank.get(idx, n)
            rrf_scores[idx] = 1.0 / (RRF_K + dr) + 1.0 / (RRF_K + br)

        top_idxs = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]  # type: ignore[arg-type]

        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(top_idxs, start=1):
            chunk = self.chunks[idx]
            results.append({
                "rank":        rank,
                "score":       round(rrf_scores[idx], 6),
                "company":     str(chunk.get("company", "?")),
                "year":        str(chunk.get("year", "?")),
                "section":     str(chunk.get("section", "?")),
                "chunk_index": int(chunk.get("chunk_index", -1)),
                "text":        str(chunk.get("text", "")),
            })
        return results

    def semantic_search(self, query: str, k: int = TOP_K) -> list[dict[str, Any]]:
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
            results.append({
                "rank":        rank,
                "score":       float(score),
                "company":     str(chunk.get("company", "?")),
                "year":        str(chunk.get("year", "?")),
                "section":     str(chunk.get("section", "?")),
                "chunk_index": int(chunk.get("chunk_index", -1)),
                "text":        str(chunk.get("text", "")),
            })
        return results

    def filter_chunks(
        self,
        company: str,
        year: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        company_lower = company.lower().strip()
        matches: list[dict[str, Any]] = []

        for chunk in self.chunks:
            if company_lower not in str(chunk.get("company", "")).lower():
                continue
            if year and str(year).strip() != str(chunk.get("year", "")).strip():
                continue
            matches.append({
                "company":     str(chunk.get("company", "?")),
                "year":        str(chunk.get("year", "?")),
                "section":     str(chunk.get("section", "?")),
                "chunk_index": int(chunk.get("chunk_index", -1)),
                "text":        str(chunk.get("text", "")),
            })
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

if _LLM_BACKEND == "openai":
    chat_model = _ChatModel(model="gpt-4o-mini", temperature=0,
                            api_key=os.environ.get("OPENAI_API_KEY", ""))
elif _LLM_BACKEND == "anthropic":
    chat_model = _ChatModel(model="claude-3-haiku-20240307", temperature=0,
                            api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
elif _LLM_BACKEND == "ollama":
    chat_model = _ChatModel(model=OLLAMA_MODEL, temperature=0)
else:
    chat_model = None

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
    """
    matches = STORE.filter_chunks(company=company, year=year, keyword=keyword, limit=5)
    if not matches:
        year_str = f" for {year}" if year else ""
        kw_str   = f" with keyword '{keyword}'" if keyword else ""
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
    """List every company-year filing currently loaded in the local corpus."""
    return "Available filings:\n" + "\n".join(STORE.available_filings)


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

_AVAILABILITY_PHRASES = frozenset([
    "what filings are available", "available filings",
    "which filings are available", "what companies do you have",
    "what filings do you have", "list available filings",
])

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
    detected_year    = _detect(q, KNOWN_YEARS)
    detected_keyword = _detect(q, CANDIDATE_KEYWORDS)

    if detected_company:
        context = lookup_company_filing.invoke({
            "company": _normalize_company(detected_company),
            "year":    detected_year,
            "keyword": detected_keyword,
        })
    else:
        context = search_sec_filings.invoke({"query": user_query, "k": TOP_K})

    if "No excerpts found" in context or len(context.strip()) < 200:
        fallback = f"{user_query} SEC 10-K business segments products strategy risk"
        context  = search_sec_filings.invoke({"query": fallback, "k": TOP_K})

    if chat_model is None:
        return (
            "No LLM backend configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY "
            "in Streamlit secrets.\n\nRaw context:\n" + context
        )
    prompt = _SYNTHESIS_TEMPLATE.format(question=user_query, context=context)
    return chat_model.invoke(prompt).content.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SEC 10-K Research Advisor v2 (Hybrid Search)")
    parser.add_argument("--query", type=str, help="Single question (non-interactive mode)")
    parser.add_argument("--build-index", action="store_true", help="Force rebuild the index from PDFs")
    args = parser.parse_args()

    if args.build_index:
        build_index(PDF_DIR, DEFAULT_DATA_DIR)
        return

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
