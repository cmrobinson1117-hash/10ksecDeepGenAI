# DAIS 10-K Research Advisor

An agentic research assistant that answers questions over a local corpus of SEC 10-K filings
(Apple, Tesla, Delta Air Lines, Home Depot, and Coca-Cola — 2023 and 2024).

---

## Architecture

```
User question
      │
      ▼
  ask_agent()                    ← routing layer
      │
      ├─ list_available_filings  ← "what filings are available?"
      ├─ lookup_company_filing   ← named-company questions (+ year/keyword filter)
      └─ search_sec_filings      ← general / cross-company questions
                                    (semantic FAISS search in v1;
                                     hybrid BM25+FAISS in v2)
      │
      ▼
  ChatOllama (llama3.1:8b)       ← synthesis only; no outside knowledge used
      │
      ▼
  Final answer  [with inline citations in v2]
```

**Data stores:** FAISS vector index + JSONL chunk store  
**Embeddings:** `all-MiniLM-L6-v2` (sentence-transformers)  
**LLM:** Ollama `llama3.1:8b` (local, zero data-egress)

---

## Project files

| File | Purpose |
|------|---------|
| `agent.py` | Baseline agent — dense FAISS search |
| `agent_v2.py` | Improved agent — hybrid BM25+FAISS, inline citations |
| `app.py` | Streamlit chat interface |
| `batch_run.py` | Batch query runner (JSON/CSV I/O) |
| `evaluate.py` | Milestone 4 evaluation framework |
| `test_set.json` | 32-question graded test set |
| `pipeline.py` | Document ingestion and FAISS indexing |
| `query.py` | Stand-alone retrieval test script |

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama and pull the model

```bash
ollama pull llama3.1:8b
```

### 3. Run the Streamlit chat interface

```bash
streamlit run app.py
```

### 4. Run the CLI agent (interactive)

```bash
python agent.py
```

### 5. Single question

```bash
python agent.py --query "Compare Apple and Tesla cybersecurity risks in 2024."
```

### 6. Batch queries

```bash
python batch_run.py                            # built-in demo questions
python batch_run.py --input queries.txt --output results.json
```

---

## Milestone 4 — Evaluation (complete workflow)

### Step 1 — Run baseline evaluation

```bash
python evaluate.py --agent v1 \
  --test_set test_set.json \
  --output baseline_results.json
```

### Step 2 — Install the improvement dependency

```bash
pip install rank-bm25
```

### Step 3 — Run improved evaluation

```bash
python evaluate.py --agent v2 \
  --test_set test_set.json \
  --output improved_results.json
```

### Step 4 — Before/after comparison table

```bash
python evaluate.py --compare baseline_results.json improved_results.json
```

### Optional — add BERTScore

```bash
pip install bert-score
python evaluate.py --agent v2 --test_set test_set.json \
  --output improved_results.json --bertscore
```

---

## Metrics computed

| Metric | Type | What it measures |
|--------|------|-----------------|
| ROUGE-1/2/L F1 | Answer quality | N-gram overlap with reference answers |
| BERTScore F1 | Answer quality | Semantic similarity (optional) |
| Precision@5 | Retrieval quality | Fraction of top-5 chunks from the correct source |
| Recall@5 | Retrieval quality | Fraction of relevant sources retrieved in top-5 |
| LLM Relevance (1-5) | Answer quality | Does the answer address the question? |
| LLM Grounding (1-5) | Answer quality | Is the answer supported by retrieved evidence? |
| LLM Completeness (1-5) | Answer quality | Does the answer cover the reference's key points? |

---

## System improvement (v1 → v2)

**Motivation from error analysis:**
The baseline (v1) struggled with:
- Year-over-year comparison questions (required chunks from 2023 AND 2024)
- Proper-noun queries (loyalty program names, product names) where dense embeddings blur over exact terminology
- Cross-company comparisons (retrieval must surface relevant chunks from multiple filings)

**Changes made in agent_v2.py:**
1. **Hybrid retrieval** — BM25 keyword search runs in parallel with FAISS dense search; ranked lists are merged with Reciprocal Rank Fusion (RRF). This directly improves Precision@k and Recall@k on proper-noun and exact-term queries.
2. **Inline citations** — the synthesis prompt now instructs the LLM to append `[Company Year 10-K, Section]` after each claim, improving LLM Grounding scores and making answers auditable.

---

## Test set coverage (32 questions)

| Category | Count | Focus |
|----------|-------|-------|
| Company-specific | 15 | Single company, single year |
| Cross-company comparison | 9 | Multiple companies in one answer |
| Year-over-year comparison | 5 | Same company, 2023 vs 2024 |
| Corpus metadata | 1 | What filings are available? |
| Adversarial | 2 | Out-of-corpus / edge cases |

---

## Example questions

```
What cybersecurity risks appear across the 2024 filings?
Compare Apple and Tesla 2024 business focus areas.
What supply chain or operational risks are mentioned most often?
What filings are available in the corpus?
How did Coca-Cola's risk language change from 2023 to 2024?
```

---

## Rebuild the FAISS index from PDFs

```bash
python pipeline.py
```
