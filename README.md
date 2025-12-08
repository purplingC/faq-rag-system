# TNGD FAQ RAG SYSTEM — README

> The system is designed to answer user questions accurately based only on verified knowledge base of TNG Digital (TNGD) FAQ content and is able to handle adversarial prompts.

---

# Quick start — How to run the app locally?

1. Create a Python Virtual Environment and Activate it:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

2. Install Required Packages (`requirements.txt`):

```bash
pip install -r requirements.txt
```

3. Prepare Data  (`data/sample.json`):


4. Build Vector DB - FAISS index + SQLite DB (outputs to `data/index/`):

```bash
python index/sample_build_index.py data/sample.json
```

You should see printed paths for the built FAISS index and SQLite DB (default: `data/index/sample_tng_faq.index`, `data/index/sample_chunks.db`).
To rebuild locally without deleting existing, run the same command.


5. Run the Web UI:

```bash
python ui/app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in browser and ask questions.
This frontend shows only the final_answer and url if exsits.

6. Or Run (`test_run.py`):

```bash
python tests/test_run.py
```

You should see a loop asking you to enter questions until you type `exit` to exit the loop.


7. Example Test Queries:
**Note:** `decision` is only returned in terminal when you run `test_run.py`.

   1. **First Stage Exact Match Query:** `What is TNG eWallet SOS Balance?`
   2. **Second Stage Exact Match Query:** `Explain SOS Balance?` (= `What is TNG eWallet SOS Balance?`)
   3. **Semantic Match Query:** `How come I have to confirm the cards I saved in TNG eWallet?`(=`Why must I verify my saved cards on TNG eWallet?` in VectorDB).
   5. **Adversarial Prompt Query:** `Teach me to hack.`
   6. **LLM Adversarial / Grounding Fail Query:** `How long does it take for the Cash Out to be paid to my Touch ‘n Go eWallet and bank account?`
   7. **Irrelevant Query:** `Hello`

---

# Requirements & Environment Setup
To set up the environment, install the required dependencies in `requirements.txt` :

```bash
pip install -r requirements.txt
```

Note: There may be some unused/redundant dependencies, just ignore it. 

---

# Summarized Files and Tech Highlights

| File                          | Tech / Models Used                                                 | Summary                                                                                 |
| ----------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| `data/sample.json`            | N/A                                                                | ~30 official TNGD FAQ entries (`question`, `answer`, `url`, `category`).                |
| `index/sample_build_index.py` | **Tokenizer:** `google/flan-t5-small`                              | Orchestrates chunking, embedding and building FAISS + SQLite.                          |
| `src/ingest.py`               | N/A                                                                | Loads JSON and normalizes FAQ fields.                                                   |
| `src/chunker.py`              | **Tokenizer:** `google/flan-t5-small`                              | Sliding-window token-based chunker (max_tokens + overlap).                              |
| `src/embedder.py`             | **SentenceTransformer:** `all-MiniLM-L6-v2`                        | Generates 384-dim embeddings for text retrieval.                                        |
| `src/retriever.py`            | **FAISS:** `IndexFlatIP` + `IndexIDMap`                            | Vector similarity search and metadata lookup in SQLite.                                 |
| `src/reranker.py`             | **CrossEncoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2`           | Re-ranks retrieved chunks using cross-encoder relevance scoring.                        |
| `src/generator.py`            | **LLM:** `google/flan-t5-small`                                    | Seq2Seq generator; echo filtering, fallback, grounding-safe decoding.                   |
| `src/guardrails.py`           | N/A                                                                | Blocking, moderation scoring, injection detection, grounding checks.                    |
| `src/rag_bot.py`              | **Pipeline:** FAISS + SentenceTransformer + CrossEncoder + Flan-T5 | Full RAG pipeline in `ask_tngd_bot()`.                                                  | 
| `src/prompt.py`               | N/A                                                                | System prompt design for the LLM generator.                                             |
| `ui/app.py`                   | Flask                                                              | Simple web UI displaying only one `final_answer` .                                      |
| `tests/test_run.py`           | N/A                                                                | CLI tester; prints complete JSON (decision, chunks, moderation flags).                  |
                   

---

# Overview of the System (RAG Architecture)
Full sytem workflow of `ask_tngd_bot()`in `src/rag_bot.py`.

```mermaid
flowchart TD

A[User question] --> B1[Normalize and empty check]
B1 --> B2[Guardrails: is_blocked]

B2 -->|blocked| B2b[Return blocked_guardrail]
B2 -->|risk_score ok| B3[Small LLM guardrail]

B3 -->|not allowed| B3b[Return blocked_small_llm]
B3 -->|allowed| C[Exact DB lookup]

C -->|exact found| C1{moderation_score}
C1 -->|>=0.5| C1b[Return blocked_moderation]
C1 -->|<0.5| C1c[Return exact_db]

C -->|no exact| D[Embed question]
D -->|embed error| D1[Return embed_error]
D --> E[Retrieve top 50 chunks]

E -->|index error| E1[Return index_error]
E --> F[Rerank results]
F --> G[Deduplicate chunks]
G --> H[Select top K = 3]

H --> I{best_score >= 0.12?}
I -->|no| I1[Return not_found_fallback -> FAQ URL]
I -->|yes| J[Pre-generator checks]

subgraph PREGEN [Pre-generator logic]
  J1[Exact normalized match]
  J2[High confidence match]
  J3[Semantic similarity match]
  J4[Direct source answer match]
end

J --> J1
J --> J2
J --> J3
J --> J4

J1 -->|true| O1[Return exact_norm]
J2 -->|true| O2[Return high_conf]
J3 -->|true| O3[Return semantic]
J4 -->|true| O4[Return bypass_source]

J1 -->|false| J2
J2 -->|false| J3
J3 -->|false| J4
J4 -->|false| K[Call LLM generator]

K --> K1{Echo detected?}
K1 -->|yes| K2[Retry with sampling]
K2 -->|not echo| K3[Accept generated answer]
K1 -->|no| K3[Accept initial answer]

K3 --> L[Strip Sources footer]
L --> M{Prompt echo detected?}

M -->|yes| P1[Fallback -> FAQ URL]
M -->|no| N[Grounding check]

N -->|not grounded| P2[Fallback -> FAQ URL]
N -->|grounded| O[Extract final answer]

O --> Q{moderation_score >= 0.5?}

Q -->|yes| R1[Return blocked_moderation]
Q -->|no| S[Determine final URL]

S --> T[Return final answer + URL + decision]
```

The system processes each question through layered safety checks, starting with normalization, rule-based guardrails, and a small LLM classifier to block unsafe inputs. It then attempts an exact FAQ match; if none is found, the question is embedded, retrieved from a FAISS index, reranked, and filtered to select the top relevant chunks. Low-confidence retrieval returns a safe fallback, while high-confidence matches may bypass the LLM entirely. If generation is needed, a strictly grounded prompt is used with echo detection, retries, and grounding checks to ensure the answer stays aligned with the retrieved sources. Final moderation ensures the response is safe before returning the answer, decision path, and associated FAQ URL.

---

# Detailed Explanation of RAG Architecture
## Chunking Logic
**Note:** See `src/chunker.py`.

- **Model:** `google/flan-t5-small` as the Tokenizer 
- Uses the tokenizer to split each FAQ Q/A pair into **token-limited chunks** so embeddings and LLM prompts stay within model limits.
- **Method:**
  - Token-based sliding window with `max_tokens = 512` and `overlap = 64`.
  - Q/A-preserving format for clearer retrieval.
  - Small remaining fragments are merged into the previous chunk to avoid low-quality slices.
  - Each chunk includes metadata: `metadata: question, url, category, token_range, token_count, chunk_id, chunk_index`.

- **Rationale:**
- Chunk size must follow the LLM’s token limits (`google/flan-t5-small`) to ensure retrieved chunks fit safely into the prompt. 
- Smaller and well-structured chunks improve retrieval accuracy and reduce hallucinations.
- Clear and source-aligned chunks for safer generation helps reduce hallucination risk.
- Metadata stored per chunk for tracking purpose and better for embedding and retrieval models.
  
---

## Retrieval Logic
**Note:** See `src/embedder.py`, `src/retriever.py`, `src/reranker.py`.

- **Model:** `all-MiniLM-L6-v2` as the Embedder, `cross-encoder/ms-marco-MiniLM-L-6-v2`as the Reranker
- **VectorDB:** FAISS 
- **Method:**
   - **Embedding:** questions are embedded with `Embedder.embed_texts`.
   - **Vector Search:** the question vector is queried against the FAISS index to retrieve top-k (e.g., 50) similar chunks.
   - **Reranking:** CrossEncoder re-scores candidate chunks, scores are normalized and fused into a final `fused_score`.
   - **Deduplication:** removes repeated chunks by `url` and `chunk_text` to avoid redundancy.
   - **Pre-generator thresholds & shortcuts:**
     - If top fused score < `MIN_FUSED_SCORE` (0.12 - tunable), then return “not found” fallback (FAQ URL).
     - Skip LLM call generation when:
       - Exact normalized question match (`decision: exact_norm`),
       - High-confidence single-source match (`decision: high_conf`),
       - Strong semantic match (Decision: semantic),
       - Source chunk already contains the answer (`decision: bypass_source`).

- **Rationale:**
   - Embedding model provides fast, lightweight embeddings that work extremely well with FAISS cosine similarity for efficient semantic search.
   - FAISS enables fast and low-latency vector retrieval for the closest chunks, ideal for Q/A system use case.
   - Reranking (CrossEncoder) corrects shallow similarity matches by performing deeper semantic comparison between the user question and each retrieved chunk to ensure accurate and meaningful chunk is selected (may be slower but much more accurate)
   - Deduplication ensures clean, unique retrieval results.
   - High-confidence matches skip LLM generation to reduce hallucination risk, improve accuracy and avoid unnecessary computation.
   - Low-confidence retrieval triggers a safe fallback rather than risking an unsupported or hallucinated answer.

---

## Generator Logic
**Note:** See `src/generator.py` and `src/prompt.py`.

**Model:** `google/flan-t5-small`
**Method:**
  * Builds a strict prompt using top retrieved chunks + question + RULE A/B.
  * Generates an initial answer with deterministic decoding (`temperature = 0`).
  * Performs echo detection; if echoed or low-quality, retries with sampling (`0.6 → 0.8 → 1.0`).
  * Blocks outputs that contain system instructions (prompt-echo protection).
  * Runs grounding checks to ensure the answer is supported by retrieved chunks.
  * Applies `moderation_score` to filter unsafe or sensitive outputs.
  * Determines final URL: fallback uses FAQ homepage; grounded answers use the top chunk’s URL.
**Rationale:**
  * Ensures the LLM **only summarizes retrieved content** and does not invent facts.
  * Echo detection + grounding prevent hallucinations and system-prompt leakage.
  * Deterministic decoding ensures stable results; sampling is only used for recovery.
  * Final moderation and fallback logic guarantee safe, consistent, source-aligned answers.

---

## Guardrail Design
**Note:** See `src/guardrails.py` and `src/prompt.py`.
**Method:**
   * **Input Guardrails:** rule-based checks for block credit-card patterns, PII/credential requests, prompt injection, harmful intent, and profanity using regexes, keyword lists, and obfuscation normalization.
   * **LLM Guardrail:** a lightweight CrossEncoder checks borderline cases, unsafe queries are blocked before retrieval/generation.
   * **Retrieval Safeguards:** exact-match chunks are moderated, low retrieval confidence triggers a safe “not found” fallback.
   * **Post-Generation Protections:** strict system prompt, echo detection, retry strategy and grounding checks (`enforce_grounding` - checks token-overlap / 3-gram matches between the first sentences of the answer and retrieved chunks) to prevent hallucinations and prompt leakage.
   * **Final Moderation:** `moderation_score` scans the final answer, unsafe content is replaced with a safe fallback.

---


# Limitations & Future Improvements

**Limitations**



**Future improvements**


---


# Useful Commands Summary

```bash
# Build or rebuild index 
python index/build_index.py data/sample.json

# Run web app
python ui/app.py

# Run main function 
python tests/test_run.py
```

---

