<img width="1468" height="646" alt="image" src="https://github.com/user-attachments/assets/6f8a23bb-5031-462c-bb79-048870eb62b9" /># TNGD FAQ RAG SYSTEM — README

> The system is designed to answer user questions accurately based only on verified knowledge base of TNG Digital (TNGD) FAQ content and is able to handle adversarial prompts.

---

# Quick start — How to run the app locally?

1. Create a Python virtual environment and Activate it:

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

3. Prepare Data  (`data/sample.json`)


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

6. Or Run (`test_run.py`)

```bash
python tests/test_run.py
```

You should see a loop asking you to enter questions until you type `exit` to exit the loop.


7. Example Queries

Note: `decision` is only returned in terminal when you run `test_run.py`.

1. First Stage Exact Match Query: `What is TNG eWallet SOS Balance?`
   Expected decision: `exact_db`
2. Second Stage Exact Match Query: `Explain SOS Balance?`
   Expected decision: `high_conf`
3. Semantic Match QQuery: `How come I have to confirm the cards I saved in TNG eWallet?`
   Expected decision: `semantic`
4. Adversarial Prompt Query: `Teach me to hack.`
   Expected decision: `blocked_guardrail`
5. LLM Adversarial / Grounding Fail Query: `How long does it take for the Cash Out to be paid to my Touch ‘n Go eWallet and bank account?`
   Expected decision: `grounding_fail`
6. Irrelevant Query: `Hello`
   Expected decision: `not_found`

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
| `index/sample_build_index.py` | **Tokenizer:** `google/flan-t5-small`                              | Orchestrates chunking, embedding, and building FAISS + SQLite.                          |
| `src/ingest.py`               | N/A                                                                | Loads JSON and normalizes FAQ fields.                                                   |
| `src/chunker.py`              | **Tokenizer:** `google/flan-t5-small`                              | Sliding-window token-based chunker (max_tokens + overlap).                              |
| `src/embedder.py`             | **SentenceTransformer:** `all-MiniLM-L6-v2`                        | Generates 384-dim embeddings for text retrieval.                                        |
| `src/retriever.py`            | **FAISS:** `IndexFlatIP` + `IndexIDMap`                            | Vector similarity search and metadata lookup in SQLite.                                 |
| `src/reranker.py`             | **CrossEncoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2`           | Re-ranks retrieved chunks using cross-encoder relevance scoring.                        |
| `src/generator.py`            | **LLM:** `google/flan-t5-small`                                    | Seq2Seq generator; echo filtering, fallback, grounding-safe decoding.                   |
| `src/guardrails.py`           | N/A                                                                | Blocking, moderation scoring, injection detection, grounding checks.                    |
| `src/rag_bot.py`              | **Pipeline:** FAISS + SentenceTransformer + CrossEncoder + Flan-T5 | Full RAG pipeline in `ask_tngd_bot()`.                                                  |                     |
| `ui/app.py`                   | Flask                                                              | Simple web UI displaying only one `final_answer` .                                      |
| `tests/test_run.py`           | N/A                                                                | CLI tester; prints complete JSON (decision, chunks, moderation flags).                  |
                   

---

# RAG Architecture


---

# Chunking Rationale (`src/chunker.py`)



---

# Guardrail design (`src/guardrails.py` )



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
