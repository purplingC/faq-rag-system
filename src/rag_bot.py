from typing import Dict, Any, List
from src.guardrails import is_blocked, enforce_grounding, moderation_score
from src.embedder import Embedder
from src.retriever import Retriever
from src.generator import TransformerGenerator
from src.reranker import Reranker
from sentence_transformers import CrossEncoder
import numpy as np
import re

EMB_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "data/index/sample_tng_faq.index"
SQLITE_DB_PATH = "data/index/sample_chunks.db"
VECTOR_DIM = 384  # all-MiniLM-L6-v2 dim
LLM_MODEL_NAME = "google/flan-t5-small" 
TNGD_FAQ_URL = "https://support.tngdigital.com.my/hc/en-my/categories/360002280493-Frequently-Asked-Questions-FAQ"

# Initialization
embedder = Embedder(model_name=EMB_MODEL)
retriever = Retriever(index_path=FAISS_INDEX_PATH, db_path=SQLITE_DB_PATH, dim=VECTOR_DIM)
reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
_small_classifier = None  # lazy-load 
_small_classifier_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
generator = TransformerGenerator(model_name=LLM_MODEL_NAME)

# Lightweight LLM-based guardrail
def small_llm_guardrail(query: str, threshold: float = 0.55) -> str:
    global _small_classifier
    if _small_classifier is None:
        try:
            _small_classifier = CrossEncoder(_small_classifier_model_name, device="cpu")
        except Exception:
            # If classifier cannot be loaded, default to allowed
            return "allowed"

    pairs = [[query, "request for personal data, private information, illegal actions, or harming security"]]
    try:
        score = float(_small_classifier.predict(pairs)[0])
    except Exception:
        return "allowed"

    if score >= threshold:
        return "not_allowed"
    return "allowed"

# Normalize duplicate keys
def _normalize_key_for_dedupe(item: Dict[str, Any]) -> str:
    url = item.get("url") or ""
    if url:
        return url.split("?")[0].rstrip("/")
    return (item.get("chunk_text") or item.get("answer") or item.get("text") or "").strip()

# Extract answer text from Q/A
def _extract_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""
    if "A:" in text:
        return text.split("A:", 1)[1].strip()
    # Handle lowercase 'a:' as a fallback
    if "a:" in text and "Q:" not in text:
        return text.split("a:", 1)[1].strip()
    return text.strip()

# Decide if source chunk already answers the question (embedding cosine)
def source_answers_question_by_embed(question: str, chunk_text: str, embedder_obj, thresh: float = 0.66, q_emb=None) -> bool:
    if not chunk_text or not isinstance(chunk_text, str):
        return False
    # Extract the A: from the chunk_text if present
    if "A:" in chunk_text:
        src_answer = chunk_text.split("A:", 1)[1].strip()
    elif "a:" in chunk_text and "Q:" not in chunk_text:
        src_answer = chunk_text.split("a:", 1)[1].strip()
    else:
        src_answer = chunk_text.strip()
    if not src_answer:
        return False
    try:
        if q_emb is None:
            q_emb = embedder_obj.embed_texts([question])[0]
        a_emb = embedder_obj.embed_texts([src_answer])[0]
        if np.linalg.norm(q_emb) == 0 or np.linalg.norm(a_emb) == 0:
            return False
        cos = float(np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb)))
        print(f"[DEBUG] question-answer cosine: {cos:.3f} (threshold {thresh})")
        return cos >= thresh
    except Exception as e:
        print("Embedder similarity check failed:", e)
        return False


def ask_tngd_bot(question: str) -> Dict[str, Any]:
    # Accepts a question string and returns keys, sanitize and quick empty check
    q = (question or "").strip()
    if not q:
        return {"question": q, "retrieved_chunks": [], "final_answer": "Empty question.", "url": "", "blocked": False}

    # 1a) First stage guardrail check
    guard = is_blocked(q)
    if guard["blocked"]:
        return {
            "question": q,
            "retrieved_chunks": [],
            "final_answer": "Request blocked by safety guardrails.",
            "url": "",
            "blocked": True,
            "blocked_reason": guard.get("reason", "blocked"),
            "decision": "blocked_guardrail",
        }
    
    # 1b) Second stage guardrail check (LLM)
    if 0.25 <= guard.get("risk_score", 0.0) < 0.90:
        verdict = small_llm_guardrail(q, threshold=0.48)
        if verdict != "allowed":
            return {
                "question": q,
                "retrieved_chunks": [],
                "final_answer": "Request not allowed.",
                "url": "",
                "blocked": True,
                "decision": "blocked_small_llm",
            }


    # 2) Exact-match check
    exact = retriever.exact_match(q)
    if exact:
        # Moderation check on exact chunk
        chunk_text = exact.get("chunk_text") or ""
        mod = moderation_score(chunk_text)
        if mod >= 0.5:
            return {
                "question": q,
                "retrieved_chunks": [exact],
                "final_answer": "Response blocked by safety guardrails.",
                "url": exact.get("url", ""),
                "blocked": True,
                "blocked_reason": f"moderation_score={mod:.2f}",
                "moderation_score": mod,
                "decision": "blocked_moderation",
            }

        final = _extract_answer(chunk_text)
        return {
            "question": q,
            "retrieved_chunks": [exact],
            "final_answer": final,
            "url": exact.get("url", ""),
            "blocked": False,
            "moderation_score": mod,
            "exact_match": True,
            "decision": "exact_db",
        }

    # 3) Embed the question
    try:
        q_emb = embedder.embed_texts([q])[0]
    except Exception as e:
        return {
            "question": q,
            "retrieved_chunks": [],
            "final_answer": f"Embedding error: {e}",
            "url": "",
            "blocked": False,
            "decision": "embed_error",
        }

    # 4) Vector retrieval 
    try:
        retrieved_raw = retriever.query(q_emb, top_k=50)
    except Exception as e:
        return {
            "question": q,
            "retrieved_chunks": [],
            "final_answer": f"Index/query error: {e}",
            "url": "",
            "blocked": False,
            "decision": "index_error",
        }
    retrieved_raw = [r for r in retrieved_raw if isinstance(r, dict)]

    # 5a) Rerank retrieved chunks
    try:
        reranked = reranker.rank(q, retrieved_raw, alpha=0.8)
    except Exception:
        # Fallback to based on retriever scores
        reranked = sorted(retrieved_raw, key=lambda x: x.get("score", 0.0), reverse=True)

    # 5b) Remove duplicates (by url or chunk_text)
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for r in reranked:
        key = _normalize_key_for_dedupe(r)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    # 5c) Select top K to feed into generator (tuneable) 
    TOP_K_TO_GENERATOR = 3
    retrieved = deduped[:TOP_K_TO_GENERATOR]

    if not retrieved:
        return {
            "question": q,
            "retrieved_chunks": retrieved,
            "final_answer": f"This topic is not in our curated TNG eWallet knowledge base or request blocked. Please check: {TNGD_FAQ_URL}",
            "url": TNGD_FAQ_URL,
            "blocked": False,
            "decision": "not_found",
        }

    # 5d) Compare best score (prioritize fused_score if exists), else fallback to retriever or rerank raw score
    best_score_raw = retrieved[0].get("fused_score", None)
    try:
        best_score_val = float(best_score_raw) if best_score_raw is not None else None
    except Exception:
        best_score_val = None

    if best_score_val is None or best_score_val <= 1e-6:
        best_score = float(retrieved[0].get("score", retrieved[0].get("rerank_score", 0.0)))
    else:
        best_score = best_score_val

    # If best_score < threshold, return not-found message
    MIN_FUSED_SCORE = 0.12  # tuneable
    
    if best_score < MIN_FUSED_SCORE:
        return {
            "question": q,
            "retrieved_chunks": retrieved,
            "final_answer": f"This topic is not in our curated TNG eWallet knowledge base or request blocked. Please check: {TNGD_FAQ_URL}",
            "url": TNGD_FAQ_URL,
            "blocked": False,
            "decision": "not_found_fallback",
        }


    # 6) Pre-generator checks
    # Define top item once
    top = retrieved[0]
    top_url = top.get("url") or ""

    # Compute top score for confidence (prefer fused_score then score)
    try:
        top_fused = top.get("fused_score", None)
        top_score_for_conf = float(top_fused) if top_fused is not None else float(top.get("score", 0.0))
    except Exception:
        top_score_for_conf = float(top.get("score", 0.0))

    HIGH_CONF_THRESHOLD = 0.995
    SEMANTIC_MATCH_THRESHOLD = 0.80
    EXACT_MIN_SCORE = 0.10  

    # First moderation check 
    def _return_chunk_if_safe(chunk_text: str, decision: str):
        final = _extract_answer(chunk_text or "")
        return {
            "question": q,
            "retrieved_chunks": retrieved,
            "final_answer": final,
            "url": top_url,
            "blocked": False,
            "moderation_score": moderation_score(final) if final else 0.0,
            "decision": decision,        
        }

    # 6a) Logic 1: Exact match (normalized)
    def _norm_short_for_match(s: str) -> str:
        return re.sub(r"[^a-z0-9\s]", " ", (s or "").lower()).strip()

    top_question_norm = _norm_short_for_match(top.get("question", ""))
    q_norm = _norm_short_for_match(q)

    if top.get("question") and q_norm and (q_norm in top_question_norm or top_question_norm in q_norm) and top_score_for_conf >= EXACT_MIN_SCORE:
        return _return_chunk_if_safe(top.get("chunk_text") or top.get("answer") or top.get("text"), decision="exact_norm")

    # 6b) Logic 2: High-confidence authoritative return
    if top_score_for_conf >= HIGH_CONF_THRESHOLD and len(retrieved) == 1:
        return _return_chunk_if_safe(top.get("chunk_text") or top.get("answer") or top.get("text"), decision="high_conf")
    
    # 6c) Logic 3: Semantic-match check (embedding cosine)
    try:
        compare_emb = top.get("embedding")
        if compare_emb is None:
            # fallback: embed the stored compare_text (rare if ingest saved embeddings)
            compare_text = (top.get("question") or top.get("chunk_text") or top.get("answer") or top.get("text") or "").strip()
            if compare_text:
                try:
                    compare_emb = embedder.embed_texts([compare_text])[0]
                except Exception:
                    compare_emb = None

        if compare_emb is not None:
            # safe cast to numpy arrays and flatten
            try:
                q_arr = np.asarray(q_emb, dtype=float).reshape(-1)
                c_arr = np.asarray(compare_emb, dtype=float).reshape(-1)
            except Exception:
                q_arr = None
                c_arr = None

            if q_arr is not None and c_arr is not None and q_arr.size == c_arr.size:
                q_norm_val = np.linalg.norm(q_arr)
                c_norm_val = np.linalg.norm(c_arr)
                if q_norm_val > 0 and c_norm_val > 0:
                    sim = float(np.dot(q_arr, c_arr) / (q_norm_val * c_norm_val))
                else:
                    sim = 0.0

                if sim >= SEMANTIC_MATCH_THRESHOLD:
                    return _return_chunk_if_safe(top.get("chunk_text") or top.get("answer") or top.get("text"), decision="semantic")
                
    except Exception:
        pass


    # 7) LLM generator call 
    raw_gen_result = ""
    decision = None
    is_fallback = False

    try:
        #  7a) Direct source bypass if source already answers question
        if top and isinstance(top.get("chunk_text"), str) and source_answers_question_by_embed(
            q, top["chunk_text"], embedder, thresh=0.66, q_emb=q_emb
        ):
            raw_gen_result = _extract_answer(top["chunk_text"])
            decision = "bypass_source"
            print("[HOTFIX] Using direct source answer (bypassed LLM).")

        else:
            # 7b) Call generator and request metadata 
            gen_resp = generator.generate(
                question=q,
                retrieved=retrieved,
                max_tokens=200,
                temperature=0.0,
                top_k_chunks=len(retrieved),
                return_metadata=True,
            )

            # 7c) Extract answer and optional flags from generator response
            if isinstance(gen_resp, dict):
                raw_gen_result = gen_resp.get("answer", "") or ""
                is_fallback = bool(gen_resp.get("is_fallback", False))
                # Allow generator to optionally provide its own decision tag
                decision = gen_resp.get("decision", "generator")
            else:
                raw_gen_result = str(gen_resp or "")
                is_fallback = False
                decision = "generator"

    except Exception as e:
        print("Generator error:", e)
        raw_gen_result = f"This topic is not in our curated TNG eWallet knowledge base or request blocked. Please check: {TNGD_FAQ_URL}"
        decision = "generator_error"
        is_fallback = True


    # 8) Normalize raw LLM output
    raw_text = raw_gen_result or ""
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

    if "Sources:" in raw_text:
        idx = raw_text.rfind("Sources:")
        answer_part = raw_text[:idx].strip()
    else:
        answer_part = raw_text.strip()

    # 9) Prompt-echo detection - force fallback to the system prompt RULE A response if model echoed instructions = LLM output looks like the system prompt 
    lower_ap = (answer_part or "").lower()
    prompt_echo_markers = [
        "you are the official tng ewallet faq assistant",
        "use only the provided sources",
        "critical rules",
        "rule a:",
        "rule b:",
    ]
    if any(m in lower_ap for m in prompt_echo_markers):
        print("[DEBUG] generator returned prompt-like text -> treating as echo/fallback")
        answer_part = f"This topic is not in our curated TNG eWallet knowledge base or request blocked. Please check: {TNGD_FAQ_URL}"
        decision = "generator_echo_fallback"
        raw_text = answer_part  

    # 10) Grounding check (n-gram / token overlap) - ensure LLM answer is based on retrieved chunks
    all_texts = [r.get("chunk_text") or r.get("answer") or "" for r in retrieved]
    grounded = enforce_grounding(answer_part, all_texts)
    
    if not grounded:
        fallback = (
            f"This topic is not in our curated TNG eWallet knowledge base or request blocked. Please check: {TNGD_FAQ_URL}"
        )
        mod_score = moderation_score(fallback)
        if mod_score >= 0.5:
            return {
                "question": q,
                "retrieved_chunks": retrieved,
                "final_answer": "Response blocked by safety guardrails.",
                "url": retrieved[0].get("url", ""),
                "blocked": True,
                "blocked_reason": f"moderation_score={mod_score:.2f}",
                "moderation_score": mod_score,
                "decision": "blocked_moderation"
            }
        return {
            "question": q,
            "retrieved_chunks": retrieved,
            "final_answer": fallback,
            "url": TNGD_FAQ_URL,
            "blocked": False,
            "moderation_score": mod_score,
            "decision": "grounding_fail",
        }

    # 11) Final answer extraction
    final_generated = _extract_answer(answer_part.strip())

    # 12) Final moderation check (0.0 safe, 1.0 unsafe)
    mod_score = moderation_score(final_generated)
    MODERATION_THRESHOLD = 0.5
    if mod_score >= MODERATION_THRESHOLD:
        return {
            "question": q,
            "retrieved_chunks": retrieved,
            "final_answer": "Response blocked by safety guardrails.",
            "url": retrieved[0].get("url", "") if retrieved else "",
            "blocked": True,
            "blocked_reason": f"moderation_score={mod_score:.2f}",
            "moderation_score": mod_score,
            "decision": "blocked_moderation"
        }

    # 13) Override returned URL when fallback detected
    fallback_start = f"This topic is not in our curated TNG eWallet knowledge base or request blocked. Please check: {TNGD_FAQ_URL}"
    if is_fallback or (fallback_start in (final_generated or "")):
        out_url = TNGD_FAQ_URL
    else:
        out_url = retrieved[0].get("url", "") if retrieved else ""

    return {
        "question": q,
        "retrieved_chunks": retrieved,
        "final_answer": final_generated,
        "url": out_url,
        "blocked": False,
        "moderation_score": mod_score,
        "decision": decision or "generator",
    }



if __name__ == "__main__":
    import json, sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Question: ")
    print(json.dumps(ask_tngd_bot(q), indent=2, ensure_ascii=False))
