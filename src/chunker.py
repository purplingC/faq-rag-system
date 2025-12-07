from typing import List, Tuple, Dict, Optional, Any
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Normalize values from input dicts if None
def _safe_strip(s: Optional[str]) -> str:
    return (s or "").strip()

def faq_to_chunks(
    entries: List[Dict[str, Any]],
    *,
    max_tokens: int = 512,
    overlap: int = 64,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    tokenizer_name: str = "google/flan-t5-small",
    min_chunk_tokens: int = 32,
) -> List[Tuple[str, Dict[str, Any]]]:
    if not isinstance(entries, list):
        raise ValueError("entries must be a list of dicts")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if max_tokens <= overlap:
        raise ValueError("max_tokens must be greater than overlap")

    out: List[Tuple[str, Dict[str, Any]]] = []
    for idx, e in enumerate(entries):
        q_text = _safe_strip(e.get("question", ""))
        a_text = _safe_strip(e.get("answer", ""))
        text = f"Q: {q_text}\nA: {a_text}"
        # Encode into token ids
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        N = len(token_ids)
        if N == 0:
            continue

        # Sliding window params
        step = max_tokens - overlap
        i = 0
        chunk_idx = 0

        # Min remainder threshold (tunable)
        min_remainder = max(min_chunk_tokens, overlap // 2, 16)

        while i < N:
            j = min(i + max_tokens, N)

            # Handle small remainder by merging into previous chunk when possible
            rem = N - j
            if rem and rem < min_remainder and chunk_idx > 0:
                # Pop previous metadata (we don't need the previous text variable)
                _, prev_meta = out.pop()

                # Defensive: prev_meta should include the previous token_range
                prev_range = prev_meta.get("token_range")
                if not prev_range:
                    raise RuntimeError("expected token_range in previous metadata while merging remainder")
                prev_start = prev_range[0]

                # Merged ids from prev_start to current end j
                merged_ids = token_ids[prev_start:j]
                merged_text = tokenizer.decode(
                    merged_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                ).strip()

                meta = {
                    "chunk_id": idx,
                    "question": q_text,
                    "url": _safe_strip(e.get("url", "")),
                    "category": _safe_strip(e.get("category", "")),
                    "chunk_index": chunk_idx - 1,
                    "token_range": (prev_start, j),
                    "token_count": j - prev_start,
                    "chunk_text": merged_text,
                }
                out.append((merged_text, meta))
                break

            # Normal chunk creation
            chunk_ids = token_ids[i:j]
            chunk_text = tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()

            meta = {
                "chunk_id": idx,
                "question": q_text,
                "url": _safe_strip(e.get("url", "")),
                "category": _safe_strip(e.get("category", "")),
                "chunk_index": chunk_idx,
                "token_range": (i, j),
                "token_count": j - i,
                "chunk_text": chunk_text,
            }
            out.append((chunk_text, meta))

            if j == N:
                break
            i += step
            chunk_idx += 1

    return out
