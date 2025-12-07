import sys
from pathlib import Path

# Ensure repo root is on sys.path 
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest import load_faq
from src.chunker import faq_to_chunks
from src.embedder import Embedder
from src.retriever import FaissSqliteBuilder
from transformers import AutoTokenizer


def build_index(faq_json_path: str):
    faq_path = Path(faq_json_path)
    if not faq_path.exists():
        raise FileNotFoundError(f"FAQ JSON not found at: {faq_json_path}")

    entries = load_faq(str(faq_path))

    MAX_TOKENS = 512
    OVERLAP_TOKENS = 64
    TOKENIZER_NAME = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

    chunks = faq_to_chunks(entries, max_tokens=MAX_TOKENS, overlap=OVERLAP_TOKENS, tokenizer=tokenizer)

    texts = [c[0] for c in chunks]
    metas = []
    for text, md in chunks:
        metas.append({
            "question": md["question"],
            "url": md["url"],
            "category": md["category"],
            "chunk_text": text,
            "token_range": md["token_range"],
            "token_count": md["token_count"],
            "chunk_id": md["chunk_id"],
            "chunk_index": md["chunk_index"],
        })

    print(f"Building embeddings for {len(texts)} chunks...")

    embedder = Embedder()
    vecs = embedder.embed_texts(texts)
    dim = vecs.shape[1]

    output_dir = Path("data/index")
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = str(output_dir / "sample_tng_faq.index")
    db_path = str(output_dir / "sample_chunks.db")

    builder = FaissSqliteBuilder(dim=dim, index_path=index_path, db_path=db_path)
    builder.add(vecs, metas, clear_existing=True)

    print("Index built.")
    print(f" - FAISS index: {index_path}")
    print(f" - SQLite DB:   {db_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} data/sample.json")
        raise SystemExit(1)
    build_index(sys.argv[1])
