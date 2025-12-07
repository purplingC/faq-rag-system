import faiss
import numpy as np
import sqlite3
from typing import List, Dict, Any, Optional
from pathlib import Path

# FAISS IndexIDMap wrapper
IndexIDMapClass = getattr(faiss, "IndexIDMap2", None) or getattr(faiss, "IndexIDMap", None)
if IndexIDMapClass is None:
    raise RuntimeError(
        "FAISS installation missing IndexIDMap/IndexIDMap2. "
        "ID-mapped indexes are required for explicit vector IDs.\n"
        "Install a supported FAISS build:\n"
        "  pip install faiss-cpu   # CPU\n"
        "  pip install faiss-gpu   # GPU\n"
        "If unavailable, rebuild FAISS from source with IDMap enabled."
    )

class FaissSqliteBuilder:
    def __init__(
        self,
        dim: int,
        index_path: str = "data/index/sample_tng_faq.index",
        db_path: str = "data/index/sample_chunks.db",
    ):
        self.dim = dim
        self.index_path = str(index_path)
        self.db_path = str(db_path)

        # 1) Ensure db exists
        self._ensure_db()

        # 2) Load or create FAISS index
        self._load_or_create_index()


    def _ensure_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                question TEXT,
                url TEXT,
                category TEXT,
                chunk_text TEXT,
                token_start INTEGER,
                token_end INTEGER,
                token_count INTEGER,
                chunk_id INTEGER,
                chunk_index INTEGER
            )
            """
        )
        self.conn.commit()

    def _next_id(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT MAX(id) FROM chunks")
        r = cur.fetchone()[0]
        return 0 if r is None else r + 1
    
    def _load_or_create_index(self) -> None:
        if Path(self.index_path).exists():
            self.index = faiss.read_index(self.index_path)
            if not isinstance(self.index, IndexIDMapClass):
                self.index = IndexIDMapClass(self.index)
        else:
            base = faiss.IndexFlatIP(self.dim)
            self.index = IndexIDMapClass(base)
    
    
    # 3) Add embeddings (vectors, metadatas)
    def add(
        self,
        vectors: np.ndarray,
        metadatas: List[Dict[str, Any]],
        *,
        clear_existing: bool = False,
    ) -> None:
        # Validate inputs
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array of shape (n, dim)")
        n, d = vectors.shape
        if d != self.dim:
            raise ValueError(f"vector dim mismatch: {d} != {self.dim}")
        if len(metadatas) != n:
            raise ValueError("Length of metadatas must equal number of vectors")

        cur = self.conn.cursor()

        # Guarantee a clean rebuild or add 
        if clear_existing:
            cur.execute("DELETE FROM chunks")
            self.conn.commit()
            base = faiss.IndexFlatIP(self.dim)
            self.index = IndexIDMapClass(base)
            start_id = 0
        else:
            start_id = self._next_id() 

        # Normalize vectors (inner product ≈ cosine)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vectors / norms

        # Compute ids for new vectors 
        ids = np.arange(start_id, start_id + n, dtype="int64")

        # Check that index must be wrapped with IndexIDMapClass
        if not isinstance(self.index, IndexIDMapClass):
            raise RuntimeError("Internal FAISS index is not wrapped with IndexIDMapClass — cannot add_with_ids()")

        # Add vectors to FAISS with explicit IDs
        self.index.add_with_ids(vecs, ids)

        # Prepare metadata and insert in batch
        rows = []
        for i, md in enumerate(metadatas):
            idx = int(ids[i])
            question = md.get("question", "")
            url = md.get("url", "")
            category = md.get("category", "")
            chunk_text = md.get("chunk_text", "")
            token_range = md.get("token_range") or (None, None)
            token_start, token_end = token_range
            token_count = md.get("token_count", None)
            chunk_id = md.get("chunk_id", None)
            chunk_index = md.get("chunk_index", None)

            rows.append(
                (
                    idx,
                    question,
                    url,
                    category,
                    chunk_text,
                    token_start,
                    token_end,
                    token_count,
                    chunk_id,
                    chunk_index,
                )
            )

        cur.executemany(
            """
            REPLACE INTO chunks(
                id, question, url, category, chunk_text,
                token_start, token_end, token_count, chunk_id, chunk_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        # Commit metadata and write index
        self.conn.commit()
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, self.index_path)


class Retriever:
    def __init__(
        self,
        index_path: str = "data/index/sample_tng_faq.index",
        db_path: str = "data/index/sample_chunks.db",
        dim: int = 384,
    ):
        self.index_path = str(index_path)
        self.db_path = str(db_path)
        self.dim = dim
        self._load()

    def _load(self):
        if not Path(self.index_path).exists():
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        # ensure index is wrapped with ID map
        if not isinstance(self.index, IndexIDMapClass):
            base = self.index
            self.index = IndexIDMapClass(base)
        self.conn = sqlite3.connect(self.db_path)

    # Input to query
    def query(
        self,
        qvec: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.45,
        dedupe_by: Optional[str] = "url",
    ) -> List[Dict[str, Any]]:
        qvec = np.asarray(qvec, dtype="float32")
        if qvec.ndim == 2 and qvec.shape[0] == 1:
            qvec = qvec.reshape(-1)
        if qvec.ndim != 1:
            raise ValueError("qvec must be a 1D vector or shape (1, dim)")

        # normalize query (since index uses normalized vectors)
        norm = np.linalg.norm(qvec)
        qv = qvec if norm == 0 else qvec / norm
        qv = qv.reshape(1, -1).astype("float32")
        if getattr(self.index, "ntotal", 0) == 0:
            return []

        D, I = self.index.search(qv, top_k)
        ids = [int(x) for x in I[0].tolist() if x is not None and int(x) >= 0]
        scores = D[0].tolist()

        if not ids:
            return []

        # Batch fetch metadata for all returned ids
        cur = self.conn.cursor()
        placeholders = ",".join("?" for _ in ids)
        cur.execute(
            f"SELECT id, question, url, category, chunk_text, token_start, token_end, token_count, chunk_id, chunk_index FROM chunks WHERE id IN ({placeholders})",
            tuple(ids),
        )
        rows = cur.fetchall()
        
        row_map = {r[0]: r for r in rows} # Map rows by id for fast lookup

        results_raw = []
        # Preserve order of FAISS results
        for idx, score in zip(ids, scores):
            row = row_map.get(idx)
            if not row:
                continue
            results_raw.append(
                {
                    "id": int(row[0]),
                    "score": float(score),
                    "question": row[1],
                    "url": row[2],
                    "category": row[3],
                    "chunk_text": row[4],
                    "token_start": row[5],
                    "token_end": row[6],
                    "token_count": row[7],
                    "chunk_id": row[8],
                    "chunk_index": row[9],
                }
            )

        # Apply minimum score filter
        filtered = [r for r in results_raw if r["score"] >= min_score]

        # Deduplicate by given metadata key, keep top scoring chunk per key
        if dedupe_by:
            best: Dict[str, Dict[str, Any]] = {}
            for r in filtered:
                key = r.get(dedupe_by) or f"__id_{r['id']}"
                existing = best.get(key)
                if existing is None or r["score"] > existing["score"]:
                    best[key] = r
            cleaned = sorted(best.values(), key=lambda x: x["score"], reverse=True)
        else:
            cleaned = sorted(filtered, key=lambda x: x["score"], reverse=True)

        return cleaned

    def exact_match(self, question: str):
        if not question:
            return None
        norm = " ".join(question.strip().split()).lower() # collapse whitespace and lowercase
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id,question,url,category,chunk_text FROM chunks WHERE lower(trim(question))=?",
            (norm,),
        )
        row = cur.fetchone()
        if row:
            return {
                "id": row[0],
                "question": row[1],
                "url": row[2],
                "category": row[3],
                "chunk_text": row[4],
                "matched_by": "exact_db",
            }
        return None

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
