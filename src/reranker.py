import numpy as np
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder 

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu", batch_size: int = 32,):
        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size)
        self.cross = CrossEncoder(self.model_name, device=self.device)

    def _batched(self, items, size):
        for i in range(0, len(items), size):
            yield items[i : i + size]

    @staticmethod
    def _minmax_norm(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        mn = float(arr.min())
        mx = float(arr.max())
        if mx - mn <= 1e-12:
            return np.full_like(arr, 0.5, dtype=float)
        return (arr - mn) / (mx - mn)
    
    def rank(self, query: str, candidates: List[Dict[str, Any]], alpha: float = 0.8) -> List[Dict[str, Any]]:
        """
        Rank candidates using CrossEncoder and fuse with retriever scores.
        Returns candidates with added fields: rerank_score, retriever_score_norm, fused_score.
        Minimal fallback: if cross.predict fails at runtime, use retriever scores as rerank_scores.
        """
        texts = [c.get("chunk_text", "") for c in candidates]
        n = len(texts)
        if n == 0:
            return []

        # Compute cross-encoder scores (batched)
        pairs = [[query, t] for t in texts]
        all_scores = []

        for batch in self._batched(pairs, self.batch_size):
            batch_scores = self.cross.predict(batch, show_progress_bar=False)
            all_scores.extend(batch_scores)
        rerank_scores = np.array(all_scores, dtype=float)

        # Retriever raw scores
        retr_scores = np.array([float(c.get("score", 0.0)) for c in candidates], dtype=float)

        # Normalization
        if n == 1:
            r_norm = np.array([1.0], dtype=float)
            retr_norm = np.array([1.0], dtype=float)
        else:
            r_norm = self._minmax_norm(rerank_scores)
            retr_norm = self._minmax_norm(retr_scores)

        # Clamp alpha
        try:
            a = float(alpha)
        except Exception:
            a = 0.8
        a = max(0.0, min(1.0, a))

        # Score-based linear fusion
        if r_norm.size == 0 or retr_norm.size == 0:
            fused = np.array([], dtype=float)
        else:
            fused = a * r_norm + (1.0 - a) * retr_norm

        # Build output
        out = []
        for i, c in enumerate(candidates):
            new = dict(c)
            new["rerank_score"] = float(r_norm[i]) if i < len(r_norm) else 0.0
            new["retriever_score_norm"] = float(retr_norm[i]) if i < len(retr_norm) else 0.0
            new["fused_score"] = float(fused[i]) if i < len(fused) else 0.0
            out.append(new)

        out_sorted = sorted(out, key=lambda x: x.get("fused_score", 0.0), reverse=True)
        return out_sorted
