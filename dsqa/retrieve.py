"""Retriever: dense / bm25 / hybrid (RRF) + optional cross-encoder rerank."""
import re
import time

import numpy as np

_TOK = re.compile(r"\d+(?:\.\d+)+|[^\W_]+")


def tokenize(text):
    return _TOK.findall((text or "").lower())


class Retriever:
    """chunks: list of dict with 'text' (and optionally 'etext' = text to embed/index).
    mode: 'dense' | 'bm25' | 'hybrid'.
    embed_model: local path or repo id (dense/hybrid). query_prefix: prepended to the question
    for dense only (bge instruction). embeddings: optional precomputed (n, d) float32 array.
    rerank_model: CrossEncoder path/id or None; rerank_n: top-N candidates reranked."""

    def __init__(self, chunks, mode="dense", embed_model=None, query_prefix="", device="cpu",
                 rerank_model=None, rerank_n=50, embeddings=None):
        if mode not in ("dense", "bm25", "hybrid"):
            raise ValueError(f"unknown mode {mode!r}")
        self.chunks, self.mode, self.query_prefix = chunks, mode, query_prefix
        self.rerank_n = rerank_n
        self.texts = [c.get("etext") or c["text"] for c in chunks]
        self.t_embed = 0.0
        self.embeddings = embeddings
        self.embedder = self.bm25 = self.cross = None

        if mode in ("dense", "hybrid"):
            if not embed_model:
                raise ValueError("embed_model required for dense/hybrid")
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(str(embed_model), device=device)
            if self.embeddings is None:
                t = time.time()
                self.embeddings = np.asarray(self.embedder.encode(
                    self.texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64),
                    dtype=np.float32)
                self.t_embed = time.time() - t
        if mode in ("bm25", "hybrid"):
            try:
                from rank_bm25 import BM25Okapi
            except ImportError as e:
                raise ImportError("rank_bm25 is not installed in this environment") from e
            self.bm25 = BM25Okapi([tokenize(t) or [""] for t in self.texts])
        if rerank_model:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError("CrossEncoder unavailable in sentence_transformers") from e
            self.cross = CrossEncoder(str(rerank_model), device=device)

    def _dense(self, q):
        qe = self.embedder.encode([self.query_prefix + q], normalize_embeddings=True,
                                  show_progress_bar=False)[0]
        return self.embeddings @ np.asarray(qe, dtype=np.float32)

    def _bm25(self, q):
        return np.asarray(self.bm25.get_scores(tokenize(q) or [""]), dtype=np.float32)

    @staticmethod
    def _top(scores, n):
        n = min(n, len(scores))
        idx = np.argpartition(-scores, n - 1)[:n] if n < len(scores) else np.arange(len(scores))
        return idx[np.argsort(-scores[idx])]

    def search(self, question, k=10):
        if not self.chunks:
            return []
        need = max(k, self.rerank_n) if self.cross else k
        if self.mode == "dense":
            s = self._dense(question)
            top = self._top(s, need)
            res = [(int(i), float(s[i])) for i in top]
        elif self.mode == "bm25":
            s = self._bm25(question)
            top = self._top(s, need)
            res = [(int(i), float(s[i])) for i in top]
        else:
            fused = {}
            for s in (self._dense(question), self._bm25(question)):
                for r, i in enumerate(self._top(s, 200)):
                    fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (60 + r + 1)
            res = sorted(fused.items(), key=lambda x: -x[1])[:need]
        if self.cross and res:
            cand = res[: self.rerank_n]
            sc = self.cross.predict([(question, self.chunks[i]["text"]) for i, _ in cand],
                                    show_progress_bar=False)
            res = sorted(((i, float(x)) for (i, _), x in zip(cand, sc)), key=lambda x: -x[1])
        return res[:k]
