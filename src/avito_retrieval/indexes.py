from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from .progress import Timer, heartbeat, progress


class BM25Index:
    def __init__(self):
        self.retriever = None
        self.doc_rows = None
        self.path = None

    def build(self, texts: list[str], doc_rows: np.ndarray, path: str | Path, k1: float, b: float) -> None:
        import bm25s

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.retriever = bm25s.BM25(method="lucene", k1=float(k1), b=float(b))
        with Timer("BM25 index"):
            tokens = bm25s.tokenize(texts)
            self.retriever.index(tokens)
        self.retriever.save(str(path / "bm25"))
        np.save(path / "doc_rows.npy", np.asarray(doc_rows, dtype=np.int32))
        self.doc_rows = np.asarray(doc_rows, dtype=np.int32)
        self.path = path

    def load(self, path: str | Path):
        import bm25s

        path = Path(path)
        self.retriever = bm25s.BM25.load(str(path / "bm25"), mmap=True)
        self.doc_rows = np.load(path / "doc_rows.npy", mmap_mode="r")
        self.path = path
        return self

    def search(self, query: str, k: int = 100):
        results = self.search_many([query], k=k)
        return results[0]

    def search_many(self, queries: list[str], k: int = 100) -> list[tuple[np.ndarray, np.ndarray]]:
        if not queries:
            return []
        if k <= 0 or len(self.doc_rows) == 0:
            empty = (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32))
            return [empty for _ in queries]
        import bm25s

        q_tokens = bm25s.tokenize(list(queries))
        try:
            docs, scores = self.retriever.retrieve(
                q_tokens, k=min(int(k), len(self.doc_rows)), show_progress=False
            )
        except TypeError:
            docs, scores = self.retriever.retrieve(q_tokens, k=min(int(k), len(self.doc_rows)))
        outputs = []
        for row_docs, row_scores in zip(docs, scores):
            idx = np.asarray(row_docs, dtype=np.int64)
            score = np.asarray(row_scores, dtype=np.float32)
            valid = idx >= 0
            outputs.append((np.asarray(self.doc_rows[idx[valid]], dtype=np.int32), score[valid]))
        return outputs


class CharTfidfIndex:
    def __init__(self):
        self.vectorizer = None
        self.matrix = None
        self.doc_rows = None
        self.path = None

    def build(
        self,
        texts: list[str],
        doc_rows: np.ndarray,
        path: str | Path,
        analyzer: str,
        ngram_min: int,
        ngram_max: int,
        min_df: int,
        max_features: int,
    ) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=(int(ngram_min), int(ngram_max)),
            min_df=int(min_df),
            max_features=int(max_features),
            dtype=np.float32,
            sublinear_tf=True,
            norm="l2",
        )
        with Timer("Character TF-IDF index"), heartbeat("Character TF-IDF index"):
            self.matrix = self.vectorizer.fit_transform(texts)
        save_npz(path / "matrix.npz", self.matrix, compressed=True)
        joblib.dump(self.vectorizer, path / "vectorizer.joblib", compress=3)
        np.save(path / "doc_rows.npy", np.asarray(doc_rows, dtype=np.int32))
        self.doc_rows = np.asarray(doc_rows, dtype=np.int32)
        self.path = path

    def load(self, path: str | Path):
        path = Path(path)
        self.vectorizer = joblib.load(path / "vectorizer.joblib")
        self.matrix = load_npz(path / "matrix.npz").tocsr()
        self.doc_rows = np.load(path / "doc_rows.npy", mmap_mode="r")
        self.path = path
        return self

    def search(self, query: str, k: int = 100):
        return self.search_many([query], k=k)[0]

    def search_many(self, queries: list[str], k: int = 100, batch_size: int = 64) -> list[tuple[np.ndarray, np.ndarray]]:
        if not queries:
            return []
        if k <= 0 or len(self.doc_rows) == 0:
            empty = (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32))
            return [empty for _ in queries]
        k = min(int(k), len(self.doc_rows))
        outputs = []
        for start in range(0, len(queries), int(batch_size)):
            batch = queries[start:start + int(batch_size)]
            q = self.vectorizer.transform(batch)
            scores_matrix = self.matrix @ q.T
            for col in range(q.shape[0]):
                scores = scores_matrix.getcol(col).toarray().ravel()
                if k == len(scores):
                    idx = np.argsort(-scores)
                else:
                    idx = np.argpartition(-scores, k - 1)[:k]
                    idx = idx[np.argsort(-scores[idx])]
                outputs.append((np.asarray(self.doc_rows[idx], dtype=np.int32), scores[idx].astype(np.float32)))
        return outputs


class HNSWIndex:
    def __init__(self):
        self.index = None
        self.doc_rows = None
        self.path = None
        self.dim = None

    def build(
        self,
        embeddings: np.ndarray,
        doc_rows: np.ndarray,
        path: str | Path,
        M: int,
        ef_construction: int,
        ef_search: int,
        num_threads: int,
        chunk_size: int = 4096,
    ) -> None:
        import hnswlib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        embeddings = np.asarray(embeddings)
        if embeddings.ndim != 2 or len(embeddings) != len(doc_rows):
            raise ValueError("HNSW embeddings/doc_rows shape mismatch")
        self.dim = int(embeddings.shape[1])
        self.index = hnswlib.Index(space="cosine", dim=self.dim)
        self.index.init_index(max_elements=len(embeddings), M=int(M), ef_construction=int(ef_construction), random_seed=42)
        self.index.set_num_threads(max(1, int(num_threads)))
        self.index.set_ef(int(ef_search))
        with Timer("HNSW index"), heartbeat("HNSW index"):
            total = (len(embeddings) + chunk_size - 1) // chunk_size
            for start in progress(range(0, len(embeddings), chunk_size), total=total, desc="HNSW add", unit="batch"):
                end = min(len(embeddings), start + chunk_size)
                self.index.add_items(
                    np.asarray(embeddings[start:end], dtype=np.float32),
                    ids=np.arange(start, end, dtype=np.int64),
                )
        self.index.save_index(str(path / "index.bin"))
        np.save(path / "doc_rows.npy", np.asarray(doc_rows, dtype=np.int32))
        (path / "meta.json").write_text(
            json.dumps({"dim": self.dim, "space": "cosine", "size": int(len(embeddings))}, indent=2),
            encoding="utf-8",
        )
        self.doc_rows = np.asarray(doc_rows, dtype=np.int32)
        self.path = path

    def load(self, path: str | Path, ef_search: int = 160, num_threads: int = 4):
        import hnswlib

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        self.dim = int(meta["dim"])
        self.index = hnswlib.Index(space="cosine", dim=self.dim)
        self.index.load_index(str(path / "index.bin"), max_elements=int(meta["size"]))
        self.index.set_num_threads(max(1, int(num_threads)))
        self.index.set_ef(int(ef_search))
        self.doc_rows = np.load(path / "doc_rows.npy", mmap_mode="r")
        self.path = path
        return self

    def search(self, vector: np.ndarray, k: int = 100):
        return self.search_many(np.asarray(vector, dtype=np.float32).reshape(1, -1), k=k)[0]

    def search_many(self, vectors: np.ndarray, k: int = 100) -> list[tuple[np.ndarray, np.ndarray]]:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 1D or 2D array")
        if k <= 0 or len(self.doc_rows) == 0:
            empty = (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32))
            return [empty for _ in range(len(vectors))]
        labels, distances = self.index.knn_query(vectors, k=min(int(k), len(self.doc_rows)))
        outputs = []
        for idx, dist in zip(labels, distances):
            outputs.append((np.asarray(self.doc_rows[idx], dtype=np.int32), (1.0 - dist).astype(np.float32)))
        return outputs
