from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .embeddings import E5Encoder
from .progress import Timer


class QueryHistory:
    """Semantic historical-query memory with exact-query exclusion to prevent training leakage."""

    def __init__(self):
        self.query_index = None
        self.query_keys = None
        self.indptr = None
        self.item_rows = None
        self.path = None

    def build(
        self,
        grouped_queries: pd.DataFrame,
        encoder: E5Encoder,
        item_id_to_row: dict[str, int],
        path: str | Path,
        top_items_per_query: int,
        desc: str = "History query embeddings",
    ) -> None:
        from .text import dense_query_text

        texts = [
            dense_query_text(r.search_query, r.search_infm_params_text, r.search_is_delivery_search)
            for r in grouped_queries.itertuples(index=False)
        ]
        with Timer(desc):
            embeddings = encoder.encode(texts, desc=desc)
        self.build_from_embeddings(grouped_queries, embeddings, item_id_to_row, path, top_items_per_query)

    def build_from_embeddings(
        self,
        grouped_queries: pd.DataFrame,
        embeddings: np.ndarray,
        item_id_to_row: dict[str, int],
        path: str | Path,
        top_items_per_query: int,
        hnsw_threads: int = 4,
    ) -> None:
        from .indexes import HNSWIndex

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        embeddings = np.asarray(embeddings)
        if len(embeddings) != len(grouped_queries):
            raise ValueError("History embeddings and grouped query rows do not align")

        q_index = HNSWIndex()
        q_rows = np.arange(len(grouped_queries), dtype=np.int32)
        q_index.build(
            embeddings,
            q_rows,
            path / "query_hnsw",
            M=32,
            ef_construction=180,
            ef_search=100,
            num_threads=int(hnsw_threads),
            chunk_size=2048,
        )

        item_lists: list[list[int]] = []
        for row in grouped_queries.itertuples(index=False):
            vals = []
            seen = set()
            for item_id in row.positive_item_ids:
                item_row = item_id_to_row.get(str(item_id))
                if item_row is not None and item_row not in seen:
                    seen.add(item_row)
                    vals.append(int(item_row))
                if len(vals) >= int(top_items_per_query):
                    break
            item_lists.append(vals)

        indptr = np.zeros(len(item_lists) + 1, dtype=np.int64)
        flat: list[int] = []
        for i, vals in enumerate(item_lists):
            flat.extend(vals)
            indptr[i + 1] = len(flat)

        np.save(path / "indptr.npy", indptr)
        np.save(path / "item_rows.npy", np.asarray(flat, dtype=np.int32))
        query_keys = np.asarray(grouped_queries["query_key"].astype(str).tolist(), dtype="<U32")
        if not np.all(np.char.str_len(query_keys) == 32):
            raise ValueError("query_key must be a 32-character hexadecimal string")
        np.save(path / "query_keys.npy", query_keys, allow_pickle=False)
        grouped_queries[
            [
                "query_key", "split_key", "search_query", "search_location_id",
                "search_is_delivery_search", "search_infm_params_text", "search_category",
            ]
        ].to_parquet(path / "queries.parquet", index=False)

        self.query_index = q_index
        self.query_keys = query_keys
        self.indptr = indptr
        self.item_rows = np.asarray(flat, dtype=np.int32)
        self.path = path

    def load(self, path: str | Path, ef_search: int = 100, num_threads: int = 4):
        from .indexes import HNSWIndex

        path = Path(path)
        self.query_index = HNSWIndex().load(path / "query_hnsw", ef_search=ef_search, num_threads=num_threads)
        self.indptr = np.load(path / "indptr.npy", mmap_mode="r")
        self.item_rows = np.load(path / "item_rows.npy", mmap_mode="r")
        self.query_keys = np.load(path / "query_keys.npy", allow_pickle=False)
        self.path = path
        return self

    def search(
        self,
        query_vector: np.ndarray,
        top_queries: int,
        top_items: int,
        min_similarity: float,
        exclude_query_key: str | None = None,
    ):
        results = self.search_many(
            np.asarray(query_vector, dtype=np.float32).reshape(1, -1),
            top_queries=top_queries,
            top_items=top_items,
            min_similarity=min_similarity,
            exclude_query_keys=[exclude_query_key],
        )
        return results[0]

    def search_many(
        self,
        query_vectors: np.ndarray,
        top_queries: int,
        top_items: int,
        min_similarity: float,
        exclude_query_keys: list[str | None] | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self.query_index is None or len(self.query_keys) == 0:
            empty = (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32))
            return [empty for _ in range(len(np.asarray(query_vectors)))]

        vectors = np.asarray(query_vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if exclude_query_keys is None:
            exclude_query_keys = [None] * len(vectors)
        if len(exclude_query_keys) != len(vectors):
            raise ValueError("exclude_query_keys and query_vectors have different lengths")

        k = min(int(top_queries) + 4, len(self.query_keys))
        labels, distances = self.query_index.index.knn_query(vectors, k=k)
        outputs = []
        for q_pos, (q_labels, q_distances) in enumerate(zip(labels, distances)):
            q_sim = (1.0 - q_distances).astype(np.float32)
            exclude_query_key = exclude_query_keys[q_pos]
            scores: dict[int, float] = {}
            source_rank: dict[int, int] = {}
            used_queries = 0
            for rank, (query_row, similarity) in enumerate(zip(q_labels, q_sim), start=1):
                if exclude_query_key is not None and str(self.query_keys[int(query_row)]) == str(exclude_query_key):
                    continue
                if used_queries >= int(top_queries):
                    break
                if float(similarity) < float(min_similarity):
                    continue
                used_queries += 1
                start, end = int(self.indptr[query_row]), int(self.indptr[query_row + 1])
                for item_row in self.item_rows[start:end]:
                    item_row = int(item_row)
                    if float(similarity) > scores.get(item_row, -1.0):
                        scores[item_row] = float(similarity)
                        source_rank[item_row] = rank
            ranked = sorted(scores, key=lambda row: (-scores[row], source_rank[row], row))[: int(top_items)]
            outputs.append((
                np.asarray(ranked, dtype=np.int32),
                np.asarray([scores[row] for row in ranked], dtype=np.float32),
                np.asarray([source_rank[row] for row in ranked], dtype=np.int32),
            ))
        return outputs
