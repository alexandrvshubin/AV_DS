from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .embeddings import E5Encoder
from .history import QueryHistory
from .indexes import BM25Index, CharTfidfIndex, HNSWIndex
from .progress import progress
from .query_expansion import HistoricalTermExpansion
from .text import compact_query_text, dense_query_text, enrich_query_with_morphology


@dataclass(frozen=True)
class RetrievalResult:
    candidates: pd.DataFrame


class HybridRetriever:
    def __init__(
        self,
        items: pd.DataFrame,
        bm25: BM25Index,
        char_index: CharTfidfIndex,
        dense_index: HNSWIndex,
        history: QueryHistory | None,
        encoder: E5Encoder,
        item_embeddings: np.ndarray | None,
        term_expander: HistoricalTermExpansion | None,
        config: dict[str, Any],
    ):
        self.items = items.reset_index(drop=True)
        self.bm25 = bm25
        self.char = char_index
        self.dense = dense_index
        self.history = history
        self.encoder = encoder
        self.item_embeddings = item_embeddings
        self.term_expander = term_expander
        self.config = config
        self.item_ids = self.items["item_id"].astype(str).to_numpy()
        self.locations = self.items.groupby("item_location_id", sort=False).indices

    @classmethod
    def from_artifacts(cls, items, encoder, artifact_root, config, history_name: str, term_expansion_name: str):
        root = Path(artifact_root)
        idx_cfg = config["indexes"]
        compute_cfg = config["compute"]
        threads = compute_cfg["hnsw_threads"]
        if threads == "auto":
            base_threads = compute_cfg.get("cpu_threads", "auto")
            if base_threads == "auto":
                import os
                base_threads = os.cpu_count() or 8
            threads = max(1, int(round(int(base_threads) * 0.8)))
        bm25 = BM25Index().load(root / "items_bm25")
        char = CharTfidfIndex().load(root / "items_char")
        dense = HNSWIndex().load(
            root / "items_dense",
            ef_search=int(idx_cfg["dense"]["ef_search"]),
            num_threads=int(threads),
        )
        history_path = root / history_name
        history = (
            QueryHistory().load(
                history_path,
                ef_search=int(idx_cfg["history"]["top_queries"]) * 4,
                num_threads=int(threads),
            )
            if (history_path / "query_hnsw" / "index.bin").exists()
            else None
        )
        embedding_path = root / "item_embeddings.npy"
        item_embeddings = np.load(embedding_path, mmap_mode="r") if embedding_path.exists() else None
        expansion_path = root / f"{term_expansion_name}.json"
        expander = HistoricalTermExpansion.load(expansion_path) if expansion_path.exists() else None
        return cls(items, bm25, char, dense, history, encoder, item_embeddings, expander, config)

    def with_label_artifacts(self, history: QueryHistory | None, term_expander: HistoricalTermExpansion | None):
        return HybridRetriever(
            self.items, self.bm25, self.char, self.dense, history,
            self.encoder, self.item_embeddings, term_expander, self.config,
        )

    @staticmethod
    def _add_source(store: dict[int, dict[str, float]], rows, scores, source: str) -> None:
        for rank, (row_idx, score) in enumerate(zip(rows, scores), start=1):
            row_idx = int(row_idx)
            entry = store.setdefault(row_idx, {})
            entry[f"{source}_score"] = float(score)
            entry[f"{source}_rank"] = float(rank)

    def _local_dense_many(
        self,
        query_vectors: np.ndarray,
        location_ids: np.ndarray,
        top_k: int,
        max_items: int,
        batch_size: int = 16,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        outputs: list[tuple[np.ndarray, np.ndarray]] = [
            (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)) for _ in range(len(query_vectors))
        ]
        if self.item_embeddings is None or len(query_vectors) == 0:
            return outputs
        groups: dict[int, list[int]] = {}
        for pos, location_id in enumerate(location_ids.astype(np.int64, copy=False)):
            groups.setdefault(int(location_id), []).append(pos)

        for location_id, positions in groups.items():
            idxs_raw = self.locations.get(location_id)
            if idxs_raw is None or len(idxs_raw) == 0 or len(idxs_raw) > int(max_items):
                continue
            idxs = np.asarray(idxs_raw, dtype=np.int32)
            item_vectors = np.asarray(self.item_embeddings[idxs], dtype=np.float32)
            for start in range(0, len(positions), int(batch_size)):
                pos_batch = positions[start:start + int(batch_size)]
                q_batch = np.asarray(query_vectors[pos_batch], dtype=np.float32)
                scores_matrix = q_batch @ item_vectors.T
                k = min(int(top_k), len(idxs))
                if k <= 0:
                    continue
                for local_pos, global_pos in enumerate(pos_batch):
                    scores = scores_matrix[local_pos]
                    if k == len(scores):
                        part = np.argsort(-scores)
                    else:
                        part = np.argpartition(-scores, k - 1)[:k]
                        part = part[np.argsort(-scores[part])]
                    outputs[global_pos] = (idxs[part], scores[part].astype(np.float32))
        return outputs

    def _local_dense(self, q_vec: np.ndarray, location_id: int, top_k: int, max_items: int):
        if self.item_embeddings is None:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        idxs = self.locations.get(location_id)
        if idxs is None or len(idxs) == 0 or len(idxs) > int(max_items):
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        idxs = np.asarray(idxs, dtype=np.int32)
        vectors = np.asarray(self.item_embeddings[idxs], dtype=np.float32)
        scores = vectors @ np.asarray(q_vec, dtype=np.float32).reshape(-1)
        k = min(int(top_k), len(scores))
        if k == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        part = np.argpartition(-scores, k - 1)[:k]
        part = part[np.argsort(-scores[part])]
        return idxs[part], scores[part].astype(np.float32)

    def _fuse(
        self,
        source_results: dict[str, tuple[np.ndarray, np.ndarray]],
        history_result: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> pd.DataFrame:
        store: dict[int, dict[str, float]] = {}
        for source, (rows, scores) in source_results.items():
            self._add_source(store, rows, scores, source)
        if history_result is not None:
            rows, scores, ranks = history_result
            for row_idx, score, rank in zip(rows, scores, ranks):
                entry = store.setdefault(int(row_idx), {})
                entry["history_score"] = float(score)
                entry["history_rank"] = float(rank)

        pool_cfg = self.config["candidate_pool"]
        weights = pool_cfg["source_weights"]
        rrf_k = float(pool_cfg["rrf_k"])
        rows_out = []
        for row_idx, source_features in store.items():
            rrf = 0.0
            for source, weight in weights.items():
                rank = source_features.get(f"{source}_rank")
                if rank is not None:
                    rrf += float(weight) / (rrf_k + float(rank))
            row = dict(source_features)
            row["item_row"] = int(row_idx)
            row["item_id"] = str(self.item_ids[row_idx])
            row["rrf_score"] = float(rrf)
            rows_out.append(row)

        frame = pd.DataFrame(rows_out)
        if frame.empty:
            return frame
        return frame.sort_values(["rrf_score", "item_row"], ascending=[False, True]).head(
            int(pool_cfg["max_candidates"])
        ).reset_index(drop=True)

    def retrieve_one(
        self,
        query_row: pd.Series | dict[str, Any],
        q_vec: np.ndarray | None = None,
        exclude_history_query: bool = False,
    ) -> RetrievalResult:
        q = dict(query_row)
        query_key = q.get("query_key")
        lexical_query = compact_query_text(q.get("search_query", ""), q.get("search_infm_params_text", ""))
        morph_query = enrich_query_with_morphology(lexical_query)
        expanded_query = self.term_expander.expand(morph_query) if self.term_expander else morph_query
        if q_vec is None:
            qdense = dense_query_text(
                q.get("search_query", ""),
                q.get("search_infm_params_text", ""),
                q.get("search_is_delivery_search", 0),
            )
            q_vec = self.encoder.encode([qdense], desc="Query embedding")[0]

        idx_cfg = self.config["indexes"]
        source_results = {
            "bm25": self.bm25.search(expanded_query, int(idx_cfg["bm25"]["top_k"])),
            "char_tfidf": self.char.search(lexical_query, int(idx_cfg["char_tfidf"]["top_k"])),
            "dense": self.dense.search(q_vec, int(idx_cfg["dense"]["top_k"])),
        }
        source_results["local_dense"] = self._local_dense(
            q_vec,
            int(q.get("search_location_id", -1)),
            top_k=int(self.config["compute"].get("local_dense_top_k", 80)),
            max_items=int(self.config["compute"]["local_dense_max_items"]),
        )
        history_result = None
        if self.history is not None:
            history_result = self.history.search(
                q_vec,
                top_queries=int(idx_cfg["history"]["top_queries"]),
                top_items=int(idx_cfg["history"]["top_k_items"]),
                min_similarity=float(idx_cfg["history"]["min_similarity"]),
                exclude_query_key=(str(query_key) if exclude_history_query and query_key is not None else None),
            )
        return RetrievalResult(self._fuse(source_results, history_result))

    def embed_queries(self, queries: pd.DataFrame) -> np.ndarray:
        texts = [
            dense_query_text(r.search_query, r.search_infm_params_text, r.search_is_delivery_search)
            for r in queries.itertuples(index=False)
        ]
        return self.encoder.encode(texts, desc="Query embeddings")

    def retrieve_many(
        self,
        queries: pd.DataFrame,
        query_vectors: np.ndarray | None = None,
        desc: str = "Retrieving candidates",
    ) -> list[pd.DataFrame]:
        """Batch the expensive ANN/sparse retrieval operations before per-query fusion.

        This retains exactly the same source definitions as retrieve_one but avoids one
        Python/library invocation for every individual query whenever a backend supports
        batched search.
        """
        queries = queries.reset_index(drop=True)
        if query_vectors is None:
            query_vectors = self.embed_queries(queries)
        if len(query_vectors) != len(queries):
            raise ValueError("query_vectors and queries have different lengths")
        idx_cfg = self.config["indexes"]
        compute_cfg = self.config["compute"]

        lexical_queries = []
        expanded_queries = []
        for row in queries.itertuples(index=False):
            lexical = compact_query_text(row.search_query, row.search_infm_params_text)
            morph = enrich_query_with_morphology(lexical)
            expanded_queries.append(self.term_expander.expand(morph) if self.term_expander else morph)
            lexical_queries.append(lexical)

        bm25_results = self.bm25.search_many(expanded_queries, int(idx_cfg["bm25"]["top_k"]))
        char_results = self.char.search_many(
            lexical_queries,
            int(idx_cfg["char_tfidf"]["top_k"]),
            batch_size=int(compute_cfg.get("retrieval_batch_size", 64)),
        )
        dense_results = self.dense.search_many(query_vectors, int(idx_cfg["dense"]["top_k"]))
        history_results = None
        if self.history is not None:
            exclude = []
            for row in queries.itertuples(index=False):
                exclude.append(str(row.query_key) if hasattr(row, "query_key") else None)
            history_results = self.history.search_many(
                query_vectors,
                top_queries=int(idx_cfg["history"]["top_queries"]),
                top_items=int(idx_cfg["history"]["top_k_items"]),
                min_similarity=float(idx_cfg["history"]["min_similarity"]),
                exclude_query_keys=exclude,
            )

        results: list[pd.DataFrame] = []
        local_top_k = int(compute_cfg.get("local_dense_top_k", 80))
        local_max_items = int(compute_cfg["local_dense_max_items"])
        local_results = self._local_dense_many(
            query_vectors,
            queries["search_location_id"].fillna(-1).to_numpy(dtype=np.int64),
            top_k=local_top_k,
            max_items=local_max_items,
            batch_size=int(compute_cfg.get("local_dense_batch_size", 16)),
        )
        for pos, _ in enumerate(progress(queries.itertuples(index=False), total=len(queries), desc=desc, unit="query")):
            sources = {
                "bm25": bm25_results[pos],
                "char_tfidf": char_results[pos],
                "dense": dense_results[pos],
                "local_dense": local_results[pos],
            }
            results.append(self._fuse(sources, history_results[pos] if history_results is not None else None))
        return results
