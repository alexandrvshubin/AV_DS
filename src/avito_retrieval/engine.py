from __future__ import annotations

from typing import Any

import pandas as pd

from .features import FeatureBuilder
from .progress import Timer, progress
from .ranker import LearningToRankModel
from .retrieval import HybridRetriever


class SearchEngine:
    def __init__(self, retriever: HybridRetriever, feature_builder: FeatureBuilder, ranker: LearningToRankModel | None, config: dict[str, Any]):
        self.retriever = retriever
        self.feature_builder = feature_builder
        self.ranker = ranker
        self.config = config

    def _ranked_frame(self, query_row, candidates, features=None):
        if self.ranker is None:
            merged = candidates.copy()
            merged["model_score"] = merged["rrf_score"]
            return merged
        if features is None:
            features = self.feature_builder.build(query_row, candidates)
        features["model_score"] = self.ranker.predict(features)
        merged = candidates.merge(features[["item_row", "item_id", "model_score"]], on=["item_row", "item_id"], how="left")
        merged["model_score"] = merged["model_score"].fillna(-1e9)
        return merged.sort_values(["model_score", "rrf_score", "item_row"], ascending=[False, False, True]).reset_index(drop=True)

    def apply_protection(self, candidates: pd.DataFrame, ranked: pd.DataFrame, features: pd.DataFrame, profile: str) -> list[str]:
        profiles = self.config["candidate_pool"]["protection_profiles"]
        quotas = profiles.get(profile, profiles[self.config["candidate_pool"]["default_protection_profile"]])
        selected: list[str] = []
        seen: set[str] = set()

        for source, quota in quotas.items():
            if source == "exact_anchor":
                anchor = features.loc[features["query_in_title_exact"] > 0].sort_values("rrf_score", ascending=False)
                ids = anchor.item_id.astype(str).tolist()
            else:
                rank_col = f"{source}_rank"
                if rank_col not in candidates.columns:
                    continue
                ids = candidates.loc[candidates[rank_col].notna()].sort_values(rank_col).item_id.astype(str).tolist()
            for item_id in ids[: int(quota)]:
                if item_id not in seen:
                    seen.add(item_id)
                    selected.append(item_id)
                    if len(selected) >= 50:
                        return selected[:50]

        for item_id in ranked.item_id.astype(str):
            if item_id not in seen:
                seen.add(item_id)
                selected.append(item_id)
                if len(selected) >= 50:
                    break
        return selected[:50]

    def rank_candidates(self, query_row: pd.Series | dict[str, Any], protection_profile: str | None = None):
        candidates = self.retriever.retrieve_one(query_row).candidates
        if candidates.empty:
            return [], candidates, candidates
        features = self.feature_builder.build(query_row, candidates)
        ranked = self._ranked_frame(query_row, candidates, features)
        if self.ranker is None:
            final = ranked.sort_values(["rrf_score", "item_row"], ascending=[False, True]).item_id.astype(str).tolist()[:50]
        else:
            profile = protection_profile or self.config["candidate_pool"]["default_protection_profile"]
            final = self.apply_protection(candidates, ranked, features, profile)
        return final, ranked, features

    def predict(self, queries: pd.DataFrame, protection_profile: str | None = None, desc: str = "Candidate generation"):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        queries = queries.reset_index(drop=True)
        profile = protection_profile or self.config["candidate_pool"]["default_protection_profile"]
        vectors = self.retriever.embed_queries(queries)
        candidates = self.retriever.retrieve_many(queries, query_vectors=vectors, desc=f"{desc} retrieval")

        def build_one(pos: int):
            query_row = queries.iloc[pos]
            candidate_frame = candidates[pos]
            if candidate_frame.empty:
                return pos, []
            features = self.feature_builder.build(query_row, candidate_frame)
            ranked = self._ranked_frame(query_row, candidate_frame, features)
            if self.ranker is None:
                result = ranked.sort_values(["rrf_score", "item_row"], ascending=[False, True]).item_id.astype(str).tolist()[:50]
            else:
                result = self.apply_protection(candidate_frame, ranked, features, profile)
            return pos, result

        workers = max(1, int(self.config["compute"].get("feature_workers", 1)))
        outputs: list[list[str]] = [[] for _ in range(len(queries))]
        with Timer(f"{desc} ranking"):
            if workers == 1:
                iterator = (build_one(pos) for pos in range(len(queries)))
                for pos, result in progress(iterator, total=len(queries), desc=desc, unit="query"):
                    outputs[pos] = result
            else:
                futures = []
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="predict-features") as pool:
                    for pos in range(len(queries)):
                        futures.append(pool.submit(build_one, pos))
                    for future in progress(as_completed(futures), total=len(futures), desc=desc, unit="query"):
                        pos, result = future.result()
                        outputs[pos] = result
        return outputs

    def inspect(self, query_row, top_n: int = 20, protection_profile: str | None = None) -> pd.DataFrame:
        final, ranked, features = self.rank_candidates(query_row, protection_profile=protection_profile)
        if not final:
            return pd.DataFrame()
        keep = ranked[ranked.item_id.astype(str).isin(final[:top_n])].copy()
        keep["final_rank"] = keep.item_id.astype(str).map({x: i + 1 for i, x in enumerate(final)})
        lookup = self.retriever.items.set_index("item_id")
        keep["title"] = keep.item_id.astype(str).map(lookup["item_title_raw"])
        keep["location_id"] = keep.item_id.astype(str).map(lookup["item_location_id"])
        keep["rating"] = keep.item_id.astype(str).map(lookup["item_rating"])
        keep["description"] = keep.item_id.astype(str).map(lookup["item_description_raw"]).fillna("").str.slice(0, 500)
        cols = [
            "final_rank", "item_id", "model_score", "rrf_score", "bm25_score", "char_score",
            "dense_score", "local_dense_score", "history_score", "title", "location_id", "rating", "description",
        ]
        return keep.sort_values("final_rank")[[c for c in cols if c in keep.columns]]
