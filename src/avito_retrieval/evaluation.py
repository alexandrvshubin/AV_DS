from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def recall_at_k(predictions: list[list[str]], truths: list[set[str]], k: int = 50) -> float:
    values = [len(set(map(str, pred[:k])) & truth) / max(1, len(truth)) for pred, truth in zip(predictions, truths)]
    return float(np.mean(values)) if values else 0.0


def hit_rate_at_k(predictions: list[list[str]], truths: list[set[str]], k: int = 50) -> float:
    values = [float(bool(set(map(str, pred[:k])) & truth)) for pred, truth in zip(predictions, truths)]
    return float(np.mean(values)) if values else 0.0


def evaluate_candidate_sources(engine, query_groups: pd.DataFrame, max_queries: int | None = None) -> dict:
    if max_queries is not None and len(query_groups) > int(max_queries):
        query_groups = query_groups.sample(int(max_queries), random_state=42).reset_index(drop=True)
    vectors = engine.retriever.embed_queries(query_groups)
    source_names = ["bm25", "char_tfidf", "dense", "local_dense", "history", "rrf"]
    predictions = {name: [] for name in source_names}
    truths = []

    from .progress import progress
    for pos, (_, q) in enumerate(progress(query_groups.iterrows(), total=len(query_groups), desc="Evaluate retrieval sources", unit="query")):
        candidates = engine.retriever.retrieve_one(q, q_vec=vectors[pos], exclude_history_query=True).candidates
        truth = set(map(str, q.positive_item_ids))
        truths.append(truth)
        for source in source_names:
            if source == "rrf":
                ids = candidates.sort_values(["rrf_score", "item_row"], ascending=[False, True]).item_id.astype(str).tolist()
            else:
                rank_col = f"{source}_rank"
                ids = candidates.loc[candidates[rank_col].notna()].sort_values(rank_col).item_id.astype(str).tolist() if rank_col in candidates.columns else []
            predictions[source].append(ids[:50])

    return {
        source: {
            "recall@50": recall_at_k(predictions[source], truths),
            "hit_rate@50": hit_rate_at_k(predictions[source], truths),
        }
        for source in source_names
    }


def evaluate_ranker_profiles(engine, query_groups: pd.DataFrame, profiles: list[str], max_queries: int | None = None):
    if max_queries is not None and len(query_groups) > int(max_queries):
        query_groups = query_groups.sample(int(max_queries), random_state=42).reset_index(drop=True)
    vectors = engine.retriever.embed_queries(query_groups)
    source_names = ["bm25", "char_tfidf", "dense", "local_dense", "history", "rrf"]
    source_preds = {name: [] for name in source_names}
    profile_preds = {profile: [] for profile in profiles}
    truths = []

    from .progress import progress
    for pos, (_, q) in enumerate(progress(query_groups.iterrows(), total=len(query_groups), desc="Evaluate validation", unit="query")):
        candidates = engine.retriever.retrieve_one(q, q_vec=vectors[pos], exclude_history_query=True).candidates
        truth = set(map(str, q.positive_item_ids))
        truths.append(truth)
        if candidates.empty:
            for source in source_names:
                source_preds[source].append([])
            for profile in profiles:
                profile_preds[profile].append([])
            continue

        rrf_ids = candidates.sort_values(["rrf_score", "item_row"], ascending=[False, True]).item_id.astype(str).tolist()
        source_preds["rrf"].append(rrf_ids[:50])
        for source in source_names[:-1]:
            rank_col = f"{source}_rank"
            ids = (
                candidates.loc[candidates[rank_col].notna()]
                .sort_values(rank_col)
                .item_id.astype(str)
                .tolist()
                if rank_col in candidates.columns else []
            )
            source_preds[source].append(ids[:50])

        features = engine.feature_builder.build(q, candidates)
        ranked = engine._ranked_frame(q, candidates, features)
        for profile in profiles:
            profile_preds[profile].append(engine.apply_protection(candidates, ranked, features, profile))

    retrieval_metrics = {
        source: {
            "recall@50": recall_at_k(source_preds[source], truths),
            "hit_rate@50": hit_rate_at_k(source_preds[source], truths),
        }
        for source in source_names
    }
    profile_metrics = {
        profile: {
            "recall@50": recall_at_k(profile_preds[profile], truths),
            "hit_rate@50": hit_rate_at_k(profile_preds[profile], truths),
        }
        for profile in profiles
    }
    order = profiles
    best = max(
        order,
        key=lambda p: (
            profile_metrics[p]["recall@50"],
            -sum(engine.config["candidate_pool"]["protection_profiles"].get(p, {}).values()),
        ),
    )
    return {
        "retrieval_sources": retrieval_metrics,
        "rrf": retrieval_metrics["rrf"],
        "profiles": profile_metrics,
        "selected_profile": best,
        "queries_evaluated": len(query_groups),
    }


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _ndcg_binary(labels: np.ndarray, scores: np.ndarray, k: int = 50) -> float:
    labels = np.asarray(labels, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if len(labels) == 0:
        return 0.0
    k = min(int(k), len(labels))
    order = np.argsort(-scores, kind="stable")[:k]
    gains = labels[order]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2, dtype=np.float32))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(labels)[::-1][:k]
    ideal_dcg = float(np.sum(ideal * discounts))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def ndcg_at_k_from_ranker_table(table, ranker, k: int = 50) -> float:
    features = table.features
    labels = np.asarray(table.labels, dtype=np.int8)
    keys = features["__query_key"].astype(str).to_numpy()
    scores = ranker.predict(features)
    values = []
    if len(keys) == 0:
        return 0.0
    boundaries = np.flatnonzero(keys[1:] != keys[:-1]) + 1
    starts = np.concatenate(([0], boundaries, [len(keys)]))
    for start, end in zip(starts[:-1], starts[1:]):
        values.append(_ndcg_binary(labels[start:end], scores[start:end], k=k))
    return float(np.mean(values)) if values else 0.0


def evaluate_cached_ranker_table(table, truths: dict[str, set[str]], ranker, config: dict, profiles: list[str] | None = None):
    """Evaluate a materialized validation table without rerunning retrieval.

    The truth dictionary intentionally contains every selected validation query. Queries
    whose relevant items were not recovered by retrieval are represented by empty
    candidate lists, so Recall@50 retains the correct zero contribution for those queries.
    """
    features = table.features
    profiles = profiles or list(config["candidate_pool"]["protection_profiles"].keys())
    source_names = ["bm25", "char_tfidf", "dense", "local_dense", "history", "rrf"]
    source_preds = {name: [] for name in source_names}
    profile_preds = {name: [] for name in profiles}
    truth_list = []

    from .engine import SearchEngine
    helper = SearchEngine(None, None, ranker, config)

    keys = features["__query_key"].astype(str).to_numpy()
    boundaries = np.flatnonzero(keys[1:] != keys[:-1]) + 1 if len(keys) else np.empty(0, dtype=np.int64)
    starts = np.concatenate(([0], boundaries, [len(keys)]))
    groups_by_key = {}
    for left, right in zip(starts[:-1], starts[1:]):
        groups_by_key[str(keys[left])] = (int(left), int(right))

    for key, truth in truths.items():
        truth = set(map(str, truth))
        truth_list.append(truth)
        bounds = groups_by_key.get(str(key))
        if bounds is None:
            for source in source_names:
                source_preds[source].append([])
            for profile in profiles:
                profile_preds[profile].append([])
            continue

        left, right = bounds
        group = features.iloc[left:right].copy()
        for source in source_names[:-1]:
            rank_col = f"{source}_rank"
            ids = (
                group.loc[group[rank_col].notna()]
                .sort_values(rank_col)
                .item_id.astype(str)
                .tolist()
                if rank_col in group.columns else []
            )
            source_preds[source].append(ids[:50])
        source_preds["rrf"].append(
            group.sort_values(["rrf_score", "item_row"], ascending=[False, True]).item_id.astype(str).tolist()[:50]
        )

        scores = ranker.predict(group)
        ranked = group.copy()
        ranked["model_score"] = scores
        for profile in profiles:
            profile_preds[profile].append(helper.apply_protection(group, ranked, group, profile))

    retrieval_metrics = {
        source: {
            "recall@50": recall_at_k(source_preds[source], truth_list),
            "hit_rate@50": hit_rate_at_k(source_preds[source], truth_list),
        }
        for source in source_names
    }
    profile_metrics = {
        profile: {
            "recall@50": recall_at_k(profile_preds[profile], truth_list),
            "hit_rate@50": hit_rate_at_k(profile_preds[profile], truth_list),
        }
        for profile in profiles
    }
    selected = max(
        profiles,
        key=lambda p: (
            profile_metrics[p]["recall@50"],
            -sum(config["candidate_pool"]["protection_profiles"].get(p, {}).values()),
        ),
    )
    return {
        "retrieval_sources": retrieval_metrics,
        "rrf": retrieval_metrics["rrf"],
        "profiles": profile_metrics,
        "selected_profile": selected,
        "queries_evaluated": len(truth_list),
        "queries_with_candidate_features": len(groups_by_key),
        "queries_without_retrieved_positive": max(0, len(truth_list) - len(groups_by_key)),
        "ranker_ndcg@50": ndcg_at_k_from_ranker_table(table, ranker, k=50),
    }
