from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, FeatureBuilder
from .progress import Timer, heartbeat, progress


_CACHE_KEY_COL = "__query_key"
_CACHE_LABEL_COL = "__label"
_CACHE_POS_COL = "__query_pos"


class LearningToRankModel:
    """LightGBM LambdaRank model with explicit train/validation monitoring."""

    def __init__(self, booster=None, feature_names: list[str] | None = None, training_report: dict | None = None):
        self.booster = booster
        self.feature_names = feature_names or list(FEATURE_NAMES)
        self.training_report = training_report or {}

    @staticmethod
    def _n_jobs(config: dict) -> int:
        value = config.get("n_jobs", "auto")
        if value == "auto":
            import os
            return max(1, int(round((os.cpu_count() or 8) * 0.75)))
        return max(1, int(value))

    @staticmethod
    def _make_model(config: dict, n_estimators: int | None = None):
        import lightgbm as lgb

        return lgb.LGBMRanker(
            n_jobs=1,
            force_col_wise=True,
            deterministic=True,
            objective=str(config.get("objective", "lambdarank")),
            metric=str(config.get("metric", "ndcg")),
            num_leaves=int(config.get("num_leaves", 63)),
            min_child_samples=int(config.get("min_child_samples", 40)),
            learning_rate=float(config.get("learning_rate", 0.045)),
            n_estimators=int(n_estimators if n_estimators is not None else config.get("n_estimators", 650)),
            max_depth=int(config.get("max_depth", -1)),
            subsample=float(config.get("subsample", 0.90)),
            subsample_freq=int(config.get("subsample_freq", 1)),
            colsample_bytree=float(config.get("colsample_bytree", 0.90)),
            reg_alpha=float(config.get("reg_alpha", 0.25)),
            reg_lambda=float(config.get("reg_lambda", 2.0)),
            random_state=int(config.get("random_state", 42)),
            verbosity=-1,
            lambdarank_truncation_level=int(config.get("lambdarank_truncation_level", 53)),
        )

    @staticmethod
    def _validate_table(features: pd.DataFrame, labels: np.ndarray, groups: list[int], name: str) -> None:
        if not groups:
            raise ValueError(f"{name}: no ranking groups")
        if sum(groups) != len(features):
            raise ValueError(f"{name}: sum(groups)={sum(groups)} but rows={len(features)}")
        if len(labels) != len(features):
            raise ValueError(f"{name}: labels and feature rows have different lengths")
        if int(np.sum(labels)) <= 0:
            raise ValueError(f"{name}: no positive labels")
        offset = 0
        for group_size in groups:
            group_labels = labels[offset:offset + group_size]
            offset += group_size
            if int(np.sum(group_labels)) <= 0:
                raise ValueError(f"{name}: a query group has no positive candidate")

    def fit_with_validation(
        self,
        train_features: pd.DataFrame,
        train_labels: np.ndarray,
        train_groups: list[int],
        val_features: pd.DataFrame,
        val_labels: np.ndarray,
        val_groups: list[int],
        config: dict,
    ) -> dict:
        import lightgbm as lgb

        self._validate_table(train_features, train_labels, train_groups, "train")
        self._validate_table(val_features, val_labels, val_groups, "validation")

        x_train = train_features[self.feature_names].astype(np.float32)
        x_val = val_features[self.feature_names].astype(np.float32)
        y_train = np.asarray(train_labels, dtype=np.int8)
        y_val = np.asarray(val_labels, dtype=np.int8)

        model = self._make_model(config)
        stopping = int(config.get("early_stopping_rounds", 60))
        log_period = int(config.get("log_period", 25))
        print(
            f"LambdaRank TRAIN: {len(train_groups):,} queries / {len(x_train):,} rows; "
            f"VAL: {len(val_groups):,} queries / {len(x_val):,} rows"
        )
        print("objective=lambdarank; loss=LambdaRank pairwise/listwise surrogate; validation metric=NDCG@50")

        with Timer("LightGBM train+validation"), heartbeat("LightGBM train+validation"):
            model.fit(
                x_train,
                y_train,
                group=train_groups,
                eval_set=[(x_train, y_train), (x_val, y_val)],
                eval_names=["train", "validation"],
                eval_group=[train_groups, val_groups],
                eval_at=[50],
                callbacks=[
                    lgb.early_stopping(stopping, first_metric_only=True, verbose=False),
                    lgb.log_evaluation(period=log_period),
                ],
            )

        best_iteration = int(getattr(model, "best_iteration_", 0) or 0)
        if best_iteration <= 0:
            best_iteration = int(config.get("n_estimators", 650))

        validation_scores = model.best_score_.get("validation", {}) if getattr(model, "best_score_", None) else {}
        best_ndcg = float("nan")
        for key, value in validation_scores.items():
            if "ndcg@50" in key.lower():
                best_ndcg = float(value)
                break

        self.booster = model
        self.training_report = {
            "mode": "tuning_with_validation",
            "objective": str(config.get("objective", "lambdarank")),
            "loss_description": "LambdaRank ranking objective; no BCE/MSE scalar loss is used",
            "monitor_metric": "ndcg@50",
            "best_iteration": best_iteration,
            "best_validation_ndcg_at_50": best_ndcg,
            "train_rows": int(len(x_train)),
            "train_queries": int(len(train_groups)),
            "train_positive_rows": int(np.sum(y_train)),
            "validation_rows": int(len(x_val)),
            "validation_queries": int(len(val_groups)),
            "validation_positive_rows": int(np.sum(y_val)),
            "requested_estimators": int(config.get("n_estimators", 650)),
            "early_stopping_rounds": stopping,
            "eval_history": getattr(model, "evals_result_", {}),
            "feature_importance_gain": {
                name: float(value)
                for name, value in zip(self.feature_names, model.booster_.feature_importance(importance_type="gain"))
            },
        }
        return self.training_report

    def fit_full(self, features: pd.DataFrame, labels: np.ndarray, groups: list[int], config: dict, n_estimators: int) -> dict:
        import lightgbm as lgb

        self._validate_table(features, labels, groups, "production")
        x = features[self.feature_names].astype(np.float32)
        y = np.asarray(labels, dtype=np.int8)
        n_estimators = max(1, int(n_estimators))
        model = self._make_model(config, n_estimators=n_estimators)
        with Timer(f"LightGBM production fit ({n_estimators} trees)"), heartbeat("LightGBM production fit"):
            model.fit(
                x,
                y,
                group=groups,
                eval_set=[(x, y)],
                eval_names=["train"],
                eval_group=[groups],
                eval_at=[50],
                callbacks=[lgb.log_evaluation(period=int(config.get("log_period", 25)))],
            )
        self.booster = model
        self.training_report = {
            "mode": "production_fit",
            "objective": str(config.get("objective", "lambdarank")),
            "loss_description": "LambdaRank ranking objective; label-derived retrieval features use exact-query exclusion",
            "monitor_metric": "ndcg@50",
            "n_estimators": n_estimators,
            "train_rows": int(len(x)),
            "train_queries": int(len(groups)),
            "train_positive_rows": int(np.sum(y)),
            "eval_history": getattr(model, "evals_result_", {}),
            "feature_importance_gain": {
                name: float(value)
                for name, value in zip(self.feature_names, model.booster_.feature_importance(importance_type="gain"))
            },
        }
        return self.training_report

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Ranker is not fitted")
        return np.asarray(self.booster.predict(features[self.feature_names].astype(np.float32)), dtype=np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        path.with_name(path.stem + "_training.json").write_text(
            json.dumps(self.training_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        path.with_name(path.stem + "_features.json").write_text(
            json.dumps(self.feature_names, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path):
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Unexpected ranker artifact type: {type(model)!r}")
        return model


@dataclass(frozen=True)
class RankerTableResult:
    features: pd.DataFrame
    labels: np.ndarray
    groups: list[int]
    stats: dict[str, object]


def _select_training_candidates(result: pd.DataFrame, positives: set[str], max_candidates: int, seed: int) -> pd.DataFrame:
    if len(result) <= max_candidates:
        return result
    positive = result[result.item_id.astype(str).isin(positives)]
    keep = set(positive.item_id.astype(str))
    selected = [positive]
    budget = max(0, int(max_candidates) - len(keep))

    sources = ["bm25_rank", "char_tfidf_rank", "dense_rank", "local_dense_rank", "history_rank", "rrf_score"]
    remaining = result.drop(index=positive.index)
    for col in sources:
        if budget <= 0 or col not in result.columns:
            continue
        pool = remaining[remaining[col].notna()]
        pool = pool.sort_values(col, ascending=(col != "rrf_score"))
        take = min(budget, max(8, int(max_candidates) // (len(sources) + 1)))
        part = pool[~pool.item_id.astype(str).isin(keep)].head(take)
        if not part.empty:
            selected.append(part)
            keep.update(part.item_id.astype(str))
            budget = int(max_candidates) - len(keep)

    if budget > 0:
        pool = remaining[~remaining.item_id.astype(str).isin(keep)]
        if not pool.empty:
            rng = np.random.default_rng(seed)
            n = min(budget, len(pool))
            selected.append(pool.iloc[rng.choice(len(pool), size=n, replace=False)])

    return pd.concat(selected, ignore_index=True).drop_duplicates("item_id").head(max_candidates).reset_index(drop=True)


def _build_one_feature_group(
    pos: int,
    query_row: pd.Series,
    candidate_result: pd.DataFrame,
    feature_builder: FeatureBuilder,
    max_candidates_per_query: int,
    seed: int,
):
    positives = set(map(str, query_row.positive_item_ids))
    if candidate_result.empty:
        return pos, None, 0.0, False, 0
    found = positives & set(candidate_result.item_id.astype(str))
    recall = len(found) / max(1, len(positives))
    if not found:
        return pos, None, recall, False, 0
    selected = _select_training_candidates(candidate_result, positives, max_candidates_per_query, seed + pos)
    feats = feature_builder.build(query_row, selected)
    if feats.empty:
        return pos, None, recall, False, 0
    y = feats.item_id.astype(str).isin(positives).astype(np.int8).to_numpy()
    if int(y.sum()) == 0:
        return pos, None, recall, False, 0
    feats[_CACHE_KEY_COL] = str(query_row.query_key)
    feats[_CACHE_POS_COL] = int(pos)
    return pos, feats, recall, True, int(y.sum()), y


def make_ranker_training_table(
    query_groups: pd.DataFrame,
    retriever,
    feature_builder: FeatureBuilder,
    max_queries: int | None,
    max_candidates_per_query: int,
    seed: int,
    query_vectors: np.ndarray | None = None,
    desc: str = "Build ranker table",
    feature_workers: int = 1,
) -> RankerTableResult:
    query_groups = query_groups.reset_index(drop=True)
    if max_queries is not None and len(query_groups) > int(max_queries):
        query_groups = query_groups.sample(int(max_queries), random_state=int(seed)).reset_index(drop=True)
        query_vectors = None

    if query_vectors is None:
        query_vectors = retriever.embed_queries(query_groups)
    if len(query_vectors) != len(query_groups):
        raise ValueError("query_vectors and query_groups have different lengths")

    with Timer(f"{desc}: batched retrieval"), heartbeat(f"{desc}: batched retrieval"):
        candidate_results = retriever.retrieve_many(query_groups, query_vectors=query_vectors, desc=f"{desc} retrieval")

    feature_rows: list[tuple[int, pd.DataFrame, float, bool, int, np.ndarray]] = []
    seen = 0
    candidate_recall_values = []
    workers = max(1, int(feature_workers))

    with Timer(f"{desc}: feature construction"), heartbeat(f"{desc}: feature construction"):
        if workers == 1:
            iterator = []
            for pos in range(len(query_groups)):
                iterator.append(_build_one_feature_group(
                    pos, query_groups.iloc[pos], candidate_results[pos], feature_builder,
                    int(max_candidates_per_query), int(seed),
                ))
            completed = iterator
        else:
            completed = []
            futures = []
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="features") as pool:
                for pos in range(len(query_groups)):
                    futures.append(pool.submit(
                        _build_one_feature_group,
                        pos, query_groups.iloc[pos], candidate_results[pos], feature_builder,
                        int(max_candidates_per_query), int(seed),
                    ))
                for future in progress(as_completed(futures), total=len(futures), desc=f"{desc} features", unit="query"):
                    completed.append(future.result())
            completed.sort(key=lambda item: item[0])

    labels: list[np.ndarray] = []
    groups: list[int] = []
    usable_rows: list[pd.DataFrame] = []
    positive = with_positive = 0
    for result in completed:
        _, feats, recall, usable, positive_count, *rest = result
        seen += 1
        candidate_recall_values.append(recall)
        if not usable or feats is None:
            continue
        y = rest[0]
        usable_rows.append(feats)
        labels.append(y)
        groups.append(len(feats))
        with_positive += 1
        positive += int(positive_count)

    if not usable_rows:
        raise RuntimeError("No training query groups retained: retrieval did not recover any positives")
    features = pd.concat(usable_rows, ignore_index=True)
    label_array = np.concatenate(labels).astype(np.int8)
    stats = {
        "queries_seen": int(seen),
        "queries_with_positive_in_candidates": int(with_positive),
        "queries_without_positive_in_candidates": int(seen - with_positive),
        "candidate_recall_at_pool_mean": float(np.mean(candidate_recall_values)) if candidate_recall_values else 0.0,
        "rows": int(len(features)),
        "positive_rows": int(positive),
        "groups": int(len(groups)),
    }
    return RankerTableResult(features, label_array, groups, stats)


def save_ranker_table(result: RankerTableResult, path: str | Path, signature: str, table_name: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = result.features.copy()
    if _CACHE_LABEL_COL not in frame.columns:
        raise ValueError("Ranker cache requires labels aligned with feature rows")
    frame.to_parquet(path, index=False)
    meta = {
        "signature": signature,
        "table_name": table_name,
        "rows": int(len(frame)),
        "groups": int(len(result.groups)),
        "positive_rows": int(result.labels.sum()),
        "stats": result.stats,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def attach_labels_and_save(result: RankerTableResult, path: str | Path, signature: str, table_name: str) -> None:
    frame = result.features.copy()
    frame[_CACHE_LABEL_COL] = np.asarray(result.labels, dtype=np.int8)
    if _CACHE_KEY_COL not in frame.columns:
        raise ValueError("Ranker cache requires query keys")
    save_ranker_table(
        RankerTableResult(frame, result.labels, result.groups, result.stats), path, signature, table_name
    )


def load_ranker_table(path: str | Path, signature: str) -> RankerTableResult | None:
    path = Path(path)
    meta_path = path.with_suffix(".json")
    if not path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("signature") != signature:
            return None
        frame = pd.read_parquet(path)
        if _CACHE_LABEL_COL not in frame.columns or _CACHE_KEY_COL not in frame.columns:
            return None
        labels = frame.pop(_CACHE_LABEL_COL).to_numpy(dtype=np.int8)
        keys = frame[_CACHE_KEY_COL].astype(str).to_numpy()
        groups = []
        if len(keys):
            boundary = np.flatnonzero(keys[1:] != keys[:-1]) + 1
            starts = np.concatenate(([0], boundary, [len(keys)]))
            groups = [int(b - a) for a, b in zip(starts[:-1], starts[1:])]
        return RankerTableResult(frame, labels, groups, meta.get("stats", {}))
    except Exception:
        return None
