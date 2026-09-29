from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .data import (
    filter_pairs_to_items,
    group_positive_items,
    load_benchmark_queries,
    load_items,
    load_train_pairs,
    make_query_key,
    split_query_groups,
)
from .embeddings import E5Encoder
from .features import FeatureBuilder
from .history import QueryHistory
from .indexes import BM25Index, CharTfidfIndex, HNSWIndex
from .progress import Timer, heartbeat
from .query_expansion import HistoricalTermExpansion
from .ranker import (
    RankerTableResult,
    load_ranker_table,
    make_ranker_training_table,
    attach_labels_and_save,
)
from .retrieval import HybridRetriever
from .runtime import configure_runtime, print_runtime
from .text import char_text, dense_item_text, lexical_text


def ensure_dirs(cfg: Config) -> Path:
    root = cfg.artifact_root
    for name in [
        "items_bm25", "items_char", "items_dense", "history_train", "history_all", "models",
        "reports", "cache", "metadata", "oof",
    ]:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def make_encoder(cfg: Config) -> E5Encoder:
    compute = cfg.raw["compute"]
    return E5Encoder(
        cfg.raw["models"]["embedding_model"],
        device=compute["device"],
        batch_size=int(compute["dense_batch_size"]),
        max_length=int(compute["dense_max_length"]),
        truncate_dim=cfg.raw["models"].get("embedding_truncate_dim"),
        cache_dir=cfg._resolve(compute.get("model_cache_dir", "artifacts/cache/huggingface")),
    )


def _sha256_item_ids(series: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in series.astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _artifact_signature(cfg: Config) -> str:
    payload = {
        "package_version": "2.1.0",
        "train": {
            "path": str(cfg.train_path),
            "size": cfg.train_path.stat().st_size,
            "mtime_ns": cfg.train_path.stat().st_mtime_ns,
        },
        "items": {
            "path": str(cfg.items_path),
            "size": cfg.items_path.stat().st_size,
            "mtime_ns": cfg.items_path.stat().st_mtime_ns,
        },
        "embedding_model": cfg.raw["models"]["embedding_model"],
        "embedding_truncate_dim": cfg.raw["models"].get("embedding_truncate_dim"),
        "bm25": cfg.raw["indexes"]["bm25"],
        "char_tfidf": cfg.raw["indexes"]["char_tfidf"],
        "dense": cfg.raw["indexes"]["dense"],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def training_signature(cfg: Config) -> str:
    payload = {
        "code_version": "3.0.0-fast-ranker-cache",
        "item_index_signature": _artifact_signature(cfg),
        "ranker": cfg.raw["ranker"],
        "features": cfg.raw["features"],
        "candidate_pool": cfg.raw["candidate_pool"],
        "training": cfg.raw["training"],
        "compute_retrieval": {
            "retrieval_batch_size": cfg.raw["compute"].get("retrieval_batch_size", 64),
            "local_dense_top_k": cfg.raw["compute"].get("local_dense_top_k", 80),
            "local_dense_max_items": cfg.raw["compute"]["local_dense_max_items"],
        },
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def item_index_exists(cfg: Config) -> bool:
    root = cfg.artifact_root
    required = [
        "index_manifest.json", "items_bm25/bm25", "items_char/matrix.npz", "items_dense/index.bin", "item_embeddings.npy",
    ]
    if not all((root / path).exists() for path in required):
        return False
    try:
        manifest = json.loads((root / "index_manifest.json").read_text(encoding="utf-8"))
        return manifest.get("signature") == _artifact_signature(cfg)
    except Exception:
        return False


def build_item_indexes(cfg: Config) -> None:
    root = ensure_dirs(cfg)
    runtime = configure_runtime(cfg.raw["compute"]["device"], cfg.raw["compute"]["cpu_threads"])
    print_runtime(runtime)
    items = load_items(cfg.items_path)
    item_rows = np.arange(len(items), dtype=np.int32)

    with Timer("Prepare lexical texts"), heartbeat("Prepare lexical texts"):
        lexical_texts = [lexical_text(t, p) for t, p in zip(items.item_title_raw, items.item_infm_params_text)]
        char_texts = [char_text(t, p) for t, p in zip(items.item_title_raw, items.item_infm_params_text)]

    index_cfg = cfg.raw["indexes"]
    BM25Index().build(
        lexical_texts, item_rows, root / "items_bm25",
        k1=float(index_cfg["bm25"]["k1"]), b=float(index_cfg["bm25"]["b"]),
    )
    del lexical_texts

    CharTfidfIndex().build(
        char_texts, item_rows, root / "items_char",
        analyzer=str(index_cfg["char_tfidf"]["analyzer"]),
        ngram_min=int(index_cfg["char_tfidf"]["ngram_min"]),
        ngram_max=int(index_cfg["char_tfidf"]["ngram_max"]),
        min_df=int(index_cfg["char_tfidf"]["min_df"]),
        max_features=int(index_cfg["char_tfidf"]["max_features"]),
    )
    del char_texts

    dense_texts = [
        dense_item_text(t, p, d, max_description_chars=2200)
        for t, p, d in zip(items.item_title_raw, items.item_infm_params_text, items.item_description_raw)
    ]
    encoder = make_encoder(cfg)
    embedding_path = root / "item_embeddings.npy"
    embeddings = encoder.encode(dense_texts, desc="Item dense embeddings", output_path=embedding_path)
    del dense_texts

    threads = cfg.raw["compute"]["hnsw_threads"]
    if threads == "auto":
        threads = runtime.cpu_threads
    HNSWIndex().build(
        embeddings, item_rows, root / "items_dense",
        M=int(index_cfg["dense"]["M"]),
        ef_construction=int(index_cfg["dense"]["ef_construction"]),
        ef_search=int(index_cfg["dense"]["ef_search"]),
        num_threads=int(threads), chunk_size=4096,
    )

    items.to_parquet(root / "item_metadata.parquet", index=False)
    manifest = {
        "item_count": int(len(items)),
        "embedding_model": cfg.raw["models"]["embedding_model"],
        "embedding_dimension": int(embeddings.shape[1]),
        "item_ids_sha256": _sha256_item_ids(items["item_id"]),
        "signature": _artifact_signature(cfg),
    }
    (root / "index_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def history_artifact_exists(cfg: Config, name: str, expected_query_count: int | None = None) -> bool:
    root = cfg.artifact_root / name
    required = [
        "query_hnsw/index.bin", "query_hnsw/meta.json", "indptr.npy", "item_rows.npy", "query_keys.npy", "queries.parquet",
    ]
    if not all((root / rel).exists() for rel in required):
        return False
    if expected_query_count is None:
        return True
    try:
        query_count = len(np.load(root / "query_keys.npy", allow_pickle=False))
        indptr_count = len(np.load(root / "indptr.npy", mmap_mode="r")) - 1
        return query_count == int(expected_query_count) and indptr_count == int(expected_query_count)
    except Exception:
        return False


def _query_embeddings_for_groups(encoder: E5Encoder, groups: pd.DataFrame) -> np.ndarray:
    from .text import dense_query_text
    texts = [
        dense_query_text(r.search_query, r.search_infm_params_text, r.search_is_delivery_search)
        for r in groups.itertuples(index=False)
    ]
    return encoder.encode(texts, desc="Ranker query embeddings")


def build_history_artifact(
    cfg: Config,
    grouped: pd.DataFrame,
    encoder: E5Encoder,
    item_id_to_row: dict[str, int],
    name: str,
    embeddings: np.ndarray | None = None,
) -> None:
    if history_artifact_exists(cfg, name):
        print(f"Reusing existing {name} artifact")
        return
    if embeddings is None:
        embeddings = _query_embeddings_for_groups(encoder, grouped)
    threads = cfg.raw["compute"]["hnsw_threads"]
    if threads == "auto":
        import os
        threads = max(1, int(round((os.cpu_count() or 8) * 0.8)))
    QueryHistory().build_from_embeddings(
        grouped,
        embeddings,
        item_id_to_row,
        cfg.artifact_root / name,
        top_items_per_query=int(cfg.raw["indexes"]["history"]["top_items_per_query"]),
        hnsw_threads=int(threads),
    )


def build_training_history(cfg: Config, train_pairs: pd.DataFrame, history_name: str) -> None:
    if history_artifact_exists(cfg, history_name):
        print(f"Reusing existing {history_name}")
        return
    items = load_items(cfg.items_path)
    filtered = filter_pairs_to_items(train_pairs, set(items.item_id.astype(str)))
    grouped = group_positive_items(filtered)
    encoder = make_encoder(cfg)
    item_map = {item_id: i for i, item_id in enumerate(items.item_id.astype(str))}
    build_history_artifact(cfg, grouped, encoder, item_map, history_name)


def build_term_expansion(cfg: Config, train_pairs: pd.DataFrame, name: str = "term_expansion") -> None:
    path = cfg.artifact_root / f"{name}.json"
    if path.exists():
        print(f"Reusing existing {name}.json")
        return
    items = load_items(cfg.items_path)
    expander = HistoricalTermExpansion.fit(train_pairs, items)
    expander.save(path)


def load_retriever(
    cfg: Config,
    items: pd.DataFrame | None = None,
    history_name: str = "history_all",
    term_expansion_name: str = "term_expansion",
    encoder: E5Encoder | None = None,
) -> HybridRetriever:
    root = ensure_dirs(cfg)
    if items is None:
        items = load_items(cfg.items_path)
    if encoder is None:
        encoder = make_encoder(cfg)
    return HybridRetriever.from_artifacts(items, encoder, root, cfg.raw, history_name, term_expansion_name)


def make_base_retriever(cfg: Config, items: pd.DataFrame, encoder: E5Encoder) -> HybridRetriever:
    root = cfg.artifact_root
    index_cfg = cfg.raw["indexes"]
    compute = cfg.raw["compute"]
    threads = compute["hnsw_threads"]
    if threads == "auto":
        import os
        threads = max(1, int(round((os.cpu_count() or 8) * 0.8)))
    return HybridRetriever(
        items=items,
        bm25=BM25Index().load(root / "items_bm25"),
        char_index=CharTfidfIndex().load(root / "items_char"),
        dense_index=HNSWIndex().load(root / "items_dense", ef_search=int(index_cfg["dense"]["ef_search"]), num_threads=int(threads)),
        history=None,
        encoder=encoder,
        item_embeddings=np.load(root / "item_embeddings.npy", mmap_mode="r"),
        term_expander=None,
        config=cfg.raw,
    )


def _pairs_for_groups(train_pairs: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    keys = set(groups.query_key.astype(str))
    tmp = train_pairs.copy()
    tmp["query_key"] = make_query_key(tmp)
    return tmp.loc[tmp.query_key.astype(str).isin(keys)].drop(columns=["query_key"]).reset_index(drop=True)


def _save_truths(groups: pd.DataFrame, path: Path) -> None:
    truth = {
        str(row.query_key): [str(x) for x in row.positive_item_ids]
        for row in groups.itertuples(index=False)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(truth, ensure_ascii=False), encoding="utf-8")


def load_cached_truths(path: str | Path) -> dict[str, set[str]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(key): set(map(str, values)) for key, values in raw.items()}


def _cached_table_is_ready(path: Path, signature: str) -> bool:
    meta = path.with_suffix(".json")
    if not path.exists() or not meta.exists():
        return False
    try:
        return json.loads(meta.read_text(encoding="utf-8")).get("signature") == signature
    except Exception:
        return False


def _make_direct_ranker_table(
    cfg: Config,
    query_groups: pd.DataFrame,
    retriever: HybridRetriever,
    feature_builder: FeatureBuilder,
    max_queries: int,
    max_candidates: int,
    seed: int,
    desc: str,
) -> RankerTableResult:
    return make_ranker_training_table(
        query_groups,
        retriever,
        feature_builder,
        max_queries=None if max_queries is None else int(max_queries),
        max_candidates_per_query=int(max_candidates),
        seed=int(seed),
        feature_workers=int(cfg.raw["compute"].get("feature_workers", 1)),
        desc=desc,
    )


def _build_tuning_validation_tables(
    cfg: Config,
    train_groups: pd.DataFrame,
    val_groups: pd.DataFrame,
    train_pairs: pd.DataFrame,
    items: pd.DataFrame,
    feature_builder: FeatureBuilder,
):
    signature = training_signature(cfg)
    cache_root = cfg.artifact_root / "cache"
    train_cache = cache_root / "ranker_train_table.parquet"
    val_cache = cache_root / "ranker_validation_table.parquet"
    val_truth_path = cache_root / "ranker_validation_truth.json"

    # The history artifact is safe because retrieve_many excludes the current exact query key.
    # Reuse the artifact produced by the interrupted run when its query count matches the split.
    fit_pairs = _pairs_for_groups(train_pairs, train_groups)
    fit_grouped = group_positive_items(fit_pairs)
    encoder = make_encoder(cfg)
    item_map = {item_id: i for i, item_id in enumerate(items.item_id.astype(str))}
    if not history_artifact_exists(cfg, "history_train", expected_query_count=len(fit_grouped)):
        build_history_artifact(cfg, fit_grouped, encoder, item_map, "history_train")

    validation_term_path = cfg.artifact_root / "term_expansion_train.json"
    if not validation_term_path.exists():
        HistoricalTermExpansion.fit(fit_pairs, items).save(validation_term_path)

    # Training and validation use the same train-split memory, but ranker-training lexical
    # expansion is fitted on a disjoint subset of train queries to prevent self-label expansion.
    ranker_term_path = cfg.artifact_root / "term_expansion_ranker_train.json"
    if not ranker_term_path.exists():
        term_fit_fraction = float(cfg.raw["training"].get("ranker_term_fit_fraction", 0.25))
        term_fit_groups = train_groups.sample(
            max(1, int(len(train_groups) * term_fit_fraction)),
            random_state=int(cfg.raw["training"]["split_seed"]) + 17,
        ).reset_index(drop=True)
        term_fit_pairs = _pairs_for_groups(train_pairs, term_fit_groups)
        HistoricalTermExpansion.fit(term_fit_pairs, items).save(ranker_term_path)

    training_cfg = cfg.raw["training"]
    train_max_queries = int(training_cfg["ranker_train_max_queries"])
    val_max_queries = int(training_cfg.get("validation_tune_max_queries", training_cfg.get("validation_max_queries", 1500)))
    max_train_candidates = int(training_cfg.get("max_candidates_per_query_train", training_cfg.get("max_candidates_per_query", 192)))
    max_val_candidates = int(training_cfg.get("max_candidates_per_query_val", training_cfg.get("max_candidates_per_query", 192)))

    train_retriever = load_retriever(
        cfg, items, history_name="history_train", term_expansion_name="term_expansion_ranker_train", encoder=encoder
    )
    val_retriever = load_retriever(
        cfg, items, history_name="history_train", term_expansion_name="term_expansion_train", encoder=encoder
    )

    train_table = load_ranker_table(train_cache, signature)
    if train_table is None:
        selected_train = train_groups
        train_table = _make_direct_ranker_table(
            cfg, selected_train, train_retriever, feature_builder,
            max_queries=train_max_queries,
            max_candidates=max_train_candidates,
            seed=int(training_cfg["negative_sampling_seed"]),
            desc="TRAIN ranker table",
        )
        attach_labels_and_save(train_table, train_cache, signature, "train")
    else:
        print(f"Reusing cached ranker train table: {train_cache}")

    val_table = load_ranker_table(val_cache, signature)
    if val_table is None:
        val_subset = val_groups.sample(
            min(len(val_groups), val_max_queries),
            random_state=int(training_cfg["split_seed"]),
        ).reset_index(drop=True)
        val_table = _make_direct_ranker_table(
            cfg, val_subset, val_retriever, feature_builder,
            max_queries=None,
            max_candidates=max_val_candidates,
            seed=int(training_cfg["negative_sampling_seed"]),
            desc="VALIDATION ranker table",
        )
        attach_labels_and_save(val_table, val_cache, signature, "validation")
        _save_truths(val_subset, val_truth_path)
    else:
        print(f"Reusing cached validation ranker table: {val_cache}")
        if not val_truth_path.exists():
            key_set = set(val_table.features["__query_key"].astype(str))
            _save_truths(val_groups[val_groups.query_key.astype(str).isin(key_set)], val_truth_path)
        val_subset = val_groups[val_groups.query_key.astype(str).isin(set(val_table.features["__query_key"].astype(str)))].reset_index(drop=True)

    return encoder, train_table, val_table, val_subset, signature


def save_predictions(query_ids: list[str], predictions: list[list[str]], path: str | Path) -> pd.DataFrame:
    if len(query_ids) != len(predictions):
        raise ValueError("query_ids and predictions lengths differ")
    cleaned = []
    for values in predictions:
        seen = set()
        row = []
        for item_id in values:
            item_id = str(item_id)
            if item_id not in seen:
                seen.add(item_id)
                row.append(item_id)
            if len(row) == 50:
                break
        cleaned.append(" ".join(row))
    answer = pd.DataFrame({"query_id": [str(x) for x in query_ids], "answer": cleaned})
    validate_answer_frame(answer)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    answer.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return answer


def validate_answer_frame(answer: pd.DataFrame, valid_item_ids: set[str] | None = None, expected_query_ids: set[str] | None = None) -> None:
    if list(answer.columns) != ["query_id", "answer"]:
        raise ValueError("answer.csv must contain exactly query_id,answer")
    if answer["query_id"].isna().any() or answer["query_id"].astype(str).str.len().ne(16).any():
        raise ValueError("Invalid query_id: every query_id must be a 16-character string")
    if answer["query_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate query_id")
    if expected_query_ids is not None and set(answer.query_id.astype(str)) != set(expected_query_ids):
        raise ValueError("answer.csv query_id set differs from benchmark_queries.parquet")
    item_re = __import__("re").compile(r"[0-9a-f]{16}\Z")
    for idx, value in enumerate(answer["answer"].astype(str)):
        ids = value.split()
        if len(ids) > 50:
            raise ValueError(f"Row {idx}: more than 50 item_id values")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Row {idx}: duplicate item_id")
        if any(item_re.fullmatch(item) is None for item in ids):
            raise ValueError(f"Row {idx}: invalid item_id")
        if valid_item_ids is not None and not set(ids).issubset(valid_item_ids):
            raise ValueError(f"Row {idx}: item_id absent from benchmark_items.parquet")


def validate_answer_file(cfg: Config, path: str | Path) -> dict[str, Any]:
    answer = pd.read_csv(path, dtype=str, keep_default_na=False)
    queries = load_benchmark_queries(cfg.queries_path)
    items = load_items(cfg.items_path)
    validate_answer_frame(answer, set(items.item_id.astype(str)), set(queries.query_id.astype(str)))
    result = {
        "valid": True,
        "rows": int(len(answer)),
        "benchmark_queries": int(len(queries)),
        "nonempty_answers": int((answer["answer"].str.len() > 0).sum()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
