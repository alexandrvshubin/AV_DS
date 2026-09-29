from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .text import normalize_text

SEARCH_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

ITEM_COLUMNS = [
    "item_title_raw",
    "item_rating_reviews_count",
    "item_rating",
    "item_price",
    "item_microcat_id",
    "item_longitude",
    "item_location_id",
    "item_latitude",
    "item_is_phone_hidden",
    "item_is_message_forbidden",
    "item_infm_params_text",
    "item_id",
    "item_description_raw",
    "item_category_id",
]


def _safe_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str)


def _to_float32(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")


def load_items(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=ITEM_COLUMNS)
    for col in ["item_title_raw", "item_infm_params_text", "item_id", "item_description_raw"]:
        df[col] = _safe_text(df[col])
    _to_float32(df, ["item_price", "item_longitude", "item_latitude", "item_rating_reviews_count", "item_rating"])
    df["item_id"] = df["item_id"].astype(str)
    if df["item_id"].duplicated().any():
        raise ValueError("benchmark_items.parquet contains duplicate item_id values")
    bad = ~df["item_id"].str.fullmatch(r"[0-9a-f]{16}", na=False)
    if bad.any():
        raise ValueError(f"Invalid item_id values: {df.loc[bad, 'item_id'].head(10).tolist()}")
    df.reset_index(drop=True, inplace=True)
    return df


def load_train_pairs(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=SEARCH_COLUMNS + ["item_id"])
    df["search_query"] = _safe_text(df["search_query"])
    df["search_infm_params_text"] = _safe_text(df["search_infm_params_text"])
    df["item_id"] = _safe_text(df["item_id"])
    return df


def load_benchmark_queries(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["query_id"] + SEARCH_COLUMNS)
    df["query_id"] = _safe_text(df["query_id"])
    df["search_query"] = _safe_text(df["search_query"])
    df["search_infm_params_text"] = _safe_text(df["search_infm_params_text"])
    bad = df["query_id"].str.len().ne(16)
    if bad.any():
        raise ValueError(f"Invalid benchmark query_id length: {df.loc[bad, 'query_id'].head(10).tolist()}")
    if df["query_id"].duplicated().any():
        raise ValueError("benchmark_queries.parquet contains duplicate query_id values")
    return df


def make_query_key(frame: pd.DataFrame) -> pd.Series:
    query = frame["search_query"].fillna("").astype(str).map(normalize_text)
    location = frame["search_location_id"].fillna(-1).astype(str)
    delivery = frame["search_is_delivery_search"].fillna(0).astype(str)
    params = frame["search_infm_params_text"].fillna("").astype(str).map(normalize_text)
    category = frame["search_category"].fillna(-1).astype(str)
    raw = query + "||" + location + "||" + delivery + "||" + params + "||" + category
    return raw.map(lambda x: hashlib.blake2b(x.encode("utf-8"), digest_size=16).hexdigest())


def make_split_key(frame: pd.DataFrame) -> pd.Series:
    # Split by normalized query text, not by query rows. Therefore the same textual query
    # cannot occur in both train and validation even when location/filters differ.
    return frame["search_query"].fillna("").astype(str).map(normalize_text).replace("", "__EMPTY_QUERY__")


def add_query_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["query_key"] = make_query_key(out)
    out["split_key"] = make_split_key(out)
    return out


def group_positive_items(train_pairs: pd.DataFrame) -> pd.DataFrame:
    tmp = add_query_keys(train_pairs)
    result = (
        tmp.groupby("query_key", sort=False, observed=True)
        .agg(
            split_key=("split_key", "first"),
            search_query=("search_query", "first"),
            search_location_id=("search_location_id", "first"),
            search_is_delivery_search=("search_is_delivery_search", "first"),
            search_infm_params_text=("search_infm_params_text", "first"),
            search_category=("search_category", "first"),
            positive_item_ids=("item_id", lambda x: pd.unique(x.astype(str)).tolist()),
        )
        .reset_index()
    )
    result["positive_count"] = result["positive_item_ids"].str.len().astype(np.int32)
    return result


def split_query_groups(groups: pd.DataFrame, fit_fraction: float = 0.8, seed: int = 42):
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must be between 0 and 1")
    unique_keys = pd.Series(groups["split_key"].astype(str).unique())
    rng = np.random.default_rng(seed)
    order = np.arange(len(unique_keys))
    rng.shuffle(order)
    cut = int(len(order) * fit_fraction)
    train_keys = set(unique_keys.iloc[order[:cut]].tolist())
    mask = groups["split_key"].astype(str).isin(train_keys)
    train = groups[mask].reset_index(drop=True)
    val = groups[~mask].reset_index(drop=True)
    if train.empty or val.empty:
        raise ValueError("Train/validation split produced an empty split")
    return train, val


def filter_pairs_to_items(train_pairs: pd.DataFrame, valid_item_ids: set[str]) -> pd.DataFrame:
    mask = train_pairs["item_id"].astype(str).isin(valid_item_ids)
    return train_pairs.loc[mask].reset_index(drop=True)


def validate_dataset_schema(train_path: Path, items_path: Path, queries_path: Path) -> dict:
    import pyarrow.parquet as pq

    required_train = set(SEARCH_COLUMNS + ["item_id"])
    required_items = set(ITEM_COLUMNS)
    required_queries = set(["query_id"] + SEARCH_COLUMNS)

    schemas = {
        "train": pq.read_schema(train_path),
        "items": pq.read_schema(items_path),
        "queries": pq.read_schema(queries_path),
    }
    for name, required in [("train", required_train), ("items", required_items), ("queries", required_queries)]:
        actual = set(schemas[name].names)
        missing = sorted(required - actual)
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
    return {
        "train_columns": schemas["train"].names,
        "items_columns": schemas["items"].names,
        "queries_columns": schemas["queries"].names,
    }
