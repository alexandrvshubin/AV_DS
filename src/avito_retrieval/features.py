from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .text import lexical_tokens, normalize_text

PRICE_RE = re.compile(r"(?:цена|стоимость)[^0-9]{0,30}(?:от\s*)?(\d+(?:[.,]\d+)?)?(?:[^0-9]{0,15}до\s*(\d+(?:[.,]\d+)?))?", re.IGNORECASE)
RATING_RE = re.compile(r"рейтинг[^0-9]{0,30}(\d(?:[.,]\d)?)", re.IGNORECASE)


FEATURE_NAMES = [
    "bm25_score", "bm25_rank", "char_score", "char_rank", "dense_score", "dense_rank",
    "local_dense_score", "local_dense_rank", "history_score", "history_rank", "rrf_score",
    "query_token_coverage", "title_token_coverage", "param_token_coverage", "token_jaccard",
    "title_fuzzy_ratio", "title_fuzzy_partial", "title_token_set_ratio", "query_in_title_exact",
    "query_any_token_in_title", "category_match", "location_match", "rating", "log_reviews",
    "quality_score", "rating_pass", "price_log", "price_pass", "contactable", "phone_available",
    "message_available", "title_length_log", "description_length_log", "params_length_log",
]


def parse_rating_min(params: str) -> float:
    match = RATING_RE.search(normalize_text(params))
    if not match:
        return np.nan
    return float(match.group(1).replace(",", "."))


def parse_price_bounds(params: str) -> tuple[float, float]:
    text = normalize_text(params)
    match = PRICE_RE.search(text)
    if not match:
        return np.nan, np.nan
    lower = float(match.group(1).replace(",", ".")) if match.group(1) else np.nan
    upper = float(match.group(2).replace(",", ".")) if match.group(2) else np.nan
    return lower, upper


def _log1p_safe(value: float) -> float:
    return math.log1p(max(0.0, float(value)))


class FeatureBuilder:
    """Fast feature builder for small candidate sets.

    The item dataframe is converted to compact numpy arrays once. Fuzzy string scores
    are calculated with RapidFuzz's C++-backed cdist instead of three Python calls per row.
    Token sets are cached lazily to keep memory bounded on 24 GB Apple Silicon machines.
    """

    def __init__(self, items: pd.DataFrame, fuzzy: bool = True):
        self.items = items.reset_index(drop=True)
        self.fuzzy = bool(fuzzy)
        self.item_id = self.items["item_id"].astype(str).to_numpy()
        self.title = self.items["item_title_raw"].fillna("").astype(str).map(normalize_text).to_numpy(dtype=object)
        self.params = self.items["item_infm_params_text"].fillna("").astype(str).map(normalize_text).to_numpy(dtype=object)
        self.description = self.items["item_description_raw"].fillna("").astype(str).map(normalize_text).to_numpy(dtype=object)
        self.location = self.items["item_location_id"].fillna(-1).to_numpy(dtype=np.int64)
        self.category = self.items["item_category_id"].fillna(-1).to_numpy(dtype=np.int64)
        self.rating = pd.to_numeric(self.items["item_rating"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        self.reviews = pd.to_numeric(self.items["item_rating_reviews_count"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        self.price = pd.to_numeric(self.items["item_price"], errors="coerce").to_numpy(dtype=np.float32)
        self.phone_available = (~self.items["item_is_phone_hidden"].fillna(False)).to_numpy(dtype=np.float32)
        self.message_available = (~self.items["item_is_message_forbidden"].fillna(False)).to_numpy(dtype=np.float32)
        self.title_lengths = np.fromiter((len(x) for x in self.title), dtype=np.int32, count=len(self.title))
        self.description_lengths = np.fromiter((len(x) for x in self.description), dtype=np.int32, count=len(self.description))
        self.params_lengths = np.fromiter((len(x) for x in self.params), dtype=np.int32, count=len(self.params))

    @lru_cache(maxsize=300_000)
    def _tokens(self, item_row: int) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
        title_tokens = frozenset(lexical_tokens(self.title[item_row]))
        param_tokens = frozenset(lexical_tokens(self.params[item_row]))
        item_tokens = title_tokens | param_tokens
        return title_tokens, param_tokens, item_tokens

    @staticmethod
    def _fuzzy_scores(query_text: str, titles: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not query_text or not titles:
            zeros = np.zeros(len(titles), dtype=np.float32)
            return zeros, zeros.copy(), zeros.copy()
        ratio = np.asarray(
            process.cdist([query_text], titles, scorer=fuzz.ratio, dtype="float32", workers=1)[0],
            dtype=np.float32,
        ) / 100.0
        partial = np.asarray(
            process.cdist([query_text], titles, scorer=fuzz.partial_ratio, dtype="float32", workers=1)[0],
            dtype=np.float32,
        ) / 100.0
        token_set = np.asarray(
            process.cdist([query_text], titles, scorer=fuzz.token_set_ratio, dtype="float32", workers=1)[0],
            dtype=np.float32,
        ) / 100.0
        return ratio, partial, token_set

    def build(self, query_row: pd.Series | dict[str, Any], candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            return pd.DataFrame(columns=["item_row", "item_id", *FEATURE_NAMES])

        q = dict(query_row)
        qtext = normalize_text(q.get("search_query", ""))
        qparams = normalize_text(q.get("search_infm_params_text", ""))
        query_tokens = frozenset(lexical_tokens(f"{qtext} {qparams}"))
        title_query_tokens = frozenset(lexical_tokens(qtext))
        filter_tokens = frozenset(lexical_tokens(qparams))
        rating_min = parse_rating_min(qparams)
        price_low, price_high = parse_price_bounds(qparams)
        query_category = int(q.get("search_category", -1))
        query_location = int(q.get("search_location_id", -1))

        row_ids = candidates["item_row"].to_numpy(dtype=np.int32, copy=False)
        titles = [self.title[i] for i in row_ids]
        fuzzy_ratio, fuzzy_partial, fuzzy_set = self._fuzzy_scores(qtext, titles) if self.fuzzy else (
            np.zeros(len(row_ids), dtype=np.float32),
            np.zeros(len(row_ids), dtype=np.float32),
            np.zeros(len(row_ids), dtype=np.float32),
        )

        rows = []
        for j, candidate in enumerate(candidates.itertuples(index=False)):
            row_idx = int(row_ids[j])
            title_tokens, param_tokens, item_tokens = self._tokens(row_idx)
            token_intersection = len(query_tokens & item_tokens)
            coverage = token_intersection / max(1, len(query_tokens))
            title_token_coverage = len(title_query_tokens & title_tokens) / max(1, len(title_query_tokens))
            param_coverage = len(filter_tokens & param_tokens) / max(1, len(filter_tokens)) if filter_tokens else 0.0
            union_size = len(query_tokens) + len(item_tokens) - token_intersection
            jaccard = token_intersection / max(1, union_size)

            rating = float(self.rating[row_idx])
            reviews = float(self.reviews[row_idx])
            price = float(self.price[row_idx])
            category_match = float(query_category >= 0 and query_category == self.category[row_idx])
            location_match = float(query_location >= 0 and query_location == self.location[row_idx])
            rating_pass = 1.0 if np.isnan(rating_min) else float(rating + 1e-6 >= rating_min)
            has_price_filter = not np.isnan(price_low) or not np.isnan(price_high)
            if not has_price_filter:
                price_pass = 1.0
            elif np.isnan(price):
                price_pass = 0.0
            else:
                price_pass = float(
                    (np.isnan(price_low) or price >= price_low)
                    and (np.isnan(price_high) or price <= price_high)
                )

            rows.append({
                "item_row": row_idx,
                "item_id": str(self.item_id[row_idx]),
                "bm25_score": float(getattr(candidate, "bm25_score", 0.0)),
                "bm25_rank": float(getattr(candidate, "bm25_rank", 9999.0)),
                "char_score": float(getattr(candidate, "char_score", 0.0)),
                "char_rank": float(getattr(candidate, "char_rank", 9999.0)),
                "dense_score": float(getattr(candidate, "dense_score", 0.0)),
                "dense_rank": float(getattr(candidate, "dense_rank", 9999.0)),
                "local_dense_score": float(getattr(candidate, "local_dense_score", 0.0)),
                "local_dense_rank": float(getattr(candidate, "local_dense_rank", 9999.0)),
                "history_score": float(getattr(candidate, "history_score", 0.0)),
                "history_rank": float(getattr(candidate, "history_rank", 9999.0)),
                "rrf_score": float(getattr(candidate, "rrf_score", 0.0)),
                "query_token_coverage": coverage,
                "title_token_coverage": title_token_coverage,
                "param_token_coverage": param_coverage,
                "token_jaccard": jaccard,
                "title_fuzzy_ratio": float(fuzzy_ratio[j]),
                "title_fuzzy_partial": float(fuzzy_partial[j]),
                "title_token_set_ratio": float(fuzzy_set[j]),
                "query_in_title_exact": float(bool(title_query_tokens) and title_query_tokens.issubset(title_tokens)),
                "query_any_token_in_title": float(bool(title_query_tokens & title_tokens)),
                "category_match": category_match,
                "location_match": location_match,
                "rating": rating,
                "log_reviews": _log1p_safe(reviews),
                "quality_score": (rating / 5.0) * math.log1p(reviews) if rating > 0 else 0.0,
                "rating_pass": rating_pass,
                "price_log": _log1p_safe(price) if not np.isnan(price) else 0.0,
                "price_pass": price_pass,
                "contactable": float(self.phone_available[row_idx] > 0 or self.message_available[row_idx] > 0),
                "phone_available": float(self.phone_available[row_idx]),
                "message_available": float(self.message_available[row_idx]),
                "title_length_log": _log1p_safe(self.title_lengths[row_idx]),
                "description_length_log": _log1p_safe(self.description_lengths[row_idx]),
                "params_length_log": _log1p_safe(self.params_lengths[row_idx]),
            })
        return pd.DataFrame(rows)
