from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .data import group_positive_items
from .progress import progress
from .text import lexical_tokens


class HistoricalTermExpansion:
    """Data-driven lexical expansion learned from query -> selected-item title associations."""

    def __init__(self, associations: dict[str, list[str]] | None = None):
        self.associations = associations or {}

    @classmethod
    def fit(
        cls,
        train_pairs: pd.DataFrame,
        item_lookup: pd.DataFrame,
        min_support: int = 3,
        top_terms: int = 5,
    ):
        item_title = dict(zip(item_lookup["item_id"].astype(str), item_lookup["item_title_raw"].astype(str)))
        grouped = group_positive_items(train_pairs)
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        query_support = Counter()
        term_support = Counter()

        for row in progress(grouped.itertuples(index=False), total=len(grouped), desc="Learn query terms", unit="query"):
            q_tokens = set(lexical_tokens(f"{row.search_query} {row.search_infm_params_text}"))
            if not q_tokens:
                continue
            clicked_tokens = set()
            for item_id in row.positive_item_ids[:20]:
                clicked_tokens.update(lexical_tokens(item_title.get(str(item_id), "")))
            clicked_tokens = {x for x in clicked_tokens if len(x) >= 3}
            for q_token in q_tokens:
                query_support[q_token] += 1
                for term in clicked_tokens:
                    if term != q_token:
                        counts[q_token][term] += 1
                        term_support[term] += 1

        total_queries = max(1, len(grouped))
        associations: dict[str, list[str]] = {}
        for q_token, counter in counts.items():
            q_support = max(1, query_support[q_token])
            scored = []
            for term, support in counter.items():
                if support < int(min_support):
                    continue
                t_support = max(1, term_support[term])
                pmi = (support * total_queries) / max(1.0, q_support * t_support)
                score = support * (1.0 + max(0.0, math.log(max(pmi, 1.0))))
                scored.append((score, term))
            scored.sort(key=lambda x: (-x[0], x[1]))
            associations[q_token] = [term for _, term in scored[: int(top_terms)]]
        return cls(associations)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.associations, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def expand(self, query: str) -> str:
        tokens = lexical_tokens(query)
        additions = []
        for token in tokens:
            additions.extend(self.associations.get(token, []))
        additions = list(dict.fromkeys(additions))
        return f"{query} {' '.join(additions)}".strip()
