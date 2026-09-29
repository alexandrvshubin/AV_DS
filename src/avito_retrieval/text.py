from __future__ import annotations

import re
from functools import lru_cache

RU_STOPWORDS = {
    "а", "без", "бы", "был", "быть", "в", "во", "вот", "всего", "вы", "да", "для",
    "до", "же", "за", "и", "из", "или", "к", "как", "ко", "ли", "мне", "мы", "на",
    "над", "не", "него", "нет", "ни", "но", "ну", "о", "об", "от", "по", "под", "при",
    "про", "с", "со", "так", "там", "то", "тоже", "только", "у", "уже", "хотя", "что",
    "чтобы", "это", "я",
}

TOKEN_RE = re.compile(r"[\w+#.-]+", flags=re.UNICODE)
SPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+7|8)[\s()\-\d]{8,}")


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.lower().replace("ё", "е")
    text = URL_RE.sub(" ", text)
    text = PHONE_RE.sub(" phone ", text)
    text = text.replace("№", " номер ")
    text = text.replace("–", "-").replace("—", "-")
    text = SPACE_RE.sub(" ", text).strip()
    return text


def lexical_tokens(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = [x for x in TOKEN_RE.findall(normalize_text(text)) if x]
    if remove_stopwords:
        tokens = [x for x in tokens if x not in RU_STOPWORDS or len(x) <= 2]
    return tokens


def lexical_text(title: object, params: object) -> str:
    title_text = normalize_text(title)
    params_text = normalize_text(params)
    # BM25 field weighting: title is repeated because it is a compact high-signal field.
    return f"{title_text} {title_text} {params_text}".strip()


def char_text(title: object, params: object) -> str:
    return f"{normalize_text(title)} {normalize_text(params)}".strip()


def dense_item_text(title: object, params: object, description: object, max_description_chars: int = 2200) -> str:
    title_text = normalize_text(title)
    params_text = normalize_text(params)
    description_text = normalize_text(description)[:max_description_chars]
    return f"passage: {title_text}. {params_text}. {description_text}".strip()


def dense_query_text(query: object, params: object, delivery: object = 0) -> str:
    query_text = normalize_text(query)
    params_text = normalize_text(params)
    delivery_text = " поиск с доставкой" if int(delivery or 0) else ""
    return f"query: {query_text}. {params_text}.{delivery_text}".strip()


def compact_query_text(query: object, params: object) -> str:
    return f"{normalize_text(query)} {normalize_text(params)}".strip()


class RussianLemmatizer:
    def __init__(self):
        try:
            import pymorphy3
            self._morph = pymorphy3.MorphAnalyzer(lang="ru")
            self.available = True
        except Exception:
            self._morph = None
            self.available = False

    @lru_cache(maxsize=200_000)
    def lemma(self, token: str) -> str:
        if not self.available or not token.isalpha() or len(token) < 3:
            return token
        try:
            return self._morph.parse(token)[0].normal_form
        except Exception:
            return token

    def enrich_query(self, text: str) -> str:
        tokens = lexical_tokens(text)
        additions = [self.lemma(token) for token in tokens if token != self.lemma(token)]
        additions = list(dict.fromkeys(additions))
        return f"{text} {' '.join(additions)}".strip()


_LEMMATIZER = None


def enrich_query_with_morphology(text: str) -> str:
    global _LEMMATIZER
    if _LEMMATIZER is None:
        _LEMMATIZER = RussianLemmatizer()
    return _LEMMATIZER.enrich_query(text)
