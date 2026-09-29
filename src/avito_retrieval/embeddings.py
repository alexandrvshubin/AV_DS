from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from .progress import Timer


class E5Encoder:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 256,
        truncate_dim: int | None = None,
        cache_dir: str | Path | None = None,
    ):
        if self_cache := cache_dir:
            self_cache = Path(self_cache).expanduser()
            self_cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(self_cache))
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(self_cache / "hub"))
            os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(self_cache / "sentence-transformers"))

        from sentence_transformers import SentenceTransformer

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.truncate_dim = truncate_dim
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        print(f"Loading embedding model: {model_name}")
        print(f"Embedding device={device}; batch_size={self.batch_size}; max_length={self.max_length}")
        kwargs = {"device": device}
        if self.cache_dir:
            kwargs["cache_folder"] = str(self.cache_dir / "sentence-transformers")
        self.model = SentenceTransformer(model_name, **kwargs)
        self.model.eval()
        self.model.max_seq_length = self.max_length
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], desc: str = "Embedding", output_path: str | Path | None = None) -> np.ndarray:
        if output_path is None:
            with Timer(desc):
                result = self.model.encode(
                    list(texts),
                    batch_size=self.batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    truncate_dim=self.truncate_dim,
                )
            self._clear_device_cache()
            return np.asarray(result, dtype=np.float32)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dimension = int(self.truncate_dim or self.dimension)
        mmap = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(len(texts), dimension))
        for start in tqdm(
            range(0, len(texts), self.batch_size),
            desc=desc,
            unit="batch",
            dynamic_ncols=True,
            mininterval=1.5,
        ):
            batch = texts[start:start + self.batch_size]
            current = self.model.encode(
                list(batch),
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
                truncate_dim=self.truncate_dim,
            )
            mmap[start:start + len(batch)] = np.asarray(current, dtype=np.float32)
        mmap.flush()
        del mmap
        self._clear_device_cache()
        return np.load(output_path, mmap_mode="r")

    @staticmethod
    def _clear_device_cache() -> None:
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
