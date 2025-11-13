"""Vector search backend using sentence-transformers + hnswlib."""
from __future__ import annotations

import ast
import logging
import os
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:  # hnswlib is optional in offline environments
    import hnswlib  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    hnswlib = None

from ..paths import data_path, env_or_path

INDEX_PATH = env_or_path("VEC_INDEX", data_path("index", "vec.index"))
META_PATH = env_or_path("VEC_META", data_path("index", "vec.meta.csv"))
MODEL_NAME = os.getenv("EMB_MODEL", "BAAI/bge-small-zh-v1.5")

LOGGER = logging.getLogger(__name__)
_FALLBACK_WARNED = False


@lru_cache()
def _load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@lru_cache()
def _load_index() -> hnswlib.Index:
    if hnswlib is None:  # pragma: no cover - exercised in offline envs
        raise RuntimeError("hnswlib is not available; use fallback search")
    model = _load_model()
    dim = model.get_sentence_embedding_dimension()
    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(str(INDEX_PATH))
    index.set_ef(64)
    return index


@lru_cache()
def _load_meta() -> pd.DataFrame:
    return pd.read_csv(META_PATH)


def search(query_texts: List[str], topk: int = 20) -> List[Dict]:
    if not query_texts:
        return []
    if hnswlib is None:
        return _fallback_search(query_texts, topk)

    model = _load_model()
    index = _load_index()
    meta = _load_meta()

    embeddings = model.encode(query_texts, normalize_embeddings=True)
    labels, distances = index.knn_query(embeddings, k=topk)

    hits: Dict[str, Dict] = {}
    for q_idx, _ in enumerate(query_texts):
        for label, dist in zip(labels[q_idx], distances[q_idx]):
            if label < 0:
                continue
            row = meta.iloc[int(label)]
            _upsert_hit(hits, row, 1.0 - float(dist))
    return list(hits.values())


@lru_cache()
def _meta_embeddings() -> Tuple[pd.DataFrame, np.ndarray]:  # pragma: no cover - heavy init
    meta = _load_meta()
    model = _load_model()
    texts = meta["text"].fillna("").astype(str).tolist()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return meta, np.asarray(embeddings)


def _fallback_search(query_texts: List[str], topk: int) -> List[Dict]:
    global _FALLBACK_WARNED
    if not _FALLBACK_WARNED:
        LOGGER.warning("hnswlib unavailable; using in-memory cosine search fallback")
        _FALLBACK_WARNED = True
    meta, embeddings = _meta_embeddings()
    model = _load_model()
    query_embs = model.encode(query_texts, normalize_embeddings=True)
    hits: Dict[str, Dict] = {}
    for q_idx, q_emb in enumerate(np.asarray(query_embs)):
        sims = np.dot(embeddings, q_emb)
        if sims.ndim == 1:
            sims = sims.reshape(-1)
        top_n = min(len(meta), max(topk, 1) * 3)
        best_idx = np.argsort(-sims)[:top_n]
        for idx in best_idx:
            row = meta.iloc[int(idx)]
            _upsert_hit(hits, row, float(sims[idx]))
    ranked = sorted(hits.values(), key=lambda item: item.get("score_semantic", 0.0), reverse=True)
    return ranked[:topk]


def _upsert_hit(hits: Dict[str, Dict], row: pd.Series, score: float) -> None:
    chunk_id = row["chunk_id"]
    if chunk_id in hits and score <= hits[chunk_id]["score_semantic"]:
        return
    bbox = row.get("bbox")
    if isinstance(bbox, str):
        try:
            bbox = ast.literal_eval(bbox)
        except Exception:  # pragma: no cover - defensive
            bbox = []
    hits[chunk_id] = {
        "chunk_id": chunk_id,
        "doc_id": row.get("doc_id"),
        "page_no": int(row.get("page_no", 0) or 0),
        "bbox": bbox,
        "text": str(row.get("text", ""))[:240],
        "type": row.get("type", ""),
        "score_semantic": float(score),
    }
