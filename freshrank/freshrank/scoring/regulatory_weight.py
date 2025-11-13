"""Helpers to load regulatory tags and compute weighting adjustments."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


@dataclass
class RegulatoryHit:
    tag: str
    confidence: float
    evidence: List[str]
    source: Optional[str] = None


TagIndex = Dict[Tuple[str, Optional[str]], List[RegulatoryHit]]


def load_tag_index(path: Path) -> TagIndex:
    index: TagIndex = {}
    for candidate in _iter_tag_files(path):
        with candidate.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = (record.get("doc_id"), record.get("chunk_id"))
                hits = _extract_hits(record, source=candidate.stem)
                if not hits:
                    continue
                index.setdefault(key, []).extend(hits)
    return index


def _iter_tag_files(path: Path) -> Iterable[Path]:
    if path.is_dir():
        for candidate in sorted(path.glob("*tag*.jsonl")):
            if candidate.is_file():
                yield candidate
        return
    if path.exists():
        yield path
    for candidate in sorted(path.parent.glob("*tag*.jsonl")):
        if candidate != path and candidate.is_file():
            yield candidate


def _extract_hits(record: dict, *, source: str) -> List[RegulatoryHit]:
    hits: List[RegulatoryHit] = []
    details = record.get("tag_details")
    if isinstance(details, list) and details:
        for item in details:
            if not isinstance(item, dict):
                continue
            tag = item.get("name") or item.get("tag")
            if not tag:
                continue
            hits.append(
                RegulatoryHit(
                    tag=tag,
                    confidence=float(item.get("confidence", record.get("confidence", 0.5))),
                    evidence=item.get("evidence") or record.get("evidence") or [],
                    source=item.get("source") or record.get("doc_type") or source,
                )
            )
        return hits
    tags = record.get("tags") or []
    evidence = record.get("evidence") or []
    confidence = float(record.get("confidence", 0.5))
    for tag in tags:
        if isinstance(tag, str):
            hits.append(
                RegulatoryHit(
                    tag=tag,
                    confidence=confidence,
                    evidence=evidence,
                    source=record.get("doc_type") or source,
                )
            )
    return hits


def lookup_hits(index: TagIndex, doc_id: str, chunk_id: Optional[str]) -> List[RegulatoryHit]:
    if not index:
        return []
    if (doc_id, chunk_id) in index:
        return index[(doc_id, chunk_id)]
    if chunk_id is not None and (doc_id, None) in index:
        return index[(doc_id, None)]
    return index.get((doc_id, chunk_id), [])


def compute_adjustment(
    hits: List[RegulatoryHit],
    weight_config: dict,
) -> dict:
    if not hits:
        return {"multiplier": 1.0, "bonus": 0.0, "tags": [], "evidence": [], "w_regulatory": 0.0}

    reg_cfg = weight_config.get("regulatory", {})
    tag_weights = dict(reg_cfg.get("tag_weights", {}))
    if not tag_weights:
        tag_weights = _fallback_tag_weights(reg_cfg)

    summary: Dict[str, Dict[str, Any]] = {}
    for hit in hits:
        entry = summary.setdefault(
            hit.tag,
            {"confidence": 0.0, "evidence": [], "sources": []},
        )
        if hit.confidence > entry["confidence"]:
            entry["confidence"] = hit.confidence
        for snippet in hit.evidence[:2]:
            if snippet and snippet not in entry["evidence"]:
                entry["evidence"].append(snippet)
        if hit.source and hit.source not in entry["sources"]:
            entry["sources"].append(hit.source)

    multiplier = 1.0
    bonus = 0.0
    evidence: List[str] = []
    tags: List[str] = []

    max_bonus = reg_cfg.get("max_bonus")
    min_bonus = reg_cfg.get("min_bonus")

    for tag, info in summary.items():
        tags.append(tag)
        evidence.extend(info.get("evidence", [])[:2])
        weights = tag_weights.get(tag, {})
        conf = float(info.get("confidence", reg_cfg.get("default_confidence", 0.6)))
        min_conf = float(weights.get("min_confidence", 0.0))
        if conf < min_conf:
            continue
        if "multiplier" in weights:
            base_multiplier = float(weights["multiplier"])
            multiplier *= 1.0 + (base_multiplier - 1.0) * conf
        if "bonus" in weights:
            bonus += float(weights["bonus"]) * conf

    if max_bonus is not None:
        bonus = min(bonus, float(max_bonus))
    if min_bonus is not None:
        bonus = max(bonus, float(min_bonus))

    w_regulatory = (multiplier - 1.0) + bonus
    return {
        "multiplier": multiplier,
        "bonus": bonus,
        "tags": sorted(set(tags)),
        "evidence": evidence,
        "w_regulatory": w_regulatory,
    }


def ensure_hits(candidates: Iterable[Any]) -> List[RegulatoryHit]:
    """Coerce user-provided payloads into `RegulatoryHit` objects."""

    hits: List[RegulatoryHit] = []
    for item in candidates or []:
        if isinstance(item, RegulatoryHit):
            hits.append(item)
            continue
        if isinstance(item, dict):
            tag = item.get("tag") or item.get("name")
            if not tag:
                continue
            hits.append(
                RegulatoryHit(
                    tag=tag,
                    confidence=float(item.get("confidence", item.get("score", 0.7) or 0.7)),
                    evidence=list(item.get("evidence", []) or []),
                    source=item.get("source"),
                )
            )
            continue
        if isinstance(item, str):
            hits.append(RegulatoryHit(tag=item, confidence=0.8, evidence=[]))
    return hits


def _fallback_tag_weights(reg_cfg: dict) -> Dict[str, Dict[str, float]]:
    weights: Dict[str, Dict[str, float]] = {}
    esg_multiplier = float(reg_cfg.get("esg_multiplier", 1.0))
    if esg_multiplier > 1.0:
        weights.setdefault("esg", {})["multiplier"] = esg_multiplier
    rating_bonus = float(reg_cfg.get("rating_bonus", 0.0))
    if rating_bonus:
        weights.setdefault("rating", {})["bonus"] = rating_bonus
        weights.setdefault("consumer_protection", {})["bonus"] = rating_bonus
        weights.setdefault("sales_compliance", {})["bonus"] = rating_bonus
    return weights
