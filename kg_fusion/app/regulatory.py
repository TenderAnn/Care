"""Regulatory metadata loaders and scoring utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Iterator, List, Optional

import yaml

from .. import PACKAGE_ROOT
from ..paths import env_or_path

_REG_TAGS_PATH = env_or_path(
    "REG_TAGS",
    PACKAGE_ROOT.parent / "freshrank" / "data" / "metadata" / "esg_tags.jsonl",
)
_RULEBOOK_PATH = env_or_path(
    "REG_RULEBOOK",
    PACKAGE_ROOT.parent / "freshrank" / "regulatory" / "weights.yaml",
)

_POSITIVE_MULTIPLIER_TAGS = {"esg", "disclosure_gov", "green_finance"}
_RATING_TAGS = {"rating", "consumer_protection", "sales_compliance"}


@dataclass
class RegulatoryScore:
    bonus: float
    multiplier: float
    tags: List[str]
    confidence: float
    contributions: List[Dict[str, Any]]

    def to_debug(self) -> Dict[str, Any]:
        return {
            "bonus": round(self.bonus, 4),
            "multiplier": round(self.multiplier, 4),
            "tags": self.tags,
            "confidence": round(self.confidence, 4),
            "contributions": self.contributions,
        }


@lru_cache()
def _load_rulebook() -> Dict[str, Any]:
    path = Path(_RULEBOOK_PATH)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@lru_cache()
def _load_regulatory_meta() -> Dict[str, Dict[str, Any]]:
    payload: Dict[str, Dict[str, Any]] = {}
    for path in _iter_tag_paths(Path(_REG_TAGS_PATH)):
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                doc_id = str(record.get("doc_id") or "").strip()
                if not doc_id:
                    continue
                entry = payload.setdefault(doc_id, {"tags": {}})
                for detail in _normalise_details(record, source=path.stem):
                    tag_name = detail.get("tag")
                    if not tag_name:
                        continue
                    tag_entry = entry["tags"].setdefault(
                        tag_name,
                        {"confidence": 0.0, "evidence": [], "sources": []},
                    )
                    conf = float(detail.get("confidence", record.get("confidence", 0.5)))
                    if conf > tag_entry["confidence"]:
                        tag_entry["confidence"] = conf
                    for snippet in detail.get("evidence", [])[:3]:
                        if snippet and snippet not in tag_entry["evidence"]:
                            tag_entry["evidence"].append(snippet)
                    source = detail.get("source") or record.get("doc_type") or path.stem
                    if source and source not in tag_entry["sources"]:
                        tag_entry["sources"].append(source)
    return payload


def has_regulatory_meta() -> bool:
    return bool(_load_regulatory_meta())


def regulatory_score(doc_id: str, base_score: float) -> RegulatoryScore | None:
    meta = _load_regulatory_meta().get(doc_id)
    if not meta:
        return None

    tags: Dict[str, Dict[str, Any]] = meta.get("tags", {})
    if not tags:
        return None

    rulebook = _load_rulebook().get("regulatory", {})
    tag_weights = dict(rulebook.get("tag_weights", {}))
    if not tag_weights:
        tag_weights = _fallback_tag_weights(rulebook)

    effective_base = max(base_score, float(rulebook.get("base_floor", 0.5)))
    max_bonus = float(rulebook.get("max_bonus", 0.0)) or None
    min_bonus = rulebook.get("min_bonus")
    default_conf = float(rulebook.get("default_confidence", 0.6))

    multiplier = 1.0
    bonus = 0.0
    contributions: List[Dict[str, Any]] = []
    confidences: List[float] = []

    for tag_name, info in tags.items():
        conf = float(info.get("confidence", default_conf) or default_conf)
        confidences.append(conf)
        weights = tag_weights.get(tag_name, {})
        detail: Dict[str, Any] = {
            "tag": tag_name,
            "confidence": round(conf, 4),
            "sources": info.get("sources", []),
            "evidence": info.get("evidence", [])[:3],
        }
        min_conf = float(weights.get("min_confidence", 0.0))
        if conf < min_conf:
            detail["applied"] = False
            contributions.append(detail)
            continue
        applied: Dict[str, Any] = {"confidence_used": round(conf, 4)}
        if "multiplier" in weights:
            base_multiplier = float(weights["multiplier"])
            applied_multiplier = 1.0 + (base_multiplier - 1.0) * conf
            multiplier *= applied_multiplier
            applied["multiplier"] = round(applied_multiplier, 4)
            applied["base_multiplier"] = base_multiplier
        if "bonus" in weights:
            per_bonus = float(weights["bonus"]) * conf
            bonus += per_bonus
            applied["bonus"] = round(per_bonus, 4)
            applied["base_bonus"] = float(weights["bonus"])
        if applied:
            detail["applied"] = applied
        contributions.append(detail)

    if multiplier != 1.0:
        mult_bonus = (multiplier - 1.0) * effective_base
        contributions.append(
            {
                "tag": "__multiplier__",
                "applied": {
                    "multiplier": round(multiplier, 4),
                    "applied_on": round(effective_base, 4),
                    "bonus": round(mult_bonus, 4),
                },
            }
        )
        bonus += mult_bonus

    if max_bonus is not None:
        bonus = min(bonus, max_bonus)
    if min_bonus is not None:
        bonus = max(bonus, float(min_bonus))

    confidence = mean(confidences) if confidences else default_conf

    return RegulatoryScore(
        bonus=bonus,
        multiplier=multiplier,
        tags=sorted(tags.keys()),
        confidence=confidence,
        contributions=contributions,
    )


def _fallback_tag_weights(rulebook: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    weights: Dict[str, Dict[str, Any]] = {}
    esg_multiplier = float(rulebook.get("esg_multiplier", 1.0))
    if esg_multiplier > 1.0:
        for tag in _POSITIVE_MULTIPLIER_TAGS:
            weights.setdefault(tag, {})["multiplier"] = esg_multiplier
    rating_bonus = float(rulebook.get("rating_bonus", 0.0))
    if rating_bonus:
        for tag in _RATING_TAGS:
            weights.setdefault(tag, {})["bonus"] = rating_bonus
    return weights


def _iter_tag_paths(path: Path) -> Iterable[Path]:
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


def _normalise_details(record: Dict[str, Any], *, source: str) -> Iterator[Dict[str, Any]]:
    details = record.get("tag_details")
    if isinstance(details, list) and details:
        for item in details:
            if not isinstance(item, dict):
                continue
            tag = item.get("name") or item.get("tag")
            if not tag:
                continue
            yield {
                "tag": tag,
                "confidence": item.get("confidence", record.get("confidence", 0.5)),
                "evidence": item.get("evidence") or record.get("evidence") or [],
                "source": item.get("source") or record.get("doc_type") or source,
            }
        return
    tags = record.get("tags") or []
    evidence = record.get("evidence") or []
    confidence = record.get("confidence", 0.5)
    for tag in tags:
        if not isinstance(tag, str):
            continue
        yield {
            "tag": tag,
            "confidence": confidence,
            "evidence": evidence,
            "source": record.get("doc_type") or source,
        }
