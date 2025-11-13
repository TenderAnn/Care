"""Recency metadata loading and scoring utilities."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .. import PACKAGE_ROOT
from ..paths import data_path, env_or_path

RECENCY_CSV = env_or_path("RECENCY_CSV", data_path("kg", "recency_meta.csv"))
LIFECYCLE_LOG = env_or_path(
    "LIFECYCLE_LOG",
    PACKAGE_ROOT.parent / "freshrank" / "data" / "metadata" / "ingestion_events.jsonl",
)
RULEBOOK_PATH = env_or_path(
    "REG_RULEBOOK",
    PACKAGE_ROOT.parent / "freshrank" / "regulatory" / "weights.yaml",
)


@dataclass
class LifecycleInfo:
    doc_id: str
    ingested_at: Optional[str] = None
    parsed_at: Optional[str] = None
    metadata_ready_at: Optional[str] = None
    served_at: Optional[str] = None


@dataclass
class RecencyScore:
    bonus: float
    flags: Dict[str, bool]
    curve: Dict[str, Any]
    timeline: Dict[str, Any]
    sla_flags: Dict[str, bool]

    def to_debug(self) -> Dict[str, Any]:
        return {
            "bonus": round(self.bonus, 4),
            "flags": self.flags,
            "curve": self.curve,
            "timeline": self.timeline,
            "sla_flags": self.sla_flags,
        }


def load_recency() -> Dict[str, Dict[str, Any]]:
    """Load recency metadata combined with lifecycle tracking."""

    recency = _load_recency_csv()
    lifecycle = _load_lifecycle()
    merged: Dict[str, Dict[str, Any]] = {}
    for doc_id, payload in recency.items():
        merged[doc_id] = {**payload, "lifecycle": lifecycle.get(doc_id)}
    for doc_id, info in lifecycle.items():
        merged.setdefault(doc_id, {"effective_date": "", "discontinue_date": ""})["lifecycle"] = info
    return merged


def recency_score(meta: Dict[str, Any]) -> RecencyScore:
    """Return bonus, flags, curve diagnostics, and SLA checks based on recency."""

    config = _recency_config()
    curve_cfg = config.get("curve", {})
    sla_cfg = config.get("sla_minutes", {})

    today = date.today()
    flags: Dict[str, bool] = {}
    sla_flags: Dict[str, bool] = {}

    eff = _parse_date(meta.get("effective_date"))
    dis = _parse_date(meta.get("discontinue_date"))

    age_days = (today - eff).days if eff else None
    decay = _compute_decay(age_days, curve_cfg)

    bonus = 0.0
    curve_debug = {
        "age_days": age_days,
        "fresh_window_days": curve_cfg.get("fresh_window_days"),
        "stale_after_days": curve_cfg.get("stale_after_days"),
        "decay_weight": round(decay, 4) if decay is not None else None,
    }

    if age_days is not None:
        fresh_window = int(curve_cfg.get("fresh_window_days", 180))
        stale_after = int(curve_cfg.get("stale_after_days", fresh_window))
        if age_days <= fresh_window:
            freshness = max(0.0, 1.0 - (age_days / max(fresh_window, 1)))
            bonus += float(curve_cfg.get("fresh_bonus", 0.3)) * freshness
            flags["fresh_window"] = True
            curve_debug["freshness"] = round(freshness, 4)
        elif stale_after > fresh_window:
            progress = min(1.0, (age_days - fresh_window) / (stale_after - fresh_window))
            penalty = float(curve_cfg.get("stale_penalty", -0.4)) * progress
            bonus += penalty
            flags["aging"] = True
            curve_debug["aging_progress"] = round(progress, 4)
        else:
            curve_debug["freshness"] = 0.0

    if dis and (today - dis).days >= 0:
        bonus += float(curve_cfg.get("expiry_penalty", -0.5))
        flags["expired"] = True

    if decay is not None:
        decay_scale = float(curve_cfg.get("decay_scale", 0.2))
        bonus += (decay - 1.0) * decay_scale
        curve_debug["decay_scale"] = decay_scale

    bonus = _clamp(bonus, curve_cfg.get("min_bonus"), curve_cfg.get("max_bonus"))

    lifecycle: Optional[LifecycleInfo] = meta.get("lifecycle")
    timeline_debug = _timeline_debug(lifecycle)
    if timeline_debug:
        sla_flags = _compute_sla_flags(timeline_debug, sla_cfg)

    return RecencyScore(bonus=round(bonus, 4), flags=flags, curve=curve_debug, timeline=timeline_debug, sla_flags=sla_flags)


# ---------------------------------------------------------------------------
# Helpers


def _load_recency_csv() -> Dict[str, Dict[str, str]]:
    path = Path(RECENCY_CSV)
    if not path.exists():
        return {}
    data: Dict[str, Dict[str, str]] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            data[row["doc_id"]] = {
                "effective_date": (row.get("effective_date") or "").strip(),
                "discontinue_date": (row.get("discontinue_date") or "").strip(),
            }
    return data


def _load_lifecycle() -> Dict[str, LifecycleInfo]:
    path = Path(LIFECYCLE_LOG)
    if not path.exists():
        return {}
    records: Dict[str, LifecycleInfo] = {}
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            doc_id = str(payload.get("doc_id"))
            if not doc_id:
                continue
            records[doc_id] = LifecycleInfo(
                doc_id=doc_id,
                ingested_at=payload.get("ingested_at"),
                parsed_at=payload.get("parsed_at"),
                metadata_ready_at=payload.get("metadata_ready_at"),
                served_at=payload.get("served_at"),
            )
    return records


@lru_cache()
def _recency_config() -> Dict[str, Any]:
    path = Path(RULEBOOK_PATH)
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload.get("recency", {})


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def _parse_iso(iso_str: Optional[str]) -> Optional[datetime]:
    if not iso_str:
        return None
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def _compute_decay(age_days: Optional[int], curve_cfg: Dict[str, Any]) -> Optional[float]:
    if age_days is None:
        return None
    half_life = float(curve_cfg.get("decay_half_life_days", 365))
    if half_life <= 0:
        return 1.0
    age = max(age_days, 0)
    return math.exp(-age / half_life)


def _clamp(value: float, min_value: Optional[float], max_value: Optional[float]) -> float:
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _timeline_debug(lifecycle: Optional[LifecycleInfo]) -> Dict[str, Any]:
    if lifecycle is None:
        return {}
    return {
        "ingested_at": lifecycle.ingested_at,
        "parsed_at": lifecycle.parsed_at,
        "metadata_ready_at": lifecycle.metadata_ready_at,
        "served_at": lifecycle.served_at,
        "durations_minutes": {
            "ingest_to_parse": _duration_minutes(lifecycle.ingested_at, lifecycle.parsed_at),
            "parse_to_metadata": _duration_minutes(lifecycle.parsed_at, lifecycle.metadata_ready_at),
            "metadata_to_served": _duration_minutes(lifecycle.metadata_ready_at, lifecycle.served_at),
        },
    }


def _duration_minutes(start: Optional[str], end: Optional[str]) -> Optional[float]:
    start_dt = _parse_iso(start)
    end_dt = _parse_iso(end)
    if not start_dt or not end_dt:
        return None
    delta = end_dt - start_dt
    return round(delta.total_seconds() / 60.0, 2)


def _compute_sla_flags(timeline: Dict[str, Any], sla_cfg: Dict[str, Any]) -> Dict[str, bool]:
    flags: Dict[str, bool] = {}
    durations = timeline.get("durations_minutes", {})
    thresholds = {
        "ingest_to_parse": float(sla_cfg.get("parse", 15) or 0),
        "parse_to_metadata": float(sla_cfg.get("metadata", 20) or 0),
        "metadata_to_served": float(sla_cfg.get("serve", 30) or 0),
    }
    for key, threshold in thresholds.items():
        duration = durations.get(key)
        if duration is None or threshold <= 0:
            continue
        if duration > threshold:
            flags[f"sla_{key}"] = True
    return flags
