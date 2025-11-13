"""Lifecycle tracking aligned with freshrank ingestion semantics."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from .. import PACKAGE_ROOT
from .schema import LifecycleRecord


class LifecycleTracker:
    """Utility to keep ingestion_events.jsonl in sync with fresh parses."""

    def __init__(self, log_path: Path | str | None = None) -> None:
        if log_path is None:
            log_path = PACKAGE_ROOT.parent / "freshrank" / "data" / "metadata" / "ingestion_events.jsonl"
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _read_events(self) -> Dict[str, Dict[str, str]]:
        events: Dict[str, Dict[str, str]] = {}
        if not self.log_path.exists():
            return events
        with self.log_path.open(encoding="utf-8") as fp:
            for line in fp:
                data = json.loads(line)
                doc_id = data.get("doc_id")
                if not doc_id:
                    continue
                events[doc_id] = data
        return events

    def _write_events(self, events: Iterable[Dict[str, str]]) -> None:
        with self.log_path.open("w", encoding="utf-8") as fp:
            for event in sorted(events, key=lambda item: item.get("doc_id", "")):
                fp.write(json.dumps(event, ensure_ascii=False) + "\n")

    def update(self, doc_id: str, *, parsed_at: str | None = None, metadata_ready_at: str | None = None) -> LifecycleRecord:
        events = self._read_events()
        existing = events.get(doc_id)
        timestamp = parsed_at or self._now_iso()
        meta_ready = metadata_ready_at or timestamp
        ingested = (existing or {}).get("ingested_at") or timestamp
        served = (existing or {}).get("served_at")
        record = LifecycleRecord(
            doc_id=doc_id,
            ingested_at=ingested,
            parsed_at=timestamp,
            metadata_ready_at=meta_ready,
            served_at=served,
        )
        events[doc_id] = record.to_dict()
        self._write_events(events.values())
        return record

    def mark_served(self, doc_id: str, *, served_at: str | None = None) -> LifecycleRecord:
        """Mark a document as served after graph/index pipelines complete."""

        events = self._read_events()
        existing = events.get(doc_id, {})
        served_time = served_at or self._now_iso()
        record = LifecycleRecord(
            doc_id=doc_id,
            ingested_at=existing.get("ingested_at") or served_time,
            parsed_at=existing.get("parsed_at") or served_time,
            metadata_ready_at=existing.get("metadata_ready_at") or served_time,
            served_at=served_time,
        )
        events[doc_id] = record.to_dict()
        self._write_events(events.values())
        return record
