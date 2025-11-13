"""Slot alignment helpers that reuse the intent dataset as supervision."""
from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

LOGGER = logging.getLogger(__name__)


@dataclass
class _AliasEntry:
    alias: str
    slots: Dict[str, str]
    priority: int


class SlotMapper:
    """Infer canonical slots from free text based on the intent dataset."""

    def __init__(
        self,
        dataset_path: Path | str,
        synonyms_path: Path | str | None = None,
        *,
        min_alias_len: int = 2,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.synonyms_path = Path(synonyms_path) if synonyms_path else None
        self.min_alias_len = min_alias_len
        self._alias_index: List[_AliasEntry] = []
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Intent dataset not found: {self.dataset_path}")
        self._build_alias_index()

    @staticmethod
    def _normalise(text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"\s+", "", text)
        pattern = r"""["'`·•†‡°℃()（）［］\[\]{}<>《》“”‘’、，,。:：；;?!？！~—\-]"""
        text = re.sub(pattern, "", text)
        return text

    def _register_alias(self, alias: str, slots: Dict[str, str], priority: int) -> None:
        alias_norm = self._normalise(alias)
        if len(alias_norm) < self.min_alias_len:
            return
        self._alias_index.append(_AliasEntry(alias=alias_norm, slots=slots, priority=priority))

    def _build_alias_index(self) -> None:
        product_seen: Dict[str, Dict[str, str]] = {}
        with self.dataset_path.open(encoding="utf-8") as fp:
            for line in fp:
                record = json.loads(line)
                raw_slots = record.get("slots")
                slots: Dict[str, str] = {}
                if isinstance(raw_slots, str):
                    try:
                        slots = json.loads(raw_slots)
                    except json.JSONDecodeError:
                        LOGGER.debug("Unable to decode slots payload: %s", raw_slots)
                        slots = {}
                elif isinstance(raw_slots, dict):
                    slots = raw_slots
                canonical_product = (slots.get("product_name") or record.get("source_hint") or "").strip()
                if not canonical_product:
                    continue
                slots = {k: v for k, v in slots.items() if isinstance(v, str) and v}
                slots.setdefault("intent", record.get("intent", ""))
                product_seen.setdefault(canonical_product, slots)
                aliases: List[str] = [canonical_product]
                source_hint = record.get("source_hint")
                if source_hint:
                    aliases.append(source_hint)
                canonical_query = record.get("canonical_query")
                if canonical_query:
                    aliases.append(canonical_query)
                for alias in aliases:
                    self._register_alias(alias, slots, priority=2)
        if self.synonyms_path and self.synonyms_path.exists():
            with self.synonyms_path.open(encoding="utf-8") as fp:
                reader = csv.DictReader(fp, delimiter="\t")
                for row in reader:
                    canonical = row.get("canonical", "").strip()
                    synonym = row.get("synonym", "").strip()
                    if not canonical or not synonym:
                        continue
                    slots = {"product_line": canonical}
                    self._register_alias(synonym, slots, priority=1)
        # Sort aliases so that higher priority and longer matches are checked first
        self._alias_index.sort(key=lambda entry: (entry.priority, len(entry.alias)), reverse=True)
        LOGGER.info("Loaded %d slot aliases from %s", len(self._alias_index), self.dataset_path)

    def infer_slots(self, text: str) -> tuple[Dict[str, str], List[str]]:
        """Infer slots from `text`.

        Returns a tuple of `(slots, matched_aliases)`.
        """
        if not text:
            return {}, []
        norm_text = self._normalise(text)
        matched_aliases: List[str] = []
        slots: Dict[str, str] = {}
        for entry in self._alias_index:
            if entry.alias in norm_text:
                merged = False
                for key, value in entry.slots.items():
                    if value and key not in slots:
                        slots[key] = value
                        merged = True
                if merged:
                    matched_aliases.append(entry.alias)
        return slots, matched_aliases
