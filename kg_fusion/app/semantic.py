"""Semantic retrieval helpers for GraphRAG fusion (Phase 3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .graph.schema import GraphQueryFilters
from . import vector_backend


@dataclass(frozen=True)
class QueryCandidate:
    """Single semantic query with metadata and weighting."""

    text: str
    weight: float = 1.0
    source: str = "user"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_debug(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "weight": round(self.weight, 4),
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class SemanticResult:
    """Container returned by :class:`SemanticRetriever`."""

    hits: List[Dict[str, Any]]
    candidates: List[QueryCandidate]
    debug: Dict[str, Any]


class SemanticRetriever:
    """Build weighted semantic queries and aggregate vector results."""

    def __init__(
        self,
        *,
        base_weight: float = 1.0,
        slot_weight: float = 0.65,
        combo_weight: float = 0.85,
        intent_weight: float = 0.7,
    ) -> None:
        self.base_weight = base_weight
        self.slot_weight = slot_weight
        self.combo_weight = combo_weight
        self.intent_weight = intent_weight

    def retrieve(
        self,
        filters: GraphQueryFilters,
        query_texts: Sequence[str],
        *,
        topk: int = 20,
    ) -> SemanticResult:
        candidates = self._build_candidates(filters, query_texts)
        hits, debug = self._collect_hits(candidates, topk=topk)
        return SemanticResult(hits=hits, candidates=candidates, debug=debug)

    # ------------------------------------------------------------------
    # candidate generation
    # ------------------------------------------------------------------
    def _build_candidates(
        self,
        filters: GraphQueryFilters,
        query_texts: Sequence[str],
    ) -> List[QueryCandidate]:
        seen: set[str] = set()
        candidates: List[QueryCandidate] = []

        def _push(text: str, weight: float, source: str, metadata: Dict[str, Any] | None = None) -> None:
            normalised = text.strip()
            if not normalised:
                return
            key = normalised.lower()
            if key in seen:
                return
            seen.add(key)
            candidates.append(
                QueryCandidate(
                    text=normalised,
                    weight=max(weight, 0.0),
                    source=source,
                    metadata=metadata or {},
                )
            )

        for original in query_texts:
            if isinstance(original, str):
                _push(original, self.base_weight, "user", {"role": "original"})

        slot_texts: List[Tuple[str, Dict[str, Any]]] = []
        if filters.product_name:
            slot_texts.append((filters.product_name, {"slot": "product_name"}))
            if filters.field:
                slot_texts.append((f"{filters.product_name} {filters.field}", {"slot": "product_name+field"}))
        if filters.product_line:
            slot_texts.append((f"{filters.product_line} 产品", {"slot": "product_line"}))
        if filters.benefit_type:
            slot_texts.append((filters.benefit_type, {"slot": "benefit_type"}))
            slot_texts.append((f"{filters.benefit_type} 条款", {"slot": "benefit_type"}))
        if filters.field:
            slot_texts.append((filters.field, {"slot": "field"}))
        if filters.version_year:
            slot_texts.append((f"{filters.version_year} 版本", {"slot": "version_year"}))

        for text, meta in slot_texts:
            _push(text, self.slot_weight, "slot", meta)

        combos: List[Tuple[str, Dict[str, Any]]] = []
        if filters.product_name and filters.benefit_type:
            combos.append(
                (
                    f"{filters.product_name} {filters.benefit_type}",
                    {"slot": "product_name+benefit_type"},
                )
            )
        if filters.product_name and filters.version_year:
            combos.append(
                (
                    f"{filters.product_name} {filters.version_year}",
                    {"slot": "product_name+version_year"},
                )
            )
        if filters.product_line and filters.field:
            combos.append(
                (
                    f"{filters.product_line} {filters.field}",
                    {"slot": "product_line+field"},
                )
            )

        for text, meta in combos:
            _push(text, self.combo_weight, "slot-combo", meta)

        intent_hints = _intent_hints(filters.intent)
        for hint in intent_hints:
            _push(hint, self.intent_weight, "intent", {"intent": filters.intent})

        if not candidates:
            # ensure at least one placeholder query so downstream logic can still run
            _push("保险 产品", self.base_weight, "fallback", {"reason": "no-input"})

        return candidates

    # ------------------------------------------------------------------
    # retrieval aggregation
    # ------------------------------------------------------------------
    def _collect_hits(self, candidates: Sequence[QueryCandidate], *, topk: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        aggregated: Dict[str, Dict[str, Any]] = {}
        candidate_debug: List[Dict[str, Any]] = []

        for candidate in candidates:
            raw_hits = vector_backend.search([candidate.text], topk=topk)
            candidate_summary = candidate.to_debug()
            candidate_summary["hit_count"] = len(raw_hits)
            candidate_debug.append(candidate_summary)

            for hit in raw_hits:
                chunk_id = hit.get("chunk_id")
                if not chunk_id:
                    continue
                weighted = float(hit.get("score_semantic", 0.0)) * candidate.weight
                entry = aggregated.setdefault(
                    chunk_id,
                    {
                        "chunk_id": chunk_id,
                        "doc_id": hit.get("doc_id"),
                        "page_no": hit.get("page_no"),
                        "bbox": hit.get("bbox", []),
                        "text": hit.get("text", ""),
                        "type": hit.get("type", "vector"),
                        "score_semantic": 0.0,
                        "semantic_sources": [],
                    },
                )
                # Keep the strongest preview/bbox info encountered
                if weighted > entry["score_semantic"]:
                    entry.update(
                        {
                            "doc_id": hit.get("doc_id"),
                            "page_no": hit.get("page_no"),
                            "bbox": hit.get("bbox", []),
                            "text": hit.get("text", ""),
                            "type": hit.get("type", "vector"),
                        }
                    )
                entry["score_semantic"] = max(entry["score_semantic"], weighted)
                entry.setdefault("semantic_sources", []).append(
                    {
                        "text": candidate.text,
                        "weight": round(candidate.weight, 4),
                        "raw_score": float(hit.get("score_semantic", 0.0)),
                        "weighted_score": round(weighted, 4),
                        "source": candidate.source,
                    }
                )

        hits = sorted(
            aggregated.values(),
            key=lambda item: item.get("score_semantic", 0.0),
            reverse=True,
        )
        debug = {
            "candidates": candidate_debug,
            "unique_chunks": len(hits),
        }
        return hits[:topk], debug


def _intent_hints(intent: str | None) -> Iterable[str]:
    if not intent:
        return []
    intent_upper = intent.upper()
    mapping = {
        "COVERAGE": ["保障责任", "赔付范围"],
        "COVERAGE_EXCLUSION": ["除外责任", "免责条款"],
        "ELIGIBILITY": ["投保条件", "承保规则"],
        "CLAIM_PROCESS": ["理赔流程", "理赔材料"],
        "PREMIUM_RATE": ["费率表", "保费"],
    }
    return mapping.get(intent_upper, [])
