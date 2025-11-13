"""GraphRAG index and planner built on top of structured artefacts."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from .schema import (
    Anchor,
    GraphEdge,
    GraphNode,
    GraphQueryFilters,
    GraphQueryHit,
    GraphQueryPlan,
    GraphQueryResponse,
    ScoreBreakdown,
)
from ..semantic import SemanticRetriever


def _safe_lower(value: Optional[str]) -> str:
    return value.lower() if isinstance(value, str) else ""


class GraphRAGIndex:
    """Lightweight in-memory index for graph-aware retrieval."""

    def __init__(self, nodes: Sequence[GraphNode], edges: Sequence[GraphEdge]) -> None:
        self.nodes = list(nodes)
        self.edges = list(edges)
        self._edges_by_src: Dict[str, List[GraphEdge]] = defaultdict(list)
        for edge in self.edges:
            self._edges_by_src[edge.src_id].append(edge)

    def filter_chunks(self, filters: GraphQueryFilters) -> List[GraphNode]:
        """Return chunk nodes matching the slot filters."""

        results: List[GraphNode] = []
        for node in self.nodes:
            if not node.node_id.startswith("chunk:"):
                continue
            slots = node.properties.get("slots", {}) if isinstance(node.properties, dict) else {}
            if self._match_slots(slots, filters):
                results.append(node)
        return results

    @staticmethod
    def _match_slots(slots: Dict[str, str], filters: GraphQueryFilters) -> bool:
        tests: List[Tuple[Optional[str], str]] = [
            (filters.product_name, "product_name"),
            (filters.product_line, "product_line"),
            (filters.version_year, "version_year"),
            (filters.benefit_type, "benefit_type"),
            (filters.field, "field"),
        ]
        for expected, key in tests:
            if expected is None:
                continue
            actual = slots.get(key)
            if not isinstance(actual, str):
                return False
            if _safe_lower(actual) != _safe_lower(expected):
                return False
        return True

    def neighbourhood(self, node_id: str) -> List[GraphEdge]:
        return list(self._edges_by_src.get(node_id, []))


class GraphRAGPlanner:
    """Planner combining graph filtering with vector search."""

    def __init__(self, index: GraphRAGIndex, *, semantic_retriever: SemanticRetriever | None = None) -> None:
        self.index = index
        self.semantic = semantic_retriever or SemanticRetriever()

    def query(
        self,
        filters: GraphQueryFilters,
        query_texts: Sequence[str],
        *,
        topk: int = 20,
    ) -> GraphQueryResponse:
        plan = GraphQueryPlan(
            filters=filters,
            query_texts=list(query_texts),
            strategy=["slot-filter", "graph-neighbourhood", "semantic-retrieve"],
        )

        chunk_candidates = self.index.filter_chunks(filters)
        plan.add_step(
            "slot-filter",
            "Filter chunk nodes using slot-aligned constraints",
            {"candidates": len(chunk_candidates)},
        )

        neighbourhood_edges = [self.index.neighbourhood(node.node_id) for node in chunk_candidates]
        plan.add_step(
            "graph-neighbourhood",
            "Collect adjacent entity and regulation evidence for the filtered chunks",
            {"edges": sum(len(edges) for edges in neighbourhood_edges)},
        )

        semantic_result = self.semantic.retrieve(filters, query_texts, topk=topk)
        plan.add_step(
            "semantic-retrieve",
            "Retrieve semantic neighbours using weighted query expansion",
            {
                "semantic_hits": len(semantic_result.hits),
                "candidates": len(semantic_result.candidates),
            },
        )

        hits = self._merge_hits(
            chunk_candidates,
            neighbourhood_edges,
            semantic_result.hits,
            topk=topk,
        )
        debug = {
            "filtered_nodes": [node.node_id for node in chunk_candidates],
            "semantic_chunk_ids": [hit.get("chunk_id") for hit in semantic_result.hits],
            "semantic": semantic_result.debug,
        }
        return GraphQueryResponse(plan=plan, hits=hits, debug=debug)

    def _merge_hits(
        self,
        chunk_candidates: Sequence[GraphNode],
        neighbourhood_edges: Sequence[Sequence[GraphEdge]],
        semantic_hits: Sequence[Dict],
        *,
        topk: int,
    ) -> List[GraphQueryHit]:
        results: Dict[str, GraphQueryHit] = {}

        for node, edges in zip(chunk_candidates, neighbourhood_edges):
            anchor = node.anchors[0] if node.anchors else Anchor(
                doc_id=node.properties.get("doc_id", ""),
                chunk_id=node.node_id.split(":", 1)[-1],
                page_no=int(node.properties.get("page_no", 0)),
                bbox=node.properties.get("bbox", []),
                text=node.label,
            )
            score = ScoreBreakdown(graph=1.0)
            results[node.node_id] = GraphQueryHit(
                doc_id=anchor.doc_id,
                chunk_id=anchor.chunk_id,
                preview=anchor.text,
                page_no=anchor.page_no,
                bbox=anchor.bbox,
                source=node.node_type,
                edge=None,
                node_type=node.node_type,
                anchors=[anchor] + node.anchors[1:],
                score=score,
                extra={
                    "neighbours": [edge.to_dict() for edge in edges],
                },
            )

        for semantic in semantic_hits:
            chunk_id = semantic.get("chunk_id") or semantic.get("doc_id")
            key = f"chunk:{chunk_id}"
            anchor = Anchor(
                doc_id=semantic.get("doc_id", ""),
                chunk_id=str(chunk_id),
                page_no=int(semantic.get("page_no", 0)),
                bbox=semantic.get("bbox", []),
                text=semantic.get("text", ""),
            )
            existing = results.get(key)
            semantic_score = float(semantic.get("score_semantic", 0.0))
            semantic_sources = semantic.get("semantic_sources") or [semantic]
            if existing:
                existing.score.semantic = max(existing.score.semantic, semantic_score)
                if not existing.preview:
                    existing.preview = anchor.text
                existing.extra.setdefault("semantic_sources", []).extend(semantic_sources)
            else:
                results[key] = GraphQueryHit(
                    doc_id=anchor.doc_id,
                    chunk_id=anchor.chunk_id,
                    preview=anchor.text,
                    page_no=anchor.page_no,
                    bbox=anchor.bbox,
                    source=semantic.get("type", "vector"),
                    anchors=[anchor],
                    score=ScoreBreakdown(semantic=semantic_score),
                    extra={"semantic_sources": list(semantic_sources)},
                )

        scored = sorted(
            results.values(),
            key=lambda item: item.score.graph + item.score.semantic,
            reverse=True,
        )
        return scored[:topk]
