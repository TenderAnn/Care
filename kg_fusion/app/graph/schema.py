"""Dataclasses capturing the graph schema and query responses."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional


@dataclass
class Anchor:
    """Anchor links a graph artefact back to a parsed chunk."""

    doc_id: str
    chunk_id: str
    page_no: int
    bbox: List[float]
    text: str
    source: str = "chunk"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "page_no": self.page_no,
            "bbox": self.bbox,
            "text": self.text,
            "source": self.source,
        }


@dataclass
class GraphNode:
    """Graph node enriched with anchors and embeddings."""

    node_id: str
    node_type: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    anchors: List[Anchor] = field(default_factory=list)
    embeddings: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "properties": self.properties,
            "embeddings": self.embeddings,
        }
        if self.anchors:
            payload["anchors"] = [anchor.to_dict() for anchor in self.anchors]
        return payload


@dataclass
class GraphEdge:
    """Relationship between two graph nodes."""

    edge_id: str
    edge_type: str
    src_id: str
    dst_id: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "edge_type": self.edge_type,
            "src_id": self.src_id,
            "dst_id": self.dst_id,
            "properties": self.properties,
        }


@dataclass
class GraphBuildArtifacts:
    """Container produced after processing a parsed document."""

    doc_id: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }


@dataclass
class GraphQueryFilters:
    """Filters supplied by upstream intent/slot parsing."""

    intent: Optional[str] = None
    product_name: Optional[str] = None
    product_line: Optional[str] = None
    version_year: Optional[str] = None
    benefit_type: Optional[str] = None
    field: Optional[str] = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "GraphQueryFilters":
        return cls(
            intent=(mapping.get("intent") or None),
            product_name=_normalise(mapping.get("product_name")),
            product_line=_normalise(mapping.get("product_line")),
            version_year=_normalise(mapping.get("version_year")),
            benefit_type=_normalise(mapping.get("benefit_type")),
            field=_normalise(mapping.get("field")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalise(value: Any) -> Optional[str]:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


@dataclass
class GraphQueryStep:
    """Single step inside a query plan."""

    name: str
    description: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
        }


@dataclass
class ScoreBreakdown:
    """Detailed score components for a hit."""

    graph: float = 0.0
    semantic: float = 0.0
    recency: float = 0.0
    regulatory: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "graph": round(self.graph, 4),
            "semantic": round(self.semantic, 4),
            "recency": round(self.recency, 4),
            "regulatory": round(self.regulatory, 4),
        }


@dataclass
class GraphQueryHit:
    """Unified hit structure returned to downstream consumers."""

    doc_id: str
    chunk_id: str
    preview: str
    page_no: int
    bbox: List[float]
    source: str
    edge: Optional[str] = None
    node_type: Optional[str] = None
    anchors: List[Anchor] = field(default_factory=list)
    score: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "preview": self.preview,
            "page_no": self.page_no,
            "bbox": self.bbox,
            "source": self.source,
            "score_breakdown": self.score.to_dict(),
            "score_total": round(
                self.score.graph + self.score.semantic + self.score.recency + self.score.regulatory,
                4,
            ),
            "anchors": [anchor.to_dict() for anchor in self.anchors],
        }
        if self.edge:
            payload["edge"] = self.edge
        if self.node_type:
            payload["node_type"] = self.node_type
        if self.extra:
            payload["extra"] = self.extra
        return payload


@dataclass
class GraphQueryPlan:
    """Structured plan combining filters, strategy and executed steps."""

    filters: GraphQueryFilters
    query_texts: List[str]
    strategy: List[str]
    steps: List[GraphQueryStep] = field(default_factory=list)

    def add_step(self, name: str, description: str, arguments: Optional[MutableMapping[str, Any]] = None) -> None:
        self.steps.append(
            GraphQueryStep(
                name=name,
                description=description,
                arguments=dict(arguments or {}),
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filters": self.filters.to_dict(),
            "query_texts": self.query_texts,
            "strategy": self.strategy,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass
class GraphQueryResponse:
    """Full response consumed by the FastAPI service."""

    plan: GraphQueryPlan
    hits: List[GraphQueryHit]
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "hits": [hit.to_dict() for hit in self.hits],
            "debug": self.debug,
        }
