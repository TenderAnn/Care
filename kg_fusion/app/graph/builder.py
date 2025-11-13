"""Graph builder that consumes structured parse artefacts."""
from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Optional, Tuple

from .schema import Anchor, GraphBuildArtifacts, GraphEdge, GraphNode
from ..ingest.schema import DocumentChunk, DocumentParseResult


class EntityCandidate:
    """Intermediate structure returned by entity extractors."""

    def __init__(self, *, label: str, entity_type: str, properties: Optional[Dict[str, str]] = None) -> None:
        self.label = label
        self.entity_type = entity_type
        self.properties = properties or {}

    def to_node(self, *, anchor: Anchor, prefix: str) -> GraphNode:
        node_id = f"{prefix}:{hashlib.md5(self.label.encode('utf-8')).hexdigest()}"
        return GraphNode(
            node_id=node_id,
            node_type=self.entity_type,
            label=self.label,
            properties=dict(self.properties),
            anchors=[anchor],
        )


class EntityExtractor:
    """Base protocol for entity extraction strategies."""

    def extract(self, chunk: DocumentChunk) -> Iterable[EntityCandidate]:  # pragma: no cover - protocol
        raise NotImplementedError


class SlotBackedExtractor(EntityExtractor):
    """Entity extractor that infers nodes purely from slot annotations."""

    SLOT_TO_TYPE = {
        "product_name": "Product",
        "product_line": "ProductLine",
        "benefit_type": "Benefit",
        "version_year": "Version",
        "field": "Field",
    }

    def extract(self, chunk: DocumentChunk) -> Iterable[EntityCandidate]:
        slots = chunk.slots or {}
        for key, value in slots.items():
            if not value:
                continue
            entity_type = self.SLOT_TO_TYPE.get(key)
            if not entity_type:
                continue
            label = str(value).strip()
            if not label:
                continue
            yield EntityCandidate(label=label, entity_type=entity_type, properties={"slot": key})


class HybridExtractor(EntityExtractor):
    """Hybrid extractor using slots with optional LLM or regex enrichments."""

    def __init__(self, *, slot_extractor: Optional[EntityExtractor] = None, llm_hook: Optional[EntityExtractor] = None) -> None:
        self.slot_extractor = slot_extractor or SlotBackedExtractor()
        self.llm_hook = llm_hook

    def extract(self, chunk: DocumentChunk) -> Iterable[EntityCandidate]:
        yielded = list(self.slot_extractor.extract(chunk))
        if self.llm_hook is not None:
            try:
                yielded.extend(list(self.llm_hook.extract(chunk)))
            except Exception:  # pragma: no cover - defensive guard for external hooks
                pass
        return yielded


class GraphBuilder:
    """Transform structured parse results into nodes and edges."""

    def __init__(self, extractor: Optional[EntityExtractor] = None) -> None:
        self.extractor = extractor or HybridExtractor()

    def build(self, parse: DocumentParseResult) -> GraphBuildArtifacts:
        doc_node = GraphNode(
            node_id=f"doc:{parse.doc_id}",
            node_type="Document",
            label=parse.doc_id,
            properties={
                "source_path": parse.source_path,
                "parser": parse.parser_info,
                "parsed_at": parse.parsed_at,
                "metadata_ready_at": parse.metadata_ready_at,
            },
        )

        nodes: Dict[str, GraphNode] = {doc_node.node_id: doc_node}
        edges: Dict[str, GraphEdge] = {}

        heading_stack: Dict[int, GraphNode] = {}

        for chunk in parse.chunks:
            chunk_anchor = Anchor(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                page_no=chunk.page_no,
                bbox=chunk.bbox,
                text=chunk.text[:240],
            )
            chunk_node = GraphNode(
                node_id=f"chunk:{chunk.chunk_id}",
                node_type=chunk.chunk_type.value,
                label=(chunk.heading.path[-1] if chunk.heading and chunk.heading.path else chunk.chunk_id),
                properties={
                    "chunk_type": chunk.chunk_type.value,
                    "page_no": chunk.page_no,
                    "tokens": chunk.tokens,
                    "slots": chunk.slots,
                    "metadata": chunk.metadata,
                },
                anchors=[chunk_anchor],
            )
            nodes[chunk_node.node_id] = chunk_node

            edge_id = f"edge:{doc_node.node_id}->{chunk_node.node_id}"
            edges[edge_id] = GraphEdge(
                edge_id=edge_id,
                edge_type="HAS_CHUNK",
                src_id=doc_node.node_id,
                dst_id=chunk_node.node_id,
                properties={"order": chunk.chunk_id},
            )

            if chunk.heading:
                heading_level = chunk.heading.level
                heading_node = GraphNode(
                    node_id=f"heading:{chunk.doc_id}:{chunk.chunk_id}",
                    node_type="Heading",
                    label=" > ".join(chunk.heading.path) or chunk.chunk_id,
                    properties={
                        "level": heading_level,
                        "path": chunk.heading.path,
                        "parent_ids": chunk.heading.parent_ids,
                    },
                    anchors=[chunk_anchor],
                )
                nodes[heading_node.node_id] = heading_node
                heading_stack[heading_level] = heading_node
                if heading_level > 1 and (heading_level - 1) in heading_stack:
                    parent = heading_stack[heading_level - 1]
                    parent_edge_id = f"edge:{parent.node_id}->{heading_node.node_id}"
                    edges[parent_edge_id] = GraphEdge(
                        edge_id=parent_edge_id,
                        edge_type="HAS_SUBHEADING",
                        src_id=parent.node_id,
                        dst_id=heading_node.node_id,
                        properties={"level": heading_level},
                    )
                heading_edge_id = f"edge:{heading_node.node_id}->{chunk_node.node_id}"
                edges[heading_edge_id] = GraphEdge(
                    edge_id=heading_edge_id,
                    edge_type="HAS_SECTION",
                    src_id=heading_node.node_id,
                    dst_id=chunk_node.node_id,
                    properties={"level": heading_level},
                )

            entity_nodes, entity_edges = self._build_entities(chunk, chunk_anchor, chunk_node)
            for node in entity_nodes:
                nodes.setdefault(node.node_id, node)
            for edge in entity_edges:
                edges[edge.edge_id] = edge

        return GraphBuildArtifacts(doc_id=parse.doc_id, nodes=list(nodes.values()), edges=list(edges.values()))

    def _build_entities(
        self,
        chunk: DocumentChunk,
        chunk_anchor: Anchor,
        chunk_node: GraphNode,
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        entity_nodes: Dict[str, GraphNode] = {}
        entity_edges: List[GraphEdge] = []
        regulation_label = chunk.metadata.get("regulation") if isinstance(chunk.metadata, dict) else None
        regulation_node_id: Optional[str] = None
        if regulation_label:
            regulation_node_id = f"regulation:{regulation_label}"
            entity_nodes.setdefault(
                regulation_node_id,
                GraphNode(
                    node_id=regulation_node_id,
                    node_type="Regulation",
                    label=str(regulation_label),
                    properties={"source": chunk.metadata.get("regulation_source")},
                    anchors=[chunk_anchor],
                ),
            )
        for candidate in self.extractor.extract(chunk):
            node = candidate.to_node(anchor=chunk_anchor, prefix=candidate.entity_type.lower())
            entity_nodes.setdefault(node.node_id, node)
            edge_id = f"edge:{chunk_node.node_id}->{node.node_id}"
            entity_edges.append(
                GraphEdge(
                    edge_id=edge_id,
                    edge_type="MENTIONS_ENTITY",
                    src_id=chunk_node.node_id,
                    dst_id=node.node_id,
                    properties={"slot": candidate.properties.get("slot")},
                )
            )
            if candidate.entity_type == "Benefit" and regulation_node_id:
                reg_edge_id = f"edge:{node.node_id}->{regulation_node_id}"
                entity_edges.append(
                    GraphEdge(
                        edge_id=reg_edge_id,
                        edge_type="SUPPORTS_REGULATION",
                        src_id=node.node_id,
                        dst_id=regulation_node_id,
                        properties={"evidence": regulation_label},
                    )
                )
        return list(entity_nodes.values()), entity_edges
