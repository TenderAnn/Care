"""Persistence helpers for graph nodes, edges and metadata."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .schema import Anchor, GraphBuildArtifacts, GraphEdge, GraphNode
from ...paths import data_path


class GraphStore:
    """File-based store for graph nodes and edges."""

    def __init__(self, *, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root is not None else data_path("structured")
        self.nodes_path = self.root / "graph_nodes.jsonl"
        self.edges_path = self.root / "graph_edges.jsonl"
        self.metadata_path = self.root / "graph_metadata.json"
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, artefacts: Iterable[GraphBuildArtifacts]) -> None:
        """Persist a collection of artefacts replacing prior content."""

        nodes: List[Dict] = []
        edges: List[Dict] = []
        doc_index: Dict[str, Dict[str, int]] = {}
        for artefact in artefacts:
            doc_nodes = [node.to_dict() for node in artefact.nodes]
            doc_edges = [edge.to_dict() for edge in artefact.edges]
            doc_index[artefact.doc_id] = {
                "nodes": len(doc_nodes),
                "edges": len(doc_edges),
            }
            nodes.extend(doc_nodes)
            edges.extend(doc_edges)
        self._write_jsonl(self.nodes_path, nodes)
        self._write_jsonl(self.edges_path, edges)
        self.metadata_path.write_text(
            json.dumps({"docs": doc_index}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def append(self, artefact: GraphBuildArtifacts) -> None:
        """Append artefacts to the store without rewriting the full files."""

        self._append_jsonl(self.nodes_path, (node.to_dict() for node in artefact.nodes))
        self._append_jsonl(self.edges_path, (edge.to_dict() for edge in artefact.edges))
        meta = self._read_metadata()
        meta.setdefault("docs", {})[artefact.doc_id] = {
            "nodes": len(artefact.nodes),
            "edges": len(artefact.edges),
        }
        self.metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_nodes(self) -> List[GraphNode]:
        nodes: List[GraphNode] = []
        for record in self._read_jsonl(self.nodes_path):
            anchors = [
                Anchor(
                    doc_id=anchor.get("doc_id", ""),
                    chunk_id=anchor.get("chunk_id", ""),
                    page_no=int(anchor.get("page_no", 0)),
                    bbox=list(anchor.get("bbox", [])),
                    text=anchor.get("text", ""),
                    source=anchor.get("source", "chunk"),
                )
                for anchor in record.pop("anchors", [])
            ]
            nodes.append(
                GraphNode(
                    node_id=record.get("node_id", ""),
                    node_type=record.get("node_type", ""),
                    label=record.get("label", ""),
                    properties=record.get("properties", {}),
                    anchors=anchors,
                    embeddings=record.get("embeddings", {}),
                )
            )
        return nodes

    def load_edges(self) -> List[GraphEdge]:
        return [GraphEdge(**record) for record in self._read_jsonl(self.edges_path)]

    def _read_metadata(self) -> Dict[str, Dict]:
        if not self.metadata_path.exists():
            return {}
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_jsonl(path: Path, records: Iterable[Dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            for record in records:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _append_jsonl(path: Path, records: Iterable[Dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fp:
            for record in records:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_jsonl(path: Path) -> Iterable[Dict]:
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
