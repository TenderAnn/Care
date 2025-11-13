"""Dataclasses shared by the ingestion and parsing stack."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class ChunkType(str, Enum):
    """Enumerated chunk categories recognised by downstream services."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    FOOTNOTE = "footnote"


@dataclass
class HeadingInfo:
    """Heading metadata produced during parsing."""

    level: int
    path: List[str] = field(default_factory=list)
    is_heading: bool = False
    parent_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "path": self.path,
            "is_heading": self.is_heading,
            "parent_ids": self.parent_ids,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "HeadingInfo":
        return cls(
            level=int(payload.get("level", 0)),
            path=list(payload.get("path", [])),
            is_heading=bool(payload.get("is_heading", False)),
            parent_ids=list(payload.get("parent_ids", [])),
        )


@dataclass
class DocumentChunk:
    """Representation of a structured chunk with layout and semantic context."""

    doc_id: str
    chunk_id: str
    page_no: int
    chunk_type: ChunkType
    text: str
    bbox: List[float]
    tokens: List[str] = field(default_factory=list)
    heading: Optional[HeadingInfo] = None
    slots: Dict[str, str] = field(default_factory=dict)
    anchors: Dict[str, Any] = field(default_factory=dict)
    layout: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "page_no": self.page_no,
            "chunk_type": str(self.chunk_type),
            "text": self.text,
            "bbox": self.bbox,
            "tokens": self.tokens,
            "slots": self.slots,
            "anchors": self.anchors,
            "layout": self.layout,
            "metadata": self.metadata,
        }
        if self.heading is not None:
            payload["heading"] = self.heading.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DocumentChunk":
        heading_payload = payload.get("heading")
        heading = HeadingInfo.from_dict(heading_payload) if isinstance(heading_payload, dict) else None
        return cls(
            doc_id=payload.get("doc_id", ""),
            chunk_id=payload.get("chunk_id", ""),
            page_no=int(payload.get("page_no", 0)),
            chunk_type=ChunkType(payload.get("chunk_type", ChunkType.PARAGRAPH)),
            text=payload.get("text", ""),
            bbox=list(payload.get("bbox", [])),
            tokens=list(payload.get("tokens", [])),
            heading=heading,
            slots=dict(payload.get("slots", {})),
            anchors=dict(payload.get("anchors", {})),
            layout=dict(payload.get("layout", {})),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class DocumentParseResult:
    """Container for the full parse artefacts of a single document."""

    doc_id: str
    source_path: str
    parsed_at: str
    metadata_ready_at: str
    parser_info: Dict[str, Any]
    chunks: List[DocumentChunk] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "parsed_at": self.parsed_at,
            "metadata_ready_at": self.metadata_ready_at,
            "parser": self.parser_info,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DocumentParseResult":
        return cls(
            doc_id=payload.get("doc_id", ""),
            source_path=payload.get("source_path", ""),
            parsed_at=payload.get("parsed_at", ""),
            metadata_ready_at=payload.get("metadata_ready_at", ""),
            parser_info=dict(payload.get("parser", {})),
            chunks=[DocumentChunk.from_dict(item) for item in payload.get("chunks", [])],
        )


@dataclass
class LifecycleRecord:
    """Lifecycle entry aligned with freshrank ingestion semantics."""

    doc_id: str
    ingested_at: str
    parsed_at: str
    metadata_ready_at: str
    served_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        record = asdict(self)
        # keep explicit None instead of dropping the field for transparency
        return record
