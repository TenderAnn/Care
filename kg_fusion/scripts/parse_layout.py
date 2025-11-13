"""Parse PDFs into structured multi-modal chunks with slot and lifecycle alignment."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Optional

from kg_fusion.app.ingest import (
    ChunkType,
    HybridPDFParser,
    LifecycleTracker,
    SlotMapper,
)
from kg_fusion.paths import data_path

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTENT_DATASET = REPO_ROOT / "intent_dataset_zh_insurance_v1.jsonl"
DEFAULT_SYNONYMS = REPO_ROOT / "synonyms_insurance_zh.tsv"
DEFAULT_LIFECYCLE_LOG = REPO_ROOT / "freshrank" / "data" / "metadata" / "ingestion_events.jsonl"


def write_structured(result, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fp:
        json.dump(result.to_dict(), fp, ensure_ascii=False, indent=2)
    LOGGER.info("Structured artefact written to %s", out_file)


def write_sections(result, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fp:
        for idx, chunk in enumerate(result.chunks):
            heading = chunk.heading or None
            record = {
                "doc_id": chunk.doc_id,
                "page_no": chunk.page_no,
                "section_id": chunk.chunk_id,
                "order": idx,
                "bbox": chunk.bbox,
                "is_heading": chunk.chunk_type == ChunkType.HEADING,
                "heading_level": heading.level if heading else 0,
                "heading_path": " > ".join(heading.path) if heading else "",
                "text": chunk.text,
                "chunk_type": str(chunk.chunk_type),
                "slots": chunk.slots,
                "layout": chunk.layout,
                "metadata": chunk.metadata,
            }
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    LOGGER.info("Legacy sections written to %s", out_file)


def iter_pdfs(in_dir: Path) -> Iterable[Path]:
    return sorted(p for p in in_dir.glob("*.pdf") if p.is_file())


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", default=str(data_path("raw")))
    parser.add_argument("--structured-out", default=str(data_path("structured")))
    parser.add_argument("--sections-out", default=str(data_path("parsed")))
    parser.add_argument("--intent-dataset", default=str(DEFAULT_INTENT_DATASET))
    parser.add_argument("--synonyms", default=str(DEFAULT_SYNONYMS))
    parser.add_argument("--lifecycle-log", default=str(DEFAULT_LIFECYCLE_LOG))
    parser.add_argument("--disable-layout", action="store_true", help="Disable LayoutLM/Detectron based layout cues")
    parser.add_argument("--disable-ocr", action="store_true", help="Disable PaddleOCR fallbacks")
    parser.add_argument("--no-slot-mapper", action="store_true", help="Skip slot inference based on the intent dataset")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing artefacts")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    in_dir = Path(args.in_dir)
    structured_dir = Path(args.structured_out)
    sections_dir = Path(args.sections_out)

    slot_mapper: Optional[SlotMapper] = None
    if not args.no_slot_mapper:
        try:
            slot_mapper = SlotMapper(Path(args.intent_dataset), Path(args.synonyms))
        except FileNotFoundError as exc:
            LOGGER.warning("Slot mapper initialisation skipped: %s", exc)
            slot_mapper = None

    parser = HybridPDFParser(
        enable_layout_model=not args.disable_layout,
        enable_ocr=not args.disable_ocr,
    )
    tracker = LifecycleTracker(Path(args.lifecycle_log))

    pdfs = list(iter_pdfs(in_dir))
    if not pdfs:
        LOGGER.warning("No PDF files found under %s", in_dir)
        return

    for pdf in pdfs:
        structured_file = structured_dir / f"{pdf.stem}.structured.json"
        sections_file = sections_dir / f"{pdf.stem}.sections.jsonl"
        if not args.overwrite and structured_file.exists():
            LOGGER.info("Skipping %s (existing structured artefact)", pdf.name)
            continue
        LOGGER.info("Parsing %s", pdf.name)
        result = parser.parse(pdf, slot_mapper=slot_mapper)
        write_structured(result, structured_file)
        write_sections(result, sections_file)
        tracker.update(result.doc_id, parsed_at=result.parsed_at, metadata_ready_at=result.metadata_ready_at)


if __name__ == "__main__":
    main()
