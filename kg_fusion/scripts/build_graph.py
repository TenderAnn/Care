"""Build graph artefacts and GraphRAG indices from structured parses."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List

from kg_fusion.app.graph import GraphBuildArtifacts, GraphBuilder, GraphStore
from kg_fusion.app.ingest import DocumentParseResult, LifecycleTracker
from kg_fusion.paths import data_path

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRUCTURED_IN = data_path("structured")
DEFAULT_GRAPH_ROOT = data_path("structured")
DEFAULT_LIFECYCLE_LOG = REPO_ROOT / "freshrank" / "data" / "metadata" / "ingestion_events.jsonl"


def iter_structured(path: Path) -> Iterable[Path]:
    return sorted(p for p in path.glob("*.structured.json") if p.is_file())


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured-in", default=str(DEFAULT_STRUCTURED_IN))
    parser.add_argument("--graph-root", default=str(DEFAULT_GRAPH_ROOT))
    parser.add_argument("--lifecycle-log", default=str(DEFAULT_LIFECYCLE_LOG))
    parser.add_argument("--overwrite", action="store_true", help="Rewrite the graph store instead of appending")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on the number of documents processed")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_parse(path: Path) -> DocumentParseResult:
    with path.open(encoding="utf-8") as fp:
        payload = json.load(fp)
    return DocumentParseResult.from_dict(payload)


def main() -> None:
    args = parse_arguments()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    structured_dir = Path(args.structured_in)
    graph_root = Path(args.graph_root)
    tracker = LifecycleTracker(Path(args.lifecycle_log))
    builder = GraphBuilder()
    store = GraphStore(root=graph_root)

    structured_files = list(iter_structured(structured_dir))
    if not structured_files:
        LOGGER.warning("No structured artefacts found under %s", structured_dir)
        return

    artefacts: List[GraphBuildArtifacts] = []
    processed = 0
    for idx, structured_file in enumerate(structured_files, start=1):
        if args.limit and idx > args.limit:
            break
        LOGGER.info("Building graph for %s", structured_file.name)
        parse_result = load_parse(structured_file)
        artefact = builder.build(parse_result)
        if args.overwrite:
            artefacts.append(artefact)
        else:
            store.append(artefact)
        tracker.mark_served(parse_result.doc_id)
        processed += 1

    if args.overwrite and artefacts:
        store.save(artefacts)

    total = len(artefacts) if args.overwrite else processed
    LOGGER.info("Graph build completed. Processed %d documents", total)


if __name__ == "__main__":
    main()
