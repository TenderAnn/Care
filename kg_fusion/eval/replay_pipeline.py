"""Phase 5 evaluation harness for GraphRAG + freshrank integration."""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from fastapi.testclient import TestClient

import sys

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[2]
FRESHRANK_PKG = REPO_ROOT / "freshrank"
for candidate in (str(REPO_ROOT), str(FRESHRANK_PKG)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from freshrank.service.pipeline import RankingPipeline
from kg_fusion.app.main import app

CASE_PATH = Path(__file__).resolve().parent / "cases" / "phase5_pipeline.jsonl"
META_PATH = Path(__file__).resolve().parent / "cases" / "doc_metadata.json"
REPORT_PATH = Path(__file__).resolve().parent / "reports" / "phase5_eval.json"


@dataclass
class EvalCase:
    query: str
    expected_doc_ids: List[str]
    topk: int = 5
    intent: str | None = None
    notes: str | None = None


@dataclass
class CaseResult:
    case: EvalCase
    kg_hits: List[str]
    rerank_hits: List[str]
    kg_precision: float
    kg_recall: float
    rerank_precision: float
    rerank_recall: float
    breakdown_ok: bool


def load_cases(path: Path) -> List[EvalCase]:
    cases: List[EvalCase] = []
    if not path.exists():
        raise FileNotFoundError(f"Evaluation cases not found: {path}")
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            payload = json.loads(line)
            cases.append(
                EvalCase(
                    query=payload["query"],
                    expected_doc_ids=list(payload.get("expected_doc_ids", [])),
                    topk=int(payload.get("topk", 5)),
                    intent=payload.get("intent"),
                    notes=payload.get("notes"),
                )
            )
    return cases


def load_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def precision_recall(retrieved: Sequence[str], expected: Sequence[str]) -> tuple[float, float]:
    if not retrieved:
        if expected:
            return 0.0, 0.0
        return 0.0, 1.0
    retrieved_set = list(dict.fromkeys(retrieved))
    expected_set = set(expected)
    true_pos = [doc for doc in retrieved_set if doc in expected_set]
    precision = len(true_pos) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(true_pos) / len(expected_set) if expected_set else 1.0
    return precision, recall


def build_rerank_input(results: Sequence[Dict[str, Any]], metadata: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for hit in results:
        doc_id = hit.get("doc_id")
        if not doc_id:
            continue
        meta = metadata.get(doc_id, {})
        payload = {
            "doc_id": doc_id,
            "chunk_id": hit.get("chunk_id"),
            "relevance": float(hit.get("score_total", 0.0)),
            "publish_date": meta.get("publish_date"),
            "effective_date": meta.get("effective_date"),
            "expired": meta.get("expired", False),
        }
        if meta.get("regulatory_hits"):
            payload["regulatory_hits"] = meta["regulatory_hits"]
        elif meta.get("regulatory_tags"):
            payload["regulatory_tags"] = meta["regulatory_tags"]
        docs.append(payload)
    return docs


def verify_breakdown(items: Iterable[Dict[str, Any]], rules: Dict[str, Any]) -> bool:
    ok = True
    expiry_cfg = (rules or {}).get("expiry_handling", {})
    auto_demotion = expiry_cfg.get("auto_demotion_score")
    min_allowed = expiry_cfg.get("min_allowed_score", 0.0)
    tol = 1e-3
    for doc in items:
        breakdown = doc.get("score_breakdown", {})
        base = float(breakdown.get("base", 0.0))
        recency_multiplier = float(breakdown.get("recency_multiplier", 1.0))
        regulatory_multiplier = float(breakdown.get("regulatory_multiplier", 1.0))
        regulatory_bonus = float(breakdown.get("regulatory_bonus", 0.0))
        computed = base * recency_multiplier
        computed = computed * regulatory_multiplier + regulatory_bonus
        final_score = float(doc.get("final_score", 0.0))
        diff = computed - final_score
        if abs(diff) <= tol:
            continue
        if diff >= 0 and auto_demotion is not None and abs(final_score - float(auto_demotion)) <= tol:
            continue
        if diff >= 0 and min_allowed is not None and abs(final_score - float(min_allowed)) <= tol:
            continue
        ok = False
        break
    return ok


def evaluate(cases: Sequence[EvalCase], metadata: Dict[str, Dict[str, Any]], *, topk_override: int | None = None, verbose: bool = True) -> Dict[str, Any]:
    client = TestClient(app)
    pipeline = RankingPipeline()
    case_results: List[CaseResult] = []

    for case in cases:
        topk = topk_override or case.topk
        payload = {"text": case.query}
        if case.intent:
            payload["intent"] = case.intent
        response = client.post("/kg/query", json=payload)
        response.raise_for_status()
        body = response.json()
        kg_results = body.get("results", [])
        kg_hits = [item.get("doc_id") for item in kg_results[:topk] if item.get("doc_id")]
        rerank_input = build_rerank_input(kg_results[:topk], metadata)
        rerank_results = pipeline.rank(rerank_input)
        rerank_hits = [item.get("doc_id") for item in rerank_results[:topk] if item.get("doc_id")]

        kg_precision, kg_recall = precision_recall(kg_hits, case.expected_doc_ids)
        rerank_precision, rerank_recall = precision_recall(rerank_hits, case.expected_doc_ids)
        breakdown_ok = verify_breakdown(rerank_results, pipeline.rules)

        case_results.append(
            CaseResult(
                case=case,
                kg_hits=kg_hits,
                rerank_hits=rerank_hits,
                kg_precision=kg_precision,
                kg_recall=kg_recall,
                rerank_precision=rerank_precision,
                rerank_recall=rerank_recall,
                breakdown_ok=breakdown_ok,
            )
        )

        if verbose:
            print(f"Query: {case.query}")
            if case.notes:
                print(f"  Notes: {case.notes}")
            print(f"  KG hits: {kg_hits}")
            print(f"  KG precision={kg_precision:.2f} recall={kg_recall:.2f}")
            print(f"  Rerank hits: {rerank_hits}")
            print(f"  Rerank precision={rerank_precision:.2f} recall={rerank_recall:.2f}")
            print(f"  Score breakdown consistent: {breakdown_ok}")
            print("  ---")

    agg = summarise(case_results)
    agg["cases"] = [
        {
            "query": result.case.query,
            "expected_doc_ids": result.case.expected_doc_ids,
            "topk": result.case.topk,
            "kg_hits": result.kg_hits,
            "rerank_hits": result.rerank_hits,
            "kg_precision": result.kg_precision,
            "kg_recall": result.kg_recall,
            "rerank_precision": result.rerank_precision,
            "rerank_recall": result.rerank_recall,
            "breakdown_ok": result.breakdown_ok,
            "notes": result.case.notes,
        }
        for result in case_results
    ]
    return agg


def summarise(results: Sequence[CaseResult]) -> Dict[str, Any]:
    if not results:
        return {"kg_precision": 0.0, "kg_recall": 0.0, "rerank_precision": 0.0, "rerank_recall": 0.0, "breakdown_ok": True}
    kg_precisions = [item.kg_precision for item in results]
    kg_recalls = [item.kg_recall for item in results]
    rr_precisions = [item.rerank_precision for item in results]
    rr_recalls = [item.rerank_recall for item in results]
    breakdown_ok = all(item.breakdown_ok for item in results)
    return {
        "kg_precision": statistics.mean(kg_precisions) if kg_precisions else 0.0,
        "kg_recall": statistics.mean(kg_recalls) if kg_recalls else 0.0,
        "rerank_precision": statistics.mean(rr_precisions) if rr_precisions else 0.0,
        "rerank_recall": statistics.mean(rr_recalls) if rr_recalls else 0.0,
        "breakdown_ok": breakdown_ok,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay /kg/query → /rerank for regression checks")
    parser.add_argument("--cases", type=Path, default=CASE_PATH, help="Path to JSONL evaluation cases")
    parser.add_argument("--metadata", type=Path, default=META_PATH, help="Document metadata overrides")
    parser.add_argument("--output", type=Path, default=REPORT_PATH, help="Where to write the aggregated report")
    parser.add_argument("--topk", type=int, default=None, help="Override the per-case top-k")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-case logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    metadata = load_metadata(args.metadata)
    report = evaluate(cases, metadata, topk_override=args.topk, verbose=not args.quiet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAverages → KG precision={report['kg_precision']:.2f} recall={report['kg_recall']:.2f}; "
          f"Rerank precision={report['rerank_precision']:.2f} recall={report['rerank_recall']:.2f}; "
          f"Score breakdown OK={report['breakdown_ok']}")
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
