"""Fusion strategy combining graph hits, vector search, and recency boosts."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .graph_backend import query as graph_query
from .recency import load_recency, recency_score
from .regulatory import regulatory_score

RECENCY_META = load_recency()


def run(filters: Dict[str, Any], query_texts: List[str], topk: int = 20) -> Dict[str, Any]:
    """Execute the fusion strategy returning results and debug metadata."""

    graph_response = graph_query(filters, query_texts, topk=topk)
    graph_hits = graph_response.get("hits", [])
    plan = graph_response.get("plan", {})
    debug = graph_response.get("debug", {})

    results: List[Dict[str, Any]] = []
    regulatory_debug: List[Dict[str, Any]] = []
    recency_debug: List[Dict[str, Any]] = []
    for raw in graph_hits:
        result, reg_debug = _normalise_hit(raw)
        if reg_debug:
            regulatory_debug.append(reg_debug)
        rec_debug = _apply_recency(result)
        if rec_debug:
            recency_debug.append(rec_debug)
        results.append(result)

    results.sort(key=lambda item: item.get("score_total", 0.0), reverse=True)
    if regulatory_debug or recency_debug:
        fusion_debug = debug.setdefault("fusion", {})
        if regulatory_debug:
            fusion_debug["regulatory"] = regulatory_debug
        if recency_debug:
            fusion_debug["recency"] = recency_debug
    return {"plan": plan, "results": results[:topk], "debug": debug}


def _normalise_hit(hit: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Convert graph/raw hits into unified payload expected by rerankers."""

    anchors = hit.get("anchors") or []
    if anchors:
        anchor = anchors[0]
        doc_id = anchor.get("doc_id")
        page_no = anchor.get("page_no")
        bbox = anchor.get("bbox")
        preview = (anchor.get("text") or "")[:240]
        chunk_id = anchor.get("chunk_id") or hit.get("chunk_id")
    else:
        doc_id = hit.get("doc_id")
        page_no = hit.get("page_no")
        bbox = hit.get("bbox")
        preview = (hit.get("preview") or hit.get("text") or "")[:240]
        chunk_id = hit.get("chunk_id")

    score_breakdown = hit.get("score_breakdown")
    breakdown = dict(score_breakdown) if isinstance(score_breakdown, dict) else {}
    if breakdown:
        score_graph = float(breakdown.get("graph", 0.0))
        score_semantic = float(breakdown.get("semantic", 0.0))
        raw_reg = breakdown.get("regulatory", 0.0)
        score_regulatory = float(raw_reg.get("score", raw_reg) if isinstance(raw_reg, dict) else raw_reg)
    else:
        score_graph = float(hit.get("score_graph", 1.0 if anchors else 0.0))
        score_semantic = float(hit.get("score_semantic", 0.0))
        score_regulatory = float(hit.get("score_regulatory", 0.0))

    score_total = score_graph + score_semantic + score_regulatory

    extra = dict(hit.get("extra", {}))
    regulatory_debug: Optional[Dict[str, Any]] = None
    if doc_id:
        reg_score = regulatory_score(doc_id, score_total)
        if reg_score:
            score_regulatory = round(score_regulatory + reg_score.bonus, 4)
            score_total = score_graph + score_semantic + score_regulatory
            reg_debug = reg_score.to_debug()
            reg_debug["score"] = score_regulatory
            extra["regulatory"] = reg_debug
            regulatory_debug = {"doc_id": doc_id, **reg_debug}
    if regulatory_debug is None and isinstance(breakdown.get("regulatory"), dict):
        reg_debug = dict(breakdown.get("regulatory", {}))
    elif regulatory_debug is not None:
        reg_debug = dict(extra.get("regulatory", {}))
    else:
        reg_debug = None

    breakdown["graph"] = round(score_graph, 4)
    breakdown["semantic"] = round(score_semantic, 4)
    if reg_debug:
        reg_debug.setdefault("score", score_regulatory)
        breakdown["regulatory"] = reg_debug
    else:
        breakdown["regulatory"] = round(score_regulatory, 4)

    payload = {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "page_no": page_no,
        "bbox": bbox,
        "preview": preview,
        "edge": hit.get("edge"),
        "source": hit.get("source") or hit.get("node_type"),
        "score_graph": score_graph,
        "score_semantic": score_semantic,
        "score_regulatory": score_regulatory,
        "score_total": round(score_total, 4),
        "anchors": anchors,
        "extra": extra,
        "score_breakdown": breakdown,
    }
    return payload, regulatory_debug


def _apply_recency(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    doc_id = result.get("doc_id")
    if not doc_id:
        result["score_recency"] = 0.0
        result["recency_flags"] = []
        return None
    recency_meta = RECENCY_META.get(doc_id, {})
    rec_score = recency_score(recency_meta)
    result["score_recency"] = rec_score.bonus
    result["recency_flags"] = rec_score.flags
    result.setdefault("extra", {})["recency"] = rec_score.to_debug()
    breakdown = result.setdefault("score_breakdown", {})
    rec_debug = {**rec_score.to_debug(), "score": rec_score.bonus}
    breakdown["recency"] = rec_debug
    result["score_total"] = round(result.get("score_total", 0.0) + rec_score.bonus, 4)
    if not recency_meta:
        return None
    debug_payload = {"doc_id": doc_id, **rec_debug, "metadata": recency_meta}
    return debug_payload
