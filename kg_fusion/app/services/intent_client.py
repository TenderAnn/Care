from typing import Any, Dict, Optional

import httpx

from ..config import INTENT_BASE_URL

try:  # local fallback for offline evaluation
    from ..ingest.slot_mapper import SlotMapper
except Exception:  # pragma: no cover - optional dependency
    SlotMapper = None  # type: ignore

_SLOT_MAPPER: Optional[SlotMapper] = None


async def parse(text: str) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{INTENT_BASE_URL}/intent/parse", json={"text": text})
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return _fallback_parse(text)


async def rewrite(
    text: str, intent: Optional[str], slots: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"text": text}
    if intent:
        payload["intent"] = intent
    if slots:
        payload["slots"] = slots

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{INTENT_BASE_URL}/intent/rewrite", json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return _fallback_rewrite(payload)


def _fallback_parse(text: str) -> Dict[str, Any]:
    mapper = _ensure_mapper()
    slots: Dict[str, Any] = {}
    intent = None
    if mapper is not None:
        inferred, _ = mapper.infer_slots(text)
        if inferred:
            intent = inferred.pop("intent", None)
            slots.update(inferred)
    return {
        "intent": intent,
        "slots": slots,
        "confidence": 0.5,
        "threshold": 0.35,
        "need_clarify": False,
        "normalized": text,
    }


def _fallback_rewrite(payload: Dict[str, Any]) -> Dict[str, Any]:
    intent = payload.get("intent")
    slots = payload.get("slots") or {}
    base = payload.get("text")
    rewrites = []
    if isinstance(base, str) and base.strip():
        rewrites.append(base.strip())
    if intent and isinstance(intent, str):
        rewrites.insert(0, intent)
    return {"rewrites": rewrites[:3]}


def _ensure_mapper() -> Optional[SlotMapper]:
    global _SLOT_MAPPER
    if _SLOT_MAPPER is not None or SlotMapper is None:
        return _SLOT_MAPPER
    try:
        from .. import PACKAGE_ROOT

        dataset_path = PACKAGE_ROOT.parent / "intent_dataset_zh_insurance_v1.jsonl"
        synonyms_path = PACKAGE_ROOT.parent / "synonyms_insurance_zh.tsv"
        _SLOT_MAPPER = SlotMapper(dataset_path, synonyms_path)
    except Exception:
        _SLOT_MAPPER = None
    return _SLOT_MAPPER
