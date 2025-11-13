"""Hybrid PDF parser that combines layout cues, OCR fallbacks and semantic tagging."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import fitz  # PyMuPDF
import numpy as np

from .schema import ChunkType, DocumentChunk, DocumentParseResult, HeadingInfo

if TYPE_CHECKING:  # pragma: no cover - optional dependency hints
    from .slot_mapper import SlotMapper

LOGGER = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _norm_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\t", " ")
    text = " ".join(text.split())
    text = text.replace("保 费", "保费").replace("年 金", "年金")
    return text.strip()


def _looks_like_heading(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("第") and stripped.endswith("章"):
        return True
    if stripped.startswith(tuple("一二三四五六七八九十")) and stripped.endswith("、"):
        return True
    if stripped[:2].lower() in {"ch", "se", "ap"} and stripped.lower().startswith(("chapter", "section", "appendix")):
        return True
    return False


def _tokenise(text: str) -> List[str]:
    if not text:
        return []
    tokens: List[str] = []
    buff = ""
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            if buff:
                tokens.append(buff)
                buff = ""
            tokens.append(char)
        elif char.isalnum():
            buff += char
        else:
            if buff:
                tokens.append(buff)
                buff = ""
    if buff:
        tokens.append(buff)
    return tokens


@dataclass
class _Line:
    text: str
    bbox: Tuple[float, float, float, float]
    y0: float
    y1: float
    avg_size: float
    x_center: float
    block_type: int


class HybridPDFParser:
    """Parse PDF documents into structured chunks with optional VDU/OCR support."""

    def __init__(
        self,
        *,
        enable_layout_model: bool = True,
        enable_ocr: bool = True,
        layout_model: str | None = None,
        ocr_lang: str = "ch",
        column_margin: float = 0.08,
    ) -> None:
        self.enable_layout_model = enable_layout_model
        self.enable_ocr = enable_ocr
        self.layout_model_name = layout_model or "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x"
        self.ocr_lang = ocr_lang
        self.column_margin = column_margin
        self._layout_predictor = None
        self._ocr = None
        self._bootstrap_optional_models()

    # Optional dependency wiring -------------------------------------------------
    def _bootstrap_optional_models(self) -> None:
        if self.enable_layout_model:
            try:
                import layoutparser as lp  # type: ignore

                self._layout_predictor = lp.Detectron2LayoutModel(
                    self.layout_model_name,
                    extra_config={"MODEL.ROI_HEADS.SCORE_THRESH_TEST": 0.5},
                )
                LOGGER.info("Layout model initialised: %s", self.layout_model_name)
            except Exception as exc:  # pragma: no cover - optional
                LOGGER.warning("Layout model unavailable (%s); falling back to heuristic layout.", exc)
                self._layout_predictor = None
        if self.enable_ocr:
            try:
                from paddleocr import PaddleOCR  # type: ignore

                self._ocr = PaddleOCR(use_angle_cls=True, lang=self.ocr_lang, show_log=False)
                LOGGER.info("PaddleOCR initialised for lang=%s", self.ocr_lang)
            except Exception as exc:  # pragma: no cover - optional
                LOGGER.warning("PaddleOCR unavailable (%s); OCR fallbacks disabled.", exc)
                self._ocr = None

    # Core parsing ----------------------------------------------------------------
    def parse(self, pdf_path: Path | str, *, slot_mapper: "SlotMapper" | None = None) -> DocumentParseResult:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)
        doc = fitz.open(pdf_path)
        doc_id = pdf_path.stem
        parsed_at = _now_iso()
        size_threshold = self._estimate_heading_threshold(doc)
        chunks: List[DocumentChunk] = []
        heading_stack: List[Tuple[int, str, str]] = []
        global_index = 0

        for page_idx, page in enumerate(doc, start=1):
            page_width = page.rect.width
            page_height = page.rect.height
            lines = self._page_to_lines(page)
            columns = self._split_columns(lines, page_width)
            layout_predictions = self._predict_layout(page)
            for column_index, column_lines in enumerate(columns):
                paragraphs = self._merge_lines(column_lines)
                for order_in_column, para in enumerate(paragraphs):
                    text = para.text.strip()
                    if not text:
                        continue
                    heading_flag, level = self._is_heading(text, para.avg_size, size_threshold)
                    parent_ids: List[str] = []
                    if heading_flag:
                        while heading_stack and heading_stack[-1][0] >= level:
                            heading_stack.pop()
                        chunk_level = level if level > 0 else 1
                    else:
                        chunk_level = heading_stack[-1][0] if heading_stack else 0
                    if heading_flag:
                        heading_path = [item[1] for item in heading_stack] + [text]
                    else:
                        heading_path = [item[1] for item in heading_stack]
                    parent_ids = [item[2] for item in heading_stack]
                    global_index += 1
                    chunk_id = f"{doc_id}-{page_idx:03d}-{global_index:04d}"
                    if heading_flag:
                        heading_stack.append((chunk_level, text, chunk_id))
                    heading_info = HeadingInfo(
                        level=chunk_level,
                        path=heading_path,
                        is_heading=heading_flag,
                        parent_ids=parent_ids,
                    )
                    chunk_type = self._infer_chunk_type(para.bbox, heading_flag, layout_predictions)
                    anchors = {
                        "page": page_idx,
                        "bbox": [round(coord, 2) for coord in para.bbox],
                        "page_width": round(float(page_width), 2),
                        "page_height": round(float(page_height), 2),
                    }
                    metadata = {
                        "avg_font_size": round(float(para.avg_size), 2),
                        "column_index": column_index,
                        "order_in_column": order_in_column,
                        "reading_order": global_index,
                        "source": "pymupdf",
                    }
                    slot_matches: List[str] = []
                    slots: Dict[str, str] = {}
                    if slot_mapper is not None:
                        slots, slot_matches = slot_mapper.infer_slots(text)
                        if slot_matches:
                            metadata["slot_matches"] = slot_matches
                    tokens = _tokenise(text)
                    chunk = DocumentChunk(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        page_no=page_idx,
                        chunk_type=chunk_type,
                        text=text,
                        bbox=[round(coord, 2) for coord in para.bbox],
                        tokens=tokens,
                        heading=heading_info,
                        slots=slots,
                        anchors=anchors,
                        layout={
                            "heading_level": heading_info.level,
                            "heading_path": heading_info.path,
                            "column": column_index,
                            "order": order_in_column,
                        },
                        metadata=metadata,
                    )
                    chunks.append(chunk)
            # optionally capture OCR artefacts for image-only pages
            if self._ocr:
                global_index = self._attach_ocr_chunks(
                    doc_id,
                    page_idx,
                    page,
                    chunks,
                    heading_stack,
                    global_index,
                    slot_mapper,
                )

        doc.close()
        parser_info = {
            "name": "hybrid-pdf-parser",
            "layout_model": self.layout_model_name if self._layout_predictor else None,
            "ocr": bool(self._ocr),
            "column_margin": self.column_margin,
        }
        result = DocumentParseResult(
            doc_id=doc_id,
            source_path=str(pdf_path),
            parsed_at=parsed_at,
            metadata_ready_at=parsed_at,
            parser_info=parser_info,
            chunks=chunks,
        )
        return result

    # Helper methods --------------------------------------------------------------
    def _estimate_heading_threshold(self, doc: fitz.Document) -> float:
        sizes: List[float] = []
        for page in doc:
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block.get("type", 0) != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        sizes.append(float(span.get("size", 10)))
        if not sizes:
            return 14.0
        percentile = float(np.percentile(sizes, 85))
        median = float(np.median(sizes))
        return max(percentile, median + 1.0)

    def _page_to_lines(self, page: fitz.Page) -> List[_Line]:
        data = page.get_text("dict")
        lines: List[_Line] = []
        for block in data.get("blocks", []):
            block_type = block.get("type", 0)
            if block_type not in (0, 1):  # text or image as OCR candidate
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(span.get("text", "") for span in spans)
                if not text.strip():
                    continue
                x0 = min(span.get("bbox", [0, 0, 0, 0])[0] for span in spans)
                y0 = min(span.get("bbox", [0, 0, 0, 0])[1] for span in spans)
                x1 = max(span.get("bbox", [0, 0, 0, 0])[2] for span in spans)
                y1 = max(span.get("bbox", [0, 0, 0, 0])[3] for span in spans)
                sizes = [span.get("size", 10) for span in spans]
                lines.append(
                    _Line(
                        text=text,
                        bbox=(x0, y0, x1, y1),
                        y0=y0,
                        y1=y1,
                        avg_size=float(np.mean(sizes) if sizes else 10.0),
                        x_center=(x0 + x1) / 2,
                        block_type=block_type,
                    )
                )
        return lines

    def _split_columns(self, lines: Sequence[_Line], page_width: float) -> List[List[_Line]]:
        if not lines:
            return [[]]
        xs = np.array([line.x_center for line in lines])
        if len(xs) < 40:
            return [sorted(lines, key=lambda ln: (ln.y0, ln.bbox[0]))]
        mid = page_width / 2
        margin = page_width * self.column_margin
        left = [ln for ln in lines if ln.x_center < mid - margin]
        right = [ln for ln in lines if ln.x_center > mid + margin]
        if len(left) > 0.2 * len(lines) and len(right) > 0.2 * len(lines):
            left.sort(key=lambda ln: (ln.y0, ln.bbox[0]))
            right.sort(key=lambda ln: (ln.y0, ln.bbox[0]))
            return [left, right]
        return [sorted(lines, key=lambda ln: (ln.y0, ln.bbox[0]))]

    @dataclass
    class _Paragraph:
        text: str
        bbox: Tuple[float, float, float, float]
        avg_size: float

    def _merge_lines(self, lines: Sequence[_Line], gap_factor: float = 1.2) -> List["HybridPDFParser._Paragraph"]:
        paragraphs: List[List[_Line]] = []
        buffer: List[_Line] = []
        last_line: Optional[_Line] = None
        for line in lines:
            text = line.text.strip()
            if not text:
                continue
            current_heading = _looks_like_heading(text)
            if last_line is None:
                buffer = [line]
            else:
                gap = line.y0 - last_line.y1
                avg_size = (line.avg_size + last_line.avg_size) / 2 or 10
                threshold = gap_factor * avg_size
                prev_text = last_line.text.strip()
                join_hint = not prev_text.endswith(("。", "；", ";", "!", "？", "?", ")"))
                if gap < threshold or join_hint or current_heading:
                    buffer.append(line)
                else:
                    paragraphs.append(buffer)
                    buffer = [line]
            last_line = line
        if buffer:
            paragraphs.append(buffer)
        merged: List[HybridPDFParser._Paragraph] = []
        for group in paragraphs:
            text = " ".join(line.text.rstrip("-‐—~") for line in group)
            text = text.replace("- ", "").replace("‐ ", "").replace("— ", "").replace("~ ", "")
            text = _norm_text(text)
            x0 = min(line.bbox[0] for line in group)
            y0 = min(line.bbox[1] for line in group)
            x1 = max(line.bbox[2] for line in group)
            y1 = max(line.bbox[3] for line in group)
            avg_size = float(np.mean([line.avg_size for line in group]) if group else 10.0)
            merged.append(HybridPDFParser._Paragraph(text=text, bbox=(x0, y0, x1, y1), avg_size=avg_size))
        return merged

    def _is_heading(self, text: str, avg_size: float, threshold: float) -> Tuple[bool, int]:
        stripped = text.strip()
        if not stripped:
            return False, 0
        if avg_size >= threshold:
            return True, 1
        if _looks_like_heading(stripped):
            return True, 2
        return False, 0

    def _predict_layout(self, page: fitz.Page) -> List[Dict[str, Any]]:
        if not self._layout_predictor:
            return []
        try:
            from PIL import Image  # type: ignore

            pix = page.get_pixmap()
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            layout = self._layout_predictor.detect(np.array(image))
            predictions: List[Dict[str, Any]] = []
            for block in layout:
                bbox = [float(block.block.x_1), float(block.block.y_1), float(block.block.x_2), float(block.block.y_2)]
                predictions.append({"type": block.type, "bbox": bbox, "score": float(getattr(block, "score", 1.0))})
            return predictions
        except Exception as exc:  # pragma: no cover - optional dependency path
            LOGGER.warning("Layout prediction failed: %s", exc)
            return []

    def _infer_chunk_type(
        self,
        bbox: Tuple[float, float, float, float],
        is_heading: bool,
        layout_predictions: Sequence[Dict[str, Any]],
    ) -> ChunkType:
        if is_heading:
            return ChunkType.HEADING
        for prediction in layout_predictions:
            p_type = (prediction.get("type") or "").lower()
            if p_type not in {"table", "figure"}:
                continue
            if self._iou(bbox, prediction.get("bbox", (0, 0, 0, 0))) > 0.35:
                if p_type == "table":
                    return ChunkType.TABLE
                if p_type == "figure":
                    return ChunkType.FIGURE
        return ChunkType.PARAGRAPH

    @staticmethod
    def _iou(a: Sequence[float], b: Sequence[float]) -> float:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        inter_x0 = max(ax0, bx0)
        inter_y0 = max(ay0, by0)
        inter_x1 = min(ax1, bx1)
        inter_y1 = min(ay1, by1)
        if inter_x0 >= inter_x1 or inter_y0 >= inter_y1:
            return 0.0
        inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
        a_area = (ax1 - ax0) * (ay1 - ay0)
        b_area = (bx1 - bx0) * (by1 - by0)
        union = a_area + b_area - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _attach_ocr_chunks(
        self,
        doc_id: str,
        page_idx: int,
        page: fitz.Page,
        chunks: List[DocumentChunk],
        heading_stack: List[Tuple[int, str, str]],
        global_index: int,
        slot_mapper: "SlotMapper" | None,
    ) -> int:
        if not self._ocr:
            return global_index
        try:
            from PIL import Image  # type: ignore

            pix = page.get_pixmap()
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            np_image = np.array(image)
            ocr_result = self._ocr.ocr(np_image, cls=True)  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency path
            LOGGER.debug("Skipping OCR for page %s: %s", page_idx, exc)
            return global_index
        heading_path = [item[1] for item in heading_stack]
        parent_ids = [item[2] for item in heading_stack]
        for line in ocr_result or []:
            for box, text, confidence in line:
                text = (text or "").strip()
                if not text:
                    continue
                bbox = [float(coord) for coord in (box[0][0], box[0][1], box[2][0], box[2][1])]
                if any(self._bbox_contains(chunk.bbox, bbox) for chunk in chunks if chunk.page_no == page_idx):
                    continue
                global_index += 1
                chunk_id = f"{doc_id}-{page_idx:03d}-ocr{global_index:04d}"
                slots: Dict[str, str] = {}
                slot_matches: List[str] = []
                if slot_mapper is not None:
                    slots, slot_matches = slot_mapper.infer_slots(text)
                metadata = {
                    "confidence": float(confidence) if confidence is not None else None,
                    "source": "paddleocr",
                }
                if slot_matches:
                    metadata["slot_matches"] = slot_matches
                chunk = DocumentChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    page_no=page_idx,
                    chunk_type=ChunkType.PARAGRAPH,
                    text=text,
                    bbox=[round(coord, 2) for coord in bbox],
                    tokens=_tokenise(text),
                    heading=HeadingInfo(level=heading_stack[-1][0] if heading_stack else 0, path=heading_path, is_heading=False, parent_ids=parent_ids),
                    slots=slots,
                    anchors={"page": page_idx, "bbox": [round(coord, 2) for coord in bbox]},
                    layout={"heading_path": heading_path, "column": None, "order": None},
                    metadata=metadata,
                )
                chunks.append(chunk)
        return global_index

    @staticmethod
    def _bbox_contains(container: Sequence[float], candidate: Sequence[float]) -> bool:
        cx0, cy0, cx1, cy1 = container
        dx0, dy0, dx1, dy1 = candidate
        return cx0 <= dx0 and cy0 <= dy0 and cx1 >= dx1 and cy1 >= dy1
