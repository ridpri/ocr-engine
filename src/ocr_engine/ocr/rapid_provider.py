from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ocr_engine.ocr.base import OcrDependencyError, OcrResult, OcrToken


class RapidOcrProvider:
    def __init__(self, engine_factory: Callable[[], Any] | None = None) -> None:
        self._engine_factory = engine_factory
        self._engine: Any | None = None

    def extract_text(self, image_path: str) -> OcrResult:
        engine = self._get_engine()
        try:
            raw = engine(image_path)
        except Exception as exc:
            raise OcrDependencyError(f"RapidOCR failed to process the image. Root cause: {exc}") from exc
        return normalize_rapid_output(raw)

    def warm_up(self) -> None:
        self._get_engine()

    def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        try:
            if self._engine_factory is not None:
                self._engine = self._engine_factory()
            else:
                from rapidocr import RapidOCR

                self._engine = RapidOCR()
        except Exception as exc:
            raise OcrDependencyError(
                "RapidOCR is not ready. Install rapidocr and onnxruntime, then allow "
                f"RapidOCR to download its ONNX model files on first run. Root cause: {exc}"
            ) from exc
        return self._engine


def normalize_rapid_output(raw: Any) -> OcrResult:
    tokens: list[OcrToken] = []

    texts = getattr(raw, "txts", None)
    scores = getattr(raw, "scores", None)
    boxes = getattr(raw, "boxes", None)
    if texts is not None:
        for index, text in enumerate(texts):
            text_value = _text_value(text)
            if not text_value:
                continue
            confidence = _indexed_float(scores, index)
            bbox = _to_jsonable(boxes[index]) if boxes is not None and index < len(boxes) else None
            tokens.append(OcrToken(text=text_value, confidence=confidence, bbox=bbox))
        return OcrResult(
            raw_text="\n".join(token.text for token in tokens if token.text),
            tokens=tokens,
            provider="rapidocr",
        )

    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                bbox, text, confidence = item[:3]
                text_value = _text_value(text)
                if text_value:
                    tokens.append(OcrToken(text=text_value, confidence=_safe_float(confidence), bbox=_to_jsonable(bbox)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                text, confidence = item[:2]
                text_value = _text_value(text)
                if text_value:
                    tokens.append(OcrToken(text=text_value, confidence=_safe_float(confidence), bbox=None))

    return OcrResult(
        raw_text="\n".join(token.text for token in tokens if token.text),
        tokens=tokens,
        provider="rapidocr",
    )


def _indexed_float(values: Any, index: int) -> float:
    if values is None or index >= len(values):
        return 0.0
    return _safe_float(values[index])


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
