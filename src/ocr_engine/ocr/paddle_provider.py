from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from ocr_engine.ocr.base import OcrDependencyError, OcrResult, OcrToken


class PaddleOcrProvider:
    def __init__(
        self,
        engine_factory: Callable[[], Any] | None = None,
        lang: str = "en",
        use_angle_cls: bool = False,
        preset: str = "fast",
    ) -> None:
        self._engine_factory = engine_factory
        self._engine: Any | None = None
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.preset = preset

    def extract_text(self, image_path: str) -> OcrResult:
        engine = self._get_engine()
        if hasattr(engine, "ocr"):
            try:
                raw = engine.ocr(image_path, cls=self.use_angle_cls)
            except TypeError:
                raw = engine.ocr(image_path)
        elif hasattr(engine, "predict"):
            raw = engine.predict(image_path)
        else:
            raise OcrDependencyError("PaddleOCR engine does not expose ocr() or predict().")
        return normalize_paddle_output(raw)

    def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine

        try:
            if self._engine_factory is not None:
                self._engine = self._engine_factory()
            else:
                os.environ.setdefault("FLAGS_use_mkldnn", "0")
                os.environ.setdefault("FLAGS_use_onednn", "0")
                os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
                os.environ.setdefault("FLAGS_enable_pir_api", "0")
                from paddleocr import PaddleOCR

                with _optional_insecure_model_downloads():
                    try:
                        self._engine = PaddleOCR(**build_paddle_kwargs(self.lang, self.use_angle_cls, self.preset))
                    except TypeError:
                        self._engine = PaddleOCR(use_angle_cls=self.use_angle_cls, lang=self.lang)
        except Exception as exc:  # Paddle can raise import/runtime errors during model setup.
            raise OcrDependencyError(
                "PaddleOCR is not ready. Install dependencies from requirements.txt, "
                "make sure paddlepaddle is available for this Python version, and allow "
                "PaddleOCR to download its official model files on first run. If your "
                "network blocks model hosts or SSL verification, pre-download the models "
                f"or run on a network that can reach Paddle model sources. Root cause: {exc}"
            ) from exc

        return self._engine


def build_paddle_kwargs(lang: str = "en", use_angle_cls: bool = False, preset: str = "fast") -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "lang": lang,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": use_angle_cls,
    }
    if preset == "fast":
        kwargs.update(
            {
                "text_detection_model_name": "PP-OCRv5_mobile_det",
                "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
                "text_det_limit_side_len": 1280,
                "text_det_limit_type": "max",
                "text_recognition_batch_size": 4,
            }
        )
    return kwargs


@contextmanager
def _optional_insecure_model_downloads():
    if os.getenv("OCR_ENGINE_ALLOW_INSECURE_MODEL_DOWNLOADS") != "1":
        yield
        return

    import requests
    import urllib3

    original_request = requests.sessions.Session.request

    def request_with_verify_disabled(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("verify", False)
        return original_request(self, method, url, **kwargs)

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    requests.sessions.Session.request = request_with_verify_disabled
    try:
        yield
    finally:
        requests.sessions.Session.request = original_request


def normalize_paddle_output(raw: Any) -> OcrResult:
    tokens: list[OcrToken] = []
    _collect_tokens(raw, tokens)
    raw_text = "\n".join(token.text for token in tokens if token.text)
    return OcrResult(raw_text=raw_text, tokens=tokens, provider="paddleocr")


def _collect_tokens(node: Any, tokens: list[OcrToken]) -> None:
    if node is None:
        return

    if isinstance(node, dict):
        _collect_dict_tokens(node, tokens)
        return

    if _looks_like_legacy_line(node):
        bbox = node[0]
        text, confidence = node[1]
        tokens.append(OcrToken(text=str(text), confidence=float(confidence), bbox=bbox))
        return

    if isinstance(node, (list, tuple)):
        for item in node:
            _collect_tokens(item, tokens)


def _collect_dict_tokens(node: dict[str, Any], tokens: list[OcrToken]) -> None:
    texts = _first_present(node, ["rec_texts", "texts"], [])
    scores = _first_present(node, ["rec_scores", "scores"], [])
    boxes = _first_present(node, ["rec_boxes", "dt_polys", "boxes"], [])

    if _safe_len(texts) > 0:
        for index, text in enumerate(texts):
            confidence = float(scores[index]) if index < len(scores) else 0.0
            bbox = _to_jsonable(boxes[index]) if index < len(boxes) else None
            tokens.append(OcrToken(text=str(text), confidence=confidence, bbox=bbox))
        return

    for value in node.values():
        _collect_tokens(value, tokens)


def _looks_like_legacy_line(node: Any) -> bool:
    if not isinstance(node, (list, tuple)) or len(node) != 2:
        return False
    text_payload = node[1]
    return (
        isinstance(text_payload, (list, tuple))
        and len(text_payload) >= 2
        and isinstance(text_payload[0], str)
        and isinstance(text_payload[1], (int, float))
    )


def _first_present(node: dict[str, Any], keys: list[str], default: Any) -> Any:
    for key in keys:
        if key in node and node[key] is not None:
            return node[key]
    return default


def _safe_len(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
