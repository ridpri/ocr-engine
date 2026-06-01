from __future__ import annotations

import inspect
import os
import threading
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
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
        self._inference_warmed = False
        self._engine_lock = threading.Lock()

    def extract_text(self, image_path: str) -> OcrResult:
        engine = self._get_engine()
        if hasattr(engine, "predict"):
            raw = engine.predict(image_path)
        elif hasattr(engine, "ocr"):
            try:
                raw = engine.ocr(image_path, cls=self.use_angle_cls)
            except TypeError:
                raw = engine.ocr(image_path)
        else:
            raise OcrDependencyError("PaddleOCR engine does not expose ocr() or predict().")
        return normalize_paddle_output(raw)

    def warm_up(self) -> None:
        engine = self._get_engine()
        if self._inference_warmed or not (hasattr(engine, "predict") or hasattr(engine, "ocr")):
            return

        with TemporaryDirectory() as tmpdir:
            for image_path in _write_warmup_images(Path(tmpdir)):
                self.extract_text(str(image_path))
        self._inference_warmed = True

    def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine

        with self._engine_lock:
            if self._engine is not None:
                return self._engine
            try:
                if self._engine_factory is not None:
                    self._engine = self._engine_factory()
                else:
                    enable_mkldnn = _env_bool("OCR_PADDLE_ENABLE_MKLDNN", default=False)
                    acceleration_flag = "1" if enable_mkldnn else "0"
                    os.environ.setdefault("FLAGS_use_mkldnn", acceleration_flag)
                    os.environ.setdefault("FLAGS_use_onednn", acceleration_flag)
                    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", acceleration_flag)
                    os.environ.setdefault("FLAGS_enable_pir_api", "0")
                    from paddleocr import PaddleOCR

                    with _optional_insecure_model_downloads():
                        try:
                            kwargs = build_paddle_kwargs(self.lang, self.use_angle_cls, self.preset)
                            self._engine = PaddleOCR(**kwargs)
                        except TypeError:
                            filtered_kwargs = _filter_constructor_kwargs(PaddleOCR, kwargs)
                            if filtered_kwargs and filtered_kwargs != kwargs:
                                try:
                                    self._engine = PaddleOCR(**filtered_kwargs)
                                except TypeError:
                                    self._engine = PaddleOCR(use_angle_cls=self.use_angle_cls, lang=self.lang)
                            else:
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
    enable_mkldnn = _env_bool("OCR_PADDLE_ENABLE_MKLDNN", default=False)
    cpu_threads = _env_int("OCR_PADDLE_CPU_THREADS", default=_default_cpu_threads())
    enable_hpi = _env_bool("OCR_PADDLE_ENABLE_HPI", default=False)
    kwargs: dict[str, Any] = {
        "lang": lang,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": use_angle_cls,
        "enable_mkldnn": enable_mkldnn,
        "cpu_threads": cpu_threads,
    }
    if enable_hpi:
        kwargs["enable_hpi"] = True
    if preset == "fast":
        kwargs.update(
            {
                "text_detection_model_name": "PP-OCRv5_mobile_det",
                "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
                "text_det_limit_side_len": _env_int("OCR_PADDLE_TEXT_DET_LIMIT", default=960),
                "text_det_limit_type": "max",
                "text_recognition_batch_size": _env_int("OCR_PADDLE_REC_BATCH_SIZE", default=8),
            }
        )
    return kwargs


def _default_cpu_threads() -> int:
    cpu_count = os.cpu_count() or 4
    if cpu_count <= 4:
        return max(1, cpu_count)
    return min(cpu_count - 1, 10)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


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


def _write_warmup_images(directory: Path) -> list[Path]:
    ktp_path = directory / "warmup-ktp-fast.jpg"
    stnk_path = directory / "warmup-stnk-fast.jpg"
    _write_warmup_image(
        ktp_path,
        (496, 340),
        [
            "PROVINSI JAWA BARAT",
            "KABUPATEN BEKASI",
            "NIK 3275081705690023",
            "Nama SYUKRI SE AK",
            "Tempat Tgl Lahir SIGLI 17-05-1969",
            "Alamat PERUM BUMIJATI",
            "RT RW 003 006",
            "Kel Desa JATIWARINGIN",
            "Kecamatan PONDOK GEDE",
            "Berlaku Hingga SEUMUR HIDUP",
        ],
    )
    _write_warmup_image(
        stnk_path,
        (720, 480),
        [
            "SURAT TANDA NOMOR KENDARAAN",
            "NO POLISI B 1234 ABC",
            "NAMA PEMILIK BUDI SANTOSO",
            "MERK TOYOTA",
            "TYPE AVANZA",
            "TAHUN PEMBUATAN 2020",
            "NO RANGKA MHRRU1860KJ302319",
            "NO MESIN L15Z61219016",
            "BERLAKU SAMPAI 01-01-2027",
        ],
    )
    return [ktp_path, stnk_path]


def _write_warmup_image(path: Path, size: tuple[int, int], lines: list[str]) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        draw.text((12, 12 + index * 28), line, fill="black")
    image.save(path, format="JPEG", quality=90)


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
        text_value = _text_value(text)
        if text_value:
            tokens.append(OcrToken(text=text_value, confidence=_safe_float(confidence), bbox=bbox))
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
            text_value = _text_value(text)
            if not text_value:
                continue
            confidence = _safe_float(scores[index]) if index < len(scores) else 0.0
            bbox = _to_jsonable(boxes[index]) if index < len(boxes) else None
            tokens.append(OcrToken(text=text_value, confidence=confidence, bbox=bbox))
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


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _filter_constructor_kwargs(constructor: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(constructor)
    except (TypeError, ValueError):
        return {}
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
