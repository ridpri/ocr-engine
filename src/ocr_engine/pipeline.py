from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from PIL import Image, ImageOps

from ocr_engine.image_utils import prepare_image, prepare_stnk_fast_roi_image
from ocr_engine.nik_image_fallback import repair_ktp_nik_from_image
from ocr_engine.ocr.base import OcrProvider, OcrResult
from ocr_engine.parsers.ktp_layout import apply_ktp_layout_hints
from ocr_engine.quality import analyze_image_quality
from ocr_engine.schemas import DocumentResult
from ocr_engine.service import (
    STNK_RETRY_PREPARE_MAX_SIDE,
    build_input_assessment,
    choose_parse_document_type,
    document_result_score,
    parse_document_text,
    select_prepare_max_side,
    should_retry_stnk_highres,
    should_run_ktp_nik_fallback,
)


STNK_RETRY_SOURCE_HEADROOM_RATIO = 1.25
STNK_FAST_ROI_MAX_SIDE = 512
KTP_FAST_MAX_SIDE = 960


@dataclass(slots=True)
class OcrPipelineResult:
    ocr_result: OcrResult
    parsed: DocumentResult
    assessment: dict
    quality: dict
    nik_fallback: dict
    preprocess: dict
    timings: dict
    processing_mode: str


@dataclass(slots=True)
class _OcrAttempt:
    index: int
    max_side: int
    ocr_result: OcrResult
    parsed: DocumentResult
    detected_type: str
    assessment: dict
    quality: dict
    timings: dict
    strategy: str


def run_ocr_pipeline(
    provider: OcrProvider,
    raw_path: Path,
    requested_document_type: str,
    workdir: Path,
    processing_mode: str = "accurate",
) -> OcrPipelineResult:
    pipeline_started = time.perf_counter()
    mode = _normalize_processing_mode(processing_mode)
    attempts: list[_OcrAttempt] = []
    first_strategy = _first_attempt_strategy(requested_document_type, mode)
    first_max_side = _first_attempt_max_side(requested_document_type, mode, first_strategy)
    attempts.append(
        _run_attempt(
            0,
            provider,
            raw_path,
            workdir / "prepared.jpg",
            requested_document_type,
            first_max_side,
            first_strategy,
        )
    )

    first = attempts[0]
    if (
        mode == "accurate"
        and should_retry_stnk_highres(requested_document_type, first.parsed, first.assessment)
        and _has_stnk_retry_headroom(raw_path)
    ):
        attempts.append(
            _run_attempt(
                1,
                provider,
                raw_path,
                workdir / "prepared-stnk-retry.jpg",
                requested_document_type,
                STNK_RETRY_PREPARE_MAX_SIDE,
                "full_page",
            )
        )

    selected = max(attempts, key=lambda attempt: document_result_score(attempt.parsed, attempt.assessment))
    nik_fallback = {"attempted": False, "passes": 0, "value": None}
    nik_fallback_started = time.perf_counter()
    if should_run_ktp_nik_fallback(requested_document_type, selected.parsed.document_type):
        nik_fallback = repair_ktp_nik_from_image(provider, raw_path, selected.parsed, workdir / "nik-fallback")
    nik_fallback_ms = _elapsed_ms(nik_fallback_started)

    return OcrPipelineResult(
        ocr_result=selected.ocr_result,
        parsed=selected.parsed,
        assessment=selected.assessment,
        quality=selected.quality,
        nik_fallback=nik_fallback,
        preprocess={
            "selected_max_side": selected.max_side,
            "retry_count": max(0, len(attempts) - 1),
            "attempts": [_attempt_summary(attempt) for attempt in attempts],
        },
        timings={
            "total_ms": _elapsed_ms(pipeline_started),
            "selected_attempt_index": selected.index,
            "nik_fallback_ms": nik_fallback_ms,
            "attempts": [dict(attempt.timings) for attempt in attempts],
        },
        processing_mode=mode,
    )


def _run_attempt(
    index: int,
    provider: OcrProvider,
    raw_path: Path,
    prepared_path: Path,
    requested_document_type: str,
    max_side: int,
    strategy: str,
) -> _OcrAttempt:
    attempt_started = time.perf_counter()
    stage_started = time.perf_counter()
    if strategy == "stnk_fast_roi":
        prepare_stnk_fast_roi_image(raw_path, prepared_path, max_side=max_side)
    else:
        prepare_image(raw_path, prepared_path, max_side=max_side)
    prepare_ms = _elapsed_ms(stage_started)

    stage_started = time.perf_counter()
    ocr_result = provider.extract_text(str(prepared_path))
    ocr_ms = _elapsed_ms(stage_started)

    stage_started = time.perf_counter()
    parse_hint, detected_type = choose_parse_document_type(ocr_result.raw_text, requested_document_type)
    parsed = parse_document_text(ocr_result.raw_text, document_type_hint=parse_hint)
    apply_ktp_layout_hints(parsed, ocr_result.tokens)
    parse_ms = _elapsed_ms(stage_started)

    stage_started = time.perf_counter()
    quality = analyze_image_quality(raw_path, ocr_result)
    quality_ms = _elapsed_ms(stage_started)

    stage_started = time.perf_counter()
    assessment = build_input_assessment(
        ocr_result.raw_text,
        parsed,
        requested_document_type,
        detected_type,
        quality=quality,
    )
    assessment_ms = _elapsed_ms(stage_started)
    return _OcrAttempt(
        index=index,
        max_side=max_side,
        ocr_result=ocr_result,
        parsed=parsed,
        detected_type=detected_type,
        assessment=assessment,
        quality=quality,
        strategy=strategy,
        timings={
            "prepare_ms": prepare_ms,
            "ocr_ms": ocr_ms,
            "parse_ms": parse_ms,
            "quality_ms": quality_ms,
            "assessment_ms": assessment_ms,
            "total_ms": _elapsed_ms(attempt_started),
        },
    )


def _attempt_summary(attempt: _OcrAttempt) -> dict:
    return {
        "index": attempt.index,
        "max_side": attempt.max_side,
        "strategy": attempt.strategy,
        "document_type": attempt.parsed.document_type,
        "detected_document_type": attempt.detected_type,
        "decision": attempt.assessment.get("decision"),
        "warnings": list(attempt.parsed.warnings),
    }


def _has_stnk_retry_headroom(raw_path: Path) -> bool:
    with Image.open(raw_path) as image:
        image = ImageOps.exif_transpose(image)
        return max(image.size) > STNK_RETRY_PREPARE_MAX_SIDE * STNK_RETRY_SOURCE_HEADROOM_RATIO


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _normalize_processing_mode(processing_mode: str | None) -> str:
    mode = (processing_mode or "accurate").lower()
    if mode not in {"fast", "accurate"}:
        raise ValueError("processing_mode must be fast or accurate")
    return mode


def _first_attempt_strategy(requested_document_type: str | None, processing_mode: str) -> str:
    requested = requested_document_type.upper() if requested_document_type else "AUTO"
    if requested == "STNK" and processing_mode == "fast":
        return "stnk_fast_roi"
    return "full_page"


def _first_attempt_max_side(requested_document_type: str | None, processing_mode: str, strategy: str) -> int:
    requested = requested_document_type.upper() if requested_document_type else "AUTO"
    if strategy == "stnk_fast_roi":
        return STNK_FAST_ROI_MAX_SIDE
    if requested == "KTP" and processing_mode == "fast":
        return KTP_FAST_MAX_SIDE
    return select_prepare_max_side(requested_document_type)
