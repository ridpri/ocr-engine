from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time

from ocr_engine.image_utils import (
    prepare_ktp_fast_image,
    prepare_image,
    prepare_stnk_fast_roi_image,
    prepare_stnk_full_page_image,
    prepare_stnk_official_roi_image,
)
from ocr_engine.nik_image_fallback import repair_ktp_nik_from_image
from ocr_engine.ocr.base import OcrProvider, OcrResult
from ocr_engine.parsers.ktp_layout import apply_ktp_layout_hints
from ocr_engine.quality import analyze_image_preflight, analyze_image_quality
from ocr_engine.schemas import DocumentResult
from ocr_engine.service import (
    build_input_assessment,
    choose_parse_document_type,
    document_result_score,
    parse_document_text,
    select_prepare_max_side,
    should_retry_stnk_highres,
    should_run_ktp_nik_fallback,
)


STNK_OFFICIAL_ROI_MAX_SIDE = 1200
STNK_FAST_ROI_MAX_SIDE = 820
STNK_FAST_ROI_RIGHT_RATIO = 0.78
STNK_FULL_PAGE_MAX_SIDE = 1600
KTP_FAST_MAX_SIDE = 496
KTP_FAST_RIGHT_RATIO = 0.72
KTP_FAST_BOTTOM_RATIO = 1.0
KTP_FULL_PAGE_RETRY_MAX_SIDE = 1280


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
    run_nik_fallback: bool = True,
    force_strategy: str | None = None,
) -> OcrPipelineResult:
    pipeline_started = time.perf_counter()
    mode = _normalize_processing_mode(processing_mode)
    preflight_started = time.perf_counter()
    preflight_quality = analyze_image_preflight(raw_path)
    preflight_ms = _elapsed_ms(preflight_started)
    if _should_reject_before_ocr(requested_document_type, mode, preflight_quality):
        return _preflight_rejected_result(
            raw_path,
            requested_document_type,
            preflight_quality,
            preflight_ms,
            _elapsed_ms(pipeline_started),
            mode,
        )

    attempts: list[_OcrAttempt] = []
    first_strategy = force_strategy or _first_attempt_strategy(requested_document_type, mode)
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
            preflight_quality,
        )
    )

    first = attempts[0]
    if (
        force_strategy is None
        and
        mode == "accurate"
        and should_retry_stnk_highres(requested_document_type, first.parsed, first.assessment)
    ):
        attempts.append(
            _run_attempt(
                1,
                provider,
                raw_path,
                workdir / "prepared-stnk-retry.jpg",
                requested_document_type,
                STNK_FULL_PAGE_MAX_SIDE,
                "stnk_full_page",
                preflight_quality,
            )
        )
    elif (
        force_strategy is None
        and mode == "fast"
        and _should_retry_ktp_full_page(requested_document_type, first)
    ):
        attempts.append(
            _run_attempt(
                1,
                provider,
                raw_path,
                workdir / "prepared-ktp-retry.jpg",
                requested_document_type,
                _env_int("OCR_KTP_RETRY_MAX_SIDE", KTP_FULL_PAGE_RETRY_MAX_SIDE),
                "full_page",
                preflight_quality,
            )
        )

    selected = max(attempts, key=lambda attempt: document_result_score(attempt.parsed, attempt.assessment))
    nik_fallback = {"attempted": False, "passes": 0, "value": None}
    nik_fallback_started = time.perf_counter()
    if _should_attempt_ktp_nik_image_fallback(run_nik_fallback, requested_document_type, selected):
        nik_fallback = repair_ktp_nik_from_image(provider, raw_path, selected.parsed, workdir / "nik-fallback")
        if nik_fallback.get("value"):
            selected.assessment = build_input_assessment(
                selected.ocr_result.raw_text,
                selected.parsed,
                requested_document_type,
                selected.detected_type,
                selected.quality,
                selected.ocr_result.provider,
            )
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
            "preflight_ms": preflight_ms,
            "selected_attempt_index": selected.index,
            "nik_fallback_ms": nik_fallback_ms,
            "attempts": [dict(attempt.timings) for attempt in attempts],
        },
        processing_mode=mode,
    )


def _should_reject_before_ocr(requested_document_type: str | None, processing_mode: str, quality: dict) -> bool:
    requested = requested_document_type.upper() if requested_document_type else "AUTO"
    if requested not in {"KTP", "STNK"} or processing_mode != "fast":
        return False
    flags = set(quality.get("flags") or [])
    return bool(flags.intersection({"document_too_small", "blur_detected"}))


def _preflight_rejected_result(
    raw_path: Path,
    requested_document_type: str,
    quality: dict,
    preflight_ms: float,
    total_ms: float,
    mode: str,
) -> OcrPipelineResult:
    quality = {
        "image": dict(quality.get("image") or {}),
        "flags": [*quality.get("flags", []), "pre_ocr_rejected"],
        "metrics": dict(quality.get("metrics") or {}),
    }
    parse_hint = requested_document_type.upper() if requested_document_type.upper() in {"KTP", "STNK"} else "AUTO"
    parsed = parse_document_text("", document_type_hint=parse_hint)
    assessment = build_input_assessment(
        "",
        parsed,
        requested_document_type,
        "UNKNOWN",
        quality=quality,
    )
    return OcrPipelineResult(
        ocr_result=OcrResult(raw_text="", tokens=[], provider="preflight"),
        parsed=parsed,
        assessment=assessment,
        quality=quality,
        nik_fallback={"attempted": False, "passes": 0, "value": None},
        preprocess={
            "selected_max_side": None,
            "retry_count": 0,
            "attempts": [],
            "pre_ocr_rejected": True,
            "source": str(raw_path),
        },
        timings={
            "total_ms": total_ms,
            "preflight_ms": preflight_ms,
            "selected_attempt_index": None,
            "nik_fallback_ms": 0.0,
            "attempts": [],
        },
        processing_mode=mode,
    )


def _should_attempt_ktp_nik_image_fallback(
    run_nik_fallback: bool,
    requested_document_type: str | None,
    selected: _OcrAttempt,
) -> bool:
    if not run_nik_fallback:
        return False
    if not selected.ocr_result.tokens:
        return False
    if selected.detected_type == "UNKNOWN" and selected.assessment.get("decision") == "rejected_input":
        return False
    return should_run_ktp_nik_fallback(requested_document_type, selected.parsed.document_type)


def _should_retry_ktp_full_page(requested_document_type: str | None, attempt: _OcrAttempt) -> bool:
    requested = requested_document_type.upper() if requested_document_type else "AUTO"
    if requested != "KTP":
        return False
    if not attempt.ocr_result.tokens:
        return False
    assessment = attempt.assessment
    if assessment.get("decision") == "approved_for_auto":
        return False
    reasons = set(assessment.get("reason_codes") or [])
    retry_prefixes = (
        "missing_required:nama",
        "ktp_auto_missing:nama",
        "ktp_auto_missing:berlaku_hingga",
        "ktp_auto_missing:kecamatan",
        "ktp_auto_missing:kelurahan_desa",
        "ktp_auto_missing:alamat",
        "ktp_auto_low_confidence:nama",
        "ktp_suspicious_field:tempat_tanggal_lahir",
        "ktp_nik_ttl_mismatch",
    )
    return any(reason.startswith(retry_prefixes) for reason in reasons)


def _run_attempt(
    index: int,
    provider: OcrProvider,
    raw_path: Path,
    prepared_path: Path,
    requested_document_type: str,
    max_side: int,
    strategy: str,
    preflight_quality: dict | None = None,
) -> _OcrAttempt:
    attempt_started = time.perf_counter()
    stage_started = time.perf_counter()
    if strategy == "stnk_fast_roi":
        prepare_stnk_fast_roi_image(
            raw_path,
            prepared_path,
            max_side=max_side,
            right_ratio=_env_float("OCR_STNK_FAST_RIGHT_RATIO", STNK_FAST_ROI_RIGHT_RATIO),
        )
    elif strategy == "stnk_official_roi":
        prepare_stnk_official_roi_image(raw_path, prepared_path, max_side=max_side)
    elif strategy == "stnk_full_page":
        prepare_stnk_full_page_image(raw_path, prepared_path, max_side=max_side)
    elif strategy == "ktp_fast":
        prepare_ktp_fast_image(
            raw_path,
            prepared_path,
            max_side=max_side,
            right_ratio=_env_float("OCR_KTP_FAST_RIGHT_RATIO", KTP_FAST_RIGHT_RATIO),
            bottom_ratio=_env_float("OCR_KTP_FAST_BOTTOM_RATIO", KTP_FAST_BOTTOM_RATIO),
        )
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
    quality = analyze_image_quality(raw_path, ocr_result, preflight_quality=preflight_quality)
    quality_ms = _elapsed_ms(stage_started)

    stage_started = time.perf_counter()
    assessment = build_input_assessment(
        ocr_result.raw_text,
        parsed,
        requested_document_type,
        detected_type,
        quality=quality,
        ocr_provider=ocr_result.provider,
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


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _normalize_processing_mode(processing_mode: str | None) -> str:
    mode = (processing_mode or "accurate").lower()
    if mode not in {"fast", "accurate"}:
        raise ValueError("processing_mode must be fast or accurate")
    return mode


def _first_attempt_strategy(requested_document_type: str | None, processing_mode: str) -> str:
    requested = requested_document_type.upper() if requested_document_type else "AUTO"
    if requested == "KTP" and processing_mode == "fast":
        return "ktp_fast"
    if requested == "STNK" and processing_mode == "fast":
        return "stnk_official_roi"
    if requested == "STNK":
        return "stnk_official_roi"
    return "full_page"


def _first_attempt_max_side(requested_document_type: str | None, processing_mode: str, strategy: str) -> int:
    requested = requested_document_type.upper() if requested_document_type else "AUTO"
    if strategy == "stnk_official_roi":
        if requested == "STNK" and processing_mode == "fast":
            return _env_int("OCR_STNK_FAST_MAX_SIDE", STNK_FAST_ROI_MAX_SIDE)
        return STNK_OFFICIAL_ROI_MAX_SIDE
    if strategy == "stnk_fast_roi":
        return _env_int("OCR_STNK_FAST_MAX_SIDE", STNK_FAST_ROI_MAX_SIDE)
    if requested == "KTP" and processing_mode == "fast":
        return _env_int("OCR_KTP_FAST_MAX_SIDE", KTP_FAST_MAX_SIDE)
    if strategy == "stnk_full_page":
        return STNK_FULL_PAGE_MAX_SIDE
    return select_prepare_max_side(requested_document_type)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if 0.1 <= parsed <= 1.0 else default
