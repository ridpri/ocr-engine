from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any


def summarize_records(records: Iterable[dict[str, Any]]) -> dict:
    rows = list(records)
    decisions: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    quality_flags: Counter[str] = Counter()
    document_types: Counter[str] = Counter()
    field_status: dict[str, Counter[str]] = defaultdict(Counter)
    processing_times: list[float] = []
    token_counts: list[float] = []
    retry_counts: list[float] = []
    selected_max_sides: list[float] = []
    quality_scores: list[float] = []
    pipeline_total_times: list[float] = []
    prepare_times: list[float] = []
    ocr_times: list[float] = []
    parse_times: list[float] = []
    quality_times: list[float] = []
    assessment_times: list[float] = []
    attempt_total_times: list[float] = []
    nik_fallback_times: list[float] = []

    for row in rows:
        document_types[row.get("document_type", "UNKNOWN")] += 1
        assessment = row.get("input_assessment") or {}
        decisions[assessment.get("decision", "unknown")] += 1
        warnings.update(row.get("warnings") or [])
        for flag in (row.get("quality") or {}).get("flags") or []:
            quality_flags[flag] += 1
        for field_name, field in (row.get("fields") or {}).items():
            field_status[field_name][field.get("status", "unknown")] += 1
        _append_number(processing_times, row.get("processing_time_ms"))
        ocr = row.get("ocr") or {}
        preprocess = ocr.get("preprocess") or {}
        _append_number(token_counts, ocr.get("token_count"))
        _append_number(retry_counts, preprocess.get("retry_count"))
        _append_number(selected_max_sides, preprocess.get("selected_max_side"))
        timings = ocr.get("timings") or {}
        _append_number(pipeline_total_times, timings.get("total_ms"))
        _append_number(nik_fallback_times, timings.get("nik_fallback_ms"))
        for attempt in timings.get("attempts") or []:
            _append_number(prepare_times, attempt.get("prepare_ms"))
            _append_number(ocr_times, attempt.get("ocr_ms"))
            _append_number(parse_times, attempt.get("parse_ms"))
            _append_number(quality_times, attempt.get("quality_ms"))
            _append_number(assessment_times, attempt.get("assessment_ms"))
            _append_number(attempt_total_times, attempt.get("total_ms"))
        _append_number(quality_scores, ((row.get("quality") or {}).get("metrics") or {}).get("overall_score"))

    return {
        "total": len(rows),
        "document_types": dict(document_types),
        "decisions": dict(decisions),
        "warnings": dict(warnings),
        "quality_flags": dict(quality_flags),
        "field_status": {field: dict(statuses) for field, statuses in field_status.items()},
        "processing_time_ms": _number_summary(processing_times),
        "ocr_tokens": _number_summary(token_counts),
        "ocr_retry_count": _number_summary(retry_counts),
        "selected_max_side": _number_summary(selected_max_sides),
        "quality_score": _number_summary(quality_scores),
        "stage_timings_ms": {
            "pipeline_total": _number_summary(pipeline_total_times),
            "prepare": _number_summary(prepare_times),
            "ocr": _number_summary(ocr_times),
            "parse": _number_summary(parse_times),
            "quality": _number_summary(quality_times),
            "assessment": _number_summary(assessment_times),
            "attempt_total": _number_summary(attempt_total_times),
            "nik_fallback": _number_summary(nik_fallback_times),
        },
    }


def _append_number(values: list[float], value: Any) -> None:
    if isinstance(value, (int, float)):
        values.append(float(value))


def _number_summary(values: list[float]) -> dict:
    if not values:
        return {"min": None, "max": None, "avg": None}
    return {
        "min": min(values),
        "max": max(values),
        "avg": round(sum(values) / len(values), 2),
    }
