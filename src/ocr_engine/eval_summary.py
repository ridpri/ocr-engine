from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
import math
from typing import Any


def summarize_records(records: Iterable[dict[str, Any]]) -> dict:
    rows = list(records)
    decisions: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    reason_codes: Counter[str] = Counter()
    quality_flags: Counter[str] = Counter()
    auto_publish: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    stnk_usage_classes: Counter[str] = Counter()
    stnk_usage_reasons: Counter[str] = Counter()
    document_types: Counter[str] = Counter()
    field_status: dict[str, Counter[str]] = defaultdict(Counter)
    processing_times: list[float] = []
    stnk_structure_scores: list[float] = []
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
        statuses[str(row.get("status", "unknown"))] += 1
        document_types[row.get("document_type", "UNKNOWN")] += 1
        assessment = row.get("input_assessment") or {}
        decisions[assessment.get("decision", "unknown")] += 1
        reason_codes.update(assessment.get("reason_codes") or [])
        auto_publish[str(bool(assessment.get("can_auto_publish"))).lower()] += 1
        warnings.update(row.get("warnings") or [])
        if row.get("stnk_usage_class"):
            stnk_usage_classes[str(row.get("stnk_usage_class"))] += 1
        stnk_usage_reasons.update(row.get("stnk_usage_reasons") or [])
        for flag in (row.get("quality") or {}).get("flags") or []:
            quality_flags[flag] += 1
        for field_name, field in (row.get("fields") or {}).items():
            field_status[field_name][field.get("status", "unknown")] += 1
        _append_number(processing_times, row.get("processing_time_ms"))
        _append_number(stnk_structure_scores, row.get("stnk_structure_score"))
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
        "statuses": dict(statuses),
        "decisions": dict(decisions),
        "reason_codes": dict(reason_codes),
        "auto_publish": dict(auto_publish),
        "warnings": dict(warnings),
        "quality_flags": dict(quality_flags),
        "stnk_usage_classes": dict(stnk_usage_classes),
        "stnk_usage_reasons": dict(stnk_usage_reasons),
        "field_status": {field: dict(statuses) for field, statuses in field_status.items()},
        "processing_time_ms": _number_summary(processing_times),
        "stnk_structure_score": _number_summary(stnk_structure_scores),
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
        "slowest_records": _slowest_records(rows),
    }


def _append_number(values: list[float], value: Any) -> None:
    if isinstance(value, (int, float)):
        values.append(float(value))


def _number_summary(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "avg": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": round(sum(values) / len(values), 2),
        "p50": _percentile(sorted_values, 50),
        "p95": _percentile(sorted_values, 95),
        "p99": _percentile(sorted_values, 99),
    }


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return round(sorted_values[0], 2)
    rank = (percentile / 100) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return round(sorted_values[lower], 2)
    weight = rank - lower
    value = sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight
    return round(value, 2)


def _slowest_records(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    ranked = sorted(
        (row for row in rows if isinstance(row.get("processing_time_ms"), (int, float))),
        key=lambda row: float(row.get("processing_time_ms")),
        reverse=True,
    )
    return [_slow_record_summary(row) for row in ranked[:limit]]


def _slow_record_summary(row: dict[str, Any]) -> dict[str, Any]:
    assessment = row.get("input_assessment") or {}
    ocr = row.get("ocr") or {}
    preprocess = ocr.get("preprocess") or {}
    timings = ocr.get("timings") or {}
    attempts = timings.get("attempts") or []
    return {
        "file": row.get("file"),
        "document_type": row.get("document_type"),
        "processing_time_ms": row.get("processing_time_ms"),
        "decision": assessment.get("decision"),
        "reason_codes": assessment.get("reason_codes") or [],
        "quality_flags": (row.get("quality") or {}).get("flags") or [],
        "retry_count": preprocess.get("retry_count"),
        "selected_max_side": preprocess.get("selected_max_side"),
        "pipeline_total_ms": timings.get("total_ms"),
        "ocr_ms_total": round(
            sum(float(attempt.get("ocr_ms", 0)) for attempt in attempts if isinstance(attempt.get("ocr_ms"), (int, float))),
            2,
        ),
        "nik_fallback_ms": timings.get("nik_fallback_ms"),
    }
