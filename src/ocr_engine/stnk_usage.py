from __future__ import annotations

from typing import Any


STNK_WEB_REQUIRED_FIELDS = ("nomor_polisi", "nama_pemilik", "tahun_pembuatan", "nomor_rangka", "nomor_mesin")
WEB_STRUCTURE_SCORE_MIN = 0.7
BAD_INPUT_QUALITY_FLAGS = {"screen_or_desktop_capture", "document_too_small", "blur_detected", "low_text_density"}


def apply_stnk_web_usage_gate(record: dict[str, Any]) -> dict[str, Any]:
    document_type = str(record.get("document_type") or "").upper()
    if document_type != "STNK":
        return record

    usage_class = str(record.get("stnk_usage_class") or "").strip().lower()
    if not usage_class or usage_class == "web_usable":
        return record

    assessment = record.setdefault("input_assessment", {})
    reason_codes = assessment.setdefault("reason_codes", [])
    _append_unique(reason_codes, f"stnk_web_usage_class:{usage_class}")
    for reason in record.get("stnk_usage_reasons") or []:
        _append_unique(reason_codes, f"stnk_web_usage:{reason}")

    assessment["can_auto_publish"] = False
    if assessment.get("decision") == "approved_for_auto":
        if usage_class == "bad_input":
            assessment["decision"] = "rejected_input"
            assessment["message"] = "Foto STNK belum layak untuk pembelian web. Minta upload ulang."
        else:
            assessment["decision"] = "needs_review"
            assessment["message"] = "Data STNK terbaca, tetapi belum memenuhi syarat pembelian web otomatis."
    record["needs_review"] = True
    return record


def classify_stnk_record(record: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    assessment = record.get("input_assessment") or {}
    reason_codes = set(assessment.get("reason_codes") or [])
    document_type = str(record.get("document_type") or "UNKNOWN").upper()
    decision = assessment.get("decision")
    structure_score = _float_value(record.get("stnk_structure_score"))
    fields = record.get("fields") or {}
    quality_flags = set((record.get("quality") or {}).get("flags") or [])
    ocr_token_count = _float_value((record.get("ocr") or {}).get("token_count"))

    if record.get("status") is not None and record.get("status") != "ok":
        return "bad_input", ["processing_failed"]
    bad_quality_flags = sorted(quality_flags.intersection(BAD_INPUT_QUALITY_FLAGS))
    if bad_quality_flags:
        return "bad_input", [f"quality:{flag}" for flag in bad_quality_flags]
    if document_type != "STNK" and structure_score < 0.5:
        return "bad_input", ["not_stnk"]
    if "document_type_mismatch" in reason_codes:
        return "bad_input", ["document_type_rejected"]
    if structure_score < 0.1 and ocr_token_count < 15:
        return "bad_input", ["low_ocr_signal"]

    missing_required = [
        field_name
        for field_name in STNK_WEB_REQUIRED_FIELDS
        if (fields.get(field_name) or {}).get("status") != "ok"
    ]
    if missing_required:
        reasons.extend(f"field_not_ok:{field_name}" for field_name in missing_required)

    if decision == "rejected_input" and "document_type_unknown" in reason_codes:
        reasons.append("document_type_unknown")
    elif decision != "approved_for_auto":
        reasons.append("needs_review")
    if structure_score < WEB_STRUCTURE_SCORE_MIN and missing_required:
        reasons.append("structure_score_below_web_threshold")
    reasons.extend(f"quality:{flag}" for flag in sorted(quality_flags))

    if reasons:
        return "internal_only", reasons
    return "web_usable", []


def _float_value(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)
