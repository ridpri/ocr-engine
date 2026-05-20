from __future__ import annotations

import re

from ocr_engine.parsers.ktp import parse_ktp_text
from ocr_engine.parsers.stnk import parse_stnk_text
from ocr_engine.schemas import DocumentResult


SUPPORTED_DOCUMENT_TYPES = {"KTP", "STNK", "AUTO", None}
FIXED_DOCUMENT_TYPES = {"KTP", "STNK"}
DEFAULT_PREPARE_MAX_SIDE = 1280
STNK_PREPARE_MAX_SIDE = 1200
STNK_RETRY_PREPARE_MAX_SIDE = DEFAULT_PREPARE_MAX_SIDE
STNK_REQUIRED_RETRY_FIELDS = ("nomor_polisi", "nama_pemilik", "tahun_pembuatan", "nomor_rangka", "nomor_mesin")
STNK_MARKER_PATTERNS = [
    r"SURAT\s+TANDA\s+NOMOR\s+KENDARAAN",
    r"TANDA\s+BUKTI\s+PELUNASAN",
    r"\bNO\.?\s*POL(?:ISI)?\b",
    r"\bNO\.?\s*RANGKA\b",
    r"\bNOMOR\s+RANGKA\b",
    r"\bNO\.?\s*MESIN\b",
    r"\bNOMOR\s+MESIN\b",
    r"\bNOMOR\s*MES\w*\b",
    r"\bNOMOR\s+BPKB\b",
    r"\bNAMA\s+PEMILIK\b",
    r"\bTNKB\b",
    r"\bSWDKLLJ\b",
    r"\bBBN\s*-?\s*KB\b",
    r"\bMERK\s*/?\s*TYPE\b",
    r"\bKENDARAAN\s+KHUSUS\b",
    r"\bBERLAKU\s+SAMP\w*\b",
]
KTP_MARKER_PATTERNS = [
    r"\bPROVINSI\b",
    r"\bKABUPATEN\b",
    r"\bKOTA\b",
    r"\bKECAMATAN\b",
    r"\bKEL\s*/?\s*DESA\b",
    r"\bTEMPAT\s*/?\s*TGL\s+LAHIR\b",
    r"\bTEMPAT\s*/?\s*TANGGAL\s+LAHIR\b",
    r"\bJENIS\s+KELAMIN\b",
    r"\bAGAMA\b",
    r"\bSTATUS\s+PERKAWINAN\b",
    r"\bBERLAKU\s+HINGGA\b",
]


def detect_document_type(raw_text: str) -> str:
    text = raw_text.upper()
    if any(re.search(pattern, text) for pattern in STNK_MARKER_PATTERNS):
        return "STNK"
    has_ktp_marker = any(re.search(pattern, text) for pattern in KTP_MARKER_PATTERNS)
    if has_ktp_marker and ("NIK" in text or re.search(r"\b\d{16}\b", re.sub(r"\D", "", text))):
        return "KTP"
    if any(marker in text for marker in ["PROVINSI", "KABUPATEN", "KECAMATAN", "KEL/DESA"]):
        return "KTP"
    return "UNKNOWN"


def parse_document_text(raw_text: str, document_type_hint: str | None = None) -> DocumentResult:
    hint = document_type_hint.upper() if document_type_hint else "AUTO"
    if hint not in {"KTP", "STNK", "AUTO"}:
        raise ValueError("document_type_hint must be KTP, STNK, or AUTO")

    document_type = hint if hint != "AUTO" else detect_document_type(raw_text)
    if document_type == "KTP":
        return parse_ktp_text(raw_text)
    if document_type == "STNK":
        return parse_stnk_text(raw_text)

    ktp_result = parse_ktp_text(raw_text)
    stnk_result = parse_stnk_text(raw_text)
    best = max([ktp_result, stnk_result], key=_result_score)
    best.warnings.append("document_type:auto_guess")
    return best


def select_prepare_max_side(document_type_hint: str | None = None) -> int:
    hint = document_type_hint.upper() if document_type_hint else "AUTO"
    if hint == "STNK":
        return STNK_PREPARE_MAX_SIDE
    return DEFAULT_PREPARE_MAX_SIDE


def choose_parse_document_type(raw_text: str, document_type_hint: str | None = None) -> tuple[str, str]:
    hint = document_type_hint.upper() if document_type_hint else "AUTO"
    if hint not in {"KTP", "STNK", "AUTO"}:
        raise ValueError("document_type_hint must be KTP, STNK, or AUTO")

    detected_type = detect_document_type(raw_text)
    if hint in FIXED_DOCUMENT_TYPES and detected_type in FIXED_DOCUMENT_TYPES and detected_type != hint:
        return detected_type, detected_type
    return hint, detected_type


def should_run_ktp_nik_fallback(requested_document_type: str | None, parsed_document_type: str) -> bool:
    requested = requested_document_type.upper() if requested_document_type else "AUTO"
    parsed = parsed_document_type.upper()
    return parsed == "KTP" and requested in {"AUTO", "KTP"}


def should_retry_stnk_highres(
    requested_document_type: str | None,
    parsed: DocumentResult,
    assessment: dict,
) -> bool:
    requested = requested_document_type.upper() if requested_document_type else "AUTO"
    if requested != "STNK" or parsed.document_type != "STNK":
        return False

    reason_codes = set(assessment.get("reason_codes") or [])
    if "document_type_mismatch" in reason_codes or "screen_or_desktop_capture" in reason_codes:
        return False

    if assessment.get("decision") == "approved_for_auto":
        return False

    return any(
        parsed.fields[field_name].status in {"missing", "invalid"}
        for field_name in STNK_REQUIRED_RETRY_FIELDS
        if field_name in parsed.fields
    )


def document_result_score(parsed: DocumentResult, assessment: dict | None = None) -> float:
    decision = (assessment or {}).get("decision")
    decision_score = {
        "approved_for_auto": 100.0,
        "needs_review": 25.0,
        "rejected_input": -100.0,
    }.get(decision, 0.0)

    score = decision_score
    for field_name, field in parsed.fields.items():
        is_required = field_name in STNK_REQUIRED_RETRY_FIELDS if parsed.document_type == "STNK" else False
        if field.status == "ok":
            score += 12.0 if is_required else 2.0
            score += field.confidence
        elif field.status == "invalid":
            score -= 8.0 if is_required else 2.0
        elif field.status == "missing":
            score -= 5.0 if is_required else 0.5
    score -= len(parsed.warnings) * 2.0
    return score


def build_input_assessment(
    raw_text: str,
    parsed: DocumentResult,
    expected_document_type: str | None,
    detected_document_type: str,
    quality: dict | None = None,
) -> dict:
    expected = expected_document_type.upper() if expected_document_type else "AUTO"
    reason_codes: list[str] = []

    if expected in FIXED_DOCUMENT_TYPES and detected_document_type in FIXED_DOCUMENT_TYPES and detected_document_type != expected:
        reason_codes.append("document_type_mismatch")

    if expected in FIXED_DOCUMENT_TYPES and detected_document_type == "UNKNOWN" and parsed.needs_review:
        reason_codes.append("document_type_unknown")

    if detect_screen_or_desktop_capture(raw_text):
        reason_codes.append("screen_or_desktop_capture")

    for flag in (quality or {}).get("flags", []):
        if flag not in reason_codes:
            reason_codes.append(flag)

    for warning in parsed.warnings:
        if warning not in reason_codes:
            reason_codes.append(warning)

    rejected = "document_type_mismatch" in reason_codes or (
        "screen_or_desktop_capture" in reason_codes and parsed.needs_review
    ) or (
        "document_type_unknown" in reason_codes and parsed.needs_review
    )
    quality_review_flags = {"screen_or_desktop_capture", "document_too_small", "blur_detected", "low_text_density"}
    needs_review = rejected or parsed.needs_review or bool(quality_review_flags.intersection(reason_codes))

    if rejected:
        decision = "rejected_input"
    elif needs_review:
        decision = "needs_review"
    else:
        decision = "approved_for_auto"

    return {
        "decision": decision,
        "can_auto_publish": decision == "approved_for_auto",
        "expected_document_type": expected,
        "detected_document_type": detected_document_type,
        "reason_codes": reason_codes,
        "message": _assessment_message(decision, reason_codes, expected, detected_document_type),
    }


def detect_screen_or_desktop_capture(raw_text: str) -> bool:
    upper = raw_text.upper()
    markers = [
        "TYPE HERE",
        "TOSHIBA",
        "DRIVE",
        "MANAGE",
        ".JPG",
        ".JPEG",
        ".PNG",
        "100%",
        "WINDOWS",
        "ZOOM",
    ]
    return sum(1 for marker in markers if marker in upper) >= 2


def _assessment_message(decision: str, reason_codes: list[str], expected: str, detected: str) -> str:
    if "document_type_mismatch" in reason_codes:
        return f"Dokumen yang diupload terdeteksi {detected}, bukan {expected}. Mohon upload dokumen yang sesuai."
    if "document_type_unknown" in reason_codes:
        return f"Jenis dokumen tidak bisa dipastikan sebagai {expected}. Mohon upload ulang dokumen yang sesuai dan jelas."
    if "screen_or_desktop_capture" in reason_codes and decision == "rejected_input":
        return "Foto terlihat seperti foto layar atau viewer. Mohon upload foto dokumen asli secara langsung."
    if "screen_or_desktop_capture" in reason_codes:
        return "Foto terindikasi dari layar, sehingga hasil tidak boleh auto-terbit dan perlu review."
    if decision == "needs_review":
        return "Data berhasil diekstrak sebagian, tetapi perlu review karena ada field wajib atau kualitas yang belum memenuhi syarat."
    return "Dokumen memenuhi syarat untuk auto-processing."


def _result_score(result: DocumentResult) -> float:
    ok_fields = sum(1 for field in result.fields.values() if field.status == "ok")
    confidence = sum(field.confidence for field in result.fields.values())
    return ok_fields * 2 + confidence - len(result.warnings) * 1.5
