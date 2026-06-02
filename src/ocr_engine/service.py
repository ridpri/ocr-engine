from __future__ import annotations

from datetime import date
import os
import re

from ocr_engine.parsers.ktp import parse_ktp_text
from ocr_engine.parsers.stnk import parse_stnk_text
from ocr_engine.schemas import DocumentResult
from ocr_engine.validators import normalize_nik


SUPPORTED_DOCUMENT_TYPES = {"KTP", "STNK", "AUTO", None}
FIXED_DOCUMENT_TYPES = {"KTP", "STNK"}
DEFAULT_PREPARE_MAX_SIDE = 1280
STNK_PREPARE_MAX_SIDE = 1200
STNK_RETRY_PREPARE_MAX_SIDE = DEFAULT_PREPARE_MAX_SIDE
STNK_REQUIRED_RETRY_FIELDS = ("nomor_polisi", "nama_pemilik", "tahun_pembuatan", "nomor_rangka", "nomor_mesin")
KTP_AUTO_APPROVAL_FIELDS = (
    "nik",
    "nama",
    "tempat_tanggal_lahir",
    "jenis_kelamin",
    "alamat",
    "rt_rw",
    "kelurahan_desa",
    "kecamatan",
    "pekerjaan",
    "kewarganegaraan",
    "berlaku_hingga",
)
KTP_AUTO_MIN_CONFIDENCE = {
    "nama": 0.80,
}
KTP_NAME_BORDERLINE_CONFIDENCE = 0.70
KTP_REGION_LABEL_FRAGMENT_PATTERN = re.compile(
    r"\b(?:AGA[MN]\w*|K[EO]WARG\w*|WARGANEGARA\w*|STATUS|PEKER\w*|BERLAKU|HINGGA|NAMA|TEMPAT|TGL|LAHIR|JENIS|KELAMIN)\b",
    flags=re.IGNORECASE,
)
STNK_AUTO_APPROVAL_FIELDS = (
    "nomor_polisi",
    "nama_pemilik",
    "tahun_pembuatan",
    "nomor_rangka",
    "nomor_mesin",
)
STNK_AUTO_MIN_CONFIDENCE = {
    "nomor_polisi": 0.85,
    "nama_pemilik": 0.75,
}
STNK_SUSPICIOUS_FIELD_VALUES = {
    "nama_pemilik": {
        "A ALAMAT",
        "A ALLAMAT",
        "A LAMAT",
        "ALAMAT",
        "ALLAMAT",
        "LAMAT",
        "NAMA PEMILIK",
        "BN KENDARAAN BARU",
        "BN. KENDARAAN BARU",
    },
}
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
    ocr_provider: str | None = None,
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

    if _looks_like_unreliable_ktp_for_auto(raw_text, parsed):
        reason_codes.append("suspicious_ktp_output")

    if _provider_requires_review(ocr_provider):
        reason_codes.append(f"ocr_provider_needs_review:{ocr_provider}")

    for reason in _ktp_auto_approval_reason_codes(parsed):
        if reason not in reason_codes:
            reason_codes.append(reason)

    for reason in _stnk_auto_approval_reason_codes(parsed):
        if reason not in reason_codes:
            reason_codes.append(reason)

    rejected = "document_type_mismatch" in reason_codes or (
        "screen_or_desktop_capture" in reason_codes and parsed.needs_review
    ) or (
        "document_type_unknown" in reason_codes and parsed.needs_review
    ) or (
        "suspicious_ktp_output" in reason_codes
    )
    quality_review_flags = {"screen_or_desktop_capture", "document_too_small", "blur_detected", "low_text_density"}
    ktp_review = any(
        reason.startswith("ktp_auto_missing:")
        or reason.startswith("ktp_auto_low_confidence:")
        or reason.startswith("ktp_suspicious_field:")
        or reason
        in {
            "ktp_invalid_nik_birth_date",
            "ktp_nik_ttl_mismatch",
            "ktp_nik_gender_mismatch",
            "ktp_unusual_rt_rw",
        }
        for reason in reason_codes
    )
    stnk_review = any(
        reason.startswith("stnk_auto_missing:")
        or reason.startswith("stnk_auto_low_confidence:")
        or reason.startswith("stnk_suspicious_field:")
        for reason in reason_codes
    )
    needs_review = (
        rejected
        or parsed.needs_review
        or bool(quality_review_flags.intersection(reason_codes))
        or any(reason.startswith("ocr_provider_needs_review:") for reason in reason_codes)
        or ktp_review
        or stnk_review
    )

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
        " JPG",
        " JPEG",
        " PNG",
        "100%",
        "WINDOWS",
        "ZOOM",
    ]
    return sum(1 for marker in markers if marker in upper) >= 2


def _provider_requires_review(ocr_provider: str | None) -> bool:
    if (ocr_provider or "").lower() != "rapidocr":
        return False
    return os.getenv("OCR_RAPID_AUTO_PUBLISH", "").strip().lower() not in {"1", "true", "yes", "on"}


def _assessment_message(decision: str, reason_codes: list[str], expected: str, detected: str) -> str:
    if "document_type_mismatch" in reason_codes:
        return f"Dokumen yang diupload terdeteksi {detected}, bukan {expected}. Mohon upload dokumen yang sesuai."
    if "pre_ocr_rejected" in reason_codes:
        return "Foto terlalu kecil atau buram untuk diproses otomatis. Mohon upload ulang foto dokumen yang lebih jelas."
    if "document_type_unknown" in reason_codes:
        return f"Jenis dokumen tidak bisa dipastikan sebagai {expected}. Mohon upload ulang dokumen yang sesuai dan jelas."
    if "screen_or_desktop_capture" in reason_codes and decision == "rejected_input":
        return "Foto terlihat seperti foto layar atau viewer. Mohon upload foto dokumen asli secara langsung."
    if "screen_or_desktop_capture" in reason_codes:
        return "Foto terindikasi dari layar, sehingga hasil tidak boleh auto-terbit dan perlu review."
    if "suspicious_ktp_output" in reason_codes:
        return "Hasil baca KTP terlihat tidak andal. Mohon upload ulang foto KTP yang lebih jelas."
    if decision == "needs_review":
        return "Data berhasil diekstrak sebagian, tetapi perlu review karena ada field wajib atau kualitas yang belum memenuhi syarat."
    return "Dokumen memenuhi syarat untuk auto-processing."


def _ktp_auto_approval_reason_codes(parsed: DocumentResult) -> list[str]:
    if parsed.document_type != "KTP":
        return []

    reasons: list[str] = []
    for field_name in KTP_AUTO_APPROVAL_FIELDS:
        if not _field_ok(parsed, field_name):
            reasons.append(f"ktp_auto_missing:{field_name}")

    reasons.extend(_ktp_suspicious_field_reason_codes(parsed))

    nik = _field_text(parsed, "nik")
    ttl = _field_text(parsed, "tempat_tanggal_lahir")
    if nik and ttl:
        expected_date = _birth_date_from_nik(nik)
        actual_date = _birth_date_from_ttl(ttl)
        if expected_date is None:
            reasons.append("ktp_invalid_nik_birth_date")
        elif actual_date and expected_date != actual_date:
            reasons.append("ktp_nik_ttl_mismatch")

    gender = _field_text(parsed, "jenis_kelamin")
    if nik and gender:
        expected_gender = _gender_from_nik(nik)
        if expected_gender and gender not in {"", expected_gender}:
            reasons.append("ktp_nik_gender_mismatch")

    rt_rw = _field_text(parsed, "rt_rw")
    rt_match = re.fullmatch(r"(\d{3})/(\d{3})", rt_rw)
    if rt_match:
        rt_number = int(rt_match.group(1))
        rw_number = int(rt_match.group(2))
        if rt_number > 150 or rw_number > 250:
            reasons.append("ktp_unusual_rt_rw")

    for field_name, min_confidence in KTP_AUTO_MIN_CONFIDENCE.items():
        field = parsed.fields.get(field_name)
        if not (field and field.status == "ok" and field.value and field.confidence < min_confidence):
            continue
        if field_name == "nama" and _ktp_can_trust_borderline_name(parsed, reasons):
            continue
        reasons.append(f"ktp_auto_low_confidence:{field_name}")

    return reasons


def _ktp_suspicious_field_reason_codes(parsed: DocumentResult) -> list[str]:
    reasons: list[str] = []
    name = _field_text(parsed, "nama")
    if name and _ktp_name_has_suspicious_joined_token(name):
        reasons.append("ktp_suspicious_field:nama")

    for field_name in ("kelurahan_desa", "kecamatan"):
        value = _field_text(parsed, field_name)
        if value and KTP_REGION_LABEL_FRAGMENT_PATTERN.search(value):
            reasons.append(f"ktp_suspicious_field:{field_name}")

    ttl = _field_text(parsed, "tempat_tanggal_lahir")
    if ttl and _ktp_ttl_has_label_fragment(ttl):
        reasons.append("ktp_suspicious_field:tempat_tanggal_lahir")

    return reasons


def _ktp_name_has_suspicious_joined_token(value: str) -> bool:
    tokens = value.split()
    return len(tokens) == 1 and bool(re.fullmatch(r"[A-Z']{12,}", tokens[0]))


def _ktp_can_trust_borderline_name(parsed: DocumentResult, reasons: list[str]) -> bool:
    name_field = parsed.fields.get("nama")
    if not name_field or name_field.status != "ok" or not name_field.value:
        return False
    if name_field.confidence < KTP_NAME_BORDERLINE_CONFIDENCE:
        return False
    if not _field_ok(parsed, "provinsi") or not _field_ok(parsed, "kabupaten_kota"):
        return False
    if not _field_ok(parsed, "kode_pos"):
        return False
    blocked_prefixes = (
        "ktp_suspicious_field:",
        "ktp_nik_ttl_mismatch",
        "ktp_nik_gender_mismatch",
        "ktp_unusual_rt_rw",
        "missing_required:",
        "ktp_auto_missing:",
    )
    if any(reason.startswith(blocked_prefixes) for reason in reasons):
        return False
    return _ktp_name_is_plausible(name_field.value)


def _ktp_name_is_plausible(value: str) -> bool:
    upper = value.upper().strip()
    if not upper or any(char.isdigit() for char in upper):
        return False
    if re.search(r"\b(?:PROVINSI|KABUPATEN|JAKARTA|TANGERANG|BEKASI|SEMARANG|YOGYAKARTA|BATAM|JAMBI)\b", upper):
        return False
    if re.search(r"\b(?:ISLAM|KRISTEN|KATHOLIK|KATOLIK|HINDU|BUDHA|BUDDHA|KONGHUCU|WNI|WNA|SEUMUR|HIDUP)\b", upper):
        return False
    if re.search(r"\b(?:KARYAWAN|SWASTA|PELAJAR|MAHASISWA|MENGURUS|RUMAH|TANGGA|LAKI|PEREMPUAN)\b", upper):
        return False
    if "AKARTA BARA" in upper:
        return False

    tokens = re.findall(r"[A-Z][A-Z.'/-]*", upper)
    if not tokens:
        return False
    if len(tokens) == 1:
        token = tokens[0].replace(".", "").replace("'", "")
        return 5 <= len(token) <= 18

    good_tokens = 0
    for token in tokens:
        normalized = token.replace(".", "").replace("'", "").replace("-", "").replace("/", "")
        if len(normalized) < 3 and normalized not in {"TJ", "DJ", "OE"}:
            return False
        vowel_count = len(re.findall(r"[AEIOU]", normalized))
        if len(normalized) >= 8 and vowel_count / max(len(normalized), 1) > 0.7:
            return False
        if re.search(r"[AEIOU]{4,}", normalized):
            return False
        if len(normalized) >= 2:
            good_tokens += 1
    return good_tokens >= 2


def _ktp_ttl_has_label_fragment(value: str) -> bool:
    place = value.split(",", 1)[0]
    compact = re.sub(r"[^A-Z]", "", place.upper())
    if not compact:
        return False
    if "LAHIR" in compact or compact.startswith(("TEMPAT", "TGLL", "TGIL", "IGLL")):
        return True
    return bool(re.match(r"^\s*TAL\s+[A-Z]", place, flags=re.IGNORECASE))


def _stnk_auto_approval_reason_codes(parsed: DocumentResult) -> list[str]:
    if parsed.document_type != "STNK":
        return []

    reasons: list[str] = []
    for field_name in STNK_AUTO_APPROVAL_FIELDS:
        if not _field_ok(parsed, field_name):
            reasons.append(f"stnk_auto_missing:{field_name}")

    for field_name, min_confidence in STNK_AUTO_MIN_CONFIDENCE.items():
        field = parsed.fields.get(field_name)
        if field and field.status == "ok" and field.value and field.confidence < min_confidence:
            reasons.append(f"stnk_auto_low_confidence:{field_name}")

    for field_name, suspicious_values in STNK_SUSPICIOUS_FIELD_VALUES.items():
        value = _field_text(parsed, field_name)
        if value in suspicious_values:
            reasons.append(f"stnk_suspicious_field:{field_name}")

    return reasons


def _field_ok(parsed: DocumentResult, field_name: str) -> bool:
    field = parsed.fields.get(field_name)
    return bool(field and field.status == "ok" and field.value)


def _birth_date_from_nik(nik: str) -> str | None:
    normalized = normalize_nik(nik)
    if not normalized:
        return None
    day = int(normalized[6:8])
    month = int(normalized[8:10])
    year_two = int(normalized[10:12])
    if day > 40:
        day -= 40
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return None
    current_two = date.today().year % 100
    century = 2000 if year_two <= current_two else 1900
    return f"{day:02d}-{month:02d}-{century + year_two:04d}"


def _birth_date_from_ttl(ttl: str) -> str | None:
    match = re.search(r"\b(\d{2})-(\d{2})-(\d{4})\b", ttl)
    if not match:
        return None
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


def _gender_from_nik(nik: str) -> str | None:
    normalized = normalize_nik(nik)
    if not normalized:
        return None
    day = int(normalized[6:8])
    if not (1 <= day <= 71):
        return None
    return "PEREMPUAN" if day > 40 else "LAKI-LAKI"


def _looks_like_unreliable_ktp_for_auto(raw_text: str, parsed: DocumentResult) -> bool:
    if parsed.document_type != "KTP":
        return False

    ttl = parsed.fields.get("tempat_tanggal_lahir")
    ttl_missing = ttl is None or ttl.status != "ok" or not (ttl.value or "").strip()
    if not ttl_missing:
        return False

    name = _field_text(parsed, "nama")
    suspicious_name = bool(re.search(r"\b(?:SEUSUR|SEUMUR|BERLAKU|HINGGA|HIDUP)\b", name))

    address = _field_text(parsed, "alamat")
    address_letters = len(re.findall(r"[A-Z]", address))
    weak_address = bool(address) and address_letters <= 6 and not re.search(
        r"\b(?:J|JL|JLN|JALAN|KP|KAMP|BLOK|NO|PERUM|KOMP|GG|GANG)\b",
        address,
    )

    rt_rw = _field_text(parsed, "rt_rw")
    bad_rt_rw = False
    rt_match = re.fullmatch(r"(\d{3})/(\d{3})", rt_rw)
    if rt_match:
        rt_number = int(rt_match.group(1))
        rw_number = int(rt_match.group(2))
        bad_rt_rw = rt_number > 150 or rw_number > 250

    return suspicious_name or (weak_address and bad_rt_rw)


def _field_text(parsed: DocumentResult, field_name: str) -> str:
    field = parsed.fields.get(field_name)
    if not field or field.status != "ok" or not field.value:
        return ""
    return str(field.value).upper()


def _result_score(result: DocumentResult) -> float:
    ok_fields = sum(1 for field in result.fields.values() if field.status == "ok")
    confidence = sum(field.confidence for field in result.fields.values())
    return ok_fields * 2 + confidence - len(result.warnings) * 1.5
