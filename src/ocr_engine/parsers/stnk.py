from __future__ import annotations

import re

from ocr_engine.parsers.common import capture_after_label, make_invalid, make_missing, make_ok, normalized_lines
from ocr_engine.schemas import DocumentResult, FieldResult
from ocr_engine.validators import collapse_spaces, normalize_plate_number


STNK_LABELS: dict[str, list[str]] = {
    "nomor_polisi": ["NO POLISI", "NOMOR POLISI", "NOMOR POLIS", "Nomor Polisi", "No Polisi", "No. Polisi", "NO. POL", "NO POL", "NOPOL"],
    "nama_pemilik": ["NAMA PEMILIK", "NAMA PEMILI", "Nama Pemilik", "Nama"],
    "alamat": ["ALAMAT", "Alamat"],
    "merek": ["MERK", "Merek", "MEREK"],
    "tipe": ["TYPE", "TIPE", "Type", "Tipe"],
    "jenis": ["JENIS", "Jenis"],
    "tahun_pembuatan": ["TAHUN PEMBUATAN", "Tahun Pembuatan", "Tahun"],
    "warna": ["WARNA", "Warna"],
    "nomor_rangka": ["NO RANGKA", "NO. RANGKA", "NO.RANGKA", "NOMOR RANGKA", "Nomor Rangka", "No Rangka"],
    "nomor_mesin": ["NO MESIN", "NO. MESIN", "NO.MESIN", "NOMOR MESIN", "Nomor Mesin", "No Mesin"],
    "bahan_bakar": ["BAHAN BAKAR", "Bahan Bakar"],
    "berlaku_sampai": ["BERLAKU SAMPAI", "Berlaku Sampai", "Berlaku"],
}

STNK_REQUIRED_FIELDS = ["nomor_polisi", "nama_pemilik", "tahun_pembuatan", "nomor_rangka", "nomor_mesin"]
STNK_VALUE_NOISE = {
    "/NIK",
    "/NIK/VIN",
    "/TYPE",
    "/ MODEL",
    "BERLA",
    "DATEOFEXPIRE",
    "DATE OF EXPIRE",
    "IDENT",
    "KB",
    "LAMA",
    "MODEL",
    "PEMILIK",
    "PEMILIN",
    "REGISTRASI",
    "S/D",
    "SID",
    "TNKB",
}
PERSON_NOISE_WORDS = {
    "ALAMAT",
    "BAHAN",
    "BENSIN",
    "BBN",
    "BIAYA",
    "BPKB",
    "BUKTI",
    "BOJONG",
    "COKLAT",
    "DASANA",
    "DATE",
    "EXPIRE",
    "HITAM",
    "ISUZU",
    "JENIS",
    "JEEP",
    "JALAN",
    "JL",
    "KEL",
    "KEC",
    "KODE",
    "LIMA",
    "MAZDA",
    "MERAH",
    "MERK",
    "MINIBUS",
    "MOBIL",
    "MODEL",
    "NAMA",
    "NIK",
    "NOMOR",
    "OWNER",
    "PEMBAYARAN",
    "PEMILIK",
    "PELUNASAN",
    "PKB",
    "POLISI",
    "RANGKA",
    "REGISTRASI",
    "RIBU",
    "RUPIAH",
    "SAMSAT",
    "SILVER",
    "SILINDER",
    "SOLAR",
    "STNK",
    "SWDKLLJ",
    "TAHUN",
    "TANDA",
    "TNKB",
    "TYPE",
    "WARNA",
}
MONTH_MARKERS = {"JAN", "FEB", "MAR", "APR", "MEI", "MAY", "JUN", "JUL", "AUG", "AGS", "SEP", "OCT", "OKT", "NOV", "DEC", "DES"}
COMPANY_PREFIXES = {"PT", "CV", "UD", "PD", "KOPERASI", "YAYASAN"}


def parse_stnk_text(raw_text: str) -> DocumentResult:
    fields: dict[str, FieldResult] = {}
    warnings: list[str] = []
    stop_labels = [label for labels in STNK_LABELS.values() for label in labels]

    for field_name, labels in STNK_LABELS.items():
        value, raw = capture_after_label(raw_text, labels, stop_labels)
        if field_name == "nomor_polisi":
            fields[field_name] = _plate_field(value, raw)
        elif field_name == "nomor_rangka":
            fields[field_name] = _vehicle_id_field(value, raw, min_length=15, allow_numeric=False)
        elif field_name == "nomor_mesin":
            fields[field_name] = _vehicle_id_field(value, raw, min_length=5, allow_numeric=True)
        elif field_name == "tahun_pembuatan":
            fields[field_name] = _year_field(value, raw)
        else:
            fields[field_name] = make_ok(value, raw=raw) if value else make_missing()

    _apply_stnk_fallbacks(raw_text, fields)
    _apply_official_stnk_section_overrides(raw_text, fields)

    for field_name, field in fields.items():
        if field.status == "invalid":
            warnings.append(f"invalid:{field_name}")

    for required in STNK_REQUIRED_FIELDS:
        if fields[required].status == "missing":
            warnings.append(f"missing_required:{required}")

    return DocumentResult(
        document_type="STNK",
        schema_version="stnk.v1",
        fields=fields,
        warnings=warnings,
        raw_text=raw_text,
    )


def _plate_field(value: str | None, raw: str | None) -> FieldResult:
    if not value:
        return make_missing()
    if "XXX" in value.upper():
        return make_invalid(value, raw=raw)
    normalized = _normalize_stnk_plate(value)
    if normalized:
        return FieldResult(value=normalized, confidence=0.95, status="ok", evidence=[normalized], raw=raw)
    return make_invalid(value, raw=raw)


def _vehicle_id_field(value: str | None, raw: str | None, min_length: int, allow_numeric: bool) -> FieldResult:
    if not value:
        return make_missing()
    normalized = _normalize_vehicle_id(value)
    if (
        normalized
        and len(normalized) >= min_length
        and any(char.isdigit() for char in normalized)
        and (allow_numeric or any(char.isalpha() for char in normalized))
        and not _is_noise_value(normalized)
        and not _contains_month_marker(normalized)
    ):
        return FieldResult(value=normalized, confidence=0.9, status="ok", evidence=[normalized], raw=raw)
    return make_invalid(value, raw=raw)


def _year_field(value: str | None, raw: str | None) -> FieldResult:
    if not value:
        return make_missing()
    normalized = _normalize_year(value)
    if normalized:
        return FieldResult(value=normalized, confidence=0.88, status="ok", evidence=[normalized], raw=raw)
    return make_invalid(value, raw=raw)


def _normalize_vehicle_id(value: str) -> str | None:
    compact = "".join(char for char in value.upper() if char.isalnum())
    return compact or None


def _normalize_year(value: str) -> str | None:
    mapped = value.upper().translate(str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1"}))
    digits = "".join(char for char in mapped if char.isdigit())
    if len(digits) < 4:
        return None
    for index in range(0, len(digits) - 3):
        year = digits[index : index + 4]
        if 1900 <= int(year) <= 2099:
            return year
    return None


def _apply_stnk_fallbacks(raw_text: str, fields: dict[str, FieldResult]) -> None:
    lines = normalized_lines(raw_text)

    if fields["nomor_polisi"].status != "ok":
        plate = _fallback_plate(lines)
        if plate:
            fields["nomor_polisi"] = FieldResult(
                value=plate,
                confidence=0.8,
                status="ok",
                evidence=[plate, "fallback:plate_scan"],
                raw="fallback:plate_scan",
            )

    if _needs_text_repair(fields["nama_pemilik"]):
        owner = _fallback_owner(lines, fields["nomor_polisi"].value)
        if owner:
            fields["nama_pemilik"] = make_ok(owner, confidence=0.76, raw="fallback:owner_scan")
        else:
            fields["nama_pemilik"] = make_missing()

    if fields["tahun_pembuatan"].status != "ok":
        year = _fallback_manufacture_year(lines)
        if year:
            fields["tahun_pembuatan"] = FieldResult(
                value=year,
                confidence=0.78,
                status="ok",
                evidence=[year, "fallback:year_scan"],
                raw="fallback:year_scan",
            )

    if _needs_vehicle_id_repair(fields["nomor_rangka"], min_length=15):
        rangka = _fallback_vehicle_id(lines, target="rangka")
        if rangka:
            fields["nomor_rangka"] = FieldResult(
                value=rangka,
                confidence=0.78,
                status="ok",
                evidence=[rangka, "fallback:rangka_scan"],
                raw="fallback:rangka_scan",
            )

    if _needs_vehicle_id_repair(fields["nomor_mesin"], min_length=5):
        mesin = _fallback_vehicle_id(lines, target="mesin")
        if mesin:
            fields["nomor_mesin"] = FieldResult(
                value=mesin,
                confidence=0.76,
                status="ok",
                evidence=[mesin, "fallback:mesin_scan"],
                raw="fallback:mesin_scan",
            )


def _apply_official_stnk_section_overrides(raw_text: str, fields: dict[str, FieldResult]) -> None:
    section = _official_stnk_section_lines(raw_text)
    if not section:
        return

    plate = _section_plate(section)
    if plate and _should_apply_official_override(fields["nomor_polisi"]):
        fields["nomor_polisi"] = FieldResult(
            value=plate,
            confidence=0.9,
            status="ok",
            evidence=[plate, "official_section:plate"],
            raw="official_section:plate",
        )

    owner = _section_owner(section)
    if owner and _should_apply_official_override(fields["nama_pemilik"]):
        fields["nama_pemilik"] = make_ok(owner, confidence=0.9, raw="official_section:nama_pemilik")

    simple_fields = {
        "merek": r"\bMER[EK]\w*\b",
        "tipe": r"\b(?:TYPE|TIPE)\b",
        "jenis": r"\bJENIS\b",
        "warna": r"\bWARNA\w*\b",
    }
    for field_name, pattern in simple_fields.items():
        value = _section_simple_value(section, pattern)
        if value and _should_apply_official_override(fields[field_name]):
            fields[field_name] = make_ok(value, confidence=0.88, raw=f"official_section:{field_name}")

    year = _section_year(section)
    if year and _should_apply_official_override(fields["tahun_pembuatan"]):
        fields["tahun_pembuatan"] = FieldResult(
            value=year,
            confidence=0.9,
            status="ok",
            evidence=[year, "official_section:year"],
            raw="official_section:year",
        )

    rangka = _section_vehicle_id(section, r"RANGK\w*", target="rangka")
    if rangka and _should_apply_official_override(fields["nomor_rangka"]):
        fields["nomor_rangka"] = FieldResult(
            value=rangka,
            confidence=0.9,
            status="ok",
            evidence=[rangka, "official_section:rangka"],
            raw="official_section:rangka",
        )

    mesin = _section_vehicle_id(section, r"MESIN\w*", target="mesin")
    if mesin and _should_apply_official_override(fields["nomor_mesin"]):
        fields["nomor_mesin"] = FieldResult(
            value=mesin,
            confidence=0.88,
            status="ok",
            evidence=[mesin, "official_section:mesin"],
            raw="official_section:mesin",
        )

    expiry = _section_expiry_date(section)
    if expiry and _should_apply_official_override(fields["berlaku_sampai"]):
        fields["berlaku_sampai"] = make_ok(expiry, confidence=0.88, raw="official_section:berlaku_sampai")


def _should_apply_official_override(field: FieldResult) -> bool:
    value = (field.value or "").upper()
    return field.status != "ok" or not field.raw or field.raw.startswith("fallback:") or _is_noise_value(value)


def _official_stnk_section_lines(raw_text: str) -> list[str]:
    lines = normalized_lines(raw_text)
    starts = [
        index
        for index, line in enumerate(lines)
        if re.search(r"SURAT\s+TANDA\s+NOMOR\s+KENDARAAN\s+BERMOTOR", line, flags=re.IGNORECASE)
    ]
    if not starts:
        return []
    return lines[starts[-1] :]


def _section_plate(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not re.search(r"\b(?:NRKB|NOMOR\s+REGISTRAS\w*)\b", line, flags=re.IGNORECASE):
            continue
        for candidate_line in _window(lines, index + 1, index + 8):
            candidate = _plate_candidate_from_line(candidate_line, allow_digit_three=False, strict_line=False)
            if candidate:
                return candidate
    return None


def _section_owner(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not re.search(r"\bNAMA\s+PEMILI\w*\b", line, flags=re.IGNORECASE):
            continue
        for candidate in _window(lines, index + 1, index + 5):
            if re.search(r"\b(?:STNK|ALAMAT|NIK|KITAS|KITAP)\b", candidate, flags=re.IGNORECASE):
                break
            if _looks_like_owner_name(candidate):
                return candidate
    return None


def _section_simple_value(lines: list[str], label_pattern: str) -> str | None:
    for index, line in enumerate(lines):
        match = re.search(rf"{label_pattern}\s*[:\-]?\s*(.*)$", line, flags=re.IGNORECASE)
        if not match:
            continue
        inline_value = match.group(1).strip()
        if inline_value and not _is_section_label(inline_value) and not _looks_like_amount(inline_value):
            return collapse_spaces(inline_value.strip(" :.-"))
        for candidate in _window(lines, index + 1, index + 5):
            if _is_section_label(candidate) or _looks_like_amount(candidate):
                continue
            cleaned = collapse_spaces(candidate.strip(" :.-"))
            if cleaned and re.search(r"[A-Z]", cleaned, flags=re.IGNORECASE):
                return cleaned
    return None


def _section_year(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not re.search(r"\bTAHUN\s+PEMBUAT\w*", line, flags=re.IGNORECASE):
            continue
        for candidate in _window(lines, index + 1, index + 6):
            year = _normalize_standalone_year(candidate)
            if year:
                return year
    return None


def _section_vehicle_id(lines: list[str], label_pattern: str, target: str) -> str | None:
    for index, line in enumerate(lines):
        if not re.search(label_pattern, line, flags=re.IGNORECASE):
            continue
        for candidate_line in _window(lines, index + 1, index + 7):
            if _is_section_label(candidate_line) or _looks_like_amount(candidate_line):
                continue
            normalized = _normalize_vehicle_id(candidate_line)
            if not normalized or _is_noise_value(normalized) or _contains_month_marker(normalized):
                continue
            if normalize_plate_number(normalized) or not any(char.isdigit() for char in normalized):
                continue
            if target == "rangka" and _vehicle_id_score(normalized, "rangka") > 0:
                return normalized
            if target == "mesin" and any(char.isalpha() for char in normalized) and 5 <= len(normalized) <= 20:
                return normalized
    return None


def _section_expiry_date(lines: list[str]) -> str | None:
    month_pattern = r"Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember"
    for index, line in enumerate(lines):
        if not re.search(r"\bBERLAKU\s+S\w*", line, flags=re.IGNORECASE):
            continue
        for candidate in _window(lines, index + 1, index + 5):
            match = re.search(rf"\b\d{{1,2}}\s+(?:{month_pattern})\s+\d{{4}}\b", candidate, flags=re.IGNORECASE)
            if match:
                return collapse_spaces(match.group(0).rstrip("."))
            match = re.search(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b", candidate)
            if match:
                return match.group(0)
    return None


def _is_section_label(value: str) -> bool:
    return bool(
        re.search(
            r"\b(?:ALAMAT|BERLAKU|JENIS|MERK|MEREK|MODEL|NAMA|NIK|NOMOR|NRKB|RANGK|STNK|TAHUN|TYPE|TIPE|WARNA)\b",
            value,
            flags=re.IGNORECASE,
        )
    )


def _needs_text_repair(field: FieldResult) -> bool:
    value = (field.value or "").upper()
    return (
        field.status != "ok"
        or _is_noise_value(value)
        or _looks_like_amount(value)
        or any(char.isdigit() for char in value)
        or not _looks_like_owner_name(value)
    )


def _needs_vehicle_id_repair(field: FieldResult, min_length: int) -> bool:
    value = (field.value or "").upper()
    normalized = _normalize_vehicle_id(value) if value else None
    return field.status != "ok" or _is_noise_value(value) or normalized is None or len(normalized) < min_length


def _is_noise_value(value: str) -> bool:
    normalized = collapse_spaces(value.upper().replace(" ", ""))
    spaced = collapse_spaces(value.upper())
    return normalized in {item.replace(" ", "") for item in STNK_VALUE_NOISE} or spaced in STNK_VALUE_NOISE


def _fallback_plate(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not re.search(r"\b(?:NOMOR|NO)\s*\.?\s*POLIS\w*\b", line, flags=re.IGNORECASE):
            continue
        for candidate_line in _window(lines, index + 1, index + 22):
            candidate = _plate_candidate_from_line(candidate_line, allow_digit_three=True, strict_line=True)
            if candidate:
                return candidate

    for line in lines:
        candidate = _plate_candidate_from_line(line, allow_digit_three=False, strict_line=False)
        if candidate:
            return candidate
    return None


def _fallback_owner(lines: list[str], plate: str | None) -> str | None:
    for index, line in enumerate(lines):
        if not re.search(r"\bNAMA\s+PEMILI\w*\b", line, flags=re.IGNORECASE):
            continue
        for candidate in _window(lines, index - 5, index):
            if _looks_like_owner_name(candidate):
                return candidate
        for candidate in _window(lines, index + 1, index + 22):
            if _looks_like_owner_name(candidate):
                return candidate

    for index, line in enumerate(lines):
        if not re.search(r"\bREGISTER\b", line, flags=re.IGNORECASE):
            continue
        for candidate in _window(lines, index + 1, index + 5):
            if _looks_like_company_name(candidate):
                return candidate

    if plate:
        compact_plate = re.sub(r"[^A-Z0-9]", "", plate.upper())
        for index, line in enumerate(lines):
            if re.sub(r"[^A-Z0-9]", "", line.upper()) != compact_plate:
                continue
            for candidate in reversed(_window(lines, index - 4, index)):
                if _looks_like_owner_name(candidate):
                    return candidate
            for candidate in _window(lines, index + 1, index + 5):
                if _looks_like_owner_name(candidate):
                    return candidate

    return None


def _fallback_manufacture_year(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not re.search(r"\bTAHUN\s+PEMBUAT\w*", line, flags=re.IGNORECASE):
            continue
        for candidate in _window(lines, index + 1, index + 18):
            year = _normalize_standalone_year(candidate)
            if year:
                return year
    return None


def _fallback_vehicle_id(lines: list[str], target: str) -> str | None:
    label_pattern = r"(RANGKA|RANGK|RANGIA|RANGO|RANGON|RWNGKA|RNAKB|IDENT|VIN)" if target == "rangka" else r"(MESIN|ENGINE)"
    best: tuple[int, str] | None = None
    for index, line in enumerate(lines):
        if not re.search(label_pattern, line, flags=re.IGNORECASE):
            continue
        candidate_lines = _window(lines, index, index + 24)
        if target in {"mesin", "rangka"}:
            candidate_lines = _window(lines, index - 8, index) + candidate_lines
        for candidate_line in candidate_lines:
            if _looks_like_amount(candidate_line):
                continue
            for candidate in _vehicle_id_candidates(candidate_line, target):
                score = _vehicle_id_score(candidate, target)
                if score <= 0:
                    continue
                if best is None or score > best[0]:
                    best = (score, candidate)
        if target == "rangka":
            for candidate in _split_rangka_candidates(candidate_lines):
                score = _vehicle_id_score(candidate, target) + 1
                if score <= 1:
                    continue
                if best is None or score > best[0]:
                    best = (score, candidate)
    if best is None and target == "rangka":
        for candidate_line in lines:
            for candidate in _vehicle_id_candidates(candidate_line, target):
                score = _strong_rangka_score(candidate)
                if score <= 0:
                    continue
                if best is None or score > best[0]:
                    best = (score, candidate)
    return best[1] if best else None


def _vehicle_id_candidates(value: str, target: str) -> list[str]:
    candidates: list[str] = []
    normalized = _normalize_vehicle_id(value)
    if normalized:
        candidates.append(normalized)

    if target == "rangka":
        compact = "".join(char for char in value.upper() if char.isalnum())
        for match in re.finditer(r"NK([A-Z0-9]{15,18})", compact):
            candidates.append(match.group(1)[:17])
        for match in re.finditer(r"([A-Z]{2,5}[A-Z0-9]{13,16})", compact):
            candidate = match.group(1)
            if 15 <= len(candidate) <= 18:
                candidates.append(candidate)

    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique:
            unique.append(candidate)
    return unique


def _split_rangka_candidates(lines: list[str]) -> list[str]:
    fragments = [_normalize_vehicle_id(line) for line in lines if not _looks_like_amount(line)]
    fragments = [
        fragment
        for fragment in fragments
        if fragment
        and 4 <= len(fragment) <= 14
        and not _is_noise_value(fragment)
        and not any(
            noise in fragment
            for noise in ["ALAMAT", "BERLAKU", "MERK", "MODEL", "NAMA", "PEMILI", "RANG", "NOMOR", "MESIN", "SILINDER", "WARNA", "NIK"]
        )
    ]
    candidates: list[str] = []
    for left_index, left in enumerate(fragments):
        if not re.match(r"(?:MH|MP|MM|JM|LF|RF|MR|R0)", left):
            continue
        nearby = list(reversed(fragments[max(0, left_index - 4) : left_index])) + fragments[left_index + 1 : left_index + 5]
        for right in nearby:
            combined = left + right
            if 15 <= len(combined) <= 22:
                candidates.append(combined[:17])
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _vehicle_id_score(value: str, target: str) -> int:
    if any(
        noise in value
        for noise in ["DATEOFEXPIRE", "KEL", "MESIN", "PENGESAHAN", "RANGKA", "SILINDER", "STNK", "NOMOR", "POLISI", "PEMILIK", "RT", "RW"]
    ):
        return 0
    if _contains_month_marker(value):
        return 0
    if normalize_plate_number(value):
        return 0
    if not any(char.isdigit() for char in value) or not any(char.isalpha() for char in value):
        if target == "mesin" and value.isdigit() and 5 <= len(value) <= 10:
            return 3
        return 0
    if value.isdigit() and _normalize_year(value):
        return 0
    if target == "rangka":
        if len(value) == 17:
            return 8
        if 15 <= len(value) <= 18:
            return 6
        return 0
    if 5 <= len(value) <= 14 and not re.fullmatch(r"[A-Z]\d{7,}[A-Z]?", value):
        return 5 if re.match(r"(?:[A-Z]{2}\d|\d[A-Z]{2})", value) else 3
    return 0


def _strong_rangka_score(value: str) -> int:
    if _vehicle_id_score(value, "rangka") <= 0:
        return 0
    return 5 if re.match(r"(?:MH|MP|MM|JM|LF|RF|MR|R0)", value) else 0


def _window(lines: list[str], start: int, end: int) -> list[str]:
    return lines[max(0, start) : min(len(lines), end)]


def _looks_like_person_name(value: str) -> bool:
    cleaned = collapse_spaces(value.upper().replace(",", " "))
    if "XXX" in cleaned or "." in cleaned:
        return False
    if _looks_like_amount(cleaned) or any(char.isdigit() for char in cleaned):
        return False
    if not re.fullmatch(r"[A-Z][A-Z .']{2,}", cleaned):
        return False
    words = [word for word in cleaned.split() if word]
    if not 2 <= len(words) <= 5:
        return False
    if any("KOTA" in word for word in words):
        return False
    return not any(word in PERSON_NOISE_WORDS for word in words)


def _looks_like_owner_name(value: str) -> bool:
    return _looks_like_person_name(value) or _looks_like_company_name(value)


def _looks_like_company_name(value: str) -> bool:
    cleaned = collapse_spaces(value.upper().replace(",", " "))
    if "XXX" in cleaned or any(char.isdigit() for char in cleaned):
        return False
    words = [word.strip(".") for word in cleaned.replace(".", " ").split() if word.strip(".")]
    if len(words) < 3 or words[0] not in COMPANY_PREFIXES:
        return False
    return not any(word in PERSON_NOISE_WORDS for word in words[1:])


def _normalize_stnk_plate(value: str | None) -> str | None:
    normalized = normalize_plate_number(value)
    if normalized or not value:
        return normalized
    compact = re.sub(r"[\s.\-]", "", value.upper())
    match = re.fullmatch(r"([83])(\d{1,4})([A-Z]{1,3})", compact)
    if not match:
        return None
    return normalize_plate_number(f"B{match.group(2)}{match.group(3)}")


def _plate_candidate_from_line(line: str, allow_digit_three: bool, strict_line: bool) -> str | None:
    upper = line.upper()
    if "XXX" in upper:
        return None
    digit_prefix = "|8|3" if allow_digit_three else "|8"
    pattern = rf"(?<![A-Z0-9])([A-Z]{{1,2}}{digit_prefix})\s*[-.]?\s*(\d{{1,4}})\s*[-.]?\s*([A-Z]{{1,3}})(?![A-Z0-9])"
    for match in re.finditer(pattern, upper):
        prefix = "B" if match.group(1) in {"8", "3"} else match.group(1)
        if prefix in {"JL", "NO", "RT", "RW"} or any(marker in upper for marker in [" JL", "RT.", " KEL", " KEC"]):
            continue
        candidate = _normalize_stnk_plate(f"{prefix} {match.group(2)} {match.group(3)}")
        if candidate:
            if strict_line:
                line_compact = re.sub(r"[^A-Z0-9]", "", upper)
                candidate_compact = re.sub(r"[^A-Z0-9]", "", candidate)
                if len(line_compact) > len(candidate_compact) + 2:
                    continue
            return candidate
    return None


def _looks_like_amount(value: str) -> bool:
    return bool(re.fullmatch(r"[\d.]+(?:,\d+)?", value.strip()))


def _normalize_standalone_year(value: str) -> str | None:
    mapped = value.upper().translate(str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1"}))
    if not re.fullmatch(r"[:\s.\-/]*\d{4}[:\s.\-/]*", mapped):
        return None
    return _normalize_year(mapped)


def _contains_month_marker(value: str) -> bool:
    upper = value.upper()
    return any(marker in upper for marker in MONTH_MARKERS)
