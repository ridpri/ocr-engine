from __future__ import annotations

from difflib import SequenceMatcher
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
    "jenis": ["JENIS", "JENS", "BENIS", "Jenis"],
    "tahun_pembuatan": ["TAHUN PEMBUATAN", "Tahun Pembuatan", "Tahun"],
    "warna": ["WARNA", "Warna"],
    "nomor_rangka": ["NO RANGKA", "NO. RANGKA", "NO.RANGKA", "NOMOR RANGKA", "Nomor Rangka", "No Rangka"],
    "nomor_mesin": ["NO MESIN", "NO. MESIN", "NO.MESIN", "NOMOR MESIN", "Nomor Mesin", "No Mesin"],
    "bahan_bakar": ["BAHAN BAKAR", "BAHAN BAGAR", "BAWAN BAKAR", "MAHAN BAKAR", "BARN BAAR", "Bahan Bakar"],
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
    "CATEGORY",
    "COMPANY REGISTRATION NUMBER",
    "FUEL ENERGY SOURCES",
    "FUEL/ENERGY SOURCES",
    "IDENT",
    "KB",
    "LAMA",
    "MODEL",
    "KOHIR",
    "NO KOHIR",
    "NO. KOHIR",
    "PEMILIK",
    "PEMILIN",
    "REGISTRASI",
    "S/D",
    "SID",
    "TNKB",
    "BN.KENDARAAN",
    "BN KENDARAAN",
}
PERSON_NOISE_WORDS = {
    "ALAMAT",
    "ALLAMAT",
    "LAMAT",
    "BAHAN",
    "BENSIN",
    "BBN",
    "BANDUNG",
    "BARAT",
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
    "JAWA",
    "JL",
    "KEL",
    "KEC",
    "KEPALA",
    "KOHIR",
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
    "NUMBER",
    "NUNER",
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
    "VEHICLE",
    "VENICL",
    "WARNA",
}
MONTH_MARKERS = {"JAN", "FEB", "MAR", "APR", "MEI", "MAY", "JUN", "JUL", "AUG", "AGS", "SEP", "OCT", "OKT", "NOV", "DEC", "DES"}
COMPANY_PREFIXES = {"PT", "FT", "CV", "UD", "PD", "KOPERASI", "YAYASAN"}
RANGKA_PREFIXES = ("MJE", "MHR", "MPA", "LGX", "MH", "MP", "MM", "JM", "LF", "RF", "MR", "R0")
VEHICLE_COLOR_VALUES = {
    "ABU ABU",
    "BIRU",
    "COKLAT",
    "HIJAU",
    "HITAM",
    "KUNING",
    "MERAH",
    "ORANGE",
    "PUTIH",
    "SILVER",
}
_LABEL_TRANSLATION = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B"})


def match_stnk_label(text: str, field_name: str | None = None) -> dict | None:
    line = collapse_spaces(text.strip(" :.-"))
    if not line:
        return None

    label_groups = {field_name: STNK_LABELS[field_name]} if field_name else STNK_LABELS
    best: dict | None = None
    for current_field, labels in label_groups.items():
        for label in labels:
            if not _allows_stnk_label_match(line, current_field, label):
                continue
            for span_end in _label_candidate_spans(line, label):
                prefix = line[:span_end]
                score = _label_similarity(prefix, label)
                threshold = _label_match_threshold(label)
                if score < threshold:
                    continue
                if best is None or score > best["score"] or (
                    score == best["score"] and span_end > best["span_end"]
                ):
                    best = {
                        "field_name": current_field,
                        "label": label,
                        "score": round(score, 3),
                        "span_end": span_end,
                        "raw": line,
                    }
    return best


def _allows_stnk_label_match(line: str, field_name: str, label: str) -> bool:
    upper = line.upper()
    if field_name == "tahun_pembuatan" and "REG" in upper and not re.search(r"PE[MN]?B?U?AT", upper):
        return False
    if field_name == "tahun_pembuatan" and _normalize_label_text(label) == "TAHUN":
        return bool(re.search(r"PE[MN]?B?U?AT", upper))
    return True


def stnk_structure_score(raw_text: str) -> float:
    return analyze_stnk_structure(raw_text)["score"]


def analyze_stnk_structure(raw_text: str) -> dict:
    lines = normalized_lines(raw_text)
    field_scores = {field_name: 0.0 for field_name in STNK_LABELS}
    for line in lines:
        match = match_stnk_label(line)
        if match:
            field_scores[match["field_name"]] = max(field_scores[match["field_name"]], match["score"])

    required_score = sum(1 for field in STNK_REQUIRED_FIELDS if field_scores[field] >= 0.78) / len(STNK_REQUIRED_FIELDS)
    optional_fields = [field for field in STNK_LABELS if field not in STNK_REQUIRED_FIELDS]
    optional_score = (
        sum(1 for field in optional_fields if field_scores[field] >= 0.78) / len(optional_fields)
        if optional_fields
        else 0.0
    )
    marker_score = _stnk_marker_score(lines)
    score = min(1.0, required_score * 0.65 + optional_score * 0.2 + marker_score * 0.15)
    return {
        "score": round(score, 3),
        "required_label_coverage": round(required_score, 3),
        "optional_label_coverage": round(optional_score, 3),
        "marker_score": round(marker_score, 3),
        "field_scores": {key: round(value, 3) for key, value in field_scores.items() if value > 0},
    }


def parse_stnk_text(raw_text: str) -> DocumentResult:
    fields: dict[str, FieldResult] = {}
    warnings: list[str] = []
    stop_labels = [label for labels in STNK_LABELS.values() for label in labels]

    for field_name, labels in STNK_LABELS.items():
        value, raw = capture_after_label(raw_text, labels, stop_labels)
        fuzzy_value, fuzzy_raw = _capture_after_fuzzy_stnk_label(raw_text, field_name)
        if fuzzy_value and (not value or _should_prefer_fuzzy_stnk_value(field_name, value)):
            value, raw = fuzzy_value, fuzzy_raw
        if field_name == "nomor_polisi":
            fields[field_name] = _plate_field(value, raw)
        elif field_name == "nomor_rangka":
            fields[field_name] = _vehicle_id_field(value, raw, min_length=15, allow_numeric=False)
        elif field_name == "nomor_mesin":
            fields[field_name] = _vehicle_id_field(value, raw, min_length=5, allow_numeric=True)
        elif field_name == "tahun_pembuatan":
            fields[field_name] = _year_field(value, raw)
        elif field_name == "nama_pemilik":
            fields[field_name] = _owner_field(value, raw)
        elif field_name == "alamat":
            fields[field_name] = _address_field(value, raw)
        elif field_name == "jenis":
            fields[field_name] = _vehicle_category_field(value, raw)
        elif field_name == "warna":
            fields[field_name] = _color_field(value, raw)
        elif field_name == "bahan_bakar":
            fields[field_name] = _fuel_field(value, raw)
        elif field_name == "berlaku_sampai":
            fields[field_name] = _expiry_field(value, raw)
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
        confidence = 0.95 if _plate_has_suffix(normalized) else 0.80
        return FieldResult(value=normalized, confidence=confidence, status="ok", evidence=[normalized], raw=raw)
    return make_invalid(value, raw=raw)


def _owner_field(value: str | None, raw: str | None) -> FieldResult:
    if not value:
        return make_missing()
    cleaned = _normalize_owner_candidate(value)
    if _is_official_noise_value(cleaned) or not _looks_like_owner_name_after_label(cleaned):
        return make_missing()
    return make_ok(cleaned, confidence=0.88, raw=raw)


def _address_field(value: str | None, raw: str | None) -> FieldResult:
    if not value:
        return make_missing()
    cleaned = collapse_spaces(value.strip(" :.-"))
    if _looks_like_amount(cleaned) or _is_official_noise_value(cleaned) or not _looks_like_address_value(cleaned):
        return make_missing()
    return make_ok(cleaned, confidence=0.88, raw=raw)


def _vehicle_category_field(value: str | None, raw: str | None) -> FieldResult:
    if not value:
        return make_missing()
    category = _normalize_vehicle_category(value)
    return make_ok(category, confidence=0.88, raw=raw) if category else make_missing()


def _color_field(value: str | None, raw: str | None) -> FieldResult:
    if not value:
        return make_missing()
    color = _normalize_vehicle_color(value)
    return make_ok(color, confidence=0.88, raw=raw) if color else make_missing()


def _fuel_field(value: str | None, raw: str | None) -> FieldResult:
    if not value:
        return make_missing()
    fuel = _extract_fuel_value_from_line(value)
    return make_ok(fuel, confidence=0.88, raw=raw) if fuel else make_missing()


def _expiry_field(value: str | None, raw: str | None) -> FieldResult:
    if not value:
        return make_missing()
    text_expiry = _normalize_expiry_date_text(value)
    if text_expiry:
        return make_ok(text_expiry, confidence=0.88, raw=raw)
    match = re.search(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b", value)
    if match:
        return make_ok(match.group(0), confidence=0.88, raw=raw)
    return make_missing()


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


def _capture_after_fuzzy_stnk_label(raw_text: str, field_name: str) -> tuple[str | None, str | None]:
    lines = normalized_lines(raw_text)
    for index, line in enumerate(lines):
        match = match_stnk_label(line, field_name=field_name)
        if not match:
            continue

        inline = _value_after_fuzzy_label(line, int(match["span_end"]))
        if inline and not _is_probable_stnk_label(inline):
            return collapse_spaces(inline), line

        if index + 1 < len(lines):
            next_line = lines[index + 1]
            if not _is_probable_stnk_label(next_line):
                return collapse_spaces(next_line), line

    return None, None


def _should_prefer_fuzzy_stnk_value(field_name: str, value: str) -> bool:
    if field_name == "nomor_polisi":
        return _normalize_stnk_plate(value) is None
    if field_name == "nomor_rangka":
        normalized = _normalize_vehicle_id(value)
        return normalized is None or len(normalized) < 15 or _is_noise_value(normalized)
    if field_name == "nomor_mesin":
        normalized = _normalize_vehicle_id(value)
        return normalized is None or len(normalized) < 5 or _is_noise_value(normalized)
    if field_name == "tahun_pembuatan":
        return _normalize_year(value) is None
    if field_name == "nama_pemilik":
        return not _looks_like_owner_name(value)
    return _is_probable_stnk_label(value) or _is_noise_value(value)


def _value_after_fuzzy_label(line: str, span_end: int) -> str:
    tail = line[span_end:]
    if ":" in tail:
        tail = tail.split(":", 1)[1]
    tail = re.sub(r"^(?:[/\\]?\s*(?:NIK|NKM|VIN|NIV|TDP|KITAS|KITAP|HP|CC|MODEL|TYPE|TIPE))+", "", tail, flags=re.IGNORECASE)
    return collapse_spaces(tail.strip(" :.-/\\"))


def _is_probable_stnk_label(line: str) -> bool:
    return match_stnk_label(line) is not None or _is_section_label(line)


def _label_candidate_spans(line: str, label: str) -> list[int]:
    label_length = len(collapse_spaces(label))
    minimum = max(3, int(label_length * 0.65))
    maximum = min(len(line), max(label_length + 8, int(label_length * 1.45)))
    colon_index = line.find(":")
    if colon_index > 0:
        maximum = max(maximum, colon_index)
    spans = set(range(minimum, maximum + 1))
    for separator in [":", "-", "/", "\\"]:
        separator_index = line.find(separator)
        if separator_index > 0:
            spans.add(separator_index)
    spans.add(min(len(line), label_length))
    return sorted(span for span in spans if 0 < span <= len(line))


def _label_similarity(value: str, label: str) -> float:
    left = _normalize_label_text(value)
    right = _normalize_label_text(label)
    if not left or not right:
        return 0.0
    ratio = SequenceMatcher(None, left, right).ratio()
    if left.startswith(right):
        ratio = max(ratio, min(1.0, len(right) / max(len(left), 1) + 0.05))
    if right.startswith(left) and len(left) >= len(right) * 0.7:
        ratio = max(ratio, min(1.0, len(left) / max(len(right), 1) + 0.05))
    return ratio


def _normalize_label_text(value: str) -> str:
    return "".join(char for char in value.upper().translate(_LABEL_TRANSLATION) if char.isalnum())


def _label_match_threshold(label: str) -> float:
    compact_length = len(_normalize_label_text(label))
    if compact_length <= 6:
        return 0.9
    if compact_length <= 9:
        return 0.84
    return 0.78


def _stnk_marker_score(lines: list[str]) -> float:
    compact = _normalize_label_text(" ".join(lines))
    if "SURATTANDANOMORKENDARAAN" in compact or "NOMORKENDARAANBERMOTOR" in compact:
        return 1.0
    markers = ["STNK", "SAMSAT", "TNKB", "BPKB", "SWDKLLJ", "BBNKB"]
    hits = sum(1 for marker in markers if marker in compact)
    return min(1.0, hits / 3)


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

    if fields["warna"].status != "ok" or _is_noise_value(fields["warna"].value or ""):
        color = _section_color_value(lines)
        if color:
            fields["warna"] = make_ok(color, confidence=0.78, raw="fallback:color_scan")

    if fields["tipe"].status == "ok":
        cleaned_type = _clean_type_candidate(fields["tipe"].value or "")
        if cleaned_type and cleaned_type != fields["tipe"].value:
            fields["tipe"] = make_ok(cleaned_type, confidence=fields["tipe"].confidence, raw="fallback:type_clean")

    if _needs_type_repair(fields["tipe"]):
        tipe = _section_type_value(lines) or _model_type_value(lines)
        if tipe:
            fields["tipe"] = make_ok(tipe, confidence=0.78, raw="fallback:type_scan")

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

    mesin = _fallback_vehicle_id(lines, target="mesin")
    if mesin and (
        _needs_vehicle_id_repair(fields["nomor_mesin"], min_length=5)
        or _prefer_vehicle_id_candidate(mesin, fields["nomor_mesin"].value or "", "mesin")
    ):
        fields["nomor_mesin"] = FieldResult(
            value=mesin,
            confidence=0.76,
            status="ok",
            evidence=[mesin, "fallback:mesin_scan"],
            raw="fallback:mesin_scan",
        )


def _prefer_vehicle_id_candidate(candidate: str, current: str, target: str) -> bool:
    normalized_current = _normalize_vehicle_id(current) if current else None
    if not normalized_current:
        return True
    candidate_score = _vehicle_id_score(candidate, target)
    current_score = _vehicle_id_score(normalized_current, target)
    if target == "mesin" and current_score >= 5 and candidate_score <= 6:
        return False
    if candidate_score != current_score:
        return candidate_score > current_score
    if target == "mesin" and candidate.isdigit() != normalized_current.isdigit():
        return candidate.isdigit()
    return len(candidate) > len(normalized_current)


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

    address = _section_address(section)
    if address and _should_apply_official_text_override(fields["alamat"]):
        fields["alamat"] = make_ok(address, confidence=0.88, raw="official_section:alamat")

    company_owner = _fallback_company_owner(normalized_lines(raw_text))
    if company_owner and not _looks_like_company_name(fields["nama_pemilik"].value or ""):
        fields["nama_pemilik"] = make_ok(company_owner, confidence=0.78, raw="fallback:company_owner_scan")

    simple_fields = {
        "merek": r"\bMER[EK]\w*\b",
        "jenis": r"\bJENIS\b",
    }
    for field_name, pattern in simple_fields.items():
        value = _section_simple_value(section, pattern)
        if value and _should_apply_official_override(fields[field_name]):
            fields[field_name] = make_ok(value, confidence=0.88, raw=f"official_section:{field_name}")

    color = _section_color_value(section)
    if color and _should_apply_official_override(fields["warna"]):
        fields["warna"] = make_ok(color, confidence=0.88, raw="official_section:warna")

    inferred_merek = _section_vehicle_brand(section)
    if inferred_merek and _should_apply_official_override(fields["merek"]):
        fields["merek"] = make_ok(inferred_merek, confidence=0.86, raw="official_section:merek")

    inferred_bahan_bakar = _section_fuel_value(section)
    if inferred_bahan_bakar and _should_apply_official_override(fields["bahan_bakar"]):
        fields["bahan_bakar"] = make_ok(inferred_bahan_bakar, confidence=0.86, raw="official_section:bahan_bakar")

    inferred_type = _section_type_value(section)
    if inferred_type and (_should_apply_official_override(fields["tipe"]) or _needs_type_repair(fields["tipe"])):
        fields["tipe"] = make_ok(inferred_type, confidence=0.88, raw="official_section:tipe")

    year = _section_year(section)
    if year and _should_apply_official_override(fields["tahun_pembuatan"]):
        fields["tahun_pembuatan"] = FieldResult(
            value=year,
            confidence=0.9,
            status="ok",
            evidence=[year, "official_section:year"],
            raw="official_section:year",
        )

    rangka = _section_vehicle_id(section, r"(?:RANGK\w*|VIN|IDENT)", target="rangka")
    if rangka and _should_apply_official_vehicle_id_override(fields["nomor_rangka"], rangka):
        fields["nomor_rangka"] = FieldResult(
            value=rangka,
            confidence=0.9,
            status="ok",
            evidence=[rangka, "official_section:rangka"],
            raw="official_section:rangka",
        )

    mesin = _section_vehicle_id(section, r"(?:MESIN\w*|\bMESN\b|\bMESIY\b|\bMEIN\b|\bMFIN\b|\bENGINE\b|\bESIN\b)", target="mesin")
    if mesin and _should_apply_official_vehicle_id_override(fields["nomor_mesin"], mesin):
        fields["nomor_mesin"] = FieldResult(
            value=mesin,
            confidence=0.88,
            status="ok",
            evidence=[mesin, "official_section:mesin"],
            raw="official_section:mesin",
        )

    expiry = _section_expiry_date(section)
    if expiry and (
        _should_apply_official_override(fields["berlaku_sampai"])
        or _normalize_expiry_date_text(fields["berlaku_sampai"].value or "") is None
    ):
        fields["berlaku_sampai"] = make_ok(expiry, confidence=0.88, raw="official_section:berlaku_sampai")


def _should_apply_official_override(field: FieldResult) -> bool:
    value = (field.value or "").upper()
    return (
        field.status != "ok"
        or not field.raw
        or field.raw.startswith("fallback:")
        or len(value.strip()) <= 1
        or _is_noise_value(value)
        or _is_official_noise_value(value)
        or _is_broken_official_capture(value)
    )


def _should_apply_official_text_override(field: FieldResult) -> bool:
    value = collapse_spaces((field.value or "").upper())
    return (
        _should_apply_official_override(field)
        or len(value) <= 4
        or not re.search(r"[A-Z]", value)
        or _is_official_noise_value(value)
    )


def _should_apply_official_vehicle_id_override(field: FieldResult, candidate: str) -> bool:
    if _should_apply_official_override(field):
        return True
    if field.raw and field.raw.startswith("official_section:"):
        return False
    current = _normalize_vehicle_id(field.value or "") or ""
    return bool(candidate and candidate != current)


def _is_broken_official_capture(value: str) -> bool:
    compact = collapse_spaces(value).replace(" ", "")
    return bool(compact) and not any(ch.isalnum() for ch in compact) and len(compact) <= 3


def _section_vehicle_brand(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not re.search(r"\b(?:WARNA|TYPE|TIPE|JENIS|MODEL)\b|\bW[A-Z]?\s*N\s*A\b", line, flags=re.IGNORECASE):
            continue
        for candidate in reversed(_window(lines, index - 6, index)):
            if _is_vehicle_brand_candidate(candidate):
                return candidate

    start = _label_line_index(lines, r"\bNAMA\s+PEMILI\w*\b")
    scan_start = start + 1 if start is not None else 0
    for line in lines[scan_start:]:
        if _is_section_label(line):
            if re.search(r"\b(?:MER[EK]|TIPE|TYPE|JENIS|WARNA|MODEL|NOMOR|TAHUN|BERLAKU|BENSIN|SOLAR|LISTRIK)\b", line, flags=re.IGNORECASE):
                break
            continue
        if _is_vehicle_brand_candidate(line):
            return line
    return None


def _label_line_index(lines: list[str], pattern: str) -> int | None:
    for index, line in enumerate(lines):
        if re.search(pattern, line, flags=re.IGNORECASE):
            return index
    return None


def _is_vehicle_brand_candidate(value: str) -> bool:
    candidate = collapse_spaces(value)
    return (
        bool(re.fullmatch(r"[A-Z0-9]{2,12}", candidate))
        and not _is_noise_value(candidate)
        and not _is_official_noise_value(candidate)
        and not _extract_fuel_value_from_line(candidate)
        and not _normalize_vehicle_color(candidate)
    )


def _section_address(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not re.search(r"\b(?:ALAMAT|ADDRESS|ADDRES)\b", line, flags=re.IGNORECASE):
            continue
        parts: list[str] = []
        for candidate in _window(lines, index + 1, index + 8):
            cleaned = collapse_spaces(candidate.strip(" :.-"))
            if not cleaned:
                continue
            if _is_section_label(cleaned) and not re.search(r"\b(?:JL|JALAN|RT|RW|KEL|KEC)\b", cleaned, flags=re.IGNORECASE):
                break
            if _is_official_noise_value(cleaned) or _looks_like_amount(cleaned):
                continue
            if _looks_like_address_value(cleaned):
                parts.append(cleaned)
                continue
            if parts and re.search(r"\b(?:JAKSEL|JAKARTA|BANDUNG|TANGERANG|BEKASI|BOGOR|DEPOK)\b", cleaned, flags=re.IGNORECASE):
                parts.append(cleaned)
                break
        if parts:
            return collapse_spaces(" ".join(parts))
    return None


def _looks_like_address_value(value: str) -> bool:
    upper = value.upper()
    if len(upper) <= 4:
        return False
    return bool(
        re.search(r"\b(?:JL|JALAN|GG|GANG|RT|RW|KEL|KEC|BLOK|NO\.?|RAYA)\b", upper)
        or re.search(r"\d+/\d+", upper)
    )


def _section_fuel_value(lines: list[str]) -> str | None:
    for line in lines:
        value = _extract_fuel_value_from_line(line)
        if value:
            return value
    return None


def _extract_fuel_value_from_line(line: str) -> str | None:
    candidates = ("LISTRIK", "BENSIN", "SOLAR", "DIESEL", "LPG", "ELEKTRIK", "ELEKTRIKAL")
    upper = line.upper()
    for candidate in candidates:
        if re.search(rf"\b{candidate}\b", upper):
            if candidate in {"ELEKTRIK", "ELEKTRIKAL"}:
                return "LISTRIK"
            return candidate
    return None


def _section_type_value(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        match = re.search(r"\b(?:TYPE|TIPE)\w*\b\s*[:\-]?\s*(.*)$", line, flags=re.IGNORECASE)
        if not match:
            continue
        inline = _clean_type_candidate(match.group(1))
        if _is_strong_type_value(inline):
            return inline
        for candidate in reversed(_window(lines, index - 6, index)):
            if _is_section_label(candidate) or _looks_like_amount(candidate):
                continue
            cleaned = _clean_type_candidate(candidate)
            if _is_strong_type_value(cleaned):
                return cleaned
        for candidate in _window(lines, index + 1, index + 12):
            if _is_section_label(candidate) or _looks_like_amount(candidate):
                continue
            cleaned = _clean_type_candidate(candidate)
            if _is_strong_type_value(cleaned):
                return cleaned
    for line in lines:
        if not re.search(r"(?:BAHAN\s*BAKAR|BAHANBAKAR|HAN\s*BAKAR|HANBAKAR)", line, flags=re.IGNORECASE):
            continue
        cleaned = collapse_spaces(line.strip(" :.-"))
        if _is_strong_type_value(cleaned):
            return cleaned
    return None


def _model_type_value(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not re.search(r"\bMODEL\b", line, flags=re.IGNORECASE):
            continue
        for candidate in reversed(_window(lines, index - 6, index)):
            if _is_section_label(candidate) or _looks_like_amount(candidate):
                continue
            cleaned = _clean_type_candidate(candidate)
            if _is_strong_type_value(cleaned) or _is_textual_type_value(cleaned):
                return cleaned
    return None


def _clean_type_candidate(value: str) -> str:
    cleaned = collapse_spaces(value.strip(" :.-"))
    cleaned = re.sub(
        r"(?:BAHAN\s*BAKAR|BAHANBAKAR|SUMBER\s*ENERG\w*|SUMBERENERG\w*|TYPE\s*FUEL.*|FUELENERGY.*|JENIS|WARNA\s*TNKB|TAHUN\s+REGISTRASI|TAHUN\s+PEMBUATAN|MODEL).*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return collapse_spaces(cleaned.strip(" :.-"))


def _is_strong_type_value(value: str) -> bool:
    if _is_weak_type_value(value):
        return False
    if normalize_plate_number(value):
        return False
    compact = re.sub(r"[^A-Z0-9]", "", value.upper())
    return bool(re.search(r"[A-Z]", value, flags=re.IGNORECASE)) and any(char.isdigit() for char in value) and (
        "-" in value
        or "/" in value
        or "(" in value
        or re.search(r"\d\.\d", value)
        or re.search(r"[A-Z]\d|\d[A-Z]", compact)
    )


def _is_textual_type_value(value: str) -> bool:
    if _is_weak_type_value(value) or normalize_plate_number(value):
        return False
    cleaned = collapse_spaces(value.upper())
    if not 5 <= len(cleaned) <= 40 or not re.search(r"[A-Z]", cleaned):
        return False
    return any(marker in cleaned for marker in [".", "/", "-", "("])


def _is_weak_type_value(value: str) -> bool:
    cleaned = collapse_spaces(value.strip(" :.-"))
    return not cleaned or len(cleaned) <= 4 or _is_noise_value(cleaned) or _is_type_noise_value(cleaned)


def _is_type_noise_value(value: str) -> bool:
    cleaned = collapse_spaces(value.upper())
    if _normalize_vehicle_color(cleaned):
        return True
    if _normalize_standalone_year(cleaned):
        return True
    if _extract_fuel_value_from_line(cleaned):
        return True
    generic_words = {
        "DUMP",
        "DUMPER",
        "JEEP",
        "KENDARAAN",
        "KHUSUS",
        "MB",
        "MINIBUS",
        "MOBIL",
        "PENUMPA",
        "PENUMPANG",
        "SEDAN",
        "TR",
        "TRUCK",
    }
    words = {word for word in re.findall(r"[A-Z]+", cleaned)}
    return bool(words) and words.issubset(generic_words)


def _section_color_value(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not re.search(r"\bWARNA\w*\b|\bW[A-Z]?\s*N\s*A\b", line, flags=re.IGNORECASE):
            continue
        if re.search(r"\bTNKB\b", line, flags=re.IGNORECASE):
            continue
        inline = re.sub(r"^.*?\bWARNA\w*\b\s*[:\-]?", "", line, flags=re.IGNORECASE).strip()
        color = _normalize_vehicle_color(inline)
        if color:
            return color
        for candidate in _window(lines, index + 1, index + 6):
            color = _normalize_vehicle_color(candidate)
            if color:
                return color
    return None


def _needs_type_repair(field: FieldResult) -> bool:
    if field.status != "ok":
        return True
    value = field.value or ""
    return _is_weak_type_value(value) or (not _is_strong_type_value(value) and not _is_textual_type_value(value))


def _normalize_vehicle_color(value: str) -> str | None:
    cleaned = collapse_spaces(re.sub(r"[^A-Z ]+", " ", value.upper()))
    if cleaned in VEHICLE_COLOR_VALUES:
        return cleaned
    for color in VEHICLE_COLOR_VALUES:
        if re.search(rf"\b{re.escape(color)}\b", cleaned):
            return color
    return None


def _normalize_vehicle_category(value: str) -> str | None:
    cleaned = collapse_spaces(re.sub(r"[^A-Z ]+", " ", value.upper()))
    category_patterns = [
        ("MOBIL PENUMPANG", r"\bMOBIL\s+PEN[UO]MPA\w*\b"),
        ("KENDARAAN KHUSUS", r"\bKENDARAAN\s+KHUSUS\b"),
        ("SEPEDA MOTOR", r"\bSEPEDA\s+MOTOR\b"),
        ("MOBIL BARANG", r"\bMOBIL\s+BARANG\b"),
        ("TRUCK", r"\bTRU?C?K\b"),
        ("MINIBUS", r"\bMINI\s*BUS\b"),
    ]
    for normalized, pattern in category_patterns:
        if re.search(pattern, cleaned):
            return normalized
    if cleaned in {"CATEGORY", "JENIS", "BENIS", "JENS"}:
        return None
    return None


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
        if not _is_current_plate_label(line):
            continue
        candidate = _plate_from_label_window(lines, index, allow_digit_three=False)
        if candidate:
            return candidate
    return None


def _section_owner(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if _matches_english_owner_label(line):
            owner = _owner_after_label(lines, index)
            if owner:
                return owner

    for index, line in enumerate(lines):
        if not _matches_owner_label(line):
            continue
        owner = _owner_after_label(lines, index)
        if owner:
            return owner
    return None


def _owner_after_label(lines: list[str], index: int) -> str | None:
    for offset, candidate in enumerate(_window(lines, index + 1, index + 22), start=1):
        if _matches_english_owner_label(candidate):
            nested = _owner_after_label(lines, index + offset)
            if nested:
                return nested
            continue
        if re.search(r"\b(?:ALAMAT|ADDRESS|MERK|MEREK|BRAND|WARNA|TYPE|TIPE|JENIS|CATEGORY|MODEL)\b", candidate, flags=re.IGNORECASE):
            break
        if re.search(r"\b(?:STNK|NIK|KITAS|KITAP)\b", candidate, flags=re.IGNORECASE):
            continue
        if _looks_like_owner_name_after_label(candidate):
            return _normalize_owner_candidate(candidate)
    return None


def _section_simple_value(lines: list[str], label_pattern: str) -> str | None:
    for index, line in enumerate(lines):
        match = re.search(rf"{label_pattern}\s*[:\-]?\s*(.*)$", line, flags=re.IGNORECASE)
        if not match:
            continue
        inline_value = match.group(1).strip()
        if (
            inline_value
            and not _is_section_label(inline_value)
            and not _looks_like_amount(inline_value)
            and not _is_official_noise_value(inline_value)
        ):
            return collapse_spaces(inline_value.strip(" :.-"))
        for candidate in _window(lines, index + 1, index + 5):
            if _is_section_label(candidate) or _looks_like_amount(candidate) or _is_official_noise_value(candidate):
                continue
            cleaned = collapse_spaces(candidate.strip(" :.-"))
            if cleaned and re.search(r"[A-Z]", cleaned, flags=re.IGNORECASE):
                return cleaned
    return None


def _section_year(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not _matches_manufacture_year_label(line):
            continue
        for candidate in _window(lines, index + 1, index + 6):
            year = _normalize_standalone_year(candidate)
            if year:
                return year
    for index, line in enumerate(lines):
        if not re.search(r"\bTAHUN\s+REGISTR\w*", line, flags=re.IGNORECASE):
            continue
        for candidate in reversed(_window(lines, index - 4, index)):
            year = _normalize_standalone_year(candidate)
            if year:
                return year
    return None


def _matches_manufacture_year_label(line: str) -> bool:
    if re.search(r"REG\w*", line, flags=re.IGNORECASE) and not re.search(r"PE[MN]?B?U?AT", line, flags=re.IGNORECASE):
        return False
    if re.search(r"\bTAH[U]?[NI]\s*P[EE]?M[BU]U?AT\w*", line, flags=re.IGNORECASE):
        return True
    return bool(match_stnk_label(line, field_name="tahun_pembuatan"))


def _matches_year_context_label(line: str) -> bool:
    return bool(
        re.search(
            r"\b(?:MODEL|TAH[U]?[NI]?\s*REG\w*|TA[HB][NU]?\s*REG\w*|TARUN\s*REG\w*|REGSTR\w*|REGISTR\w*)\b",
            line,
            flags=re.IGNORECASE,
        )
    )


def _matches_vehicle_id_label(line: str, target: str) -> bool:
    field_name = "nomor_rangka" if target == "rangka" else "nomor_mesin"
    return bool(match_stnk_label(line, field_name=field_name))


def _matches_owner_label(line: str) -> bool:
    return bool(
        re.search(r"\b(?:NAMA\s*PEMILI\w*|NAMAPEMILI\w*|MA\s*PEMILIK|MAPEMILIK)\b", line, flags=re.IGNORECASE)
        or match_stnk_label(line, field_name="nama_pemilik")
    )


def _matches_english_owner_label(line: str) -> bool:
    return bool(re.search(r"\bNAME\s+OF\s+OWN\w*\b", line, flags=re.IGNORECASE))


def _section_vehicle_id(lines: list[str], label_pattern: str, target: str) -> str | None:
    for index, line in enumerate(lines):
        if not re.search(label_pattern, line, flags=re.IGNORECASE) and not _matches_vehicle_id_label(line, target):
            continue
        best: tuple[int, str] | None = None
        candidate_lines = [(True, candidate) for candidate in _window(lines, index + 1, index + 7)]
        if target == "mesin":
            after_label = _best_vehicle_id_candidate(candidate_lines, target)
            if after_label:
                return after_label
            candidate_lines = [(False, candidate) for candidate in reversed(_window(lines, index - 4, index))]
        for is_after_label, candidate_line in candidate_lines:
            if _is_section_label(candidate_line) or _looks_like_amount(candidate_line):
                continue
            normalized = _normalize_vehicle_id(candidate_line)
            if not normalized or _is_noise_value(normalized) or _contains_month_marker(normalized):
                continue
            if normalize_plate_number(normalized) or not any(char.isdigit() for char in normalized):
                continue
            score = _vehicle_id_score(normalized, target)
            if score <= 0:
                continue
            if target == "mesin" and is_after_label:
                score += 1
            if best is None or score > best[0] or (score == best[0] and _prefer_vehicle_id_candidate(normalized, best[1], target)):
                best = (score, normalized)
        if best:
            return best[1]
    return None


def _best_vehicle_id_candidate(candidate_lines: list[tuple[bool, str]], target: str) -> str | None:
    best: tuple[int, str] | None = None
    for is_after_label, candidate_line in candidate_lines:
        if _is_section_label(candidate_line) or _looks_like_amount(candidate_line):
            continue
        normalized = _normalize_vehicle_id(candidate_line)
        if not normalized or _is_noise_value(normalized) or _contains_month_marker(normalized):
            continue
        if normalize_plate_number(normalized) or not any(char.isdigit() for char in normalized):
            continue
        score = _vehicle_id_score(normalized, target)
        if score <= 0:
            continue
        if target == "mesin" and is_after_label:
            score += 1
        if best is None or score > best[0] or (score == best[0] and _prefer_vehicle_id_candidate(normalized, best[1], target)):
            best = (score, normalized)
    return best[1] if best else None


def _section_expiry_date(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not re.search(r"\bBERLAKU\s+S\w*", line, flags=re.IGNORECASE):
            continue
        for candidate in _window(lines, index + 1, index + 7):
            expiry = _normalize_expiry_date_text(candidate)
            if expiry:
                return expiry
            match = re.search(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b", candidate)
            if match:
                return match.group(0)
    return None


def _normalize_expiry_date_text(value: str) -> str | None:
    month_pattern = r"Januari|Februari|Maret|April|Mei|Juni|Jund|Junt|Juli|Agustus|September|Oktober|November|Desember"
    match = re.search(rf"\b(\d{{1,2}})\s+({month_pattern})\s+(\d{{4}})\b", value, flags=re.IGNORECASE)
    if not match:
        return None
    month = _normalize_indonesian_month(match.group(2))
    if not month:
        return None
    return f"{int(match.group(1))} {month} {match.group(3)}"


def _normalize_indonesian_month(value: str) -> str | None:
    normalized = value.upper()
    months = {
        "JANUARI": "Januari",
        "FEBRUARI": "Februari",
        "MARET": "Maret",
        "APRIL": "April",
        "MEI": "Mei",
        "JUNI": "Juni",
        "JUND": "Juni",
        "JUNT": "Juni",
        "JULI": "Juli",
        "AGUSTUS": "Agustus",
        "SEPTEMBER": "September",
        "OKTOBER": "Oktober",
        "NOVEMBER": "November",
        "DESEMBER": "Desember",
    }
    return months.get(normalized)


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


def _is_official_noise_value(value: str) -> bool:
    compact = re.sub(r"[^A-Z0-9]", "", value.upper())
    spaced = collapse_spaces(re.sub(r"[^A-Z0-9 ]+", " ", value.upper()))
    if not compact:
        return True
    if compact in {"CATEGORY", "COMPANYREGISTRATIONNUMBER", "FUELENERGYSOURCES", "NAMEOFOWNER", "ADDRESS", "ADDRES", "BRAND"}:
        return True
    if compact.startswith("BNKENDARAAN"):
        return True
    return spaced in {"CATEGORY", "COMPANY REGISTRATION NUMBER", "FUEL ENERGY SOURCES", "NAME OF OWNER"}


def _fallback_plate(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not _is_current_plate_label(line):
            continue
        candidate = _plate_from_label_window(lines, index, allow_digit_three=True)
        if candidate:
            return candidate

    candidates: list[str] = []
    for line in lines:
        candidate = _plate_candidate_from_line(line, allow_digit_three=False, strict_line=True)
        if candidate:
            candidates.append(candidate)
    return _best_plate_candidate(candidates)


def _best_plate_candidate(candidates: list[str]) -> str | None:
    if not candidates:
        return None
    return max(candidates, key=_plate_candidate_score)


def _plate_candidate_score(candidate: str) -> tuple[int, int, int]:
    compact = re.sub(r"[^A-Z0-9]", "", candidate.upper())
    match = re.fullmatch(r"([A-Z]{1,2})(\d{1,4})([A-Z]{0,3})", compact)
    if not match:
        return (0, 0, 0)
    return (len(match.group(2)), len(match.group(3)), len(match.group(1)))


def _fallback_owner(lines: list[str], plate: str | None) -> str | None:
    for index, line in enumerate(lines):
        if _matches_english_owner_label(line):
            owner = _owner_after_label(lines, index)
            if owner:
                return owner

    for index, line in enumerate(lines):
        if not _matches_owner_label(line):
            continue
        owner = _owner_after_label(lines, index)
        if owner:
            return owner
        for candidate in _window(lines, index - 5, index):
            if _looks_like_owner_name_after_label(candidate):
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


def _fallback_company_owner(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not _matches_owner_label(line):
            continue
        for candidate in _window(lines, index + 1, index + 10):
            if _is_owner_scan_stop_label(candidate) and not _looks_like_company_name(candidate):
                continue
            if _looks_like_company_name(candidate):
                return _normalize_owner_candidate(candidate)
    return None


def _fallback_manufacture_year(lines: list[str]) -> str | None:
    for line in lines:
        if _looks_like_noisy_manufacture_year_line(line):
            year = _normalize_year_pair(line)
            if year:
                return year

    for index, line in enumerate(lines):
        if not _matches_manufacture_year_label(line):
            continue
        for candidate in _window(lines, index + 1, index + 18):
            year = _normalize_standalone_year(candidate)
            if year:
                return year
        for candidate in reversed(_window(lines, index - 6, index)):
            year = _normalize_standalone_year(candidate)
            if year:
                return year

    for index, line in enumerate(lines):
        if not _matches_year_context_label(line):
            continue
        year = _normalize_year_pair(line) or _normalize_standalone_year(line) or _normalize_year(line)
        if year:
            return year
        for candidate in _window(lines, index + 1, index + 8):
            year = _normalize_standalone_year(candidate)
            if year:
                return year
        for candidate in reversed(_window(lines, index - 5, index)):
            year = _normalize_standalone_year(candidate)
            if year:
                return year

    standalone_years = [_normalize_standalone_year(line) for line in lines]
    for index in range(len(standalone_years) - 1):
        if standalone_years[index] and standalone_years[index] == standalone_years[index + 1]:
            return standalone_years[index]
    return None


def _fallback_vehicle_id(lines: list[str], target: str) -> str | None:
    label_pattern = (
        r"(RANGKA|RANGK|RANGIA|RANGO|RANGON|RWNGKA|RNAKB|IDENT|VIN)"
        if target == "rangka"
        else r"(MESIN|\bMESN\b|\bMESIY\b|\bMEIN\b|\bMFIN\b|\bENGINE\b|\bESIN\b)"
    )
    best: tuple[int, str] | None = None
    for index, line in enumerate(lines):
        if not re.search(label_pattern, line, flags=re.IGNORECASE) and not _matches_vehicle_id_label(line, target):
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
                if best is None or score > best[0] or (score == best[0] and _prefer_vehicle_id_candidate(candidate, best[1], target)):
                    best = (score, candidate)
        if target == "rangka":
            for candidate in _split_rangka_candidates(candidate_lines):
                score = _vehicle_id_score(candidate, target) + 1
                if score <= 1:
                    continue
                if best is None or score > best[0] or (score == best[0] and _prefer_vehicle_id_candidate(candidate, best[1], target)):
                    best = (score, candidate)
    if best is None and target == "rangka":
        for candidate_line in lines:
            for candidate in _vehicle_id_candidates(candidate_line, target):
                score = _strong_rangka_score(candidate)
                if score <= 0:
                    continue
                if best is None or score > best[0] or (score == best[0] and _prefer_vehicle_id_candidate(candidate, best[1], target)):
                    best = (score, candidate)
    if best is None and target == "mesin":
        for candidate_line in lines:
            for candidate in _vehicle_id_candidates(candidate_line, target):
                score = _vehicle_id_score(candidate, target)
                if score < 6:
                    continue
                if best is None or score > best[0] or (score == best[0] and _prefer_vehicle_id_candidate(candidate, best[1], target)):
                    best = (score, candidate)
    return best[1] if best else None


def _vehicle_id_candidates(value: str, target: str) -> list[str]:
    candidates: list[str] = []
    normalized = _normalize_vehicle_id(value)
    if normalized:
        if target == "mesin":
            candidates.extend(_engine_ocr_variants(normalized))
        candidates.append(normalized)

    if target == "rangka":
        compact = "".join(char for char in value.upper() if char.isalnum())
        if compact.startswith("M3E") and len(compact) >= 15:
            candidates.append(("MJE" + compact[3:])[:17])
        for prefix in RANGKA_PREFIXES:
            for match in re.finditer(prefix, compact):
                candidate = compact[match.start() : match.start() + 17]
                if 15 <= len(candidate) <= 17:
                    candidates.append(candidate)
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


def _engine_ocr_variants(value: str) -> list[str]:
    variants: list[str] = []
    if value.startswith(("JOB", "J0B", "JO8")) and len(value) >= 6:
        variants.append("J08" + value[3:])
    if value.startswith("J088") and len(value) >= 7:
        variants.append("J08E" + value[4:])
    return variants


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
    noise_value = value.upper().translate(str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B"}))
    if any(
        noise in value or noise in noise_value
        for noise in [
            "BAHAN",
            "BAKAR",
            "DATEOFEXPIRE",
            "KEL",
            "KODE",
            "LOKASI",
            "MESIN",
            "MODEL",
            "PEMBUATAN",
            "PENGESAHAN",
            "RANGKA",
            "SILINDER",
            "STNK",
            "TAHUN",
            "TIPE",
            "TYPE",
            "WARNA",
            "VIN",
            "NIK",
            "KITAS",
            "KITAP",
            "NOMOR",
            "POLISI",
            "PEMILIK",
        ]
    ):
        return 0
    if _contains_month_marker(value):
        return 0
    if normalize_plate_number(value):
        return 0
    if target == "mesin" and (value.endswith("RP") or re.search(r"\d{6,}(?:19|20)\d{2}RP$", value)):
        return 0
    if target == "mesin" and re.fullmatch(r"B\d{6,}", value):
        return 0
    if value.isdigit() and _normalize_year(value):
        return 0
    if not any(char.isdigit() for char in value) or not any(char.isalpha() for char in value):
        if target == "mesin" and value.isdigit() and 5 <= len(value) <= 10:
            return 3
        return 0
    if target == "rangka":
        if re.match(r"(?:FM|UCR)", value):
            return 0
        if len(value) == 17:
            return 8
        if 15 <= len(value) <= 18:
            return 6
        return 0
    if re.match(r"N?(?:MJE|MF|MH|MP|MM|JM|LF|LG|MR|R0|RF)", value):
        return 0
    if re.match(r"(?:J0?8|JO8|W0?4D|WO4D|4D)", value):
        return 7
    if re.fullmatch(r"\d{4,}[A-Z]\d{4,}", value) and 10 <= len(value) <= 16:
        return 6
    if re.fullmatch(r"[A-Z]\d{7,}[A-Z]?", value) and 8 <= len(value) <= 14:
        return 4
    if 15 <= len(value) <= 20:
        if re.match(r"(?:JM|LF|LG|MH|MJ|MM|MP|MR|R0|RF)", value):
            return 0
        return 4
    if 5 <= len(value) <= 14 and not re.fullmatch(r"[A-Z]\d{7,}[A-Z]?", value):
        return 5 if re.match(r"(?:[A-Z]{2}\d|\d[A-Z]{2})", value) else 3
    return 0


def _strong_rangka_score(value: str) -> int:
    if _vehicle_id_score(value, "rangka") <= 0:
        return 0
    return 5 if value.startswith(RANGKA_PREFIXES) else 0


def _window(lines: list[str], start: int, end: int) -> list[str]:
    return lines[max(0, start) : min(len(lines), end)]


def _looks_like_person_name(value: str) -> bool:
    cleaned = collapse_spaces(value.upper().replace(",", " "))
    if "XXX" in cleaned:
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


def _looks_like_owner_name_after_label(value: str) -> bool:
    if _looks_like_owner_name(value):
        return True
    cleaned = collapse_spaces(value.upper().replace(",", " "))
    if "XXX" in cleaned:
        return False
    if _looks_like_amount(cleaned) or any(char.isdigit() for char in cleaned):
        return False
    if not re.fullmatch(r"[A-Z][A-Z .']{4,}", cleaned):
        return False
    words = [word for word in cleaned.split() if word]
    if not 1 <= len(words) <= 5:
        return False
    if any("KOTA" in word for word in words):
        return False
    return not any(word in PERSON_NOISE_WORDS for word in words)


def _is_owner_scan_stop_label(value: str) -> bool:
    return bool(
        re.search(
            r"\b(?:A?\s*ALAMAT|AMAT|MERK|MEREK|TYPE|TIPE|JENIS|WARNA|KODE|ISI|KEC)\b",
            value,
            flags=re.IGNORECASE,
        )
    )


def _looks_like_company_name(value: str) -> bool:
    cleaned = collapse_spaces(value.upper().replace(",", " "))
    if "XXX" in cleaned or any(char.isdigit() for char in cleaned):
        return False
    words = [word.strip(".") for word in cleaned.replace(".", " ").split() if word.strip(".")]
    if len(words) < 3 or words[0] not in COMPANY_PREFIXES:
        return False
    return not any(word in PERSON_NOISE_WORDS for word in words[1:])


def _normalize_owner_candidate(value: str) -> str:
    cleaned = collapse_spaces(value.upper().replace(",", " "))
    words = [word.strip(".") for word in cleaned.replace(".", " ").split() if word.strip(".")]
    if words and words[0] == "FT":
        words[0] = "PT"
        return " ".join(words)
    return collapse_spaces(value)


def _normalize_stnk_plate(value: str | None) -> str | None:
    normalized = normalize_plate_number(value)
    if normalized or not value:
        return normalized
    compact = re.sub(r"[\s.\-]", "", value.upper())
    suffix_translation = str.maketrans({"0": "O", "1": "I", "3": "J", "5": "S", "6": "G", "8": "B"})
    match = re.fullmatch(r"([A-Z]{1,2})(\d{1,4})([A-Z0-9]{1,3})", compact)
    if match:
        suffix = match.group(3).translate(suffix_translation)
        return normalize_plate_number(f"{match.group(1)}{match.group(2)}{suffix}")
    match = re.fullmatch(r"([83])(\d{1,4})([A-Z0-9]{1,3})", compact)
    if match:
        suffix = match.group(3).translate(suffix_translation)
        return normalize_plate_number(f"B{match.group(2)}{suffix}")
    return None


def _plate_has_suffix(value: str) -> bool:
    compact = re.sub(r"[^A-Z0-9]", "", value.upper())
    return bool(re.fullmatch(r"[A-Z]{1,2}\d{1,4}[A-Z]{1,3}", compact))


def _is_current_plate_label(line: str) -> bool:
    if re.search(r"\b(?:NO|NOMOR)\s*REGISTRASI\s+LAMA\b", line, flags=re.IGNORECASE):
        return False
    return bool(
        re.search(r"\b(?:NOMOR|NO)\s*\.?\s*POLIS\w*\b", line, flags=re.IGNORECASE)
        or re.search(r"\b(?:NRKB|NOMOR\s*REGISTRAS\w*)\b", line, flags=re.IGNORECASE)
    )


def _plate_from_label_window(lines: list[str], index: int, allow_digit_three: bool) -> str | None:
    inline = _plate_candidate_from_line(lines[index], allow_digit_three=allow_digit_three, strict_line=False)
    if inline:
        return inline

    nearby = _window(lines, index + 1, index + 22)
    for candidate_line in nearby:
        candidate = _plate_candidate_from_line(candidate_line, allow_digit_three=allow_digit_three, strict_line=True)
        if candidate:
            return candidate

    for offset in range(0, len(nearby) - 1):
        prefix = collapse_spaces(nearby[offset].upper().strip(" :.-"))
        suffix = collapse_spaces(nearby[offset + 1].upper().strip(" :.-"))
        if not re.fullmatch(r"[A-Z]{1,2}|[83]", prefix):
            continue
        if not re.fullmatch(r"\d{1,4}\s+[A-Z0-9]{1,3}", suffix):
            continue
        candidate = _normalize_stnk_plate(f"{prefix} {suffix}")
        if candidate:
            return candidate
    return None


def _plate_candidate_from_line(line: str, allow_digit_three: bool, strict_line: bool) -> str | None:
    upper = line.upper()
    if "XXX" in upper:
        return None
    if re.search(r"\b(?:ALAMAT|ADDRESS|JL|JALAN|RT|RW|KEL|KEC|ENGINE|MESIN)\b", upper):
        return None
    digit_prefix = "|8|3" if allow_digit_three else "|8"
    pattern = rf"(?<![A-Z0-9])([A-Z]{{1,2}}{digit_prefix})\s*[-.]?\s*(\d{{1,4}})\s*[-.]?\s*([A-Z0-9]{{1,3}})(?![A-Z0-9])"
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
    slash_match = re.fullmatch(r"[:\s.\-/]*(\d{4})\s*/\s*\d{4}[:\s.\-/]*", mapped)
    if slash_match:
        return _normalize_year(slash_match.group(1))
    if not re.fullmatch(r"[:\s.\-/]*\d{4}[:\s.\-/]*", mapped):
        return None
    return _normalize_year(mapped)


def _normalize_year_pair(value: str) -> str | None:
    mapped = value.upper().translate(str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1"}))
    match = re.search(r"(?<!\d)(\d{4})\s*/\s*\d{4}(?!\d)", mapped)
    if not match:
        return None
    return _normalize_year(match.group(1))


def _looks_like_noisy_manufacture_year_line(value: str) -> bool:
    compact = _normalize_label_text(value)
    if not _normalize_year_pair(value):
        return False
    return any(marker in compact for marker in ["PEMBUAT", "PERAKIT", "PERAT", "TAHUNPEM", "AMNPERAT"])


def _contains_month_marker(value: str) -> bool:
    upper = value.upper()
    return any(marker in upper for marker in MONTH_MARKERS)
