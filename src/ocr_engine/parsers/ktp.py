from __future__ import annotations

import re

from ocr_engine.parsers.common import capture_after_label, make_invalid, make_missing, make_ok, normalized_lines
from ocr_engine.postal_code import lookup_postal_code
from ocr_engine.schemas import DocumentResult, FieldResult
from ocr_engine.validators import normalize_nik


KTP_LABELS: dict[str, list[str]] = {
    "provinsi": ["Provinsi", "PROVINSI"],
    "kabupaten_kota": ["Kota Administrasi", "KOTA ADMINISTRASI", "Kabupaten", "KABUPATEN", "Kota", "KOTA"],
    "nik": ["NIK", "NTK", "HIK"],
    "nama": ["Nama", "NAMA", "Name", "Nams", "Nania", "Namat"],
    "tempat_tanggal_lahir": [
        "Tempat/Tgl Lahir",
        "Tempat Tgl Lahir",
        "Tempat/Tanggal Lahir",
        "TempatTglLahir",
    ],
    "jenis_kelamin": ["Jenis Kelamin"],
    "alamat": ["Alamat"],
    "rt_rw": ["RT/RW", "RTRW"],
    "kelurahan_desa": [
        "Kel/Desa",
        "Kel Desa",
        "KelDesa",
        "Kelurahan",
        "Desa",
        "KeWDesa",
        "KevDesa",
        "Kel/Desn",
        "Kel/Dosa",
        "Kel/Dess",
        "Kal/Dess",
        "NOesa",
        "el/Desa",
    ],
    "kecamatan": ["Kecamatan", "Kecamnatan", "Kecamatar", "Kecanatan", "KecaMatEN", "Xecamatan", "ecomatan", "ecamatan"],
    "agama": ["Agama"],
    "status_perkawinan": ["Status Perkawinan"],
    "pekerjaan": ["Pekerjaan", "Pekeriaan", "Pekerian", "Pokerjaan", "Pakeraan", "Pekerean"],
    "kewarganegaraan": ["Kewarganegaraan", "Kewargane", "Kewarganegaraar", "Kewarganegaraart", "Kevrganegaraan"],
    "kode_pos": ["Kode Pos", "Kodepos"],
    "berlaku_hingga": [
        "Berlaku Hingga",
        "BerlakuHingga",
        "Berlaku Hing",
        "Barlaku Hingga",
        "Berlaky Hingga",
        "Borlaku Hingga",
        "Berbaky Hingga",
        "Berfaku Hingga",
        "Serfaku Hingga",
        "Bertaku Hingga",
        "Berlaku Hingoa",
        "Berlaku Hingge",
        "Berlaku Hinggs",
        "Berlaku HnOgs",
    ],
}

KTP_REQUIRED_FIELDS = ["nik", "nama", "alamat"]
FUZZY_LABEL_PATTERN = re.compile(
    r"\b(?:"
    r"NIK|NAMA|TEMP\w*|TGL|TANGGAL|LAHIR|JENIS|KELAMIN|ALAMAT|"
    r"RT\s*/?\s*(?:RW|AW)|AT\s*/?\s*RW|RTRW|RTAW|"
    r"KEL(?:/|\b)|KELURAHAN|DESA|KECAM\w*|AGAMA|STATUS|PERKAW\w*|"
    r"PEKER\w*|KEWARG\w*|BERLAKU|HINGGA|PROVINSI|KABUPATEN|KOTA|GOL|DARAH"
    r")\b",
    flags=re.IGNORECASE,
)
ADDRESS_ANCHOR_PATTERN = re.compile(r"\b(?:RT\s*/?\s*(?:RW|AW|RV)|AT\s*/?\s*RW|ATIAW|RTIAW|RTRW|RTAW)\b", flags=re.IGNORECASE)
JOB_LABEL_PATTERN = re.compile(r"\b(?:PEKER\w*|POKERJAAN|PAKERAAN|PEKEREAN)\b", flags=re.IGNORECASE)
FIELD_VALUE_BLOCKLIST = {
    "LAKI-LAKI",
    "LAKI LAKI",
    "PEREMPUAN",
    "ISLAM",
    "KRISTEN",
    "KATHOLIK",
    "KATOLIK",
    "HINDU",
    "BUDDHA",
    "BUDHA",
    "KONGHUCU",
    "KAWIN",
    "BELUM KAWIN",
    "BELUMKAWIN",
    "CERAI HIDUP",
    "CERAI MATI",
    "WNI",
    "WNA",
    "SEUMUR HIDUP",
}
RELIGION_KEYWORDS = {
    "ISLAM": "ISLAM",
    "KRISTEN": "KRISTEN",
    "KATOLIK": "KATHOLIK",
    "KATHOLIK": "KATHOLIK",
    "HINDU": "HINDU",
    "BUDHA": "BUDHA",
    "BUDDHA": "BUDHA",
    "KONGHUCU": "KONGHUCU",
    "KHONGHUCU": "KONGHUCU",
}
NON_NAME_KEYWORDS = {
    "KARYAWAN",
    "SWASTA",
    "WIRASWASTA",
    "PELAJAR",
    "MAHASISWA",
    "MENGURUS",
    "RUMAH",
    "TANGGA",
    "BURUH",
    "PETANI",
    "NELAYAN",
    "PEDAGANG",
    "PNS",
    "TNI",
    "POLRI",
    "GURU",
    "DOSEN",
    "DOKTER",
    "DRIVE",
    "MANAGE",
    "TOSHIBA",
    "TYPE",
    "HERE",
    "WINDOWS",
    "ISLAM",
    "KRISTEN",
    "KATHOLIK",
    "KATOLIK",
    "HINDU",
    "BUDDHA",
    "BUDHA",
    "KONGHUCU",
    "KAWIN",
    "CERAI",
    "WNI",
    "WNA",
    "SEUMUR",
    "HIDUP",
    "LAKI",
    "PEREMPUAN",
}
JOINED_NAME_PREFIXES = [
    "MUHAMMAD",
    "MOHAMAD",
    "MOCHAMAD",
    "ABDUL",
    "AHMAD",
    "ANNISA",
    "ANISA",
    "DEWI",
    "DIAH",
    "SITI",
    "NUR",
    "SRI",
    "TRI",
    "DWI",
    "EKO",
    "BUDI",
    "AGUS",
]
JOINED_NAME_SUFFIX_PREFIXES = {
    "AGUS",
    "AN",
    "ANG",
    "BUDI",
    "CAH",
    "DEWI",
    "DIAN",
    "DWI",
    "EKA",
    "FIT",
    "HART",
    "HIDAY",
    "KURN",
    "LEST",
    "MAUL",
    "NINGS",
    "NUG",
    "PRA",
    "PRI",
    "PUT",
    "RAH",
    "RAT",
    "RET",
    "RIZ",
    "SAP",
    "SET",
    "SUL",
    "SUP",
    "SUR",
    "SUS",
    "WAH",
    "WID",
    "YUL",
    "YUN",
}


def parse_ktp_text(raw_text: str) -> DocumentResult:
    fields: dict[str, FieldResult] = {}
    warnings: list[str] = []
    stop_labels = [label for labels in KTP_LABELS.values() for label in labels]

    for field_name, labels in KTP_LABELS.items():
        if field_name == "nik":
            fields[field_name] = _parse_nik(raw_text)
            continue

        value, raw = capture_after_label(raw_text, labels, stop_labels)
        fields[field_name] = make_ok(value, raw=raw) if value else make_missing()

    _apply_ktp_fallbacks(raw_text, fields)

    if fields["nik"].status == "invalid":
        warnings.append("invalid:nik")

    if _looks_like_bad_ktp_crop(raw_text, fields):
        warnings.append("quality:possible_non_ktp_crop")

    for required in KTP_REQUIRED_FIELDS:
        if fields[required].status == "missing":
            warnings.append(f"missing_required:{required}")

    return DocumentResult(
        document_type="KTP",
        schema_version="ktp.v1",
        fields=fields,
        warnings=warnings,
        raw_text=raw_text,
    )


def _parse_nik(raw_text: str) -> FieldResult:
    labelled_value, raw = capture_after_label(raw_text, KTP_LABELS["nik"], [label for labels in KTP_LABELS.values() for label in labels])
    candidates = []
    if labelled_value:
        candidates.append(labelled_value)
    candidates.extend(_standalone_nik_candidates(raw_text))
    candidates.extend(re.findall(r"\b\d[\d\s\-]{14,24}\d\b", raw_text))

    for candidate in candidates:
        normalized = normalize_nik(candidate)
        if normalized:
            return FieldResult(value=normalized, confidence=0.98, status="ok", evidence=[normalized], raw=raw)

    if labelled_value:
        corrected = _normalize_nik_with_ocr_corrections(labelled_value)
        if corrected:
            return FieldResult(value=corrected, confidence=0.86, status="ok", evidence=[corrected], raw=raw)

    if labelled_value:
        return make_invalid(labelled_value, raw=raw)
    short_candidate = _short_nik_candidate(raw_text)
    if short_candidate:
        return make_invalid(short_candidate)
    return make_missing()


def _standalone_nik_candidates(raw_text: str) -> list[str]:
    candidates: list[str] = []
    for line in normalized_lines(raw_text):
        for match in re.finditer(r"(?<!\d)(\d{16})(?!\d)", line):
            candidates.append(match.group(1))
    return candidates


def _short_nik_candidate(raw_text: str) -> str | None:
    for line in normalized_lines(raw_text):
        digits = re.sub(r"\D", "", line)
        if 14 <= len(digits) <= 15 and re.fullmatch(r"[:\s\d.\-/]+", line):
            return digits
    return None


def _normalize_nik_with_ocr_corrections(value: str) -> str | None:
    corrections = 0
    mapped_chars: list[str] = []
    for char in value.upper():
        if char.isdigit():
            mapped_chars.append(char)
        elif char in {",", "."}:
            corrections += 1
            mapped_chars.append("1")
        elif char in {"O", "Q"}:
            corrections += 1
            mapped_chars.append("0")
        elif char in {"I", "L", "|"}:
            corrections += 1
            mapped_chars.append("1")
        elif char == "S":
            corrections += 1
            mapped_chars.append("5")
        elif char == "B":
            corrections += 1
            mapped_chars.append("8")

    candidate = "".join(mapped_chars)
    if corrections <= 2 and len(candidate) == 16:
        return candidate
    return None


def _apply_ktp_fallbacks(raw_text: str, fields: dict[str, FieldResult]) -> None:
    lines = normalized_lines(raw_text)

    normalized_province = _normalize_province(fields["provinsi"].value or "")
    if normalized_province:
        fields["provinsi"] = make_ok(normalized_province, confidence=fields["provinsi"].confidence or 0.88)
    else:
        fallback_province = _fallback_province(lines)
        if fallback_province:
            fields["provinsi"] = make_ok(fallback_province, confidence=0.78)
        elif fields["provinsi"].status == "ok":
            fields["provinsi"] = make_missing()

    normalized_admin_area = _normalize_kabupaten_kota(fields["kabupaten_kota"].value or "")
    if normalized_admin_area:
        fields["kabupaten_kota"] = make_ok(
            normalized_admin_area,
            confidence=fields["kabupaten_kota"].confidence or 0.88,
        )
    else:
        fallback_admin_area = _fallback_kabupaten_kota(lines)
        if fallback_admin_area:
            fields["kabupaten_kota"] = make_ok(fallback_admin_area, confidence=0.78)
        elif fields["kabupaten_kota"].status == "ok":
            fields["kabupaten_kota"] = make_missing()

    if fields["nama"].status == "ok" and fields["nama"].value and _name_value_needs_repair(fields["nama"].value):
        repaired_name = _fallback_name_near_label(lines)
        fields["nama"] = make_ok(repaired_name, confidence=0.7) if repaired_name else make_missing()

    if fields["nama"].status == "missing":
        fallback_name = _fallback_name(lines)
        if fallback_name:
            fields["nama"] = make_ok(fallback_name, confidence=0.72)

    if fields["nama"].status == "missing":
        fallback_name = _fallback_name_after_birth_line(lines)
        if fallback_name:
            fields["nama"] = make_ok(fallback_name, confidence=0.7)

    if fields["nama"].status == "ok" and fields["nama"].value:
        repaired_name = _repair_joined_person_name(fields["nama"].value)
        if repaired_name != fields["nama"].value:
            fields["nama"] = make_ok(
                repaired_name,
                confidence=min(fields["nama"].confidence, 0.78),
                raw=fields["nama"].raw or "fallback:joined_name_spacing",
            )

    normalized_birth_place_date = _normalize_birth_place_date(fields["tempat_tanggal_lahir"].value or "")
    if normalized_birth_place_date:
        fields["tempat_tanggal_lahir"] = make_ok(normalized_birth_place_date, confidence=fields["tempat_tanggal_lahir"].confidence)
    else:
        fallback_ttl = _fallback_birth_place_date(lines)
        if fallback_ttl:
            fields["tempat_tanggal_lahir"] = make_ok(fallback_ttl, confidence=0.72)
        elif fields["tempat_tanggal_lahir"].status == "ok":
            fields["tempat_tanggal_lahir"] = make_missing()

    normalized_gender = _normalize_gender(fields["jenis_kelamin"].value or "")
    if normalized_gender:
        fields["jenis_kelamin"] = make_ok(normalized_gender, confidence=0.84)
    else:
        fallback_gender = _fallback_gender(lines)
        if fallback_gender:
            fields["jenis_kelamin"] = make_ok(fallback_gender, confidence=0.78)
        elif fields["jenis_kelamin"].status == "ok":
            fields["jenis_kelamin"] = make_missing()

    if fields["alamat"].status == "ok" and not _looks_like_valid_address_value(fields["alamat"].value or ""):
        fallback_address = _fallback_address(lines)
        fields["alamat"] = make_ok(fallback_address, confidence=0.72) if fallback_address else make_missing()

    if fields["alamat"].status == "missing":
        fallback_address = _fallback_address(lines)
        if fallback_address:
            fields["alamat"] = make_ok(fallback_address, confidence=0.72)

    normalized_rt_rw = _normalize_rt_rw(fields["rt_rw"].value or "")
    if normalized_rt_rw:
        fields["rt_rw"] = make_ok(normalized_rt_rw, confidence=0.84)
    else:
        fallback_rt_rw = _fallback_rt_rw(lines)
        if fallback_rt_rw:
            fields["rt_rw"] = make_ok(fallback_rt_rw, confidence=0.78)
        elif fields["rt_rw"].status == "ok":
            fields["rt_rw"] = make_missing()

    normalized_kelurahan = _normalize_region_value(fields["kelurahan_desa"].value or "")
    if normalized_kelurahan:
        fields["kelurahan_desa"] = make_ok(normalized_kelurahan, confidence=fields["kelurahan_desa"].confidence or 0.84)
    else:
        transposed_kelurahan, _ = _fallback_transposed_regions(lines)
        fallback_kelurahan = _fallback_region_value(lines, "kelurahan") or transposed_kelurahan
        if fallback_kelurahan:
            fields["kelurahan_desa"] = make_ok(fallback_kelurahan, confidence=0.78)
        elif fields["kelurahan_desa"].status == "ok":
            fields["kelurahan_desa"] = make_missing()

    normalized_kecamatan = _normalize_region_value(fields["kecamatan"].value or "")
    if normalized_kecamatan:
        fields["kecamatan"] = make_ok(normalized_kecamatan, confidence=fields["kecamatan"].confidence or 0.84)
    else:
        _, transposed_kecamatan = _fallback_transposed_regions(lines)
        fallback_kecamatan = _fallback_region_value(lines, "kecamatan") or transposed_kecamatan
        if fallback_kecamatan:
            fields["kecamatan"] = make_ok(fallback_kecamatan, confidence=0.78)
        elif fields["kecamatan"].status == "ok":
            fields["kecamatan"] = make_missing()

    normalized_marital_status = _normalize_marital_status(fields["status_perkawinan"].value or "")
    if normalized_marital_status:
        fields["status_perkawinan"] = make_ok(normalized_marital_status, confidence=0.88)
    else:
        fallback_status = _fallback_marital_status(lines)
        if fallback_status:
            fields["status_perkawinan"] = make_ok(fallback_status, confidence=0.78)
        elif fields["status_perkawinan"].status == "ok":
            fields["status_perkawinan"] = make_missing()
        else:
            fields["status_perkawinan"] = FieldResult(
                value="BELUM KAWIN",
                confidence=0.35,
                status="ok",
                evidence=["fallback:default_marital_status"],
                raw="fallback:default_marital_status",
            )

    normalized_religion = _normalize_religion(fields["agama"].value or "")
    if normalized_religion:
        fields["agama"] = make_ok(normalized_religion, confidence=0.88)
    else:
        fallback_religion = _fallback_religion(lines)
        if fallback_religion:
            fields["agama"] = make_ok(fallback_religion, confidence=0.78)
        elif fields["agama"].status == "ok":
            fields["agama"] = make_missing()

    normalized_job = _normalize_job(fields["pekerjaan"].value or "")
    if normalized_job:
        fields["pekerjaan"] = make_ok(normalized_job, confidence=0.82)
    else:
        fallback_job = _fallback_job(lines)
        if fallback_job:
            fields["pekerjaan"] = make_ok(fallback_job, confidence=0.76)
        elif fields["pekerjaan"].status == "ok":
            fields["pekerjaan"] = make_missing()

    normalized_citizenship = _normalize_citizenship(fields["kewarganegaraan"].value or "")
    if normalized_citizenship:
        fields["kewarganegaraan"] = make_ok(normalized_citizenship, confidence=0.88)
    else:
        fallback_citizenship = _fallback_citizenship(lines)
        if fallback_citizenship:
            fields["kewarganegaraan"] = make_ok(fallback_citizenship, confidence=0.78)
        elif fields["kewarganegaraan"].status == "ok":
            fields["kewarganegaraan"] = make_missing()

    normalized_expiry = _normalize_expiry(fields["berlaku_hingga"].value or "")
    if normalized_expiry:
        fields["berlaku_hingga"] = make_ok(normalized_expiry, confidence=fields["berlaku_hingga"].confidence or 0.84)
    else:
        fallback_expiry = _fallback_expiry(lines) or _fallback_transposed_expiry(lines)
        if fallback_expiry:
            fields["berlaku_hingga"] = make_ok(fallback_expiry, confidence=0.78)
        elif fields["berlaku_hingga"].status == "ok":
            fields["berlaku_hingga"] = make_missing()

    if fields["kode_pos"].status != "ok":
        postal_code = lookup_postal_code(fields)
        if postal_code:
            fields["kode_pos"] = FieldResult(
                value=postal_code.kode_pos,
                confidence=postal_code.confidence,
                status="ok",
                evidence=postal_code.evidence,
                raw="db_kode_wilayah",
                metadata={
                    "kelurahan": postal_code.kelurahan,
                    "kecamatan": postal_code.kecamatan,
                    "kode_kecamatan": postal_code.kode_kecamatan,
                    "kode_kota": postal_code.kode_kota,
                    "nama_kota": postal_code.nama_kota,
                    "kode_provinsi": postal_code.kode_provinsi,
                    "nama_provinsi": postal_code.nama_provinsi,
                    "alamat_lengkap": postal_code.alamat_lengkap,
                    "total_options": postal_code.total_options,
                    "match_status": postal_code.match_status,
                },
            )


def _normalize_province(value: str) -> str | None:
    cleaned = _clean_candidate_value(value).upper().strip(" :.-")
    cleaned = re.sub(r"^PROVINSI\s*", "", cleaned)
    cleaned = re.sub(r"^PROVINSI(?=[A-Z])", "", cleaned)
    cleaned = re.sub(r"\b(?:KABUPATEN|KOTA|NIK|NAMA)\b.*$", "", cleaned).strip(" :.-")
    if not cleaned or _contains_any_label(cleaned):
        return None
    if normalize_nik(cleaned) or any(char.isdigit() for char in cleaned):
        return None
    return cleaned if re.fullmatch(r"[A-Z][A-Z .'-]{2,}", cleaned) else None


def _fallback_province(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        upper = line.upper()
        if upper.startswith("PROVINSI"):
            normalized = _normalize_province(upper)
            if normalized:
                return normalized
            if index + 1 < len(lines):
                normalized = _normalize_province(lines[index + 1])
                if normalized:
                    return normalized
    return None


def _normalize_kabupaten_kota(value: str) -> str | None:
    cleaned = _clean_candidate_value(value).upper().strip(" :.-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\b(?:NIK|NAMA|TEMP\w*|TGL|TANGGAL|LAHIR)\b.*$", "", cleaned, flags=re.IGNORECASE).strip(" :.-")
    if not cleaned or _contains_non_admin_area_value_label(cleaned):
        return None
    if normalize_nik(cleaned) or any(char.isdigit() for char in cleaned):
        return None
    if cleaned in FIELD_VALUE_BLOCKLIST:
        return None
    if _normalize_religion(cleaned) or _normalize_gender(cleaned) or _normalize_marital_status(cleaned):
        return None
    cleaned = re.sub(r"^(?:KOTA\s+ADMINISTRASI|KABUPATEN|KAB\.?|KOTA)\s+", "", cleaned).strip(" :.-")
    if not cleaned:
        return None
    return cleaned if re.fullmatch(r"[A-Z][A-Z .'\-/]{2,}", cleaned) else None


def _fallback_kabupaten_kota(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        upper = line.upper()
        normalized = _admin_area_from_header_line(upper)
        if normalized:
            return normalized

        if not _is_kabupaten_kota_label(upper):
            continue

        for candidate in lines[index + 1 : min(len(lines), index + 4)]:
            if _contains_any_label(candidate.upper()):
                break
            normalized = _normalize_kabupaten_kota(candidate)
            if normalized:
                return normalized
    return None


def _admin_area_from_header_line(line: str) -> str | None:
    match = re.search(r"\b(?:KABUPATEN|KAB\.?|KOTA(?:\s+ADMINISTRASI)?)\s+(.+)$", line, flags=re.IGNORECASE)
    if not match:
        return None
    return _normalize_kabupaten_kota(match.group(0))


def _is_kabupaten_kota_label(value: str) -> bool:
    compact = re.sub(r"[^A-Z]", "", value.upper())
    return compact in {"KABUPATEN", "KAB", "KOTA", "KOTAADMINISTRASI"}


def _contains_non_admin_area_value_label(value: str) -> bool:
    blocked_patterns = [
        r"\b(?:NIK|NAMA|TEMP\w*|TGL|TANGGAL|LAHIR|ALAMAT|RT\s*/?\s*RW|KEL(?:/|\b)|KELURAHAN|DESA|KECAM\w*|AGAMA|STATUS|PEKER\w*|KEWARG\w*|BERLAKU|HINGGA|GOL|DARAH)\b",
        r"\bJENIS\s*KELAMIN\b",
        r"\bPERKAW\w*\b",
    ]
    return any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in blocked_patterns)


def _fallback_name(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if line.upper() == "NIK":
            scan_limit = 4
        elif _is_standalone_nik_line(line):
            scan_limit = 12
        else:
            continue
        candidates: list[str] = []
        for raw_candidate in lines[index + 1 : index + 1 + scan_limit]:
            candidate = _clean_candidate_value(raw_candidate)
            upper = candidate.upper()
            if re.fullmatch(r"[:\s\d\-]+", candidate):
                continue
            if _contains_any_label(upper):
                continue
            if _looks_like_person_name(candidate):
                candidates.append(candidate)
        if candidates:
            three_word_candidates = [candidate for candidate in candidates if len(candidate.split()) >= 3]
            return three_word_candidates[0] if three_word_candidates else candidates[0]
    return None


def _fallback_name_near_label(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not _is_name_label(line):
            continue
        for raw_candidate in reversed(lines[max(0, index - 3) : index]):
            candidate = _clean_candidate_value(raw_candidate)
            upper = candidate.upper()
            if _contains_any_label(upper):
                continue
            if normalize_nik(candidate):
                continue
            if _looks_like_person_name(candidate) or _looks_like_single_person_name(candidate):
                return candidate
    return None


def _fallback_name_after_birth_line(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not _contains_birth_date(line):
            continue
        for raw_candidate in lines[index + 1 : index + 6]:
            candidate = _clean_candidate_value(raw_candidate)
            upper = candidate.upper()
            if _contains_any_label(upper):
                continue
            if _looks_like_gender(upper):
                continue
            if _looks_like_person_name(candidate) or _looks_like_single_person_name(candidate):
                return candidate
    return None


def _repair_joined_person_name(value: str) -> str:
    cleaned = _clean_candidate_value(value).upper()
    if not cleaned or _contains_any_label(cleaned) or any(char.isdigit() for char in cleaned):
        return value

    repaired_tokens = [_repair_joined_name_token(token) for token in cleaned.split()]
    return " ".join(repaired_tokens)


def _repair_joined_name_token(token: str) -> str:
    if not re.fullmatch(r"[A-Z']{9,}", token):
        return token
    if token in NON_NAME_KEYWORDS or token in FIELD_VALUE_BLOCKLIST:
        return token

    for prefix in sorted(JOINED_NAME_PREFIXES, key=len, reverse=True):
        if not token.startswith(prefix):
            continue
        suffix = token[len(prefix) :]
        if len(suffix) < 5:
            continue
        if not any(suffix.startswith(stem) for stem in JOINED_NAME_SUFFIX_PREFIXES):
            continue
        if _looks_like_single_person_name(suffix):
            return f"{prefix} {suffix}"
    return token


def _fallback_birth_place_date(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not _is_birth_label(line):
            continue
        nearby = lines[index : min(len(lines), index + 6)]
        for offset, candidate in enumerate(nearby):
            normalized = _normalize_birth_place_date(candidate)
            if normalized:
                return normalized
            if offset + 1 < len(nearby):
                combined = f"{candidate} {nearby[offset + 1]}"
                normalized = _normalize_birth_place_date(combined)
                if normalized:
                    return normalized
            if offset + 2 < len(nearby):
                combined = f"{candidate} {nearby[offset + 1]} {nearby[offset + 2]}"
                normalized = _normalize_birth_place_date(combined)
                if normalized:
                    return normalized

    for line in lines:
        normalized = _normalize_birth_place_date(line)
        if normalized:
            return normalized
    return None


def _normalize_birth_place_date(value: str) -> str | None:
    pattern = re.compile(
        r"\b([A-Z][A-Z0-9.\s'-]{2,}?)[,.\s]+([0-9OILSB]{1,2})[-/.\s]+([0-9OILSB]{1,2})[-/.\s:]*([0-9OILSB:]{4,6})\b",
        flags=re.IGNORECASE,
    )
    match = pattern.search(value)
    if not match:
        return None

    place = _clean_birth_place(match.group(1))
    if not place:
        return None

    day = _normalize_date_component(match.group(2), 2)
    month = _normalize_date_component(match.group(3), 2)
    year = _normalize_date_component(match.group(4), 4)
    if not day or not month or not year:
        return None
    if not (1 <= int(day) <= 31 and 1 <= int(month) <= 12):
        return None
    return f"{place}, {day}-{month}-{year}"


def _clean_birth_place(value: str) -> str | None:
    cleaned = _clean_candidate_value(value).upper().strip(" ,.-:")
    cleaned = re.sub(r"\b(?:TEMP\w*|TGL|TGI|TOL|TANGGAL|LAHIR|LHIR|LABE)\b", " ", cleaned)
    cleaned = collapse_birth_place_separators(_repair_birth_place_ocr_digits(cleaned))
    if not cleaned or _contains_birth_place_blocked_label(cleaned):
        return None
    if _looks_like_address(cleaned) or cleaned.startswith("PROVINSI"):
        return None
    return cleaned if re.fullmatch(r"[A-Z][A-Z .'-]{2,}", cleaned) else None


def collapse_birth_place_separators(value: str) -> str:
    value = re.sub(r"\s*\.\s*", ".", value)
    return re.sub(r"\s+", " ", value).strip(" .")


def _repair_birth_place_ocr_digits(value: str) -> str:
    return value.translate(str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B"}))


def _contains_birth_place_blocked_label(value: str) -> bool:
    blocked = [
        r"\bNIK\b",
        r"\bNAMA\b",
        r"\bJENIS\b",
        r"\bKELAMIN\b",
        r"\bALAMAT\b",
        r"\bAGAMA\b",
        r"\bSTATUS\b",
        r"\bPERKAW\w*\b",
        r"\bPEKER\w*\b",
        r"\bKEWARG\w*\b",
        r"\bBERLAKU\b",
        r"\bHINGGA\b",
        r"\bRT\s*/?\s*RW\b",
        r"\bKEL\s*/?\s*DESA\b",
        r"\bKECAM\w*\b",
        r"\bPROVINSI\b",
    ]
    return any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in blocked)


def _normalize_date_component(value: str, expected_length: int) -> str | None:
    mapped = []
    for char in value.upper():
        if char.isdigit():
            mapped.append(char)
        elif char == "O":
            mapped.append("0")
        elif char in {"I", "L", "|"}:
            mapped.append("1")
        elif char == "S":
            mapped.append("5")
        elif char == "B":
            mapped.append("8")
    digits = "".join(mapped)
    if len(digits) != expected_length:
        return None
    return digits.zfill(expected_length)


def _contains_birth_date(value: str) -> bool:
    return bool(re.search(r"\b\d{2}[-/]\d{2}[-/]\d{4}\b", value))


def _is_birth_label(value: str) -> bool:
    compact = re.sub(r"[^A-Z]", "", value.upper())
    return bool("LAHIR" in compact and ("TEMP" in compact or "TG" in compact or "TGL" in compact or "TAL" in compact))


def _normalize_gender(value: str) -> str | None:
    compact = re.sub(r"[^A-Z]", "", value.upper())
    if compact.startswith("LAK") or "LAKILAKI" in compact or "LAKLAKI" in compact:
        return "LAKI-LAKI"
    if compact.startswith("PEREMP") or "PEREMPUAN" in compact or "PEREMPUIAN" in compact or compact.startswith("PEREP"):
        return "PEREMPUAN"
    return None


def _fallback_gender(lines: list[str]) -> str | None:
    for line in lines:
        gender = _normalize_gender(_clean_candidate_value(line))
        if gender:
            return gender
    return None


def _fallback_address(lines: list[str]) -> str | None:
    for line in lines:
        candidate = _clean_candidate_value(line)
        if _contains_any_label(candidate.upper()):
            continue
        if _looks_like_address(candidate):
            return candidate

    for index, line in enumerate(lines):
        if not ADDRESS_ANCHOR_PATTERN.search(line):
            continue
        for candidate in reversed(lines[max(0, index - 4) : index]):
            candidate = _clean_candidate_value(candidate)
            upper = candidate.upper()
            if _contains_any_label(upper):
                continue
            if _looks_like_address(candidate) or _looks_like_loose_address_before_rt(candidate):
                return candidate
    return None


def _normalize_rt_rw(value: str) -> str | None:
    cleaned = _clean_candidate_value(value).upper()
    if re.search(r"\d{2}[-.]\d{2}[-.]\d{2,4}", cleaned):
        return None

    match = re.search(r"(?<!\d)([0-9OIL|.]{1,3})\s*/\s*([0-9OIL|.]{1,3})(?!\d)", cleaned)
    if match:
        left = _normalize_rt_rw_digits(match.group(1))
        right = _normalize_rt_rw_digits(match.group(2))
        if left and right:
            return f"{left.zfill(3)}/{right.zfill(3)}"

    digits = re.sub(r"\D", "", cleaned)
    if len(digits) == 6:
        return f"{digits[:3]}/{digits[3:]}"
    if len(digits) == 7 and digits[3] == "1":
        return f"{digits[:3]}/{digits[4:]}"
    return None


def _normalize_rt_rw_digits(value: str) -> str | None:
    mapped = []
    for char in value.upper():
        if char.isdigit():
            mapped.append(char)
        elif char == "O":
            mapped.append("0")
        elif char in {"I", "L", "|", "."}:
            mapped.append("1")
    digits = "".join(mapped)
    return digits if 1 <= len(digits) <= 3 else None


def _fallback_rt_rw(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not _is_rt_rw_label(line):
            continue
        for candidate in lines[index : min(len(lines), index + 12)]:
            normalized = _normalize_rt_rw(candidate)
            if normalized:
                return normalized
    return None


def _fallback_region_value(lines: list[str], kind: str) -> str | None:
    label_fn = _is_kelurahan_label if kind == "kelurahan" else _is_kecamatan_label
    for index, line in enumerate(lines):
        inline = _region_value_from_inline_label(line, kind)
        if inline:
            return inline

        if not label_fn(line):
            continue

        for candidate in lines[index + 1 : min(len(lines), index + 5)]:
            if label_fn(candidate) or _is_region_label_fragment(candidate):
                continue
            if _contains_any_label(candidate.upper()):
                break
            normalized = _normalize_region_value(candidate)
            if normalized:
                return normalized
    return None


def _region_value_from_inline_label(line: str, kind: str) -> str | None:
    if kind == "kelurahan":
        label = r"(?:KEL\s*/?\s*DES[AN]?|KELDESA|KEWDESA|KEVDESA|KELDOSA|KELDESS|KALDESS|NOESA|EL\s*/?\s*DESA)"
    else:
        inline_joined = re.match(
            r"^\s*(?:KECAMATAN|KECAMNATAN|KECAMATAR|KECANATAN|XECAMATAN|ECAMATAN|KECAMNATAN|KECAMNATAN|KECAM)\s*[_:\-]\s*(.+)$",
            line,
            flags=re.IGNORECASE,
        )
        if inline_joined:
            return _normalize_region_value(inline_joined.group(1))
        label = r"(?:KECAMATAN|KECAMNATAN|KECAMATAR|KECANATAN|KECAMNATAN|KECA\s*MATEN|XECAMATAN|ECAMATAN)"
    match = re.search(rf"{label}\s*[:_\-]?\s*(.+)$", line, flags=re.IGNORECASE)
    if not match:
        return None
    return _normalize_region_value(match.group(1))


def _normalize_region_value(value: str) -> str | None:
    cleaned = _clean_candidate_value(value).upper().strip(" :;._-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return None
    if _contains_non_region_value_label(cleaned):
        return None
    if cleaned in FIELD_VALUE_BLOCKLIST:
        return None
    if normalize_nik(cleaned) or _normalize_rt_rw(cleaned) or _is_date_like(cleaned):
        return None
    if (
        _normalize_gender(cleaned)
        or _normalize_religion(cleaned)
        or _normalize_marital_status(cleaned)
        or _normalize_citizenship(cleaned)
        or _normalize_job(cleaned)
    ):
        return None
    if _normalize_expiry(cleaned) or _looks_like_address(cleaned):
        return None
    letters = re.findall(r"[A-Z]", cleaned)
    if len(letters) < 3:
        return None
    return cleaned if re.fullmatch(r"[A-Z][A-Z0-9 .'\-/]{2,}", cleaned) else None


def _is_kelurahan_label(value: str) -> bool:
    compact = re.sub(r"[^A-Z0-9]", "", value.upper())
    return compact in {
        "KELDESA",
        "KELURAHAN",
        "DESA",
        "KEWDESA",
        "KEVDESA",
        "KELDESN",
        "KELDOSA",
        "KELDESS",
        "KALDESS",
        "NOESA",
        "N0ESA",
        "ELDESA",
    } or compact.startswith("KELD")


def _is_kecamatan_label(value: str) -> bool:
    compact = re.sub(r"[^A-Z]", "", value.upper())
    if compact in {"KEC", "EATAN", "CAMATAN"}:
        return True
    return bool(re.fullmatch(r"[KXE]?ECAM[A-Z]*|KECAM[A-Z]*|KECAN[A-Z]*|KEC[A-Z]*NATAN", compact))


def _is_region_label_fragment(value: str) -> bool:
    compact = re.sub(r"[^A-Z]", "", value.upper())
    return compact in {"EATAN", "MATAN", "CAMATAN", "DESA", "DESS", "DOSA", "DESN"}


def _fallback_transposed_regions(lines: list[str]) -> tuple[str | None, str | None]:
    label_indexes = [
        index
        for index, line in enumerate(lines)
        if _is_kecamatan_label(line) or _is_kelurahan_label(line) or _is_rt_rw_label(line)
    ]
    if len(label_indexes) < 2:
        return None, None

    start = max(label_indexes) + 1
    candidates: list[str] = []
    for raw_candidate in lines[start : min(len(lines), start + 18)]:
        candidate = _normalize_region_value(raw_candidate)
        if not candidate:
            continue
        candidates.append(candidate)
        if len(candidates) >= 2:
            break

    if len(candidates) < 2:
        return None, candidates[0] if candidates else None
    return candidates[1], candidates[0]


def _contains_non_region_value_label(value: str) -> bool:
    compact = re.sub(r"[^A-Z]", "", value.upper())
    exact_blocked = {"NIK", "NAMA", "TEMPAT", "TGL", "LAHIR", "ALAMAT", "AGAMA", "STATUS", "PEKERJAAN", "KEWARGANEGARAAN", "BERLAKU", "HINGGA"}
    if compact in exact_blocked:
        return True
    blocked_patterns = [
        r"\b(?:NIK|NAMA|TEMP\w*|TGL|TANGGAL|LAHIR|ALAMAT|AGAMA|STATUS|PEKER\w*|KEWARG\w*|BERLAKU|HINGGA|PROVINSI|KABUPATEN|KOTA|GOL|DARAH)\b",
        r"\bJENIS\s*KELAMIN\b",
        r"\bPERKAW\w*\b",
    ]
    return any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in blocked_patterns)


def _normalize_religion(value: str) -> str | None:
    cleaned = _clean_candidate_value(value)
    compact = re.sub(r"[^A-Z]", "", cleaned.upper())
    if not compact:
        return None
    for keyword, normalized in RELIGION_KEYWORDS.items():
        if keyword in compact:
            return normalized
    return None


def _fallback_religion(lines: list[str]) -> str | None:
    for line in lines:
        normalized = _normalize_religion(line)
        if normalized:
            return normalized
    return None


def _is_rt_rw_label(value: str) -> bool:
    upper = value.upper()
    return bool(ADDRESS_ANCHOR_PATTERN.search(upper) or "RT/RW" in upper or "RT/RV" in upper or "RTRW" in upper)


def _contains_any_label(value: str) -> bool:
    labels = [label.upper() for values in KTP_LABELS.values() for label in values]
    return any(label in value for label in labels) or bool(FUZZY_LABEL_PATTERN.search(value))


def _looks_like_person_name(value: str) -> bool:
    upper = value.upper()
    if upper in FIELD_VALUE_BLOCKLIST:
        return False
    if any(keyword in upper.split() for keyword in NON_NAME_KEYWORDS):
        return False
    if any(char.isdigit() for char in upper):
        return False
    if len(upper.split()) < 2:
        return False
    if _looks_like_address(upper):
        return False
    return bool(re.fullmatch(r"[A-Z .']{5,}", upper))


def _looks_like_single_person_name(value: str) -> bool:
    upper = value.upper()
    if upper in FIELD_VALUE_BLOCKLIST:
        return False
    if any(keyword in upper.split() for keyword in NON_NAME_KEYWORDS):
        return False
    if any(char.isdigit() for char in upper):
        return False
    if _contains_any_label(upper):
        return False
    if _looks_like_address(upper):
        return False
    return bool(re.fullmatch(r"[A-Z.']{5,}", upper))


def _looks_like_gender(value: str) -> bool:
    return _normalize_gender(value) is not None


def _normalize_job(value: str) -> str | None:
    cleaned = _clean_candidate_value(value)
    upper = cleaned.upper()
    compact = re.sub(r"[^A-Z]", "", upper)
    if not cleaned or _contains_any_label(upper) or _is_city_or_region_line(upper) or _is_date_like(upper):
        return None
    if "KARYAWAN" in upper and ("BUMN" in upper or "BUUN" in upper or "BUN" in upper or "BUM" in upper):
        return "KARYAWAN BUMN"
    if "KARYAWAN" in upper and "SWASTA" in upper:
        return "KARYAWAN SWASTA"
    if "WIRASWASTA" in compact:
        return "WIRASWASTA"
    if "PELAJAR" in upper or "MAHASISWA" in upper:
        return "PELAJAR/MAHASISWA"
    if "MENGURUS" in upper and ("RUMAH" in upper or "RUMAHT" in compact):
        return "MENGURUS RUMAH TANGGA"
    if "TENTARA NASIONAL" in upper or "TENTARANASIONAL" in compact or "INDONESIA (TNI)" in upper or ("INDONESIA" in upper and "TNI" in upper):
        return "TENTARA NASIONAL INDONESIA (TNI)"
    if "PENSIUN" in compact:
        return "PENSIUNAN"
    for keyword in ["POLRI", "PNS", "GURU", "DOSEN", "DOKTER", "BURUH", "PETANI", "NELAYAN", "PEDAGANG"]:
        if keyword in upper.split() or keyword in compact:
            return keyword
    return None


def _fallback_job(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not _is_job_label(line):
            continue
        nearby = list(lines[index + 1 : min(len(lines), index + 6)])
        nearby.extend(reversed(lines[max(0, index - 5) : index]))
        for raw_candidate in nearby:
            normalized = _normalize_job(raw_candidate)
            if normalized:
                return normalized

    for line in lines:
        normalized = _normalize_job(line)
        if normalized:
            return normalized
    return None


def _normalize_citizenship(value: str) -> str | None:
    compact = re.sub(r"[^A-Z0-9]", "", value.upper())
    if not compact:
        return None
    compact = compact.replace("1", "I").replace("L", "I")
    if compact in {"WNI", "VNI"}:
        return "WNI"
    if compact in {"WNA", "VNA"}:
        return "WNA"
    return None


def _normalize_citizenship_near_label(value: str) -> str | None:
    normalized = _normalize_citizenship(value)
    if normalized:
        return normalized

    compact = re.sub(r"[^A-Z0-9]", "", value.upper()).replace("1", "I").replace("L", "I")
    if compact in {"WN", "VKE"} or compact.endswith("WNI") or compact.endswith("WN"):
        return "WNI"
    return None


def _fallback_citizenship(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not _is_citizenship_label(line):
            continue
        window = list(lines[index : min(len(lines), index + 4)])
        window.extend(reversed(lines[max(0, index - 4) : index]))
        for candidate in window:
            normalized = _normalize_citizenship_near_label(candidate)
            if normalized:
                return normalized

    for line in lines:
        normalized = _normalize_citizenship(line)
        if normalized:
            return normalized
    return None


def _is_citizenship_label(value: str) -> bool:
    compact = re.sub(r"[^A-Z]", "", value.upper())
    return "KEWARG" in compact or "KEVRG" in compact


def _fallback_expiry(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        inline = _expiry_from_inline_label(line)
        if inline:
            return inline

        if not _is_expiry_label(line):
            continue

        nearby = list(lines[index + 1 : min(len(lines), index + 5)])
        nearby.extend(reversed(lines[max(0, index - 3) : index]))
        for candidate in nearby:
            if _contains_any_label(candidate.upper()) and not _is_expiry_label(candidate):
                continue
            normalized = _normalize_expiry(candidate)
            if normalized:
                return normalized
    return None


def _fallback_transposed_expiry(lines: list[str]) -> str | None:
    expiry_indexes = [index for index, line in enumerate(lines) if _is_expiry_label(line)]
    if not expiry_indexes:
        return None
    start = expiry_indexes[0] + 1
    for candidate in lines[start : min(len(lines), start + 20)]:
        if _is_expiry_label(candidate):
            continue
        normalized = _normalize_expiry(candidate)
        if normalized:
            return normalized
    return None


def _expiry_from_inline_label(line: str) -> str | None:
    label = r"(?:BERLAK\w*|BARLAK\w*|BERLAKY|BORLAK\w*|BERBAKY|BERFAKU|BERFAK\w*|SERFAKU|SERFAK\w*|BERTAKU|BERTAK\w*|BARTARAR|NAKU)"
    match = re.search(rf"{label}\s*:?\s*(?:HING\w*|HIN\w*|HNOG\w*)?\s*[:_\-]?\s*(.*)$", line, flags=re.IGNORECASE)
    if not match:
        return None
    return _normalize_expiry(match.group(1))


def _normalize_expiry(value: str) -> str | None:
    cleaned = _clean_candidate_value(value).upper().strip(" :;._-")
    if not cleaned:
        return None
    compact = re.sub(r"[^A-Z0-9]", "", cleaned).replace("1", "I").replace("0", "O")
    if compact.startswith("SEUMURHID") or compact in {"SEUMURHIOUP", "SEUMUHIDUP"}:
        return "SEUMUR HIDUP"

    match = re.search(r"(?<!\d)(\d{1,2})[-/.\s]+(\d{1,2})[-/.\s]+(\d{4})(?!\d)", cleaned)
    if not match:
        return None
    day = match.group(1).zfill(2)
    month = match.group(2).zfill(2)
    year = match.group(3)
    return f"{day}-{month}-{year}"


def _is_expiry_label(value: str) -> bool:
    compact = re.sub(r"[^A-Z]", "", value.upper())
    has_prefix = any(prefix in compact for prefix in ["BERLAKU", "BARLAKU", "BERLAKY", "BORLAKU", "BERBAKY", "BERFAKU", "SERFAKU", "BERTAKU", "BARTARAR", "NAKU"])
    return has_prefix and ("HING" in compact or "HIN" in compact or "HNOG" in compact or compact.startswith(("BERLAKU", "BARLAKU", "BERLAKY", "SERFAKU", "BERTAKU")))


def _is_job_label(value: str) -> bool:
    return bool(JOB_LABEL_PATTERN.search(value))


def _is_city_or_region_line(value: str) -> bool:
    upper = value.upper()
    return bool(re.fullmatch(r":?\s*(KOTA|KABUPATEN|KAB|PROVINSI)?\s*[A-Z .'-]{3,}\s*", upper) and any(marker in upper for marker in ["KOTA", "KABUPATEN", "PROVINSI"]))


def _is_date_like(value: str) -> bool:
    return bool(re.search(r"\d{1,2}[-./]\d{1,2}[-./]\d{2,4}", value))


def _name_value_needs_repair(value: str) -> bool:
    upper = value.upper()
    return bool(
        _contains_any_label(upper)
        or _looks_like_address(upper)
        or any(keyword in upper.split() for keyword in NON_NAME_KEYWORDS)
        or any(char.isdigit() for char in upper)
        or re.search(r"\d{2}[-/]\d{2}[-/]\d{4}", upper)
    )


def _looks_like_address(value: str) -> bool:
    upper = value.upper()
    return bool(
        re.search(r"\b(JL|JLN|JALAN|KP|KAMP|KAMPUNG|DSN|DUSUN|DUKUH|LINGK|GG|GANG|PERUM|KOMP\w*|BLOK\w*|KAV)\b", upper)
        or re.search(r"\bNO\.?\s*\d+", upper)
    )


def _looks_like_loose_address_before_rt(value: str) -> bool:
    upper = value.upper()
    if not upper or upper in FIELD_VALUE_BLOCKLIST:
        return False
    if _looks_like_person_name(upper):
        return False
    if normalize_nik(upper):
        return False
    if re.search(r"\d{2}[-/]\d{2}[-/]\d{4}", upper):
        return False
    if re.fullmatch(r"[:\s\d./\-]+", upper):
        return False
    if len(re.findall(r"[A-Z]", upper)) < 5:
        return False
    return bool(re.fullmatch(r"[A-Z0-9 .'/\-]+", upper))


def _looks_like_valid_address_value(value: str) -> bool:
    upper = _clean_candidate_value(value).upper()
    if not upper or upper in FIELD_VALUE_BLOCKLIST:
        return False
    if re.search(r"\bGOL\.?\s*DARAH\b|\bDARAH\b", upper):
        return False
    if _contains_any_label(upper):
        return False
    return _looks_like_address(upper) or _looks_like_loose_address_before_rt(upper)


def _is_standalone_nik_line(value: str) -> bool:
    upper = value.upper()
    if _contains_any_label(upper):
        return False
    digits = re.sub(r"\D", "", value)
    return len(digits) == 16 and normalize_nik(value) is not None


def _is_name_label(value: str) -> bool:
    return bool(re.fullmatch(r":?\s*(NAMA|NAME|NAMS|NANIA|NAMAT)\s*:?", value, flags=re.IGNORECASE))


def _clean_candidate_value(value: str) -> str:
    return re.sub(r"^[\s:;\-]+", "", value).strip()


def _looks_like_bad_ktp_crop(raw_text: str, fields: dict[str, FieldResult]) -> bool:
    upper = raw_text.upper()
    desktop_markers = ["DRIVE", "TOSHIBA", "TYPE HERE", "MANAGE", ".JPG", "100%"]
    if sum(1 for marker in desktop_markers if marker in upper) < 2:
        return False
    return any(fields[field].status == "missing" for field in KTP_REQUIRED_FIELDS)


def _fallback_marital_status(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        upper = line.upper()
        if not ("STATUS" in upper or "PERKAW" in upper or "PERKAWINAR" in upper):
            continue

        window = " ".join(lines[index : min(len(lines), index + 4)])
        value = _normalize_marital_status(window)
        if value:
            return value

    for line in lines:
        value = _normalize_marital_status(line)
        if value and not _contains_non_status_label(line.upper()):
            return value
    return None


def _normalize_marital_status(value: str) -> str | None:
    compact = re.sub(r"[^A-Z0-9]", "", value.upper()).replace("0", "O").replace("1", "I")
    status_value = compact
    for noisy_label in ["STATUS", "PERKAWINARE", "PERKAWINARC", "PERKAWINAR", "PERKAWINAN"]:
        status_value = status_value.replace(noisy_label, "")
    variants = {
        compact,
        compact.replace("L", "I"),
        compact.replace("I", "L"),
        status_value,
        status_value.replace("L", "I"),
        status_value.replace("I", "L"),
    }
    if any(
        "BELUMKAWIN" in variant
        or "BEIUMKAWIN" in variant
        or "BELUMKAWLN" in variant
        or "BEIUMKAWLN" in variant
        or "BELUMMENIKAH" in variant
        or "BEIUMMENIKAH" in variant
        or "BELUMMENLKAH" in variant
        for variant in variants
    ):
        return "BELUM KAWIN"
    if any("CERAIHIDUP" in variant for variant in variants):
        return "CERAI HIDUP"
    if any("CERAIMATI" in variant for variant in variants):
        return "CERAI MATI"
    if any("KAWIN" in variant and "PERKAW" not in variant for variant in variants):
        return "KAWIN"
    return None


def _contains_non_status_label(value: str) -> bool:
    blocked = ["NIK", "NAMA", "TEMPAT", "LAHIR", "ALAMAT", "AGAMA", "PEKERJAAN", "KEWARGANEGARAAN", "BERLAKU"]
    return any(label in value for label in blocked)
