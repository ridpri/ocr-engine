from __future__ import annotations

from datetime import date
import re

from ocr_engine.parsers.common import capture_after_label, make_invalid, make_missing, make_ok, normalized_lines
from ocr_engine.postal_code import lookup_postal_code
from ocr_engine.schemas import DocumentResult, FieldResult
from ocr_engine.validators import collapse_spaces, normalize_nik


KTP_LABELS: dict[str, list[str]] = {
    "provinsi": ["Provinsi", "PROVINSI"],
    "kabupaten_kota": [
        "Kota Administrasi",
        "KOTA ADMINISTRASI",
        "Kabupaten",
        "KABUPATEN",
        "Kab/Kota",
        "Kab/ Kota",
        "Kab/Kata",
        "Kota",
        "KOTA",
    ],
    "nik": ["NIK", "NTK", "HIK"],
    "nama": ["Nama", "NAMA", "Name", "Nams", "Nania", "Namat", "Nam"],
    "tempat_tanggal_lahir": [
        "Tempat/Tgl Lahir",
        "Tempat Tgl Lahir",
        "Tempat/Tanggal Lahir",
        "TempatTglLahir",
    ],
    "jenis_kelamin": ["Jenis Kelamin"],
    "alamat": ["Alamat", "Alamal", "Almal", "Alanat", "Alsmat"],
    "rt_rw": ["RT/RW", "RTRW", "TT/RW", "TIT/RW", "AT/RW"],
    "kelurahan_desa": [
        "Kel/Desa",
        "Kel Desa",
        "KelDesa",
        "Kelurahan",
        "Desa",
        "Desa/Kel",
        "KeDesa",
        "KeilDesa",
        "KeWDesa",
        "KevDesa",
        "Kel/Desn",
        "Kel/Dosa",
        "Kel/Oosa",
        "Kol/Desa",
        "Kol/Dasa",
        "Kel/Dess",
        "Kel/Dean",
        "Desa/Kol",
        "Kal/Dess",
        "Kal/Desa",
        "NOesa",
        "el/Desa",
        "VOasa",
    ],
    "kecamatan": [
        "Kecamatan",
        "Kecamnatan",
        "Kecamatar",
        "Kecanatar",
        "Kecanatan",
        "Kecametan",
        "Kecamretan",
        "Kicanalan",
        "KecaMatEN",
        "Xecamatan",
        "ecomatan",
        "ecamatan",
        "ecematan",
        "comatan",
    ],
    "agama": ["Agama"],
    "status_perkawinan": ["Status Perkawinan"],
    "pekerjaan": ["Pekerjaan", "Pekeriaan", "Pekerian", "Pokerjaan", "Pakeraan", "Pekerean", "Pakerjaan"],
    "kewarganegaraan": [
        "Kewarganegaraan",
        "Kewargane",
        "Kewarganegaraar",
        "Kewarganegaraart",
        "Kevrganegaraan",
        "Kwarganegaraan",
        "Kowarganegaraan",
    ],
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
        "Beraku Hingga",
        "Beriaku Hingga",
        "Beriaky Hingga",
        "BeriakuHingga",
        "Beriaku Hingua",
        "ak. Hingqa",
        "iedakuHingga",
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
ADDRESS_ANCHOR_PATTERN = re.compile(
    r"\b(?:RT\s*/?\s*(?:RW|AW|RV)|TT\s*/?\s*RW|TIT\s*/?\s*RW|AT\s*/?\s*RW|ATIAW|RTIAW|RTRW|RTAW|TTRW|TITRW|ATRW)\b",
    flags=re.IGNORECASE,
)
JOB_LABEL_PATTERN = re.compile(r"\b(?:PEKER\w*|PEIKER\w*|PAKER\w*|POKERJAAN|PAKERAAN|PEKEREAN)\b", flags=re.IGNORECASE)
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
    "SETIOWATI",
    "SETIO",
    "TRI",
    "DWI",
    "EKO",
    "BUDI",
    "AGUS",
    "PUTRI",
    "PUTRA",
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
    "SANT",
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
JOINED_INITIAL_NAME_STEMS = {
    "SYARIFUL",
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

    if _looks_like_bad_ktp_crop(raw_text, fields):
        fields["nama"] = make_missing()

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

    if fields["provinsi"].status != "ok":
        jakarta_admin = fields["kabupaten_kota"].value or ""
        if jakarta_admin.startswith("JAKARTA "):
            fields["provinsi"] = make_ok("DKI JAKARTA", confidence=0.78, raw="fallback:jakarta_admin_area")
        elif jakarta_admin == "JAMBI":
            fields["provinsi"] = make_ok("JAMBI", confidence=0.72, raw="fallback:jambi_admin_area")

    if (
        fields["nama"].status == "ok"
        and fields["nama"].value
        and (
            _name_value_needs_repair(fields["nama"].value)
            or fields["nama"].value in {fields["provinsi"].value, fields["kabupaten_kota"].value}
        )
    ):
        name_was_expiry_like = _normalize_expiry(fields["nama"].value) is not None
        if name_was_expiry_like:
            repaired_name = (
                _fallback_name_before_nik_line(lines)
                or _fallback_name_before_birth_line(lines)
                or _fallback_name_after_label_window(lines)
                or _fallback_name_near_label(lines)
            )
        else:
            repaired_name = (
                _fallback_name_before_nik_line(lines)
                or _fallback_name_before_birth_line(lines)
                or _fallback_transposed_person_name(lines)
                or _fallback_name_after_label_window(lines)
                or _fallback_name_near_label(lines)
            )
        fields["nama"] = make_ok(repaired_name, confidence=0.7) if repaired_name else make_missing()
    else:
        name_was_expiry_like = False

    if fields["nama"].status == "missing" and not name_was_expiry_like:
        fallback_name = (
            _fallback_transposed_person_name(lines)
            or _fallback_name_before_nik_line(lines)
            or _fallback_name_before_birth_line(lines)
            or _fallback_name(lines)
        )
        if fallback_name:
            fields["nama"] = make_ok(fallback_name, confidence=0.72)

    if fields["nama"].status == "missing" and not name_was_expiry_like:
        fallback_name = (
            _fallback_name_after_birth_line(lines)
            or _fallback_name_before_birth_line(lines)
            or _fallback_transposed_person_name(lines)
            or _fallback_name_after_label_window(lines)
        )
        if fallback_name:
            fields["nama"] = make_ok(fallback_name, confidence=0.7)

    if fields["nama"].status == "ok" and fields["nama"].value:
        original_name = fields["nama"].value
        repaired_name = _repair_joined_person_name(original_name)
        if repaired_name != original_name:
            fields["nama"] = make_ok(
                repaired_name,
                confidence=min(fields["nama"].confidence, _joined_name_confidence_cap(original_name, repaired_name)),
                raw=fields["nama"].raw or "fallback:joined_name_spacing",
            )
        extended_name = _extend_wrapped_person_name(lines, fields["nama"].value)
        if extended_name != fields["nama"].value:
            fields["nama"] = make_ok(
                extended_name,
                confidence=min(fields["nama"].confidence, 0.78),
                raw=fields["nama"].raw or "fallback:wrapped_name",
            )

    normalized_birth_place_date = _normalize_birth_place_date(fields["tempat_tanggal_lahir"].value or "")
    if normalized_birth_place_date:
        fields["tempat_tanggal_lahir"] = make_ok(normalized_birth_place_date, confidence=fields["tempat_tanggal_lahir"].confidence)
    else:
        fallback_ttl = _fallback_birth_place_date(lines)
        if fallback_ttl:
            fields["tempat_tanggal_lahir"] = make_ok(fallback_ttl, confidence=0.72)
        else:
            fallback_ttl = _fallback_truncated_birth_place_date_from_nik(lines, fields["nik"].value)
            if fallback_ttl:
                fields["tempat_tanggal_lahir"] = make_ok(fallback_ttl, confidence=0.68)
            elif fields["tempat_tanggal_lahir"].status == "ok":
                fields["tempat_tanggal_lahir"] = make_missing()
        if fields["tempat_tanggal_lahir"].status == "ok" and not (fields["tempat_tanggal_lahir"].value or ""):
            fields["tempat_tanggal_lahir"] = make_missing()
    repaired_birth_place_date = _repair_birth_place_date_from_nik(fields["tempat_tanggal_lahir"].value or "", fields["nik"].value)
    if repaired_birth_place_date:
        fields["tempat_tanggal_lahir"] = make_ok(
            repaired_birth_place_date,
            confidence=max(fields["tempat_tanggal_lahir"].confidence or 0.0, 0.74),
            raw=fields["tempat_tanggal_lahir"].raw or "fallback:nik_birth_date",
        )

    normalized_gender = _normalize_gender(fields["jenis_kelamin"].value or "")
    if normalized_gender:
        fields["jenis_kelamin"] = make_ok(normalized_gender, confidence=0.84)
    else:
        fallback_gender = _fallback_gender(lines)
        if fallback_gender:
            fields["jenis_kelamin"] = make_ok(fallback_gender, confidence=0.78)
        elif fields["jenis_kelamin"].status == "ok":
            fields["jenis_kelamin"] = make_missing()

    if fields["alamat"].status == "ok":
        labeled_address = _fallback_labeled_address(lines)
        if labeled_address and (fields["alamat"].value or "") in labeled_address and labeled_address != fields["alamat"].value:
            fields["alamat"] = make_ok(labeled_address, confidence=0.72)

    if fields["alamat"].status == "ok" and not _looks_like_valid_address_value(fields["alamat"].value or ""):
        fallback_address = _fallback_labeled_address(lines) or _fallback_address(lines)
        fields["alamat"] = make_ok(fallback_address, confidence=0.72) if fallback_address else make_missing()

    if fields["alamat"].status == "missing":
        fallback_address = _fallback_labeled_address(lines) or _fallback_address(lines)
        if fallback_address:
            fields["alamat"] = make_ok(fallback_address, confidence=0.72)

    if fields["alamat"].status == "ok" and fields["alamat"].value:
        normalized_address = _normalize_address_value(fields["alamat"].value)
        if normalized_address != fields["alamat"].value:
            fields["alamat"] = make_ok(normalized_address, confidence=fields["alamat"].confidence, raw=fields["alamat"].raw)
        extended_address = _extend_address_with_trailing_fragment(lines, fields["alamat"].value or "")
        if extended_address and extended_address != fields["alamat"].value:
            fields["alamat"] = make_ok(
                _normalize_address_value(extended_address),
                confidence=min(fields["alamat"].confidence or 0.88, 0.78),
                raw=fields["alamat"].raw or "fallback:address_continuation",
            )

    normalized_rt_rw = _normalize_rt_rw(fields["rt_rw"].value or "")
    if normalized_rt_rw:
        fields["rt_rw"] = make_ok(normalized_rt_rw, confidence=0.84)
    else:
        fallback_rt_rw = _fallback_rt_rw(lines)
        if fallback_rt_rw:
            fields["rt_rw"] = make_ok(fallback_rt_rw, confidence=0.78)
        elif fields["rt_rw"].status == "ok":
            fields["rt_rw"] = make_missing()

    kelurahan_needs_repair = (
        fields["kelurahan_desa"].status == "ok"
        and fields["kelurahan_desa"].value
        and (
            fields["kelurahan_desa"].value in {fields["nama"].value, fields["kabupaten_kota"].value, fields["provinsi"].value}
            or _is_region_label_fragment(fields["kelurahan_desa"].value)
        )
    )
    normalized_kelurahan = None if kelurahan_needs_repair else _normalize_region_value(fields["kelurahan_desa"].value or "")
    if normalized_kelurahan:
        fields["kelurahan_desa"] = make_ok(normalized_kelurahan, confidence=fields["kelurahan_desa"].confidence or 0.84)
    else:
        transposed_kelurahan, _ = _fallback_transposed_regions(lines)
        split_kelurahan, split_kecamatan = _fallback_swapped_regions_before_kecamatan_label(lines)
        nik_following_kelurahan = _fallback_kelurahan_after_nik_before_rt(lines)
        if _has_adjacent_kecamatan_before_kelurahan_label(lines):
            fallback_kelurahan = (
                split_kelurahan
                or nik_following_kelurahan
                or _fallback_transposed_kelurahan_before_rt_value(lines)
                or transposed_kelurahan
                or _fallback_region_value(lines, "kelurahan")
            )
        else:
            fallback_kelurahan = (
                split_kelurahan
                or _fallback_region_value(lines, "kelurahan")
                or nik_following_kelurahan
                or _fallback_transposed_kelurahan_before_rt_value(lines)
                or transposed_kelurahan
            )
        if fallback_kelurahan:
            fields["kelurahan_desa"] = make_ok(fallback_kelurahan, confidence=0.78)
        elif fields["kelurahan_desa"].status == "ok":
            fields["kelurahan_desa"] = make_missing()

    kecamatan_needs_repair = (
        fields["kecamatan"].status == "ok"
        and fields["kecamatan"].value
        and fields["kecamatan"].value in {fields["nama"].value, fields["kabupaten_kota"].value, fields["provinsi"].value}
    )
    normalized_kecamatan = None if kecamatan_needs_repair else _normalize_region_value(fields["kecamatan"].value or "")
    if normalized_kecamatan:
        fields["kecamatan"] = make_ok(normalized_kecamatan, confidence=fields["kecamatan"].confidence or 0.84)
    else:
        _, transposed_kecamatan = _fallback_transposed_regions(lines)
        _, split_kecamatan = _fallback_swapped_regions_before_kecamatan_label(lines)
        fallback_kecamatan = split_kecamatan or _fallback_region_value(lines, "kecamatan") or transposed_kecamatan
        if fallback_kecamatan:
            fields["kecamatan"] = make_ok(fallback_kecamatan, confidence=0.78)
        elif fields["kecamatan"].status == "ok":
            fields["kecamatan"] = make_missing()

    split_kelurahan, split_kecamatan = _fallback_swapped_regions_before_kecamatan_label(lines)
    if split_kelurahan and split_kecamatan and fields["kelurahan_desa"].value == split_kecamatan:
        fields["kelurahan_desa"] = make_ok(split_kelurahan, confidence=0.72)
        fields["kecamatan"] = make_ok(split_kecamatan, confidence=0.72)
    reversed_kelurahan, reversed_kecamatan = _fallback_reversed_regions_after_rt(lines)
    if reversed_kelurahan and (
        fields["kelurahan_desa"].status != "ok"
        or _is_region_label_fragment(fields["kelurahan_desa"].value or "")
        or fields["kelurahan_desa"].value == fields["kecamatan"].value
    ):
        fields["kelurahan_desa"] = make_ok(reversed_kelurahan, confidence=0.72)
    if reversed_kecamatan and (
        fields["kecamatan"].status != "ok"
        or _is_region_label_fragment(fields["kecamatan"].value or "")
        or fields["kecamatan"].value == fields["kelurahan_desa"].value
    ):
        fields["kecamatan"] = make_ok(reversed_kecamatan, confidence=0.72)
    adjacent_empty_kelurahan, adjacent_empty_kecamatan = _fallback_adjacent_empty_region_labels(lines)
    if (
        adjacent_empty_kelurahan
        and adjacent_empty_kecamatan
        and fields["kelurahan_desa"].value == adjacent_empty_kecamatan
        and fields["kecamatan"].value == adjacent_empty_kelurahan
    ):
        fields["kelurahan_desa"] = make_ok(adjacent_empty_kelurahan, confidence=0.72)
        fields["kecamatan"] = make_ok(adjacent_empty_kecamatan, confidence=0.72)

    stacked_kelurahan, stacked_kecamatan = _fallback_stacked_regions_after_kecamatan(lines)
    if stacked_kelurahan and not _has_adjacent_kecamatan_before_kelurahan_label(lines) and (
        fields["kelurahan_desa"].status != "ok"
        or fields["kelurahan_desa"].value in {fields["kecamatan"].value, fields["nama"].value}
        or _is_region_label_fragment(fields["kelurahan_desa"].value or "")
    ):
        fields["kelurahan_desa"] = make_ok(stacked_kelurahan, confidence=0.72)
    if stacked_kecamatan and not _has_adjacent_kecamatan_before_kelurahan_label(lines) and (
        fields["kecamatan"].status != "ok"
        or fields["kecamatan"].value == fields["kelurahan_desa"].value
        or _is_region_label_fragment(fields["kecamatan"].value or "")
    ):
        fields["kecamatan"] = make_ok(stacked_kecamatan, confidence=0.72)

    transposed_kelurahan, transposed_kecamatan = _fallback_transposed_regions(lines)
    if (
        transposed_kelurahan
        and transposed_kecamatan
        and fields["kelurahan_desa"].status == "ok"
        and fields["kecamatan"].status == "ok"
        and fields["kelurahan_desa"].value == fields["kecamatan"].value
        and transposed_kelurahan not in {fields["kabupaten_kota"].value, fields["provinsi"].value, fields["nama"].value}
        and (_fallback_region_value(lines, "kelurahan") is None or _has_adjacent_kecamatan_before_kelurahan_label(lines))
    ):
        fields["kelurahan_desa"] = make_ok(transposed_kelurahan, confidence=0.72)
        fields["kecamatan"] = make_ok(transposed_kecamatan, confidence=0.72)

    stacked_late_kelurahan, stacked_late_kecamatan = _fallback_regions_after_empty_kelurahan_before_kecamatan(lines)
    if stacked_late_kelurahan and (
        fields["kelurahan_desa"].status != "ok"
        or _is_region_label_fragment(fields["kelurahan_desa"].value or "")
        or fields["kelurahan_desa"].value == fields["kecamatan"].value
    ):
        fields["kelurahan_desa"] = make_ok(stacked_late_kelurahan, confidence=0.72)
    if stacked_late_kecamatan and (
        fields["kecamatan"].status != "ok"
        or _is_region_label_fragment(fields["kecamatan"].value or "")
        or fields["kecamatan"].value == stacked_late_kelurahan
    ):
        fields["kecamatan"] = make_ok(stacked_late_kecamatan, confidence=0.72)
    if (
        stacked_late_kelurahan
        and stacked_late_kecamatan
        and fields["kelurahan_desa"].value == stacked_late_kecamatan
        and fields["kecamatan"].value == stacked_late_kelurahan
    ):
        fields["kelurahan_desa"] = make_ok(stacked_late_kelurahan, confidence=0.72)
        fields["kecamatan"] = make_ok(stacked_late_kecamatan, confidence=0.72)

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
        elif (
            fields["nik"].status == "ok"
            and (
                any(_is_citizenship_label(line) for line in lines)
                or re.search(r"BERLAK|BERAK|HINGGA|SEUMUR|SEUSUR", raw_text, flags=re.IGNORECASE)
            )
            and not re.search(r"\bWNA\b|WARGA\s+NEGARA\s+ASING", raw_text, flags=re.IGNORECASE)
        ):
            fields["kewarganegaraan"] = FieldResult(
                value="WNI",
                confidence=0.42,
                status="ok",
                evidence=["fallback:ktp_nik_default_wni"],
                raw="fallback:ktp_nik_default_wni",
            )

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
            _canonicalize_regions_from_postal_match(fields, postal_code)
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


def _canonicalize_regions_from_postal_match(fields: dict[str, FieldResult], postal_code) -> None:
    if postal_code.confidence < 0.9 or postal_code.match_status != "exact_match":
        return
    _canonicalize_admin_field(fields, "provinsi", postal_code.nama_provinsi, _normalize_province)
    _canonicalize_admin_field(fields, "kabupaten_kota", postal_code.nama_kota, _normalize_kabupaten_kota)
    _canonicalize_region_field(fields, "kelurahan_desa", postal_code.kelurahan)
    _canonicalize_region_field(fields, "kecamatan", postal_code.kecamatan)


def _canonicalize_admin_field(
    fields: dict[str, FieldResult],
    field_name: str,
    canonical_value: str | None,
    normalizer,
) -> None:
    if not canonical_value:
        return
    normalized_canonical = normalizer(canonical_value)
    if not normalized_canonical:
        return
    field = fields.get(field_name)
    if not field:
        return
    if field.status != "ok" or not field.value:
        fields[field_name] = FieldResult(
            value=normalized_canonical,
            confidence=max(field.confidence or 0.0, 0.84),
            status="ok",
            evidence=[*field.evidence, f"db_kode_wilayah:fill_{field_name}"],
            raw=field.raw or "db_kode_wilayah",
            metadata=field.metadata,
        )
        return
    normalized_current = normalizer(field.value) or _clean_candidate_value(field.value).upper()
    if not normalized_current or not _region_text_matches_canonical(normalized_current, normalized_canonical):
        return
    if field.value == normalized_canonical:
        return
    fields[field_name] = FieldResult(
        value=normalized_canonical,
        confidence=field.confidence,
        status=field.status,
        evidence=[*field.evidence, f"db_kode_wilayah:canonical_{field_name}"],
        raw=field.raw,
        metadata=field.metadata,
    )


def _canonicalize_region_field(fields: dict[str, FieldResult], field_name: str, canonical_value: str | None) -> None:
    if not canonical_value:
        return
    field = fields.get(field_name)
    if not field or field.status != "ok" or not field.value:
        return
    normalized_canonical = _normalize_region_value(canonical_value)
    if not normalized_canonical:
        return
    current_compact = _compact_region_text(field.value)
    canonical_compact = _compact_region_text(normalized_canonical)
    if current_compact != canonical_compact and not _can_restore_kelurahan_suffix(fields, field_name, current_compact, canonical_compact):
        return
    if field.value == normalized_canonical:
        return
    fields[field_name] = FieldResult(
        value=normalized_canonical,
        confidence=field.confidence,
        status=field.status,
        evidence=[*field.evidence, f"db_kode_wilayah:canonical_{field_name}"],
        raw=field.raw,
        metadata=field.metadata,
    )


def _can_restore_kelurahan_suffix(
    fields: dict[str, FieldResult], field_name: str, current_compact: str, canonical_compact: str
) -> bool:
    if field_name != "kelurahan_desa" or not current_compact or not canonical_compact.startswith(current_compact):
        return False
    kecamatan = fields.get("kecamatan")
    if not kecamatan or kecamatan.status != "ok" or _compact_region_text(kecamatan.value or "") != current_compact:
        return False
    suffix = canonical_compact.removeprefix(current_compact)
    return suffix in {"BARAT", "TIMUR", "UTARA", "SELATAN", "TENGAH"}


def _compact_region_text(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def _region_text_matches_canonical(current: str, canonical: str) -> bool:
    current_compact = _compact_region_text(current)
    canonical_compact = _compact_region_text(canonical)
    if current_compact == canonical_compact:
        return True
    shorter, longer = sorted((current_compact, canonical_compact), key=len)
    if len(shorter) >= 5 and longer.startswith(shorter) and len(longer) - len(shorter) <= 1:
        return True
    return False


def _normalize_province(value: str) -> str | None:
    cleaned = _clean_candidate_value(value).upper().strip(" :.-")
    cleaned = (
        cleaned.replace("OKEJAKARIA", "DKI JAKARTA")
        .replace("OKEJAKARTA", "DKI JAKARTA")
        .replace("DKLJAKARTA", "DKI JAKARTA")
        .replace("DKL JAKARTA", "DKI JAKARTA")
    )
    cleaned = re.sub(r"^(?:PR[O0]V[1I]NS[1I!L])\s*", "", cleaned)
    cleaned = re.sub(r"^(?:PR[O0]V[1I]NS[1I!L])(?=[A-Z])", "", cleaned)
    cleaned = re.sub(r"\b(?:KABUPATEN|KOTA|NIK|NAMA)\b.*$", "", cleaned).strip(" :.-")
    if not cleaned or _contains_any_label(cleaned):
        return None
    if normalize_nik(cleaned) or any(char.isdigit() for char in cleaned):
        return None
    return cleaned if re.fullmatch(r"[A-Z][A-Z .'-]{2,}", cleaned) else None


def _fallback_province(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        upper = line.upper()
        if _is_province_label(upper):
            normalized = _normalize_province(upper)
            if normalized:
                return normalized
            if index + 1 < len(lines):
                normalized = _normalize_province(lines[index + 1])
                if normalized:
                    return normalized
    return None


def _is_province_label(value: str) -> bool:
    compact = re.sub(r"[^A-Z0-9!]", "", value.upper())
    compact = compact.replace("0", "O").replace("1", "I").replace("L", "I")
    return compact.startswith("PROVINSI")


def _normalize_kabupaten_kota(value: str) -> str | None:
    cleaned = _clean_candidate_value(value).upper().strip(" :.-")
    cleaned = cleaned.replace("JAKARIA", "JAKARTA").replace("JAMBL", "JAMBI")
    cleaned = (
        cleaned.replace("JAKARTAUTARA", "JAKARTA UTARA")
        .replace("JAKARTABARAT", "JAKARTA BARAT")
        .replace("JAKARTAPUSAT", "JAKARTA PUSAT")
        .replace("JAKARTASELATAN", "JAKARTA SELATAN")
        .replace("JAKARTATIMUR", "JAKARTA TIMUR")
    )
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
    cleaned = re.sub(r"^(?:KOTA\s+ADMINISTRASI|KABUPATEN|KAB\.?\s*/?\s*KOT\w*|KAB\.?|KOTA)\s*", "", cleaned).strip(" :.-")
    if not cleaned:
        return None
    return cleaned if re.fullmatch(r"[A-Z][A-Z .'\-/]{2,}", cleaned) else None


def _fallback_kabupaten_kota(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        upper = line.upper()
        normalized = _admin_area_from_header_line(upper)
        if normalized:
            return normalized

        if _is_province_label(upper):
            for candidate in lines[index + 1 : min(len(lines), index + 4)]:
                candidate_upper = candidate.upper()
                if _contains_any_label(candidate_upper):
                    break
                normalized = _normalize_kabupaten_kota(candidate)
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

    jakarta_bare_lines = {
        "JAKARTA BARAT",
        "JAKARTA UTARA",
        "JAKARTA SELATAN",
        "JAKARTA TIMUR",
        "JAKARTA PUSAT",
        "JAMBI",
    }
    for line in lines:
        normalized = _normalize_kabupaten_kota(line)
        if normalized in jakarta_bare_lines:
            return normalized
    return None


def _admin_area_from_header_line(line: str) -> str | None:
    match = re.search(r"\b(?:KABUPATEN|KAB\.?\s*/?\s*KOT\w*|KAB\.?|KOTA(?:\s+ADMINISTRASI)?)\s*(.+)$", line, flags=re.IGNORECASE)
    if not match:
        return None
    return _normalize_kabupaten_kota(match.group(0))


def _is_kabupaten_kota_label(value: str) -> bool:
    compact = re.sub(r"[^A-Z]", "", value.upper())
    return compact in {"KABUPATEN", "KAB", "KOTA", "KOTAADMINISTRASI", "KABKOTA", "KABKATA"}


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
            if raw_candidate.lstrip().startswith(":"):
                continue
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


def _fallback_name_after_label_window(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not _is_name_label(line):
            continue
        for raw_candidate in lines[index + 1 : min(len(lines), index + 14)]:
            if raw_candidate.lstrip().startswith(":"):
                continue
            candidate = _clean_candidate_value(raw_candidate)
            upper = candidate.upper()
            if not candidate or _contains_any_label(upper):
                continue
            if normalize_nik(candidate) or _normalize_rt_rw(candidate) or _is_date_like(candidate):
                continue
            if (
                _normalize_gender(candidate)
                or _normalize_religion(candidate)
                or _normalize_marital_status(candidate)
                or _normalize_citizenship(candidate)
                or _normalize_job(candidate)
                or _looks_like_address(candidate)
            ):
                continue
            if _looks_like_person_name(candidate) or _looks_like_single_person_name(candidate):
                return candidate
    return None


def _fallback_name_before_nik_line(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not normalize_nik(line):
            continue
        if not any(_is_name_label(candidate) for candidate in lines[max(0, index - 20) : index]):
            continue
        for raw_candidate in reversed(lines[max(0, index - 4) : index]):
            candidate = _clean_candidate_value(raw_candidate)
            if _looks_like_person_name(candidate) or _looks_like_single_person_name(candidate):
                return candidate
    return None


def _fallback_transposed_person_name(lines: list[str]) -> str | None:
    candidates: list[str] = []
    for raw_candidate in lines:
        candidate = _clean_candidate_value(raw_candidate)
        upper = candidate.upper()
        if _contains_any_label(upper) or normalize_nik(candidate) or _normalize_rt_rw(candidate) or _is_date_like(candidate):
            continue
        if (
            _normalize_gender(candidate)
            or _normalize_religion(candidate)
            or _normalize_marital_status(candidate)
            or _normalize_citizenship(candidate)
            or _normalize_job(candidate)
            or _looks_like_address(candidate)
            or _is_city_or_region_line(candidate)
            or _looks_like_admin_area_fragment(candidate)
            or re.fullmatch(r"JAKARTA\s+(?:BARAT|TIMUR|UTARA|SELATAN|PUSAT)", upper)
        ):
            continue
        if _looks_like_person_name(candidate):
            candidates.append(candidate)
    return candidates[-1] if candidates else None


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


def _fallback_name_before_birth_line(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not _contains_birth_date(line):
            continue
        for raw_candidate in reversed(lines[max(0, index - 3) : index]):
            candidate = _clean_candidate_value(raw_candidate)
            upper = candidate.upper()
            if _contains_any_label(upper) or normalize_nik(candidate) or _normalize_rt_rw(candidate):
                continue
            if _looks_like_person_name(candidate) or _looks_like_single_person_name(candidate):
                return candidate
    return None


def _extend_wrapped_person_name(lines: list[str], value: str) -> str:
    current = _clean_candidate_value(value).upper()
    if not (_looks_like_person_name(current) or _looks_like_single_person_name(current)):
        return value

    for index, line in enumerate(lines[:-2]):
        if _clean_candidate_value(line).upper() != current:
            continue

        continuation = _clean_candidate_value(lines[index + 1]).upper()
        if not _looks_like_name_continuation(continuation):
            continue
        if not _is_birth_label(lines[index + 2]):
            continue
        return f"{current} {continuation}"

    return value


def _looks_like_name_continuation(value: str) -> bool:
    if (
        not value
        or _contains_any_label(value)
        or normalize_nik(value)
        or _normalize_rt_rw(value)
        or _is_date_like(value)
        or _is_city_or_region_line(value)
        or _looks_like_address(value)
        or _normalize_gender(value)
        or _normalize_religion(value)
        or _normalize_marital_status(value)
        or _normalize_citizenship(value)
        or _normalize_job(value)
    ):
        return False
    return _looks_like_single_person_name(value)


def _repair_joined_person_name(value: str) -> str:
    cleaned = _clean_candidate_value(value).upper()
    if not cleaned or _contains_any_label(cleaned) or any(char.isdigit() for char in cleaned):
        return value

    repaired_tokens = [_repair_joined_name_token(_repair_joined_initial_token(token)) for token in cleaned.split()]
    return " ".join(repaired_tokens)


def _repair_joined_initial_token(token: str) -> str:
    match = re.fullmatch(r"([A-Z']{6,})([A-Z])\.", token)
    if not match:
        return token
    stem, initial = match.groups()
    if stem not in JOINED_INITIAL_NAME_STEMS:
        return token
    return f"{stem} {initial}."


def _joined_name_confidence_cap(original: str, repaired: str) -> float:
    if _has_joined_initial_repair(original, repaired):
        return 0.82
    return 0.78


def _has_joined_initial_repair(original: str, repaired: str) -> bool:
    original_upper = _clean_candidate_value(original).upper()
    repaired_upper = _clean_candidate_value(repaired).upper()
    for stem in JOINED_INITIAL_NAME_STEMS:
        if re.search(rf"\b{re.escape(stem)}[A-Z]\.", original_upper) and re.search(
            rf"\b{re.escape(stem)} [A-Z]\.", repaired_upper
        ):
            return True
    return False


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
        if suffix.startswith("SANT") and suffix != "SANTOSO":
            continue
        if not any(suffix.startswith(stem) for stem in JOINED_NAME_SUFFIX_PREFIXES):
            continue
        if _looks_like_single_person_name(suffix):
            return f"{prefix} {suffix}"
    for prefix in ("PUTRI", "PUTRA"):
        if not token.startswith(prefix):
            continue
        suffix = token[len(prefix) :]
        if len(suffix) < 4:
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


def _fallback_truncated_birth_place_date_from_nik(lines: list[str], nik: str | None) -> str | None:
    birth_date = _birth_date_from_nik(nik)
    if not birth_date:
        return None

    for index, line in enumerate(lines):
        malformed_date_pattern = r"\b\d{1,2}[-/.\s]+\d{1,2}[-/.\s]+[0-9OILSB%]{2,4}(?![0-9OILSB%])"
        if not re.search(malformed_date_pattern, line):
            continue
        for candidate in reversed(lines[max(0, index - 3) : index + 1]):
            place = _clean_birth_place(re.sub(malformed_date_pattern, "", candidate))
            if place:
                return f"{place}, {birth_date}"
    return None


def _birth_date_from_nik(nik: str | None) -> str | None:
    normalized = normalize_nik(nik or "")
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


def _normalize_birth_place_date(value: str) -> str | None:
    value = value.replace("Â·", "-").replace("·", "-").replace("•", "-")
    pattern = re.compile(
        r"\b([A-Z][A-Z0-9.\s'-]{2,}?)[,.\s]*([0-9OILSB]{1,2})[-/.\s]+([0-9OILSB]{1,2})[-/.\s:]*([0-9OILSB:]{4,6})\b",
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
    if not (1900 <= int(year) <= date.today().year):
        return None
    return f"{place}, {day}-{month}-{year}"


def _clean_birth_place(value: str) -> str | None:
    cleaned = _clean_candidate_value(value).upper().strip(" ,.-:")
    cleaned = re.sub(
        r"\b(?:TEMP\w*|TAL|[I1]G[L1I]?LAHIR|EGL+AHIR|TGL\s*LAHIR|TGI\s*LAHIR|TGLLAHIR|TGILAHIR|TOLLAHIR|TOILAHIR|TGL|TGI|TOL|TOI|TANGGAL|LAHIR|LHIR|LABE)\b",
        " ",
        cleaned,
    )
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
    return bool(re.search(r"\b\d{1,2}[-/.\s]+\d{1,2}[-/.\s]+\d{4}\b", value))


def _is_birth_label(value: str) -> bool:
    compact = re.sub(r"[^A-Z]", "", value.upper())
    return bool("LAHIR" in compact and ("TEMP" in compact or "TG" in compact or "TGL" in compact or "TAL" in compact))


def _normalize_gender(value: str) -> str | None:
    compact = re.sub(r"[^A-Z]", "", value.upper())
    if compact.startswith("LAK") or compact.startswith("TAK") or "LAKILAKI" in compact or "LAKLAKI" in compact:
        return "LAKI-LAKI"
    if (
        compact.startswith("PEREMP")
        or "PEREMPUAN" in compact
        or "PEREMPUIAN" in compact
        or compact.startswith("PEREP")
        or compact.startswith("PERENP")
        or compact.startswith("PERENPL")
        or compact.startswith("PERENU")
    ):
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
        if _contains_any_label(candidate.upper()) and not _looks_like_address(candidate):
            continue
        if _looks_like_address(candidate):
            return candidate

    for index, line in enumerate(lines):
        if not ADDRESS_ANCHOR_PATTERN.search(line):
            continue
        for candidate in reversed(lines[max(0, index - 4) : index]):
            candidate = _clean_candidate_value(candidate)
            upper = candidate.upper()
            if _contains_any_label(upper) and not _looks_like_address(candidate):
                continue
            if _looks_like_address(candidate) or _looks_like_loose_address_before_rt(candidate):
                return candidate
    return None


def _fallback_labeled_address(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not re.search(r"\b(?:ALAMAT|ALAMAL|ALMAL|ALANAT|ALSMAT)\b", line, flags=re.IGNORECASE):
            continue

        inline = re.sub(r"^.*?\b(?:ALAMAT|ALAMAL|ALMAL|ALANAT|ALSMAT)\b\s*[:\-]?\s*", "", line, flags=re.IGNORECASE).strip()
        candidates: list[str] = []
        if inline and inline.upper() != line.upper():
            candidates.append(_clean_candidate_value(inline))

        for next_line in lines[index + 1 : min(len(lines), index + 5)]:
            if (
                ADDRESS_ANCHOR_PATTERN.search(next_line)
                or _is_rt_rw_label(next_line)
                or _is_kelurahan_label(next_line)
                or _is_kecamatan_label(next_line)
            ):
                break
            cleaned = _clean_candidate_value(next_line)
            upper = cleaned.upper()
            if not cleaned or re.search(r"\bGOL\.?\s*DARAH\b|\bDARAH\b", upper):
                continue
            if _contains_any_label(upper) and not _looks_like_address(cleaned):
                break
            if _normalize_gender(cleaned) or _normalize_religion(cleaned) or _normalize_marital_status(cleaned):
                break
            candidates.append(cleaned)

        combined = " ".join(candidates).strip()
        if combined and _looks_like_valid_address_value(combined):
            return combined

    return None


def _extend_address_with_trailing_fragment(lines: list[str], value: str) -> str | None:
    current = _clean_candidate_value(value).upper()
    if not current:
        return None

    for index, line in enumerate(lines):
        if _clean_candidate_value(line).upper() != current:
            continue
        fragments = [current]
        saw_rt_rw = False
        for candidate in lines[index + 1 : min(len(lines), index + 8)]:
            if _is_kelurahan_label(candidate) or _is_kecamatan_label(candidate):
                break
            if _is_rt_rw_label(candidate):
                saw_rt_rw = True
                continue
            if saw_rt_rw and _normalize_rt_rw(candidate):
                continue
            cleaned = _clean_candidate_value(candidate)
            upper = cleaned.upper()
            if not cleaned or len(cleaned) <= 1:
                continue
            if _contains_any_label(upper):
                break
            if _normalize_gender(cleaned) or _normalize_religion(cleaned) or _normalize_marital_status(cleaned):
                break
            if _looks_like_address(cleaned) or re.search(r"\b(?:NO|N0)\.?\s*[A-Z0-9]{1,6}\b", upper):
                fragments.append(cleaned)
        if len(fragments) > 1:
            return " ".join(fragments)
    return None


def _normalize_address_value(value: str) -> str:
    cleaned = collapse_spaces(value)
    cleaned = re.sub(r"^(JALAN)(?=[A-Z0-9])", r"\1 ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(JLN)(?=[A-Z0-9])", r"\1 ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(JL)(?=[A-Z0-9])(?!N(?:\b|[.]))(?!ALAN(?:\b|[.]))", r"\1 ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^J\s+(?=[A-Z])", "JL ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^JA\s+(?=[A-Z])", "JL ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bBEND\.?\s+HIUR\b", "BEND. HILIR", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bBALIRESIDENCE\b", "BALI RESIDENCE", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bIARAT\b", "BARAT", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(BLOK|BLK)([A-Z]{1,2})(?=\b|[\s./-]|\d)", r"\1 \2", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?<=[A-Z0-9])(?=(?:NO\.?\s*\d|RT\b|RW\b|BLOK\b|BLK\b|KAV\b|GG\b))", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"([A-Z])NO\.?(?=\d)", r"\1 NO.", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bNO\.?G(?=[A-Z]\b)", "NO.6", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"ASRIX(?= NO\.?\d)", "ASRI X", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(?:AT\.?RW|RT\.?RW|RTRW)\s*$", "", cleaned, flags=re.IGNORECASE)
    return collapse_spaces(cleaned)


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
        for candidate in reversed(lines[max(0, index - 4) : index]):
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

        value_candidates: list[tuple[str, str]] = []
        for candidate in lines[index + 1 : min(len(lines), index + 5)]:
            if label_fn(candidate) or _is_region_label_fragment(candidate):
                continue
            cleaned_candidate = _clean_candidate_value(candidate).upper().strip(" :;._-")
            if kind == "kecamatan" and _looks_like_kota_suffix_region(cleaned_candidate):
                return cleaned_candidate
            if _is_job_label(candidate):
                continue
            if _contains_any_label(candidate.upper()):
                break
            normalized = _normalize_region_value(candidate)
            if normalized:
                if kind == "kecamatan":
                    value_candidates.append((candidate, normalized))
                    continue
                return normalized
            if kind == "kecamatan":
                cleaned = _clean_candidate_value(candidate).upper().strip(" :;._-")
                if cleaned.endswith(" KOTA") and re.fullmatch(r"[A-Z][A-Z .'\-/]{4,}", cleaned):
                    return cleaned
        if kind == "kecamatan" and value_candidates:
            _, first_value = value_candidates[0]
            compact_label = re.sub(r"[^A-Z]", "", line.upper())
            first_letters = len(re.findall(r"[A-Z]", first_value))
            if (
                compact_label == "KEC"
                and len(value_candidates) > 1
                and first_letters <= 5
                and len(re.findall(r"[A-Z]", value_candidates[1][1])) >= first_letters + 3
            ):
                return value_candidates[1][1]
            return first_value
        if kind == "kecamatan":
            for candidate in reversed(lines[max(0, index - 3) : index]):
                normalized = _normalize_region_value(candidate)
                if normalized:
                    return normalized
    return None


def _region_value_from_inline_label(line: str, kind: str) -> str | None:
    if kind == "kelurahan":
        label = r"(?:K[EOA]L\s*/?\s*[DO]ES[AN]?|KOL\s*/?\s*DASA|KEL\s*/?\s*OOSA|DESA\s*/?\s*K[EO]L|KELDESA|KALDESA|KOLDESA|KOLDASA|KEDESA|KEILDESA|KEWDESA|KEVDESA|KELDOSA|KELOOSA|KELDESS|KALDESS|NOESA|EL\s*/?\s*DESA)"
    else:
        inline_joined = re.match(
            r"^\s*(?:KECAMATAN|KECAMNATAN|KECAMATAR|KECAMETAN|KECAMRETAN|KECANATAN|KICANALAN|XECAMATAN|ECAMATAN|ECEMATAN|IECAMATAN|KECAMNATAN|KECAMNATAN|KECAM)\s*(?:[_:\-]|\s+)(.+)$",
            line,
            flags=re.IGNORECASE,
        )
        if inline_joined:
            return _normalize_region_value(inline_joined.group(1))
        label = r"(?:KECAMATAN|KECAMNATAN|KECAMATAR|KECANATAN|KECAMNATAN|KECA\s*MATEN|KICANALAN|XECAMATAN|ECAMATAN|ECEMATAN|IECAMATAN)"
    match = re.search(rf"{label}\s*[:_\-]?\s*(.+)$", line, flags=re.IGNORECASE)
    if not match:
        return None
    return _normalize_region_value(match.group(1))


def _normalize_region_value(value: str) -> str | None:
    cleaned = _clean_candidate_value(value).upper().strip(" :;._-")
    cleaned = cleaned.replace("!", "I")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = (
        cleaned.replace("GANDARA SELATAN", "GANDARIA SELATAN")
        .replace("CLANDAK", "CILANDAK")
        .replace("TANAHABANG", "TANAH ABANG")
        .replace("BENDUNGANHLIR", "BENDUNGAN HILIR")
        .replace("PB SELAYANGI", "PADANG BULAN SELAYANG I")
    )
    if not cleaned:
        return None
    if _is_region_label_fragment(cleaned):
        return None
    if _looks_like_kota_suffix_region(cleaned):
        return cleaned
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
        or _is_job_label(cleaned)
    ):
        return None
    if _is_rt_rw_label(cleaned):
        return None
    if _normalize_expiry(cleaned) or (_looks_like_address(cleaned) and not cleaned.startswith(("TAMAN ", "TMN ", "DUKUH"))):
        return None
    letters = re.findall(r"[A-Z]", cleaned)
    if len(letters) < 3:
        return None
    return cleaned if re.fullmatch(r"[A-Z][A-Z0-9 .'\-/]{2,}", cleaned) else None


def _is_kelurahan_label(value: str) -> bool:
    compact = re.sub(r"[^A-Z0-9]", "", value.upper())
    return compact in {
        "KELDESA",
        "KALDESA",
        "KEDESA",
        "KEILDESA",
        "KELURAHAN",
        "DESA",
        "DESAKEL",
        "KOLDESA",
        "KOLDASA",
        "KEWDESA",
        "KEVDESA",
        "KELDESN",
        "KELDOSA",
        "KELOOSA",
        "KELDESS",
        "KELDEAN",
        "DESAKOL",
        "DESAKEL",
        "KALDESS",
        "NOESA",
        "NOASA",
        "N0ESA",
        "VOASA",
        "ELDESA",
    } or compact.startswith("KELD")


def _is_kecamatan_label(value: str) -> bool:
    compact = re.sub(r"[^A-Z]", "", value.upper())
    if compact in {"KEC", "EATAN", "CAMATAN", "COMATAN", "ECEMATAN", "IECAMATAN", "KICANALAN", "CAMNATAN"}:
        return True
    return bool(re.fullmatch(r"[KXE]?ECAM[A-Z]*|KECAM[A-Z]*|KECAN[A-Z]*|CAMNATAN|KEC[A-Z]*NATAN|KECAMRETAN|KECANATAR", compact))


def _is_region_label_fragment(value: str) -> bool:
    compact = re.sub(r"[^A-Z]", "", value.upper())
    return compact in {
        "EATAN",
        "MATAN",
        "CAMATAN",
        "DESA",
        "KELDESA",
        "KEILDESA",
        "KELDEAN",
        "KICANALAN",
        "NANIA",
        "DESS",
        "DOSA",
        "DESN",
    }


def _looks_like_kota_suffix_region(value: str) -> bool:
    return bool(
        value.endswith(" KOTA")
        and not value.startswith("KOTA ")
        and "/" not in value
        and re.fullmatch(r"[A-Z][A-Z .'\-/]{4,}", value)
    )


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


def _fallback_swapped_regions_before_kecamatan_label(lines: list[str]) -> tuple[str | None, str | None]:
    for index, line in enumerate(lines):
        if not _is_kelurahan_label(line):
            continue
        values: list[str] = []
        saw_kecamatan_label = False
        for candidate in lines[index + 1 : min(len(lines), index + 6)]:
            if _is_kecamatan_label(candidate):
                saw_kecamatan_label = True
                break
            if _contains_any_label(candidate.upper()):
                if values:
                    break
                continue
            normalized = _normalize_region_value(candidate)
            if normalized:
                values.append(normalized)
        if saw_kecamatan_label and len(values) >= 2:
            return values[-1], values[0]
    return None, None


def _fallback_kelurahan_after_nik_before_rt(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not normalize_nik(line):
            continue
        last_region: str | None = None
        for candidate in lines[index + 1 : min(len(lines), index + 6)]:
            if _normalize_rt_rw(candidate):
                return last_region
            if _contains_any_label(candidate.upper()):
                return last_region
            normalized = _normalize_region_value(candidate)
            if normalized:
                last_region = normalized
    return None


def _has_adjacent_kecamatan_before_kelurahan_label(lines: list[str]) -> bool:
    for index, line in enumerate(lines[:-1]):
        if _is_kecamatan_label(line) and _is_kelurahan_label(lines[index + 1]):
            return True
    return False


def _fallback_transposed_kelurahan_before_rt_value(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not _is_rt_rw_label(line):
            continue
        last_region: str | None = None
        for candidate in lines[index + 1 : min(len(lines), index + 12)]:
            if _normalize_rt_rw(candidate):
                return last_region
            if last_region and (
                _looks_like_address(candidate)
                or _normalize_gender(candidate)
                or _is_date_like(candidate)
                or normalize_nik(candidate)
            ):
                return last_region
            normalized = _normalize_region_value(candidate)
            if normalized:
                last_region = normalized
    return None


def _fallback_stacked_regions_after_kecamatan(lines: list[str]) -> tuple[str | None, str | None]:
    for index, line in enumerate(lines):
        if not _is_kecamatan_label(line):
            continue
        values: list[str] = []
        for candidate in lines[index + 1 : min(len(lines), index + 5)]:
            normalized = _normalize_region_value(candidate)
            if normalized:
                values.append(normalized)
                if len(values) == 2:
                    return values[0], values[1]
            elif values:
                break
    return None, None


def _fallback_regions_after_empty_kelurahan_before_kecamatan(lines: list[str]) -> tuple[str | None, str | None]:
    for kelurahan_index, line in enumerate(lines):
        if not _is_kelurahan_label(line):
            continue

        kecamatan_index: int | None = None
        values_before_kecamatan: list[str] = []
        for offset, candidate in enumerate(lines[kelurahan_index + 1 : min(len(lines), kelurahan_index + 8)], start=kelurahan_index + 1):
            if _is_kecamatan_label(candidate):
                kecamatan_index = offset
                break
            if _normalize_rt_rw(candidate):
                continue
            if _contains_any_label(candidate.upper()):
                break
            normalized = _normalize_region_value(candidate)
            if normalized:
                values_before_kecamatan.append(normalized)

        if kecamatan_index is None or values_before_kecamatan:
            continue

        values_after_kecamatan: list[str] = []
        for candidate in lines[kecamatan_index + 1 : min(len(lines), kecamatan_index + 10)]:
            if _is_kelurahan_label(candidate) or _is_kecamatan_label(candidate) or _is_rt_rw_label(candidate):
                continue
            if (
                _normalize_religion(candidate)
                or _normalize_gender(candidate)
                or _normalize_marital_status(candidate)
                or _normalize_citizenship(candidate)
                or _normalize_job(candidate)
            ):
                continue
            if _contains_any_label(candidate.upper()) and values_after_kecamatan:
                break
            normalized = _normalize_region_value(candidate)
            if normalized and normalized not in values_after_kecamatan:
                values_after_kecamatan.append(normalized)
                if len(values_after_kecamatan) == 2:
                    return values_after_kecamatan[0], values_after_kecamatan[1]

    return None, None


def _fallback_adjacent_empty_region_labels(lines: list[str]) -> tuple[str | None, str | None]:
    for index, line in enumerate(lines[:-1]):
        if not _is_kelurahan_label(line) or not _is_kecamatan_label(lines[index + 1]):
            continue
        values: list[str] = []
        for candidate in lines[index + 2 : min(len(lines), index + 8)]:
            if _is_kelurahan_label(candidate) or _is_kecamatan_label(candidate) or _is_rt_rw_label(candidate):
                continue
            if (
                _normalize_religion(candidate)
                or _normalize_gender(candidate)
                or _normalize_marital_status(candidate)
                or _normalize_citizenship(candidate)
                or _normalize_job(candidate)
            ):
                break
            normalized = _normalize_region_value(candidate)
            if normalized and normalized not in values:
                values.append(normalized)
                if len(values) == 2:
                    return values[0], values[1]
    return None, None


def _fallback_reversed_regions_after_rt(lines: list[str]) -> tuple[str | None, str | None]:
    for index, line in enumerate(lines[:-2]):
        if not _is_kecamatan_label(line):
            continue
        if not _is_kelurahan_label(lines[index + 1]) or not _is_rt_rw_label(lines[index + 2]):
            continue

        values: list[str] = []
        for candidate in lines[index + 3 : min(len(lines), index + 12)]:
            if (
                _normalize_religion(candidate)
                or _normalize_gender(candidate)
                or _normalize_marital_status(candidate)
                or _normalize_citizenship(candidate)
                or _normalize_job(candidate)
            ):
                continue
            normalized = _normalize_region_value(candidate)
            if normalized and normalized not in values:
                values.append(normalized)
                if len(values) == 2:
                    return values[1], values[0]
    return None, None


def _contains_non_region_value_label(value: str) -> bool:
    compact = re.sub(r"[^A-Z]", "", value.upper())
    exact_blocked = {"NIK", "NAMA", "TEMPAT", "TGL", "LAHIR", "ALAMAT", "AGAMA", "STATUS", "PEKERJAAN", "KEWARGANEGARAAN", "BERLAKU", "HINGGA"}
    if compact in exact_blocked:
        return True
    blocked_patterns = [
        r"\b(?:NIK|NAMA|TEMP\w*|TGL|TANGGAL|LAHIR|ALAMAT|AGAMA|STATUS|PEKER\w*|PEIKER\w*|PAKER\w*|K[EO]WARG\w*|BERLAKU|HINGGA|PROVINSI|KABUPATEN|KOTA|GOL|DARAH)\b",
        r"\bJENIS\s*KELAMIN\b",
        r"\bPERKAW\w*\b",
    ]
    return any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in blocked_patterns)


def _normalize_religion(value: str) -> str | None:
    cleaned = _clean_candidate_value(value)
    compact = re.sub(r"[^A-Z]", "", cleaned.upper())
    if not compact:
        return None
    if compact in {"KATHOLK", "KATHOLC", "KATTHOLIK"}:
        return "KATHOLIK"
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
    compact = re.sub(r"[^A-Z]", "", upper)
    return bool(
        ADDRESS_ANCHOR_PATTERN.search(upper)
        or compact in {"RTRW", "RTRV", "TTRW", "TITRW", "ATRW", "RYRW"}
    )


def _contains_any_label(value: str) -> bool:
    labels = [label.upper() for values in KTP_LABELS.values() for label in values]
    strict_word_labels = {"NIK", "NAMA", "NAME", "NAMS", "NANIA", "NAMAT", "NAM"}
    for label in labels:
        if label in strict_word_labels:
            if re.search(rf"(?<![A-Z]){re.escape(label)}(?![A-Z])", value):
                return True
            continue
        if label in value:
            return True
    return bool(FUZZY_LABEL_PATTERN.search(value))


def _looks_like_person_name(value: str) -> bool:
    upper = value.upper()
    if upper in FIELD_VALUE_BLOCKLIST:
        return False
    if _normalize_expiry(upper):
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
    if _normalize_expiry(upper):
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
    cleaned = re.sub(r"\bKOTA\s+[A-Z .'-]+$", "", cleaned, flags=re.IGNORECASE).strip()
    upper = cleaned.upper()
    compact = re.sub(r"[^A-Z]", "", upper)
    if not cleaned or _contains_any_label(upper) or _is_city_or_region_line(upper) or _is_date_like(upper):
        return None
    if "KARYAWAN" in upper and ("BUMN" in upper or "BUUN" in upper or "BUN" in upper or "BUM" in upper):
        return "KARYAWAN BUMN"
    if "KARYAWAN" in upper and ("SWASTA" in upper or "SWAST" in upper):
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
    if compact in {"WNI", "VNI", "YNI"}:
        return "WNI"
    if compact in {"WNA", "VNA", "YNA"}:
        return "WNA"
    return None


def _normalize_citizenship_near_label(value: str) -> str | None:
    normalized = _normalize_citizenship(value)
    if normalized:
        return normalized

    compact = re.sub(r"[^A-Z0-9]", "", value.upper()).replace("1", "I").replace("L", "I")
    if compact in {"WN", "VKE"} or compact.endswith("WNI") or compact.endswith("WN"):
        return "WNI"
    if compact in {"U", "V", "Y"}:
        return "WNI"
    if compact in {"WI", "NI", "VN"}:
        return None
    return None


def _fallback_citizenship(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not _is_citizenship_label(line):
            continue
        short_fragments: list[str] = []
        for candidate in lines[index + 1 : min(len(lines), index + 4)]:
            if _contains_any_label(candidate.upper()) and not _is_citizenship_label(candidate):
                break
            compact = re.sub(r"[^A-Z0-9]", "", candidate.upper()).replace("1", "I").replace("L", "I")
            if 1 <= len(compact) <= 3:
                short_fragments.append(compact)
            combined = _normalize_citizenship("".join(short_fragments))
            if combined:
                return combined
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
    return "KEWARG" in compact or "KEVRG" in compact or "KWARG" in compact or "KOWARG" in compact or "KAWARG" in compact


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
    for line in lines:
        normalized = _normalize_expiry(line)
        if normalized == "SEUMUR HIDUP":
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
    label = r"(?:BERLAK\w*|BERAK\w*|BERIAK\w*|BERIAKY|BARLAK\w*|BERLAKY|BORLAK\w*|BERBAKY|BERFAKU|BERFAK\w*|SERFAKU|SERFAK\w*|BERTAKU|BERTAK\w*|BARTARAR|IEDAKU\w*|NAKU|AK\.?)"
    match = re.search(rf"{label}\s*:?\s*(?:HING\w*|HIN\w*|HNOG\w*)?\s*[:_\-]?\s*(.*)$", line, flags=re.IGNORECASE)
    if not match:
        return None
    return _normalize_expiry(match.group(1))


def _normalize_expiry(value: str) -> str | None:
    cleaned = _clean_candidate_value(value).upper().strip(" :;._-")
    if not cleaned:
        return None
    compact = re.sub(r"[^A-Z0-9]", "", cleaned).replace("1", "I").replace("0", "O")
    if compact.startswith("SEUMURHID") or compact.startswith("SEUSURH") or compact in {"SEUMURHIOUP", "SEUMUHIDUP", "SEUSURHOUP", "SEUMURHDUP"}:
        return "SEUMUR HIDUP"

    match = re.search(r"(?<!\d)(\d{1,2})[-/.\s]+(\d{1,2})[-/.\s]+(\d{4})(?!\d)", cleaned)
    if not match:
        return None
    day = match.group(1).zfill(2)
    month = match.group(2).zfill(2)
    year = match.group(3)
    if int(year) < 2000:
        return None
    return f"{day}-{month}-{year}"


def _is_expiry_label(value: str) -> bool:
    compact = re.sub(r"[^A-Z]", "", value.upper())
    has_prefix = any(prefix in compact for prefix in ["BERLAKU", "BERAKU", "BERIAKU", "BERIAKY", "BARLAKU", "BERLAKY", "BORLAKU", "BERBAKY", "BERFAKU", "SERFAKU", "BERTAKU", "BARTARAR", "IEDAKU", "NAKU"])
    return (
        has_prefix
        and ("HING" in compact or "HIN" in compact or "HNOG" in compact)
        or compact.startswith(("BERLAKU", "BERAKU", "BERIAKU", "BERIAKY", "BARLAKU", "BERLAKY", "SERFAKU", "BERTAKU"))
        or compact in {"AKHINGQA", "AKHINGGA"}
    )


def _is_job_label(value: str) -> bool:
    return bool(JOB_LABEL_PATTERN.search(value))


def _is_city_or_region_line(value: str) -> bool:
    upper = value.upper()
    return bool(re.fullmatch(r":?\s*(KOTA|KABUPATEN|KAB|PROVINSI)?\s*[A-Z .'-]{3,}\s*", upper) and any(marker in upper for marker in ["KOTA", "KABUPATEN", "PROVINSI"]))


def _looks_like_admin_area_fragment(value: str) -> bool:
    upper = _clean_candidate_value(value).upper()
    tokens = set(re.findall(r"[A-Z]+", upper))
    if "AKARTA" in upper:
        return True
    return bool(tokens & {"BARA", "BARAT", "TIMU", "TIMUR", "UTAR", "UTARA", "SELATA", "SELATAN", "PUSAT"})


def _is_date_like(value: str) -> bool:
    return bool(re.search(r"\d{1,2}[-./]\d{1,2}[-./]\d{2,4}", value))


def _name_value_needs_repair(value: str) -> bool:
    upper = value.upper()
    letter_count = len(re.findall(r"[A-Z]", upper))
    return bool(
        _contains_any_label(upper)
        or _normalize_expiry(upper)
        or _looks_like_address(upper)
        or _is_city_or_region_line(upper)
        or "AKARTA" in upper
        or any(keyword in upper.split() for keyword in NON_NAME_KEYWORDS)
        or any(char.isdigit() for char in upper)
        or re.search(r"\d{2}[-/]\d{2}[-/]\d{4}", upper)
        or letter_count <= 1
    )


def _looks_like_address(value: str) -> bool:
    upper = value.upper()
    return bool(
        re.search(r"\b(J|JL|JLN|JALAN|KP|KAMP|KAMPUNG|DSN|DUSUN|DUKUH|LINGK|GG|GANG|PERUM|KOMP\w*|BLOK\w*|KAV|TMN|TAMAN|SIMPANG)\b\.?", upper)
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
    normalized = _normalize_address_value(upper)
    if not upper or upper in FIELD_VALUE_BLOCKLIST:
        return False
    if re.search(r"\bGOL\.?\s*DARAH\b|\bDARAH\b", upper):
        return False
    if _looks_like_address(normalized):
        return True
    if _contains_any_label(upper):
        return False
    return _looks_like_address(normalized) or _looks_like_loose_address_before_rt(normalized)


def _is_standalone_nik_line(value: str) -> bool:
    upper = value.upper()
    if _contains_any_label(upper):
        return False
    digits = re.sub(r"\D", "", value)
    return len(digits) == 16 and normalize_nik(value) is not None


def _is_name_label(value: str) -> bool:
    return bool(re.fullmatch(r":?\s*(NAMA|NAME|NAMS|NANIA|NAMAT|NAM|AM)\s*:?", value, flags=re.IGNORECASE))


def _repair_birth_place_date_from_nik(value: str, nik: str | None) -> str | None:
    normalized = _normalize_birth_place_date(value)
    birth_date = _birth_date_from_nik(nik)
    if not normalized or not birth_date:
        return None
    if normalized.endswith(birth_date):
        return normalized

    match = re.fullmatch(r"(.+),\s*(\d{2}-\d{2}-\d{4})", normalized)
    if not match:
        return None
    actual_date = match.group(2)
    if actual_date[3:5] != birth_date[3:5]:
        return None
    actual_year = int(actual_date[-4:])
    expected_year = int(birth_date[-4:])
    if actual_date[:5] == birth_date[:5] and sum(1 for left, right in zip(actual_date[-4:], birth_date[-4:]) if left != right) <= 1:
        return f"{match.group(1)}, {birth_date}"
    if actual_date[:5] == birth_date[:5] and abs(actual_year - expected_year) <= 10:
        return f"{match.group(1)}, {birth_date}"
    return None


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
        or "BELUMKAWN" in variant
        or "BEIUMKAWN" in variant
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
