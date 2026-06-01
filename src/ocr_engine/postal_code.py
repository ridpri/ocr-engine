from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import csv
import os
import re
import unicodedata
import zipfile
import xml.etree.ElementTree as ET

from ocr_engine.schemas import FieldResult


DEFAULT_DB_DIR = Path(__file__).resolve().parents[3] / "DB Kode Wilayah"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "data" / "postal_code_index.tsv"
XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
OCR_ALPHA_DIGIT_MAP = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B"})


@dataclass(frozen=True, slots=True)
class PostalCodeMatch:
    kode_pos: str
    confidence: float
    evidence: list[str]
    kelurahan: str | None = None
    kecamatan: str | None = None
    kode_kecamatan: str | None = None
    kode_kota: str | None = None
    nama_kota: str | None = None
    kode_provinsi: str | None = None
    nama_provinsi: str | None = None
    alamat_lengkap: str | None = None
    total_options: int = 1
    match_status: str = "exact_match"


@dataclass(frozen=True, slots=True)
class _City:
    code: str
    name: str
    province_code: str


@dataclass(frozen=True, slots=True)
class _PostalEntry:
    kode_pos: str
    address: str
    locality: str
    sifat_pos: str
    city_code: str
    city_name: str
    province_code: str
    province_name: str
    district_code: str
    district_name: str
    normalized_address: str
    normalized_locality: str
    normalized_city: str
    normalized_province: str


class PostalCodeIndex:
    def __init__(self, entries: list[_PostalEntry]):
        self.entries = entries
        self._entries_by_locality: dict[str, list[_PostalEntry]] = {}
        for entry in entries:
            if not entry.normalized_locality:
                continue
            self._entries_by_locality.setdefault(entry.normalized_locality, []).append(entry)

    @classmethod
    def from_excel_dir(cls, db_dir: Path) -> "PostalCodeIndex":
        provinces = {
            _clean_numeric_id(row.get("KODE_PROVINSI")): str(row.get("PROVINSI") or "")
            for row in _xlsx_dict_rows(db_dir / "M_PROVINSI.xlsx")
        }
        cities = {
            _clean_numeric_id(row.get("KODE_KOTA")): _City(
                code=_clean_numeric_id(row.get("KODE_KOTA")),
                name=str(row.get("NAMA_KOTA") or ""),
                province_code=_clean_numeric_id(row.get("KODE_PROVINSI")),
            )
            for row in _xlsx_dict_rows(db_dir / "M_KOTA.xlsx")
        }
        districts = {
            _clean_numeric_id(row.get("KODE_KECAMATAN")): str(row.get("NAMA_KECAMATAN") or "")
            for row in _xlsx_dict_rows(db_dir / "M_KECAMATAN.xlsx")
        }
        entries: list[_PostalEntry] = []
        for row in _xlsx_dict_rows(db_dir / "M_KODEPOS.xlsx"):
            kode_pos = str(row.get("KODE_POS") or "").strip()
            if not re.fullmatch(r"\d{5}", kode_pos):
                continue
            city_code = _clean_numeric_id(row.get("KODE_KOTA"))
            city = cities.get(city_code)
            if not city:
                continue
            address = str(row.get("ALAMAT") or "")
            district_code = _clean_numeric_id(row.get("KODE_KECAMATAN"))
            if district_code == "0":
                district_code = ""
            province_name = provinces.get(city.province_code, "")
            locality = _extract_locality(address, city.name, kode_pos)
            normalized_address = _normalize_text(address)
            entries.append(
                _PostalEntry(
                    kode_pos=kode_pos,
                    address=address,
                    locality=locality,
                    sifat_pos=str(row.get("SIFAT_POS") or ""),
                    city_code=city_code,
                    city_name=city.name,
                    province_code=city.province_code,
                    province_name=province_name,
                    district_code=district_code,
                    district_name=districts.get(district_code, ""),
                    normalized_address=normalized_address,
                    normalized_locality=_normalize_text(locality),
                    normalized_city=_normalize_text(city.name),
                    normalized_province=_normalize_text(province_name),
                )
            )
        return cls(entries)

    @classmethod
    def from_tsv(cls, path: Path) -> "PostalCodeIndex":
        entries: list[_PostalEntry] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                address = row.get("address") or ""
                locality = row.get("locality") or ""
                city_name = row.get("city_name") or ""
                province_code = row.get("province_code") or ""
                province_name = row.get("province_name") or ""
                entries.append(
                    _PostalEntry(
                        kode_pos=row.get("kode_pos") or "",
                        address=address,
                        locality=locality,
                        sifat_pos=row.get("sifat_pos") or "",
                        city_code=row.get("city_code") or "",
                        city_name=city_name,
                        province_code=province_code,
                        province_name=province_name,
                        district_code=row.get("district_code") or "",
                        district_name=row.get("district_name") or "",
                        normalized_address=_normalize_text(address),
                        normalized_locality=_normalize_text(locality),
                        normalized_city=_normalize_text(city_name),
                        normalized_province=_normalize_text(province_name),
                    )
                )
        return cls(entries)

    def write_tsv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "kode_pos",
                    "address",
                    "locality",
                    "sifat_pos",
                    "city_code",
                    "city_name",
                    "province_code",
                    "province_name",
                    "district_code",
                    "district_name",
                ],
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for entry in self.entries:
                if not _is_kelurahan_entry(entry):
                    continue
                writer.writerow(
                    {
                        "kode_pos": entry.kode_pos,
                        "address": entry.address,
                        "locality": entry.locality,
                        "sifat_pos": entry.sifat_pos,
                        "city_code": entry.city_code,
                        "city_name": entry.city_name,
                        "province_code": entry.province_code,
                        "province_name": entry.province_name,
                        "district_code": entry.district_code,
                        "district_name": entry.district_name,
                    }
                )

    @classmethod
    def from_records(cls, records: list[dict]) -> "PostalCodeIndex":
        entries = [
            _PostalEntry(
                kode_pos=str(record["kode_pos"]),
                address=str(record.get("address") or ""),
                locality=str(record.get("locality") or ""),
                sifat_pos=str(record.get("sifat_pos") or "Kel."),
                city_code=str(record.get("city_code") or ""),
                city_name=str(record.get("city_name") or ""),
                province_code=str(record.get("province_code") or ""),
                province_name=str(record.get("province_name") or ""),
                district_code=str(record.get("district_code") or ""),
                district_name=str(record.get("district_name") or ""),
                normalized_address=_normalize_text(record.get("address") or ""),
                normalized_locality=_normalize_text(record.get("locality") or ""),
                normalized_city=_normalize_text(record.get("city_name") or ""),
                normalized_province=_normalize_text(record.get("province_name") or ""),
            )
            for record in records
        ]
        return cls(entries)

    def lookup(self, fields: dict[str, FieldResult]) -> PostalCodeMatch | None:
        query = _query_from_fields(fields)
        if not query["kelurahan"] and not query["alamat"]:
            return None

        scored: list[tuple[float, _PostalEntry, list[str]]] = []
        candidates = self._candidate_entries(query)
        for entry in candidates:
            score, evidence = _score_entry(entry, query)
            if score >= 8:
                scored.append((score, entry, evidence))
        if not scored:
            return None

        scored.sort(key=lambda item: (item[0], _is_kelurahan_entry(item[1]), -len(item[1].address)), reverse=True)
        best_score, best_entry, evidence = scored[0]
        confidence = min(0.95, 0.55 + best_score / 20)
        total_options = sum(1 for _, entry, _ in scored if entry.kode_pos == best_entry.kode_pos)
        match_status = "exact_match" if confidence >= 0.9 else "partial_match"
        return PostalCodeMatch(
            kode_pos=best_entry.kode_pos,
            confidence=round(confidence, 2),
            evidence=evidence,
            kelurahan=best_entry.locality or None,
            kecamatan=best_entry.district_name or _district_from_address(best_entry.address) or None,
            kode_kecamatan=best_entry.district_code or None,
            kode_kota=best_entry.city_code or None,
            nama_kota=best_entry.city_name or None,
            kode_provinsi=best_entry.province_code or None,
            nama_provinsi=best_entry.province_name or None,
            alamat_lengkap=_format_full_address(best_entry),
            total_options=total_options,
            match_status=match_status,
        )

    def _candidate_entries(self, query: dict[str, str]) -> list[_PostalEntry]:
        if not query["kelurahan"]:
            return self.entries
        compact_query = _compact_text(query["kelurahan"])
        candidates: list[_PostalEntry] = []
        seen: set[tuple[str, str, str]] = set()
        for locality, entries in self._entries_by_locality.items():
            if not locality:
                continue
            if not (
                query["kelurahan"] == locality
                or query["kelurahan"] in locality
                or locality in query["kelurahan"]
                or compact_query == _compact_text(locality)
            ):
                continue
            for entry in entries:
                key = (entry.kode_pos, entry.address, entry.locality)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(entry)
        return candidates


def lookup_postal_code(fields: dict[str, FieldResult]) -> PostalCodeMatch | None:
    index = get_default_postal_code_index()
    if not index:
        return None
    return index.lookup(fields)


@lru_cache(maxsize=1)
def get_default_postal_code_index() -> PostalCodeIndex | None:
    cache_path = Path(os.getenv("OCR_POSTAL_CODE_INDEX_PATH", str(DEFAULT_CACHE_PATH)))
    if cache_path.exists():
        return PostalCodeIndex.from_tsv(cache_path)

    db_dir = Path(os.getenv("OCR_REGION_DB_DIR", str(DEFAULT_DB_DIR)))
    if not db_dir.exists():
        return None
    required = ["M_PROVINSI.xlsx", "M_KOTA.xlsx", "M_KODEPOS.xlsx"]
    if not all((db_dir / filename).exists() for filename in required):
        return None
    return PostalCodeIndex.from_excel_dir(db_dir)


def _query_from_fields(fields: dict[str, FieldResult]) -> dict[str, str]:
    return {
        "provinsi": _normalize_text(_field_value(fields, "provinsi")),
        "kota": _normalize_text(_field_value(fields, "kabupaten_kota")),
        "kecamatan": _normalize_text(_field_value(fields, "kecamatan")),
        "kelurahan": _normalize_text(_field_value(fields, "kelurahan_desa")),
        "alamat": _normalize_text(_field_value(fields, "alamat")),
    }


def _field_value(fields: dict[str, FieldResult], field_name: str) -> str:
    field = fields.get(field_name)
    if not field or field.status != "ok" or not field.value:
        return ""
    return str(field.value)


def _score_entry(entry: _PostalEntry, query: dict[str, str]) -> tuple[float, list[str]]:
    score = 0.0
    evidence: list[str] = []
    if query["provinsi"]:
        if not _same_normalized_text(query["provinsi"], entry.normalized_province):
            return 0.0, []
        score += 2
        evidence.append(f"provinsi:{entry.province_name}")

    if query["kota"]:
        if _same_normalized_text(query["kota"], entry.normalized_city):
            score += 3
        elif query["kota"] in entry.normalized_city or entry.normalized_city in query["kota"]:
            score += 2
        else:
            return 0.0, []
        evidence.append(f"kabupaten_kota:{entry.city_name}")

    if query["kelurahan"]:
        if _same_normalized_text(query["kelurahan"], entry.normalized_locality):
            score += 6
        elif _contains_phrase(entry.normalized_locality, query["kelurahan"]):
            score += 4
        elif entry.normalized_locality and _contains_phrase(query["kelurahan"], entry.normalized_locality):
            score += 4
        elif _contains_phrase(entry.normalized_address, query["kelurahan"]):
            score += 3
        else:
            return 0.0, []
        evidence.append(f"kelurahan_desa:{entry.locality or entry.address}")

    if query["kecamatan"] and _contains_phrase(entry.normalized_address, query["kecamatan"]):
        score += 2
        evidence.append(f"kecamatan:{query['kecamatan']}")

    if query["alamat"]:
        address_hits = _shared_address_terms(query["alamat"], entry.normalized_address)
        if address_hits:
            score += min(2, len(address_hits) * 0.5)
            evidence.append(f"alamat:{' '.join(address_hits[:4])}")

    if _is_kelurahan_entry(entry):
        score += 1
    return score, evidence


def _contains_phrase(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    if re.search(rf"(^| ){re.escape(needle)}($| )", haystack):
        return True
    compact_haystack = _compact_text(haystack)
    compact_needle = _compact_text(needle)
    return bool(compact_haystack and compact_needle and compact_needle in compact_haystack)


def _same_normalized_text(left: str, right: str) -> bool:
    if left == right:
        return True
    if not left or not right:
        return False
    left_compact = _compact_text(left)
    right_compact = _compact_text(right)
    if left_compact == right_compact:
        return True
    left_ocr = _ocr_compact_text(left)
    right_ocr = _ocr_compact_text(right)
    if left_ocr == right_ocr:
        return True
    shorter, longer = sorted((left_ocr, right_ocr), key=len)
    if len(shorter) >= 5 and longer.startswith(shorter) and len(longer) - len(shorter) <= 1:
        return True
    return _edit_distance_at_most(left_ocr, right_ocr, max_distance=1)


def _compact_text(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value or "")


def _ocr_compact_text(value: str) -> str:
    return _compact_text(value).translate(OCR_ALPHA_DIGIT_MAP)


def _edit_distance_at_most(left: str, right: str, max_distance: int) -> bool:
    if not left or not right:
        return False
    if abs(len(left) - len(right)) > max_distance:
        return False
    if left == right:
        return True
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        row_min = current[0]
        for right_index, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current_value = min(
                previous[right_index] + 1,
                current[right_index - 1] + 1,
                previous[right_index - 1] + cost,
            )
            current.append(current_value)
            row_min = min(row_min, current_value)
        if row_min > max_distance:
            return False
        previous = current
    return previous[-1] <= max_distance


def _shared_address_terms(query_address: str, entry_address: str) -> list[str]:
    ignored = {"JL", "JLN", "JALAN", "NO", "RT", "RW", "BLOK", "KP", "KAMPUNG"}
    terms = [term for term in query_address.split() if len(term) >= 3 and term not in ignored]
    return [term for term in terms if _contains_phrase(entry_address, term)]


def _is_kelurahan_entry(entry: _PostalEntry) -> int:
    return 1 if entry.sifat_pos.strip().lower().startswith("kel") else 0


def _extract_locality(address: str, city_name: str, kode_pos: str) -> str:
    value = re.sub(rf"\b{re.escape(kode_pos)}\b", "", address or "", flags=re.IGNORECASE)
    value = re.sub(rf"\b{re.escape(city_name)}\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\bJAKARTA\b", "", value, flags=re.IGNORECASE)
    value = value.split(",")[0]
    value = re.sub(r"\s+", " ", value).strip(" ,-")
    return value


def _district_from_address(address: str) -> str | None:
    parts = [part.strip() for part in (address or "").split(",")]
    if len(parts) < 2:
        return None
    candidate = parts[1]
    candidate = re.sub(r"^(?:KECAMATAN|KEC\.?)\s+", "", candidate, flags=re.IGNORECASE).strip()
    return candidate or None


def _format_full_address(entry: _PostalEntry) -> str:
    if entry.address:
        return entry.address
    parts = [
        entry.locality,
        entry.district_name,
        entry.city_name,
        entry.province_name,
    ]
    return ", ".join(part for part in parts if part)


def _xlsx_dict_rows(path: Path) -> list[dict[str, str | None]]:
    rows = _xlsx_rows(path)
    if not rows:
        return []
    headers = [str(value or "").strip() for value in rows[0]]
    return [
        {headers[index]: row[index] if index < len(row) else None for index in range(len(headers)) if headers[index]}
        for row in rows[1:]
    ]


def _xlsx_rows(path: Path) -> list[list[str | None]]:
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str | None]] = []
        for row in root.findall(".//a:sheetData/a:row", XLSX_NS):
            values: list[str | None] = []
            for cell in row.findall("a:c", XLSX_NS):
                index = _column_index(cell.attrib.get("r", "A1"))
                while len(values) <= index:
                    values.append(None)
                values[index] = _cell_value(cell)
            rows.append(values)
    return rows


def _cell_value(cell: ET.Element) -> str | None:
    if cell.attrib.get("t") == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//a:t", XLSX_NS))
    value = cell.find("a:v", XLSX_NS)
    return value.text if value is not None else None


def _column_index(reference: str) -> int:
    match = re.match(r"[A-Z]+", reference)
    if not match:
        return 0
    index = 0
    for char in match.group(0):
        index = index * 26 + ord(char) - 64
    return index - 1


def _clean_numeric_id(value: str | None) -> str:
    if value in {None, ""}:
        return ""
    try:
        return str(int(float(str(value))))
    except ValueError:
        return str(value).strip()


def _normalize_text(value: str | None) -> str:
    ascii_text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.upper()
    ascii_text = re.sub(r"\bDAERAH\s+ISTIMEWA\s+YOGYAKARTA\b", "DI YOGYAKARTA", ascii_text)
    ascii_text = re.sub(r"\bKEPULAUAN\s+RIAU\b", "KEP RIAU", ascii_text)
    ascii_text = re.sub(r"\b(?:KABUPATEN|KAB|KOTA|KELURAHAN|KEL|DESA|KECAMATAN|KEC)\b", " ", ascii_text)
    ascii_text = re.sub(r"[^A-Z0-9]+", " ", ascii_text)
    return re.sub(r"\s+", " ", ascii_text).strip()
