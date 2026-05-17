from __future__ import annotations

import re
from dataclasses import dataclass

from ocr_engine.ocr.base import OcrToken
from ocr_engine.schemas import DocumentResult, FieldResult


@dataclass(frozen=True, slots=True)
class _PositionedToken:
    text: str
    confidence: float
    x_center: float
    y_center: float


def apply_ktp_layout_hints(parsed: DocumentResult, tokens: list[OcrToken]) -> DocumentResult:
    if parsed.document_type != "KTP":
        return parsed

    positioned = _positioned_tokens(tokens)
    if not positioned:
        return parsed

    _repair_marital_status(parsed, positioned)
    _repair_citizenship(parsed, positioned)
    _repair_region_fields(parsed, positioned)
    _repair_expiry(parsed, positioned)
    return parsed


def _repair_marital_status(parsed: DocumentResult, tokens: list[_PositionedToken]) -> None:
    field = parsed.fields.get("status_perkawinan")
    current = _normalize_marital_status(field.value if field else None)
    if current:
        parsed.fields["status_perkawinan"] = _ok(current, 0.88)
        return

    lines = _layout_lines(tokens)
    for line in lines:
        if not _line_intersects_y(line, 0.50, 0.82):
            continue
        text = " ".join(token.text for token in line)
        if not _looks_like_status_context(text):
            continue
        normalized = _normalize_marital_status(text)
        if normalized:
            parsed.fields["status_perkawinan"] = _ok(normalized, 0.78)
            return


def _repair_citizenship(parsed: DocumentResult, tokens: list[_PositionedToken]) -> None:
    field = parsed.fields.get("kewarganegaraan")
    current = _normalize_citizenship(field.value if field else None)
    if current:
        parsed.fields["kewarganegaraan"] = _ok(current, 0.88)
        return

    lines = _layout_lines(tokens)
    for line in lines:
        if not _line_intersects_y(line, 0.66, 1.0):
            continue
        text = " ".join(token.text for token in line)
        normalized = _normalize_citizenship_near_label(text)
        if normalized and (_looks_like_citizenship_context(text) or _has_nearby_citizenship_label(lines, line)):
            parsed.fields["kewarganegaraan"] = _ok(normalized, 0.78)
            return


def _repair_region_fields(parsed: DocumentResult, tokens: list[_PositionedToken]) -> None:
    lines = _layout_lines(tokens)
    repairs = {
        "kelurahan_desa": "kelurahan",
        "kecamatan": "kecamatan",
    }
    for field_name, kind in repairs.items():
        field = parsed.fields.get(field_name)
        current = _normalize_region_value(field.value if field else None)
        if current:
            parsed.fields[field_name] = _ok(current, 0.84)
            continue

        for line in lines:
            text = " ".join(token.text for token in line)
            normalized = _region_value_from_inline_label(text, kind) or _region_value_from_line_tokens(line, kind)
            if normalized:
                parsed.fields[field_name] = _ok(normalized, 0.78)
                break


def _repair_expiry(parsed: DocumentResult, tokens: list[_PositionedToken]) -> None:
    field = parsed.fields.get("berlaku_hingga")
    current = _normalize_expiry(field.value if field else None)
    if current:
        parsed.fields["berlaku_hingga"] = _ok(current, 0.84)
        return

    lines = _layout_lines(tokens)
    for index, line in enumerate(lines):
        text = " ".join(token.text for token in line)
        normalized = _expiry_from_inline_label(text) or _expiry_value_from_line_tokens(line)
        if normalized:
            parsed.fields["berlaku_hingga"] = _ok(normalized, 0.78)
            return

        if not _is_expiry_label(text):
            continue

        nearby = lines[index + 1 : min(len(lines), index + 4)]
        nearby.extend(reversed(lines[max(0, index - 2) : index]))
        for nearby_line in nearby:
            normalized = _normalize_expiry(" ".join(token.text for token in nearby_line))
            if normalized:
                parsed.fields["berlaku_hingga"] = _ok(normalized, 0.78)
                return


def _positioned_tokens(tokens: list[OcrToken]) -> list[_PositionedToken]:
    raw_positions: list[tuple[str, float, float, float]] = []
    for token in tokens:
        center = _bbox_center(token.bbox)
        if center is None:
            continue
        raw_positions.append((token.text, token.confidence, center[0], center[1]))

    if not raw_positions:
        return []

    xs = [position[2] for position in raw_positions]
    ys = [position[3] for position in raw_positions]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    return [
        _PositionedToken(text, confidence, (x - min_x) / width, (y - min_y) / height)
        for text, confidence, x, y in raw_positions
        if text
    ]


def _bbox_center(bbox: list | None) -> tuple[float, float] | None:
    if not bbox:
        return None

    if len(bbox) == 4 and all(isinstance(value, (int, float)) for value in bbox):
        left, top, right, bottom = [float(value) for value in bbox]
        return (left + right) / 2, (top + bottom) / 2

    points = []
    for point in bbox:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                points.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
    if not points:
        return None
    return sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points)


def _layout_lines(tokens: list[_PositionedToken]) -> list[list[_PositionedToken]]:
    lines: list[list[_PositionedToken]] = []
    for token in sorted(tokens, key=lambda item: (item.y_center, item.x_center)):
        for line in lines:
            if abs(_line_y(line) - token.y_center) <= 0.035:
                line.append(token)
                break
        else:
            lines.append([token])

    for line in lines:
        line.sort(key=lambda item: item.x_center)
    return lines


def _line_y(line: list[_PositionedToken]) -> float:
    return sum(token.y_center for token in line) / len(line)


def _line_intersects_y(line: list[_PositionedToken], lower: float, upper: float) -> bool:
    y = _line_y(line)
    return lower <= y <= upper


def _has_nearby_citizenship_label(lines: list[list[_PositionedToken]], target: list[_PositionedToken]) -> bool:
    target_y = _line_y(target)
    for line in lines:
        if abs(_line_y(line) - target_y) > 0.08:
            continue
        if _looks_like_citizenship_context(" ".join(token.text for token in line)):
            return True
    return False


def _looks_like_status_context(value: str) -> bool:
    compact = re.sub(r"[^A-Z]", "", value.upper())
    return "STATUS" in compact or "PERKAW" in compact or "KAWIN" in compact or "MENIKAH" in compact


def _looks_like_citizenship_context(value: str) -> bool:
    compact = re.sub(r"[^A-Z]", "", value.upper())
    return "KEWARG" in compact or "KEVRG" in compact


def _region_value_from_line_tokens(line: list[_PositionedToken], kind: str) -> str | None:
    label_fn = _is_kelurahan_label if kind == "kelurahan" else _is_kecamatan_label
    for index, token in enumerate(line):
        if not label_fn(token.text):
            continue
        right_text = " ".join(right.text for right in line[index + 1 :])
        normalized = _normalize_region_value(right_text)
        if normalized:
            return normalized
    return None


def _region_value_from_inline_label(line: str, kind: str) -> str | None:
    if kind == "kelurahan":
        label = r"(?:KEL\s*/?\s*DES[AN]?|KELDESA|KEWDESA|KEVDESA|KELDOSA|KELDESS|KALDESS|NOESA|EL\s*/?\s*DESA)"
    else:
        label = r"(?:KECAMATAN|KECAMNATAN|KECAMATAR|KECAMNATAN|KECANATAN|KECA\s*MATEN|XECAMATAN|ECAMATAN|ECOMATAN)"
    match = re.search(rf"{label}\s*[:_\-]?\s*(.+)$", line, flags=re.IGNORECASE)
    if not match:
        return None
    return _normalize_region_value(match.group(1))


def _normalize_region_value(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"^[\s:;\-]+", "", value).upper().strip(" :;._-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return None
    if _contains_non_region_label(cleaned):
        return None
    if _normalize_marital_status(cleaned) or _normalize_citizenship(cleaned) or _normalize_expiry(cleaned):
        return None
    if re.search(r"\d{1,2}[-/.\s]+\d{1,2}[-/.\s]+\d{2,4}", cleaned):
        return None
    if re.fullmatch(r"[:\s\d./\-]+", cleaned):
        return None
    if len(re.findall(r"[A-Z]", cleaned)) < 3:
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


def _expiry_value_from_line_tokens(line: list[_PositionedToken]) -> str | None:
    for index, token in enumerate(line):
        if not _is_expiry_label(token.text):
            continue
        right_text = " ".join(right.text for right in line[index + 1 :])
        normalized = _normalize_expiry(right_text)
        if normalized:
            return normalized
    return None


def _expiry_from_inline_label(line: str) -> str | None:
    label = r"(?:BERLAK\w*|BARLAK\w*|BERLAKY|BORLAK\w*|BERBAKY|BERFAKU|BERFAK\w*|BARTARAR|NAKU)"
    match = re.search(rf"{label}\s*:?\s*(?:HING\w*|HIN\w*|HNOG\w*)?\s*[:_\-]?\s*(.*)$", line, flags=re.IGNORECASE)
    if not match:
        return None
    return _normalize_expiry(match.group(1))


def _normalize_expiry(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"^[\s:;\-]+", "", value).upper().strip(" :;._-")
    if not cleaned:
        return None
    compact = re.sub(r"[^A-Z0-9]", "", cleaned).replace("1", "I").replace("0", "O")
    if compact.startswith("SEUMURHID") or compact in {"SEUMURHIOUP", "SEUMUHIDUP"}:
        return "SEUMUR HIDUP"

    match = re.search(r"(?<!\d)(\d{1,2})[-/.\s]+(\d{1,2})[-/.\s]+(\d{4})(?!\d)", cleaned)
    if not match:
        return None
    return f"{match.group(1).zfill(2)}-{match.group(2).zfill(2)}-{match.group(3)}"


def _is_expiry_label(value: str) -> bool:
    compact = re.sub(r"[^A-Z]", "", value.upper())
    has_prefix = any(prefix in compact for prefix in ["BERLAKU", "BARLAKU", "BERLAKY", "BORLAKU", "BERBAKY", "BERFAKU", "BARTARAR", "NAKU"])
    return has_prefix and ("HING" in compact or "HIN" in compact or "HNOG" in compact or compact.startswith(("BERLAKU", "BARLAKU", "BERLAKY")))


def _contains_non_region_label(value: str) -> bool:
    compact = re.sub(r"[^A-Z]", "", value.upper())
    exact_blocked = {
        "NIK",
        "NAMA",
        "TEMPAT",
        "TGL",
        "LAHIR",
        "ALAMAT",
        "AGAMA",
        "STATUS",
        "PEKERJAAN",
        "KEWARGANEGARAAN",
        "BERLAKU",
        "HINGGA",
    }
    if compact in exact_blocked:
        return True
    blocked = [
        "TEMPAT",
        "LAHIR",
        "PERKAW",
        "KEWARG",
        "BERLAKU",
        "JENISKELAMIN",
    ]
    return any(label in compact for label in blocked)


def _normalize_marital_status(value: str | None) -> str | None:
    if not value:
        return None
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


def _normalize_citizenship(value: str | None) -> str | None:
    if not value:
        return None
    compact = re.sub(r"[^A-Z0-9]", "", value.upper()).replace("1", "I").replace("L", "I")
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
    if compact in {"WN", "VKE"} or compact.endswith("WNI") or compact.endswith("WN") or "VNI" in compact:
        return "WNI"
    return None


def _ok(value: str, confidence: float) -> FieldResult:
    return FieldResult(value=value, confidence=confidence, status="ok", evidence=[value])
