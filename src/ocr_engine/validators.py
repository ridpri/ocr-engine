from __future__ import annotations

import re


def collapse_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" :;\t\r\n")


def normalize_nik(value: str | None) -> str | None:
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    return digits if len(digits) == 16 else None


def validate_plate_number(value: str | None) -> bool:
    if not value:
        return False
    normalized = re.sub(r"[\s.\-]", "", value.upper())
    match = re.fullmatch(r"[A-Z]{1,2}(\d{1,4})[A-Z]{0,3}", normalized)
    return bool(match and int(match.group(1)) > 0)


def normalize_plate_number(value: str | None) -> str | None:
    if not value:
        return None
    compact = re.sub(r"[\s.\-]", "", value.upper())
    match = re.fullmatch(r"([A-Z]{1,2})(\d{1,4})([A-Z]{0,3})", compact)
    if not match or int(match.group(2)) <= 0:
        return None
    parts = [match.group(1), match.group(2)]
    if match.group(3):
        parts.append(match.group(3))
    return " ".join(parts)


def mask_sensitive_text(value: str) -> str:
    return re.sub(r"\b(\d{6})\d{6}(\d{4})\b", r"\1******\2", value)
