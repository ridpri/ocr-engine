from __future__ import annotations

import re
from collections.abc import Iterable

from ocr_engine.schemas import FieldResult
from ocr_engine.validators import collapse_spaces

OCR_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "：": ":",
        "／": "/",
        "，": ",",
        "－": "-",
        "．": ".",
        "\u3000": " ",
    }
)


def normalized_lines(raw_text: str) -> list[str]:
    normalized_text = raw_text.translate(OCR_PUNCTUATION_TRANSLATION)
    return [
        collapse_spaces(line)
        for line in normalized_text.replace("\r", "\n").split("\n")
        if collapse_spaces(line)
    ]


def make_ok(value: str, confidence: float = 0.88, raw: str | None = None) -> FieldResult:
    clean = collapse_spaces(value)
    return FieldResult(value=clean, confidence=confidence, status="ok", evidence=[clean], raw=raw)


def make_missing() -> FieldResult:
    return FieldResult(value=None, confidence=0.0, status="missing")


def make_invalid(value: str, raw: str | None = None) -> FieldResult:
    clean = collapse_spaces(value)
    return FieldResult(value=clean or None, confidence=0.35, status="invalid", raw=raw)


def capture_after_label(raw_text: str, labels: Iterable[str], stop_labels: Iterable[str]) -> tuple[str | None, str | None]:
    lines = normalized_lines(raw_text)
    label_pattern = "|".join(re.escape(label) for label in labels)
    stop_pattern = "|".join(re.escape(label) for label in stop_labels)

    for index, line in enumerate(lines):
        match = re.search(rf"\b(?:{label_pattern})\b\s*[:\-]?\s*(.*)$", line, flags=re.IGNORECASE)
        if not match:
            continue

        value = _strip_value_prefix(match.group(1))
        if value:
            value = _trim_inline_stop(value, stop_pattern)
            if value:
                return collapse_spaces(value), line

        for next_index in range(index + 1, min(index + 3, len(lines))):
            next_line = lines[next_index]
            if not re.search(rf"\b(?:{stop_pattern})\b", next_line, flags=re.IGNORECASE):
                next_value = _strip_value_prefix(next_line)
                if next_value:
                    return collapse_spaces(next_value), line
            else:
                break

    return None, None


def _strip_value_prefix(value: str) -> str:
    return value.lstrip(" \t:-")


def _trim_inline_stop(value: str, stop_pattern: str) -> str:
    if not stop_pattern:
        return value
    match = re.search(rf"\s+\b(?:{stop_pattern})\b\s*[:\-]?", value, flags=re.IGNORECASE)
    if match:
        return value[: match.start()]
    return value
