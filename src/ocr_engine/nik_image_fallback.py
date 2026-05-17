from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageOps

from ocr_engine.ocr.base import OcrResult
from ocr_engine.schemas import DocumentResult, FieldResult
from ocr_engine.validators import normalize_nik


class ImageOcrProvider(Protocol):
    def extract_text(self, image_path: str) -> OcrResult:
        """Extract OCR text from an image path."""


@dataclass(slots=True)
class _NikCandidate:
    value: str
    confidence: float
    source: str
    near_label: bool
    birth_match: bool

    @property
    def score(self) -> float:
        score = 0.55 + min(max(self.confidence, 0.0), 1.0) * 0.2
        if self.near_label:
            score += 0.18
        if self.birth_match:
            score += 0.18
        return min(score, 0.92)


def repair_ktp_nik_from_image(
    provider: ImageOcrProvider,
    image_path: str | Path,
    parsed: DocumentResult,
    work_dir: str | Path,
) -> dict:
    metadata = {"attempted": False, "passes": 0, "value": None}
    nik_field = parsed.fields.get("nik")
    if parsed.document_type != "KTP" or nik_field is None or nik_field.status == "ok":
        return metadata

    metadata["attempted"] = True
    best: _NikCandidate | None = None
    for variant_path in create_ktp_nik_variant_images(image_path, work_dir):
        ocr_result = provider.extract_text(str(variant_path))
        metadata["passes"] += 1
        for candidate in _extract_nik_candidates(ocr_result, parsed, variant_path.name):
            if not _is_acceptable_candidate(candidate):
                continue
            if best is None or candidate.score > best.score:
                best = candidate
        if best and best.near_label and best.birth_match:
            break

    if best is None:
        return metadata

    parsed.fields["nik"] = FieldResult(
        value=best.value,
        confidence=round(best.score, 2),
        status="ok",
        evidence=[best.value, f"image_fallback:{best.source}"],
        raw="image_nik_fallback",
    )
    parsed.warnings = [
        warning
        for warning in parsed.warnings
        if warning not in {"invalid:nik", "missing_required:nik"}
    ]
    metadata["value"] = best.value
    return metadata


def create_ktp_nik_variant_images(image_path: str | Path, work_dir: str | Path) -> list[Path]:
    image_path = Path(image_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as opened:
        base = ImageOps.exif_transpose(opened).convert("RGB")

    variants: list[Path] = []
    for orientation, image in _oriented_images(base):
        width, height = image.size
        boxes = _nik_crop_boxes(width, height, orientation)
        for crop_name, box in boxes:
            crop = image.crop(box)
            output_path = work_dir / f"nik_{orientation}_{crop_name}.jpg"
            crop.save(output_path, format="JPEG", quality=95)
            variants.append(output_path)
    return variants


def _oriented_images(base: Image.Image) -> list[tuple[str, Image.Image]]:
    return [
        ("rot90", base.rotate(90, expand=True)),
        ("rot270", base.rotate(270, expand=True)),
        ("orig", base),
    ]


def _nik_crop_boxes(width: int, height: int, orientation: str) -> list[tuple[str, tuple[int, int, int, int]]]:
    common = [
        ("top45", (0, 0, width, int(height * 0.45))),
        ("mid45", (0, int(height * 0.10), width, int(height * 0.55))),
        ("left70", (0, 0, int(width * 0.70), height)),
        ("right70", (int(width * 0.30), 0, width, height)),
    ]
    if orientation == "rot270":
        return [
            ("right70", (int(width * 0.30), 0, width, height)),
            ("top45", (0, 0, width, int(height * 0.45))),
            ("mid45", (0, int(height * 0.10), width, int(height * 0.55))),
            ("left70", (0, 0, int(width * 0.70), height)),
        ]
    return common


def _extract_nik_candidates(ocr_result: OcrResult, parsed: DocumentResult, source: str) -> list[_NikCandidate]:
    candidates: list[_NikCandidate] = []
    token_texts = [token.text for token in ocr_result.tokens]
    for index, token in enumerate(ocr_result.tokens):
        normalized = normalize_nik(token.text)
        if not normalized:
            continue
        window = token_texts[max(0, index - 3) : index + 4]
        candidates.append(
            _NikCandidate(
                value=normalized,
                confidence=token.confidence,
                source=source,
                near_label=_contains_nik_label(" ".join(window)),
                birth_match=_nik_birth_segment_matches(normalized, parsed),
            )
        )

    for index, line in enumerate(_non_empty_lines(ocr_result.raw_text)):
        for match in re.finditer(r"(?<!\d)(\d[\d\s.:\-/]{14,24}\d)(?!\d)", line):
            normalized = normalize_nik(match.group(1))
            if not normalized:
                continue
            window = " ".join(_non_empty_lines(ocr_result.raw_text)[max(0, index - 2) : index + 3])
            candidates.append(
                _NikCandidate(
                    value=normalized,
                    confidence=0.78,
                    source=source,
                    near_label=_contains_nik_label(window),
                    birth_match=_nik_birth_segment_matches(normalized, parsed),
                )
            )
    return candidates


def _is_acceptable_candidate(candidate: _NikCandidate) -> bool:
    return bool(candidate.near_label or candidate.birth_match)


def _contains_nik_label(value: str) -> bool:
    compact = re.sub(r"[^A-Z0-9]", "", value.upper())
    return any(label in compact for label in ["NIK", "NTK", "HIK", "N1K", "NIKK"])


def _nik_birth_segment_matches(nik: str, parsed: DocumentResult) -> bool:
    gender = (parsed.fields.get("jenis_kelamin").value or "").upper() if parsed.fields.get("jenis_kelamin") else ""
    text = "\n".join(
        value
        for value in [
            parsed.fields.get("tempat_tanggal_lahir").value if parsed.fields.get("tempat_tanggal_lahir") else None,
            parsed.raw_text,
        ]
        if value
    )
    for day, month, year in _date_parts(text):
        possible_days = {day}
        if gender == "PEREMPUAN":
            possible_days = {day + 40}
        elif gender != "LAKI-LAKI":
            possible_days.add(day + 40)
        for candidate_day in possible_days:
            segment = f"{candidate_day:02d}{month:02d}{year % 100:02d}"
            if nik[6:12] == segment:
                return True
    return False


def _date_parts(text: str) -> list[tuple[int, int, int]]:
    dates: list[tuple[int, int, int]] = []
    for match in re.finditer(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b", text):
        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))
        if 1 <= day <= 31 and 1 <= month <= 12:
            dates.append((day, month, year))
    return dates


def _non_empty_lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]
