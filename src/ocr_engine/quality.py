from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from ocr_engine.ocr.base import OcrResult


SCREEN_CAPTURE_MARKERS = [
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


def analyze_image_quality(
    image_path: str | Path,
    ocr_result: OcrResult,
    preflight_quality: dict | None = None,
) -> dict:
    quality = preflight_quality or analyze_image_preflight(image_path)
    width = quality["image"]["width"]
    height = quality["image"]["height"]
    blur_score = quality["metrics"]["blur_score"]

    token_count = len(ocr_result.tokens)
    megapixels = max((width * height) / 1_000_000, 0.01)
    text_density = token_count / megapixels
    flags: list[str] = list(quality["flags"])

    if token_count < 6:
        flags.append("low_text_density")
    if _looks_like_screen_or_desktop_capture(ocr_result.raw_text):
        flags.append("screen_or_desktop_capture")

    metrics = {
        "ocr_token_count": token_count,
        "text_density": round(text_density, 2),
        "blur_score": blur_score,
        "overall_score": _overall_quality_score(flags),
        "pre_ocr": False,
    }
    return {
        "image": quality["image"],
        "flags": flags,
        "metrics": metrics,
    }


def analyze_image_preflight(image_path: str | Path) -> dict:
    image_path = Path(image_path)
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        blur_score = _edge_score(image)
        stnk_tax_receipt_score = _stnk_tax_receipt_only_score(image)

    flags: list[str] = []

    if min(width, height) < 350 or width * height < 150_000:
        flags.append("document_too_small")
    if blur_score < 3.5:
        flags.append("blur_detected")
    if stnk_tax_receipt_score >= 0.82:
        flags.append("stnk_tax_receipt_only")

    metrics = {
        "ocr_token_count": 0,
        "text_density": 0.0,
        "blur_score": round(blur_score, 2),
        "stnk_tax_receipt_score": round(stnk_tax_receipt_score, 2),
        "overall_score": _overall_quality_score(flags),
        "pre_ocr": True,
    }
    return {
        "image": {"width": width, "height": height},
        "flags": flags,
        "metrics": metrics,
    }


def _edge_score(image: Image.Image) -> float:
    grayscale = ImageOps.grayscale(image)
    grayscale.thumbnail((256, 256), Image.Resampling.LANCZOS)
    width, height = grayscale.size
    pixels = grayscale.load()
    if width < 2 or height < 2:
        return 0.0

    total = 0
    count = 0
    for y in range(height - 1):
        for x in range(width - 1):
            dx = abs(int(pixels[x + 1, y]) - int(pixels[x, y]))
            dy = abs(int(pixels[x, y + 1]) - int(pixels[x, y]))
            total += dx + dy
            count += 1
    return total / max(count, 1)


def _stnk_tax_receipt_only_score(image: Image.Image) -> float:
    sample = image.copy()
    sample.thumbnail((300, 300), Image.Resampling.LANCZOS)
    width, height = sample.size
    if width <= height or width / max(height, 1) < 1.25:
        return 0.0

    pixels = sample.load()
    row_coverages: list[float] = []
    official_pixels = 0
    total_pixels = max(width * height, 1)
    for y in range(height):
        paper_pixels = 0
        for x in range(width):
            red, green, blue = pixels[x, y]
            avg = (red + green + blue) / 3
            saturation = max(red, green, blue) - min(red, green, blue)
            if green > 145 and blue > 125 and red < 225 and (green >= red or blue >= red) and saturation < 95:
                official_pixels += 1
            looks_like_document = (avg > 135 and saturation < 95) or (
                red > 120 and green > 105 and blue > 75 and saturation < 130
            )
            if looks_like_document:
                paper_pixels += 1
        row_coverages.append(paper_pixels / max(width, 1))

    active_rows = [idx for idx, coverage in enumerate(row_coverages) if coverage > 0.45]
    if not active_rows:
        return 0.0

    active_span = (max(active_rows) - min(active_rows) + 1) / max(height, 1)
    top_cut = max(1, height // 4)
    bottom_start = min(height - 1, (height * 3) // 4)
    middle_start = top_cut
    middle_end = max(middle_start + 1, bottom_start)
    top_coverage = sum(row_coverages[:top_cut]) / top_cut
    middle_coverage = sum(row_coverages[middle_start:middle_end]) / (middle_end - middle_start)
    bottom_coverage = sum(row_coverages[bottom_start:]) / max(height - bottom_start, 1)
    official_coverage = official_pixels / total_pixels
    if official_coverage >= 0.13:
        return 0.0

    score = 0.0
    if 0.30 <= active_span <= 0.72:
        score += 0.35
    if top_coverage < 0.25:
        score += 0.20
    if bottom_coverage < 0.35:
        score += 0.20
    if middle_coverage > 0.45:
        score += 0.20
    if middle_coverage > top_coverage * 1.7 and middle_coverage > bottom_coverage * 1.35:
        score += 0.05
    return min(score, 1.0)


def _looks_like_screen_or_desktop_capture(raw_text: str) -> bool:
    upper = raw_text.upper()
    return sum(1 for marker in SCREEN_CAPTURE_MARKERS if marker in upper) >= 2


def _overall_quality_score(flags: list[str]) -> float:
    penalties = {
        "document_too_small": 0.2,
        "blur_detected": 0.2,
        "low_text_density": 0.15,
        "screen_or_desktop_capture": 0.35,
    }
    score = 1.0 - sum(penalties.get(flag, 0.1) for flag in flags)
    return round(max(score, 0.0), 2)
