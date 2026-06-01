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

    flags: list[str] = []

    if min(width, height) < 350 or width * height < 150_000:
        flags.append("document_too_small")
    if blur_score < 3.5:
        flags.append("blur_detected")

    metrics = {
        "ocr_token_count": 0,
        "text_density": 0.0,
        "blur_score": round(blur_score, 2),
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
