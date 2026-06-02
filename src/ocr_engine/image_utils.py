from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_INPUT_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | {".pdf"}


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def is_supported_input(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES


def prepare_image(input_path: str | Path, output_path: str | Path, max_side: int = 2400) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    with Image.open(input_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        image.save(output_path, format="JPEG", quality=92, optimize=True)
    return output_path


def prepare_ktp_fast_image(
    input_path: str | Path,
    output_path: str | Path,
    max_side: int = 560,
    right_ratio: float = 0.8,
    bottom_ratio: float = 1.0,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    with Image.open(input_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        right = max(1, min(width, int(width * right_ratio)))
        bottom = max(1, min(height, int(height * bottom_ratio)))
        image = image.crop((0, 0, right, bottom))
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        image.save(output_path, format="JPEG", quality=92, optimize=True)
    return output_path


def prepare_stnk_full_page_image(
    input_path: str | Path,
    output_path: str | Path,
    max_side: int = 1600,
    min_long_side: int = 1600,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    with Image.open(input_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        long_side = max(width, height)
        if long_side < min_long_side:
            scale = min_long_side / max(long_side, 1)
            image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        image = ImageEnhance.Contrast(image).enhance(1.15)
        image = ImageEnhance.Sharpness(image).enhance(1.2)
        image = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=90, threshold=3))
        image.save(output_path, format="JPEG", quality=94, optimize=True)
    return output_path


def prepare_stnk_fast_roi_image(
    input_path: str | Path,
    output_path: str | Path,
    max_side: int = 1200,
    right_ratio: float = 1.0,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    with Image.open(input_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        width, height = image.size
        bottom_ratio = 0.88
        right = max(1, min(width, int(width * right_ratio)))
        roi = image.crop((0, 0, right, max(1, int(height * bottom_ratio))))
        roi.save(output_path, format="JPEG", quality=92, optimize=True)
    return output_path


def prepare_stnk_official_roi_image(
    input_path: str | Path,
    output_path: str | Path,
    max_side: int = 1200,
    top_ratio: float = 0.55,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    with Image.open(input_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        width, height = image.size
        if height > width * 1.25:
            roi = image.crop((int(width * 0.48), 0, width, height)).rotate(90, expand=True)
        elif height <= width * 0.65:
            roi = image
        else:
            top = min(height - 1, max(0, int(height * top_ratio)))
            roi = image.crop((0, top, width, height))
        roi.save(output_path, format="JPEG", quality=92, optimize=True)
    return output_path
