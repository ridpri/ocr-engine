from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


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


def prepare_stnk_fast_roi_image(input_path: str | Path, output_path: str | Path, max_side: int = 1200) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    with Image.open(input_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        width, height = image.size
        bottom_ratio = 0.88
        roi = image.crop((0, 0, width, max(1, int(height * bottom_ratio))))
        roi.save(output_path, format="JPEG", quality=92, optimize=True)
    return output_path
