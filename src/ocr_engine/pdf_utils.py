from __future__ import annotations

from pathlib import Path


def render_pdf_first_page(input_path: str | Path, output_path: str | Path, dpi: int = 200) -> Path:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PDF support requires PyMuPDF. Install project dependencies again.") from exc

    input_path = Path(input_path)
    output_path = Path(output_path)
    with fitz.open(input_path) as document:
        if document.page_count < 1:
            raise ValueError("PDF does not contain any pages.")
        page = document.load_page(0)
        scale = dpi / 72
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pixmap.save(output_path)
    return output_path
