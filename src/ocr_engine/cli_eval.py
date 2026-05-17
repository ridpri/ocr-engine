from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from ocr_engine.eval_summary import summarize_records
from ocr_engine.image_utils import is_supported_input
from ocr_engine.ocr.base import OcrDependencyError
from ocr_engine.ocr.paddle_provider import PaddleOcrProvider
from ocr_engine.pdf_utils import render_pdf_first_page
from ocr_engine.pipeline import run_ocr_pipeline
from ocr_engine.validators import mask_sensitive_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local OCR POC over image samples.")
    parser.add_argument("--input", required=True, help="Image file or folder containing images.")
    parser.add_argument("--limit", type=int, default=5, help="Maximum files to process from a folder.")
    parser.add_argument("--document-type", default="AUTO", choices=["AUTO", "KTP", "STNK"])
    parser.add_argument("--mode", default="accurate", choices=["fast", "accurate"], help="OCR processing mode.")
    parser.add_argument("--jsonl", help="Optional JSONL output path.")
    parser.add_argument("--summary-json", help="Optional summary JSON output path.")
    args = parser.parse_args()

    paths = _collect_paths(Path(args.input), args.limit)
    provider = PaddleOcrProvider()
    output_path = Path(args.jsonl) if args.jsonl else None

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    records: list[dict] = []
    for path in paths:
        try:
            record = _process_file(provider, path, args.document_type, mode=args.mode)
        except OcrDependencyError as exc:
            print(f"SETUP_ERROR {path.name}: {exc}")
            return 2
        except Exception as exc:
            record = {"file": path.name, "status": "failed", "error": str(exc)}

        print(json.dumps(record, ensure_ascii=False))
        records.append(record)
        if output_path:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summarize_records(records), ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


def _collect_paths(input_path: Path, limit: int) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return [
        path
        for path in sorted(input_path.iterdir())
        if path.is_file() and is_supported_input(path)
    ][:limit]


def _process_file(provider: PaddleOcrProvider, path: Path, document_type: str, mode: str = "accurate") -> dict:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="ocr-engine-cli-") as tmpdir:
        source_path = path
        if path.suffix.lower() == ".pdf":
            source_path = render_pdf_first_page(path, Path(tmpdir) / "raw.png")
        result = run_ocr_pipeline(provider, source_path, document_type, Path(tmpdir), processing_mode=mode)
        parsed = result.parsed
        assessment = result.assessment
        processing_time_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "file": path.name,
            "status": "ok",
            "document_type": parsed.document_type,
            "needs_review": parsed.needs_review or assessment["decision"] != "approved_for_auto",
            "warnings": parsed.warnings,
            "input_assessment": assessment,
            "quality": result.quality,
            "processing_time_ms": processing_time_ms,
            "fields": {key: field.to_dict() for key, field in parsed.fields.items()},
            "ocr": {
                "provider": result.ocr_result.provider,
                "token_count": len(result.ocr_result.tokens),
                "processing_mode": result.processing_mode,
                "raw_text_masked": mask_sensitive_text(result.ocr_result.raw_text),
                "nik_fallback": result.nik_fallback,
                "preprocess": result.preprocess,
                "timings": result.timings,
            },
        }


if __name__ == "__main__":
    raise SystemExit(main())
