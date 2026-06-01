from __future__ import annotations

import argparse
import json
import random
import os
import tempfile
import time
from pathlib import Path

from ocr_engine.eval_summary import summarize_records
from ocr_engine.image_utils import is_supported_input
from ocr_engine.ocr.base import OcrDependencyError
from ocr_engine.ocr.paddle_provider import PaddleOcrProvider
from ocr_engine.ocr.rapid_provider import RapidOcrProvider
from ocr_engine.parsers.stnk import stnk_structure_score
from ocr_engine.pdf_utils import render_pdf_first_page
from ocr_engine.pipeline import run_ocr_pipeline
from ocr_engine.postal_code import get_default_postal_code_index
from ocr_engine.stnk_usage import apply_stnk_web_usage_gate, classify_stnk_record
from ocr_engine.validators import mask_sensitive_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local OCR POC over image samples.")
    parser.add_argument("--input", required=True, help="Image file or folder containing images.")
    parser.add_argument("--limit", type=int, default=5, help="Maximum files to process from a folder. Use 0 for all files.")
    parser.add_argument("--document-type", default="AUTO", choices=["AUTO", "KTP", "STNK"])
    parser.add_argument("--mode", default="accurate", choices=["fast", "accurate"], help="OCR processing mode.")
    parser.add_argument(
        "--ocr-provider",
        default=os.getenv("OCR_PROVIDER", "paddle"),
        choices=["paddle", "rapid"],
        help="OCR backend to use for extraction.",
    )
    parser.add_argument("--recursive", action="store_true", help="Collect supported files from nested folders too.")
    parser.add_argument("--skip-pdf", action="store_true", help="Ignore PDF files when collecting a folder.")
    parser.add_argument("--random-seed", type=int, help="Shuffle files with a reproducible seed before applying --limit.")
    parser.add_argument(
        "--disable-nik-fallback",
        action="store_true",
        help="Skip image-based KTP NIK fallback so CLI timing matches the purchase KTP endpoint.",
    )
    parser.add_argument("--jsonl", help="Optional JSONL output path.")
    parser.add_argument("--summary-json", help="Optional summary JSON output path.")
    args = parser.parse_args()

    paths = _collect_paths(
        Path(args.input),
        args.limit,
        recursive=args.recursive,
        include_pdf=not args.skip_pdf,
        random_seed=args.random_seed,
    )
    provider = _create_provider(args.ocr_provider)
    _warm_postal_code_index()
    output_path = Path(args.jsonl) if args.jsonl else None

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    records: list[dict] = []
    has_errors = False
    for path in paths:
        try:
            record = _process_file(
                provider,
                path,
                args.document_type,
                mode=args.mode,
                run_nik_fallback=not args.disable_nik_fallback,
            )
        except OcrDependencyError as exc:
            has_errors = True
            record = {"file": path.name, "status": "setup_error", "document_type": args.document_type, "error": str(exc)}
        except Exception as exc:
            has_errors = True
            record = {"file": path.name, "status": "failed", "error": str(exc)}

        print(json.dumps(record, ensure_ascii=True))
        records.append(record)
        if output_path:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summarize_records(records), ensure_ascii=False, indent=2), encoding="utf-8")

    return 2 if has_errors else 0


def _collect_paths(
    input_path: Path,
    limit: int,
    *,
    recursive: bool = False,
    include_pdf: bool = True,
    random_seed: int | None = None,
) -> list[Path]:
    if input_path.is_file():
        if not include_pdf and input_path.suffix.lower() == ".pdf":
            return []
        return [input_path]
    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    paths = [
        path
        for path in sorted(iterator)
        if path.is_file() and is_supported_input(path) and (include_pdf or path.suffix.lower() != ".pdf")
    ]
    if random_seed is not None:
        random.Random(random_seed).shuffle(paths)
    if limit <= 0:
        return paths
    return paths[:limit]


def _create_provider(name: str):
    if name.strip().lower() == "rapid":
        return RapidOcrProvider()
    return PaddleOcrProvider()


def _warm_postal_code_index() -> None:
    try:
        get_default_postal_code_index()
    except Exception:
        return


def _process_file(
    provider: PaddleOcrProvider,
    path: Path,
    document_type: str,
    mode: str = "accurate",
    run_nik_fallback: bool = True,
) -> dict:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="ocr-engine-cli-") as tmpdir:
        source_path = path
        if path.suffix.lower() == ".pdf":
            source_path = render_pdf_first_page(path, Path(tmpdir) / "raw.png")
        result = run_ocr_pipeline(
            provider,
            source_path,
            document_type,
            Path(tmpdir),
            processing_mode=mode,
            run_nik_fallback=run_nik_fallback,
        )
        parsed = result.parsed
        assessment = result.assessment
        processing_time_ms = round((time.perf_counter() - started) * 1000, 2)
        record = {
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
        if document_type.upper() == "STNK" or parsed.document_type == "STNK":
            record["stnk_structure_score"] = stnk_structure_score(result.ocr_result.raw_text)
            usage_class, usage_reasons = classify_stnk_record(record)
            record["stnk_usage_class"] = usage_class
            record["stnk_usage_reasons"] = usage_reasons
            apply_stnk_web_usage_gate(record)
        return record


if __name__ == "__main__":
    raise SystemExit(main())
