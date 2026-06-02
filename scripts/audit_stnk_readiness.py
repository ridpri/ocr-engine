from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_engine.parsers.stnk import STNK_REQUIRED_FIELDS  # noqa: E402
from ocr_engine.stnk_usage import STNK_WEB_REQUIRED_FIELDS  # noqa: E402


SUMMARY_BASENAME = "readiness_summary.json"
CSV_BASENAME = "field_review.csv"

STNK_FIELD_ORDER = [
    "nomor_polisi",
    "nama_pemilik",
    "alamat",
    "merek",
    "tipe",
    "jenis",
    "tahun_pembuatan",
    "warna",
    "bahan_bakar",
    "nomor_rangka",
    "nomor_mesin",
    "berlaku_sampai",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build STNK production-readiness reports from benchmark records.jsonl.",
    )
    parser.add_argument("--records", required=True, help="Path to benchmark records.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory for readiness JSON and CSV output.")
    parser.add_argument(
        "--slow-threshold-ms",
        type=float,
        default=18_000.0,
        help="Processing time threshold used to flag slow records.",
    )
    return parser.parse_args(argv)


def load_records(path: str | Path) -> list[dict[str, Any]]:
    records_path = Path(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(records_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {records_path}:{line_number}: {exc}") from exc
        if isinstance(payload, dict):
            records.append(payload)
    return records


def build_readiness_report(records: list[dict[str, Any]], *, slow_threshold_ms: float = 18_000.0) -> dict[str, Any]:
    decisions: Counter[str] = Counter()
    usage_classes: Counter[str] = Counter()
    reason_codes: Counter[str] = Counter()
    usage_reasons: Counter[str] = Counter()
    field_status: dict[str, Counter[str]] = defaultdict(Counter)
    required_missing_or_bad: dict[str, list[str]] = defaultdict(list)
    review_files: list[dict[str, Any]] = []
    rejected_files: list[dict[str, Any]] = []
    slow_files: list[dict[str, Any]] = []
    web_ready_files: list[str] = []

    for record in records:
        file_name = str(record.get("file") or "")
        assessment = record.get("input_assessment") or {}
        decision = str(assessment.get("decision") or "unknown")
        usage_class = str(record.get("stnk_usage_class") or "unknown")
        decisions[decision] += 1
        usage_classes[usage_class] += 1
        reason_codes.update(assessment.get("reason_codes") or [])
        usage_reasons.update(record.get("stnk_usage_reasons") or [])

        fields = record.get("fields") or {}
        for field_name in STNK_FIELD_ORDER:
            field = fields.get(field_name) or {}
            status = str(field.get("status") or "missing")
            field_status[field_name][status] += 1
            if field_name in STNK_WEB_REQUIRED_FIELDS and status != "ok":
                required_missing_or_bad[field_name].append(file_name)

        if decision == "approved_for_auto" and usage_class == "web_usable":
            web_ready_files.append(file_name)
        elif decision == "rejected_input" or usage_class == "bad_input":
            rejected_files.append(_record_issue_summary(record))
        else:
            review_files.append(_record_issue_summary(record))

        processing_time_ms = record.get("processing_time_ms")
        if isinstance(processing_time_ms, (int, float)) and float(processing_time_ms) > slow_threshold_ms:
            slow_files.append(_record_issue_summary(record))

    total = len(records)
    web_ready_count = len(web_ready_files)
    return {
        "total": total,
        "web_ready_count": web_ready_count,
        "web_ready_rate": round(web_ready_count / total, 4) if total else 0.0,
        "manual_review_count": len(review_files),
        "rejected_count": len(rejected_files),
        "decisions": dict(decisions),
        "stnk_usage_classes": dict(usage_classes),
        "reason_codes": dict(reason_codes),
        "stnk_usage_reasons": dict(usage_reasons),
        "required_fields": list(STNK_REQUIRED_FIELDS),
        "web_required_fields": list(STNK_WEB_REQUIRED_FIELDS),
        "field_status": {field: dict(statuses) for field, statuses in field_status.items()},
        "required_missing_or_bad": dict(required_missing_or_bad),
        "slow_threshold_ms": slow_threshold_ms,
        "slow_files": sorted(slow_files, key=lambda row: float(row.get("processing_time_ms") or 0), reverse=True),
        "review_files": review_files,
        "rejected_files": rejected_files,
        "web_ready_files": web_ready_files,
    }


def write_field_review_csv(records: list[dict[str, Any]], path: str | Path) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    field_columns: list[str] = []
    for field_name in STNK_FIELD_ORDER:
        field_columns.extend(
            [
                f"{field_name}_value",
                f"{field_name}_status",
                f"{field_name}_confidence",
                f"expected_{field_name}",
            ]
        )
    columns = [
        "file",
        "decision",
        "usage_class",
        "processing_time_ms",
        "required_complete",
        "required_problems",
        "reason_codes",
        "usage_reasons",
        "manual_verdict",
        "review_notes",
        *field_columns,
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(_csv_row(record))


def run_readiness_audit(
    records_path: str | Path,
    output_dir: str | Path,
    *,
    slow_threshold_ms: float = 18_000.0,
) -> tuple[Path, Path]:
    records = load_records(records_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = build_readiness_report(records, slow_threshold_ms=slow_threshold_ms)
    summary_path = output_path / SUMMARY_BASENAME
    csv_path = output_path / CSV_BASENAME
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_field_review_csv(records, csv_path)
    return summary_path, csv_path


def _csv_row(record: dict[str, Any]) -> dict[str, Any]:
    assessment = record.get("input_assessment") or {}
    fields = record.get("fields") or {}
    required_problems = [
        field_name
        for field_name in STNK_WEB_REQUIRED_FIELDS
        if (fields.get(field_name) or {}).get("status") != "ok"
    ]
    row: dict[str, Any] = {
        "file": record.get("file") or "",
        "decision": assessment.get("decision") or "",
        "usage_class": record.get("stnk_usage_class") or "",
        "processing_time_ms": record.get("processing_time_ms") or "",
        "required_complete": "yes" if not required_problems else "no",
        "required_problems": ";".join(required_problems),
        "reason_codes": ";".join(assessment.get("reason_codes") or []),
        "usage_reasons": ";".join(record.get("stnk_usage_reasons") or []),
        "manual_verdict": "",
        "review_notes": "",
    }
    for field_name in STNK_FIELD_ORDER:
        field = fields.get(field_name) or {}
        row[f"{field_name}_value"] = field.get("value") or ""
        row[f"{field_name}_status"] = field.get("status") or "missing"
        row[f"{field_name}_confidence"] = field.get("confidence") if field.get("confidence") is not None else ""
        row[f"expected_{field_name}"] = ""
    return row


def _record_issue_summary(record: dict[str, Any]) -> dict[str, Any]:
    assessment = record.get("input_assessment") or {}
    return {
        "file": record.get("file"),
        "decision": assessment.get("decision"),
        "usage_class": record.get("stnk_usage_class"),
        "processing_time_ms": record.get("processing_time_ms"),
        "reason_codes": assessment.get("reason_codes") or [],
        "usage_reasons": record.get("stnk_usage_reasons") or [],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_readiness_audit(args.records, args.output_dir, slow_threshold_ms=args.slow_threshold_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
