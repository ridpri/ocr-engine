import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import importlib.util


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location(
    "audit_stnk_readiness",
    str(ROOT / "scripts" / "audit_stnk_readiness.py"),
)
audit_stnk_readiness = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(audit_stnk_readiness)


def _record(
    file_name: str,
    *,
    decision: str,
    usage_class: str,
    fields: dict,
    processing_time_ms: float = 12000.0,
    reason_codes: list[str] | None = None,
) -> dict:
    return {
        "file": file_name,
        "status": "ok",
        "document_type": "STNK",
        "processing_time_ms": processing_time_ms,
        "stnk_usage_class": usage_class,
        "stnk_usage_reasons": [],
        "input_assessment": {
            "decision": decision,
            "reason_codes": reason_codes or [],
            "can_auto_publish": decision == "approved_for_auto",
        },
        "fields": fields,
    }


def _ok_field(value: str, confidence: float = 0.9) -> dict:
    return {"value": value, "status": "ok", "confidence": confidence}


def _required_fields(**overrides: dict) -> dict:
    fields = {
        "nomor_polisi": _ok_field("B 1234 ABC"),
        "nama_pemilik": _ok_field("BUDI"),
        "tahun_pembuatan": _ok_field("2024"),
        "nomor_rangka": _ok_field("MH123456789012345"),
        "nomor_mesin": _ok_field("AB12345"),
    }
    fields.update(overrides)
    return fields


class AuditStnkReadinessTests(unittest.TestCase):
    def test_build_readiness_report_counts_ready_review_and_required_problems(self):
        records = [
            _record(
                "ready.jpg",
                decision="approved_for_auto",
                usage_class="web_usable",
                fields=_required_fields(),
            ),
            _record(
                "review.jpg",
                decision="needs_review",
                usage_class="internal_only",
                fields=_required_fields(nomor_polisi={"value": "B 1234", "status": "ok", "confidence": 0.8}),
                reason_codes=["stnk_auto_low_confidence:nomor_polisi"],
                processing_time_ms=19000.0,
            ),
            _record(
                "missing.jpg",
                decision="needs_review",
                usage_class="internal_only",
                fields=_required_fields(nomor_mesin={"value": None, "status": "missing", "confidence": 0.0}),
                reason_codes=["stnk_auto_missing:nomor_mesin"],
            ),
        ]

        summary = audit_stnk_readiness.build_readiness_report(records, slow_threshold_ms=18000.0)

        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["web_ready_count"], 1)
        self.assertEqual(summary["manual_review_count"], 2)
        self.assertEqual(summary["required_missing_or_bad"], {"nomor_mesin": ["missing.jpg"]})
        self.assertEqual(summary["field_status"]["nomor_mesin"], {"ok": 2, "missing": 1})
        self.assertEqual([item["file"] for item in summary["slow_files"]], ["review.jpg"])

    def test_run_readiness_audit_writes_summary_and_manual_review_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records_path = Path(tmpdir) / "records.jsonl"
            output_dir = Path(tmpdir) / "out"
            records = [
                _record(
                    "ready.jpg",
                    decision="approved_for_auto",
                    usage_class="web_usable",
                    fields=_required_fields(),
                )
            ]
            records_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
                encoding="utf-8",
            )

            summary_path, csv_path = audit_stnk_readiness.run_readiness_audit(records_path, output_dir)

            self.assertTrue(summary_path.exists())
            self.assertTrue(csv_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["web_ready_rate"], 1.0)

            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["file"], "ready.jpg")
            self.assertEqual(rows[0]["required_complete"], "yes")
            self.assertIn("expected_nomor_polisi", rows[0])
            self.assertIn("manual_verdict", rows[0])


if __name__ == "__main__":
    unittest.main()
