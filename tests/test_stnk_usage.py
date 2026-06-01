import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_engine.stnk_usage import apply_stnk_web_usage_gate, classify_stnk_record


class StnkUsageTests(unittest.TestCase):
    def test_low_structure_with_good_quality_is_internal_only(self):
        record = {
            "status": "ok",
            "document_type": "STNK",
            "processing_time_ms": 9000,
            "stnk_structure_score": 0.18,
            "input_assessment": {
                "decision": "rejected_input",
                "reason_codes": ["document_type_unknown", "missing_required:nomor_mesin"],
            },
            "quality": {"flags": [], "metrics": {"overall_score": 1.0}},
            "fields": {
                "nomor_polisi": {"status": "ok"},
                "nama_pemilik": {"status": "missing"},
                "tahun_pembuatan": {"status": "missing"},
                "nomor_rangka": {"status": "missing"},
                "nomor_mesin": {"status": "missing"},
            },
            "ocr": {"token_count": 70},
        }

        usage_class, reasons = classify_stnk_record(record)

        self.assertEqual(usage_class, "internal_only")
        self.assertIn("document_type_unknown", reasons)
        self.assertIn("structure_score_below_web_threshold", reasons)

    def test_quality_failure_is_bad_input(self):
        record = {
            "status": "ok",
            "document_type": "STNK",
            "processing_time_ms": 9000,
            "stnk_structure_score": 0.05,
            "input_assessment": {"decision": "rejected_input", "reason_codes": ["document_too_small"]},
            "quality": {"flags": ["document_too_small"], "metrics": {"overall_score": 0.5}},
            "fields": {},
            "ocr": {"token_count": 8},
        }

        usage_class, reasons = classify_stnk_record(record)

        self.assertEqual(usage_class, "bad_input")
        self.assertIn("quality:document_too_small", reasons)

    def test_internal_only_record_is_not_auto_publishable_for_web(self):
        record = {
            "status": "ok",
            "document_type": "STNK",
            "needs_review": False,
            "stnk_usage_class": "internal_only",
            "stnk_usage_reasons": ["processing_time_over_20s"],
            "input_assessment": {
                "decision": "approved_for_auto",
                "can_auto_publish": True,
                "reason_codes": [],
                "message": "Siap auto.",
            },
        }

        apply_stnk_web_usage_gate(record)

        self.assertTrue(record["needs_review"])
        self.assertEqual(record["input_assessment"]["decision"], "needs_review")
        self.assertFalse(record["input_assessment"]["can_auto_publish"])
        self.assertIn("stnk_web_usage_class:internal_only", record["input_assessment"]["reason_codes"])
        self.assertIn("stnk_web_usage:processing_time_over_20s", record["input_assessment"]["reason_codes"])


if __name__ == "__main__":
    unittest.main()
