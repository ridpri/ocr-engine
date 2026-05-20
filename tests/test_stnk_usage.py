import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_engine.stnk_usage import classify_stnk_record


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


if __name__ == "__main__":
    unittest.main()
