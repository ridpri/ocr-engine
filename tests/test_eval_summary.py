import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_engine.eval_summary import summarize_records


class EvalSummaryTests(unittest.TestCase):
    def test_summarize_records_counts_decisions_fields_and_runtime(self):
        records = [
            {
                "status": "ok",
                "document_type": "KTP",
                "needs_review": False,
                "processing_time_ms": 100,
                "stnk_structure_score": 0.92,
                "stnk_usage_class": "web_usable",
                "stnk_usage_reasons": [],
                "input_assessment": {"decision": "approved_for_auto", "can_auto_publish": True},
                "warnings": [],
                "fields": {"nik": {"status": "ok"}, "nama": {"status": "ok"}},
                "ocr": {
                    "token_count": 30,
                    "preprocess": {"selected_max_side": 1200, "retry_count": 0},
                    "timings": {
                        "total_ms": 90,
                        "attempts": [
                            {
                                "prepare_ms": 10,
                                "ocr_ms": 60,
                                "parse_ms": 5,
                                "quality_ms": 10,
                                "assessment_ms": 5,
                                "total_ms": 90,
                            }
                        ],
                    },
                },
                "quality": {"flags": [], "metrics": {"overall_score": 0.95}},
            },
            {
                "status": "ok",
                "document_type": "KTP",
                "needs_review": True,
                "processing_time_ms": 300,
                "stnk_structure_score": 0.28,
                "stnk_usage_class": "bad_input",
                "stnk_usage_reasons": ["document_type_rejected"],
                "input_assessment": {
                    "decision": "rejected_input",
                    "can_auto_publish": False,
                    "reason_codes": ["screen_or_desktop_capture"],
                },
                "warnings": ["missing_required:nama"],
                "fields": {"nik": {"status": "ok"}, "nama": {"status": "missing"}},
                "ocr": {
                    "token_count": 12,
                    "preprocess": {"selected_max_side": 1280, "retry_count": 1},
                    "timings": {
                        "total_ms": 220,
                        "attempts": [
                            {
                                "prepare_ms": 20,
                                "ocr_ms": 120,
                                "parse_ms": 10,
                                "quality_ms": 30,
                                "assessment_ms": 10,
                                "total_ms": 190,
                            }
                        ],
                    },
                },
                "quality": {"flags": ["screen_or_desktop_capture"], "metrics": {"overall_score": 0.45}},
            },
        ]

        summary = summarize_records(records)

        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["statuses"]["ok"], 2)
        self.assertEqual(summary["decisions"]["approved_for_auto"], 1)
        self.assertEqual(summary["decisions"]["rejected_input"], 1)
        self.assertEqual(summary["auto_publish"]["true"], 1)
        self.assertEqual(summary["auto_publish"]["false"], 1)
        self.assertEqual(summary["reason_codes"]["screen_or_desktop_capture"], 1)
        self.assertEqual(summary["field_status"]["nama"]["missing"], 1)
        self.assertEqual(summary["warnings"]["missing_required:nama"], 1)
        self.assertEqual(summary["stnk_usage_classes"]["web_usable"], 1)
        self.assertEqual(summary["stnk_usage_classes"]["bad_input"], 1)
        self.assertEqual(summary["stnk_usage_reasons"]["document_type_rejected"], 1)
        self.assertAlmostEqual(summary["stnk_structure_score"]["avg"], 0.6)
        self.assertEqual(summary["quality_flags"]["screen_or_desktop_capture"], 1)
        self.assertEqual(summary["processing_time_ms"]["avg"], 200)
        self.assertEqual(summary["processing_time_ms"]["p50"], 200)
        self.assertEqual(summary["processing_time_ms"]["p95"], 290)
        self.assertEqual(summary["ocr_tokens"]["avg"], 21)
        self.assertEqual(summary["ocr_retry_count"]["avg"], 0.5)
        self.assertEqual(summary["selected_max_side"]["max"], 1280)
        self.assertAlmostEqual(summary["quality_score"]["avg"], 0.7)
        self.assertEqual(summary["stage_timings_ms"]["pipeline_total"]["avg"], 155)
        self.assertEqual(summary["stage_timings_ms"]["ocr"]["avg"], 90)
        self.assertEqual(summary["stage_timings_ms"]["quality"]["max"], 30)
        self.assertEqual(summary["slowest_records"][0]["file"], None)
        self.assertEqual(summary["slowest_records"][0]["processing_time_ms"], 300)
        self.assertEqual(summary["slowest_records"][0]["ocr_ms_total"], 120)


if __name__ == "__main__":
    unittest.main()
