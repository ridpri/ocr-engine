import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import importlib.util
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location(
    "benchmark_stnk",
    str(ROOT / "scripts" / "benchmark_stnk.py"),
)
benchmark_stnk = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(benchmark_stnk)


def _fake_stnk_record(file_name: str, mode: str = "accurate") -> dict:
    return {
        "file": file_name,
        "status": "ok",
        "document_type": "STNK",
        "needs_review": False,
        "warnings": [],
        "input_assessment": {"decision": "approved_for_auto"},
        "quality": {"flags": [], "metrics": {"overall_score": 0.99}},
        "processing_time_ms": 12.34,
        "stnk_structure_score": 0.95,
        "stnk_usage_class": "web_usable",
        "stnk_usage_reasons": [],
        "fields": {"nomor_polisi": {"value": "B 1234 ABC", "status": "ok"}},
        "ocr": {
            "provider": "fake",
            "token_count": 3,
            "processing_mode": mode,
            "raw_text_masked": "HIDDEN",
            "nik_fallback": {"attempted": False},
            "preprocess": {
                "selected_max_side": 1200,
                "retry_count": 0,
                "attempts": [
                    {
                        "strategy": "stnk_official_roi",
                        "document_type": "STNK",
                        "decision": "approved_for_auto",
                    }
                ],
            },
            "timings": {
                "total_ms": 12.34,
                "attempts": [
                    {
                        "prepare_ms": 1,
                        "ocr_ms": 2,
                        "parse_ms": 3,
                        "quality_ms": 1,
                        "assessment_ms": 1,
                        "total_ms": 8,
                    }
                ],
            },
        },
    }


class BenchmarkStnkTests(unittest.TestCase):
    def test_benchmark_stnk_accepts_file_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "sample.jpg"
            Image.new("RGB", (1280, 720), "white").save(input_file)
            output_dir = Path(tmpdir) / "out-single"

            with patch(
                "ocr_engine.cli_eval._process_file",
                side_effect=lambda provider, path, document_type, mode="accurate", run_nik_fallback=True: _fake_stnk_record(
                    path.name,
                    mode=mode,
                ),
            ):
                rc = benchmark_stnk.main(
                    [
                        "--input",
                        str(input_file),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(rc, 0)
            self.assertTrue((output_dir / benchmark_stnk.RECORDS_BASENAME).exists())

    def test_benchmark_stnk_creates_jsonl_and_summary_with_default_stnk_and_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "sample_stnk"
            input_dir.mkdir()
            output_dir = Path(tmpdir) / "out"
            for idx in range(1, 4):
                image_path = input_dir / f"sample_{idx}.jpg"
                Image.new("RGB", (1280, 720), "white").save(image_path)

            calls: list[tuple[str, str, str]] = []

            def fake_process_file(provider, path, document_type, mode="accurate", run_nik_fallback=True):
                calls.append((path.name, document_type, mode))
                return _fake_stnk_record(path.name, mode=mode)

            with patch("ocr_engine.cli_eval._process_file", side_effect=fake_process_file):
                rc = benchmark_stnk.main(
                    [
                        "--input",
                        str(input_dir),
                        "--output-dir",
                        str(output_dir),
                        "--limit",
                        "1",
                    ]
                )

            self.assertEqual(rc, 0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1], "STNK")
            self.assertEqual(calls[0][2], "accurate")

            jsonl_path = output_dir / benchmark_stnk.RECORDS_BASENAME
            summary_path = output_dir / benchmark_stnk.SUMMARY_BASENAME
            self.assertTrue(jsonl_path.exists())
            self.assertTrue(summary_path.exists())

            lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["file"], "sample_1.jpg")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["total"], 1)
            self.assertEqual(summary["decisions"]["approved_for_auto"], 1)
            self.assertEqual(summary["stnk_usage_classes"]["web_usable"], 1)

    def test_benchmark_stnk_processes_full_corpus_when_limit_is_omitted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "sample_stnk"
            input_dir.mkdir()
            output_dir = Path(tmpdir) / "out-full"
            for idx in range(7):
                image_path = input_dir / f"sample_{idx}.jpg"
                Image.new("RGB", (1280, 720), "white").save(image_path)

            calls: list[str] = []

            def fake_process_file(provider, path, document_type, mode="accurate", run_nik_fallback=True):
                calls.append(path.name)
                return _fake_stnk_record(path.name, mode=mode)

            with patch("ocr_engine.cli_eval._process_file", side_effect=fake_process_file):
                rc = benchmark_stnk.main(
                    [
                        "--input",
                        str(input_dir),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(rc, 0)
            self.assertEqual(len(calls), 7)
            summary = json.loads((output_dir / benchmark_stnk.SUMMARY_BASENAME).read_text(encoding="utf-8"))
            self.assertEqual(summary["total"], 7)

    def test_benchmark_stnk_respects_mode_argument(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "sample_stnk"
            input_dir.mkdir()
            output_dir = Path(tmpdir) / "out-fast"
            for idx in range(2):
                image_path = input_dir / f"sample_{idx}.jpg"
                Image.new("RGB", (1280, 720), "white").save(image_path)

            calls: list[tuple[str, str, str]] = []

            def fake_process_file(provider, path, document_type, mode="accurate", run_nik_fallback=True):
                calls.append((path.name, document_type, mode))
                return _fake_stnk_record(path.name, mode=mode)

            with patch("ocr_engine.cli_eval._process_file", side_effect=fake_process_file):
                rc = benchmark_stnk.main(
                    [
                        "--input",
                        str(input_dir),
                        "--output-dir",
                        str(output_dir),
                        "--mode",
                        "fast",
                        "--limit",
                        "2",
                    ]
                )

            self.assertEqual(rc, 0)
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(mode == "fast" for _, _, mode in calls))


if __name__ == "__main__":
    unittest.main()
