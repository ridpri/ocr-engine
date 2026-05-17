import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_engine.ocr.base import OcrResult, OcrToken
from ocr_engine.quality import analyze_image_quality
from ocr_engine.service import build_input_assessment, parse_document_text


class QualityTests(unittest.TestCase):
    def test_analyze_image_quality_flags_small_blurry_and_low_text_density(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "tiny.jpg"
            Image.new("RGB", (220, 120), "white").save(image_path)

            quality = analyze_image_quality(image_path, OcrResult(raw_text="", tokens=[]))

        self.assertIn("document_too_small", quality["flags"])
        self.assertIn("blur_detected", quality["flags"])
        self.assertIn("low_text_density", quality["flags"])
        self.assertEqual(quality["image"]["width"], 220)
        self.assertEqual(quality["image"]["height"], 120)

    def test_analyze_image_quality_flags_screen_capture_from_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "screen.jpg"
            Image.new("RGB", (1200, 800), "white").save(image_path)
            ocr = OcrResult(
                raw_text="Manage\nKTP SETYO.jpg\nTOSHIBA\nType here\n100%",
                tokens=[OcrToken("KTP", 0.9), OcrToken("TOSHIBA", 0.9)],
            )

            quality = analyze_image_quality(image_path, ocr)

        self.assertIn("screen_or_desktop_capture", quality["flags"])
        self.assertGreaterEqual(quality["metrics"]["ocr_token_count"], 2)

    def test_input_assessment_uses_quality_flags(self):
        raw_text = "PROVINSI DKI JAKARTA\nNIK : 3175010101900001\nNama : BUDI\nAlamat : JL MERDEKA"
        parsed = parse_document_text(raw_text, "KTP")
        assessment = build_input_assessment(
            raw_text,
            parsed,
            "KTP",
            "KTP",
            quality={"flags": ["blur_detected"]},
        )

        self.assertEqual(assessment["decision"], "needs_review")
        self.assertFalse(assessment["can_auto_publish"])
        self.assertIn("blur_detected", assessment["reason_codes"])


if __name__ == "__main__":
    unittest.main()
