import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_engine.ocr.base import OcrDependencyError
from ocr_engine.ocr.paddle_provider import PaddleOcrProvider, build_paddle_kwargs, normalize_paddle_output


class PaddleProviderTests(unittest.TestCase):
    def test_normalize_paddle_legacy_output_to_tokens_and_text(self):
        raw = [
            [
                [[[0, 0], [10, 0], [10, 10], [0, 10]], ("NIK", 0.99)],
                [[[12, 0], [80, 0], [80, 10], [12, 10]], ("3175010101900001", 0.97)],
            ]
        ]

        result = normalize_paddle_output(raw)

        self.assertEqual(result.provider, "paddleocr")
        self.assertEqual(result.raw_text, "NIK\n3175010101900001")
        self.assertEqual(result.tokens[1].text, "3175010101900001")
        self.assertEqual(result.tokens[1].confidence, 0.97)

    def test_normalize_paddle_dict_output_to_tokens_and_text(self):
        raw = [
            {
                "rec_texts": ["NAMA", "BUDI"],
                "rec_scores": [0.91, 0.88],
                "rec_boxes": [[[0, 0], [10, 0], [10, 10], [0, 10]], [[12, 0], [50, 0], [50, 10], [12, 10]]],
            }
        ]

        result = normalize_paddle_output(raw)

        self.assertEqual(result.raw_text, "NAMA\nBUDI")
        self.assertEqual(len(result.tokens), 2)

    def test_provider_raises_actionable_error_when_dependency_missing(self):
        provider = PaddleOcrProvider(engine_factory=lambda: (_ for _ in ()).throw(ImportError("missing")))

        with self.assertRaises(OcrDependencyError) as raised:
            provider.extract_text("sample.jpg")

        self.assertIn("PaddleOCR", str(raised.exception))
        self.assertIn("requirements.txt", str(raised.exception))

    def test_fast_preset_uses_mobile_detection_and_smaller_limit(self):
        kwargs = build_paddle_kwargs(lang="en", use_angle_cls=False, preset="fast")

        self.assertEqual(kwargs["text_detection_model_name"], "PP-OCRv5_mobile_det")
        self.assertEqual(kwargs["text_recognition_model_name"], "en_PP-OCRv5_mobile_rec")
        self.assertEqual(kwargs["text_det_limit_side_len"], 1280)
        self.assertEqual(kwargs["text_recognition_batch_size"], 4)
        self.assertFalse(kwargs["use_doc_orientation_classify"])

    def test_warm_up_initializes_engine_once(self):
        calls = []

        provider = PaddleOcrProvider(engine_factory=lambda: calls.append("created") or object())

        provider.warm_up()
        provider.warm_up()

        self.assertEqual(calls, ["created"])


if __name__ == "__main__":
    unittest.main()
