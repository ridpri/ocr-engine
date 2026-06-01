import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_engine.ocr.base import OcrDependencyError
from ocr_engine.ocr.paddle_provider import PaddleOcrProvider, build_paddle_kwargs, normalize_paddle_output
from ocr_engine.ocr.rapid_provider import RapidOcrProvider, normalize_rapid_output


class FakeRapidOutput:
    txts = ["NIK", "3175010101900001"]
    scores = [0.92, 0.89]
    boxes = [
        [[0, 0], [10, 0], [10, 10], [0, 10]],
        [[12, 0], [80, 0], [80, 10], [12, 10]],
    ]


class FakePredictEngine:
    def predict(self, image_path: str):
        return [{"rec_texts": [Path(image_path).name], "rec_scores": [0.9], "rec_boxes": []}]


class RecordingPredictEngine:
    def __init__(self) -> None:
        self.names = []
        self.sizes = []

    def predict(self, image_path: str):
        self.names.append(Path(image_path).name)
        with Image.open(image_path) as image:
            self.sizes.append(image.size)
        return [{"rec_texts": [Path(image_path).name], "rec_scores": [0.9], "rec_boxes": []}]


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

    def test_normalize_paddle_output_tolerates_missing_scores(self):
        raw = [{"rec_texts": ["NAMA", None, "BUDI"], "rec_scores": [None, 0.8, "bad"], "rec_boxes": []}]

        result = normalize_paddle_output(raw)

        self.assertEqual(result.raw_text, "NAMA\nBUDI")
        self.assertEqual(result.tokens[0].confidence, 0.0)
        self.assertEqual(result.tokens[1].confidence, 0.0)

    def test_provider_raises_actionable_error_when_dependency_missing(self):
        provider = PaddleOcrProvider(engine_factory=lambda: (_ for _ in ()).throw(ImportError("missing")))

        with self.assertRaises(OcrDependencyError) as raised:
            provider.extract_text("sample.jpg")

        self.assertIn("PaddleOCR", str(raised.exception))
        self.assertIn("requirements.txt", str(raised.exception))

    def test_fast_preset_uses_mobile_detection_and_smaller_limit(self):
        with patch.dict(
            "os.environ",
            {
                "OCR_PADDLE_ENABLE_MKLDNN": "0",
                "OCR_PADDLE_CPU_THREADS": "4",
                "OCR_PADDLE_ENABLE_HPI": "0",
            },
        ):
            kwargs = build_paddle_kwargs(lang="en", use_angle_cls=False, preset="fast")

        self.assertEqual(kwargs["text_detection_model_name"], "PP-OCRv5_mobile_det")
        self.assertEqual(kwargs["text_recognition_model_name"], "en_PP-OCRv5_mobile_rec")
        self.assertEqual(kwargs["text_det_limit_side_len"], 960)
        self.assertEqual(kwargs["text_recognition_batch_size"], 8)
        self.assertFalse(kwargs["enable_mkldnn"])
        self.assertEqual(kwargs["cpu_threads"], 4)
        self.assertFalse(kwargs["use_doc_orientation_classify"])

    def test_fast_preset_keeps_experimental_cpu_acceleration_off_by_default(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("ocr_engine.ocr.paddle_provider.os.cpu_count", return_value=14):
                kwargs = build_paddle_kwargs(lang="en", use_angle_cls=False, preset="fast")

        self.assertFalse(kwargs["enable_mkldnn"])
        self.assertEqual(kwargs["cpu_threads"], 10)
        self.assertNotIn("enable_hpi", kwargs)

    def test_fast_preset_caps_default_cpu_threads_to_available_vcpus(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("ocr_engine.ocr.paddle_provider.os.cpu_count", return_value=4):
                kwargs = build_paddle_kwargs(lang="en", use_angle_cls=False, preset="fast")

        self.assertEqual(kwargs["cpu_threads"], 4)

    def test_fast_preset_can_enable_mkldnn_from_environment(self):
        with patch.dict("os.environ", {"OCR_PADDLE_ENABLE_MKLDNN": "1"}, clear=True):
            kwargs = build_paddle_kwargs(lang="en", use_angle_cls=False, preset="fast")

        self.assertTrue(kwargs["enable_mkldnn"])

    def test_fast_preset_can_enable_hpi_from_environment(self):
        with patch.dict("os.environ", {"OCR_PADDLE_ENABLE_HPI": "1"}, clear=True):
            kwargs = build_paddle_kwargs(lang="en", use_angle_cls=False, preset="fast")

        self.assertTrue(kwargs["enable_hpi"])

    def test_fast_preset_can_tune_detection_limit_and_recognition_batch_size(self):
        with patch.dict(
            "os.environ",
            {"OCR_PADDLE_TEXT_DET_LIMIT": "640", "OCR_PADDLE_REC_BATCH_SIZE": "16"},
            clear=True,
        ):
            kwargs = build_paddle_kwargs(lang="en", use_angle_cls=False, preset="fast")

        self.assertEqual(kwargs["text_det_limit_side_len"], 640)
        self.assertEqual(kwargs["text_recognition_batch_size"], 16)

    def test_warm_up_initializes_engine_once(self):
        calls = []

        provider = PaddleOcrProvider(engine_factory=lambda: calls.append("created") or object())

        provider.warm_up()
        provider.warm_up()

        self.assertEqual(calls, ["created"])

    def test_warm_up_runs_representative_document_images_once(self):
        engine = RecordingPredictEngine()
        provider = PaddleOcrProvider(engine_factory=lambda: engine)

        provider.warm_up()
        provider.warm_up()

        self.assertEqual(engine.names, ["warmup-ktp-fast.jpg", "warmup-stnk-fast.jpg"])
        self.assertEqual(engine.sizes, [(496, 340), (720, 480)])

    def test_provider_initializes_engine_once_under_concurrent_calls(self):
        calls = 0
        calls_lock = threading.Lock()

        def factory():
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.05)
            return FakePredictEngine()

        provider = PaddleOcrProvider(engine_factory=factory)

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(provider.extract_text, ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]))

        self.assertEqual(calls, 1)
        self.assertEqual([result.provider for result in results], ["paddleocr"] * 4)

    def test_normalize_rapid_output_to_tokens_and_text(self):
        result = normalize_rapid_output(FakeRapidOutput())

        self.assertEqual(result.provider, "rapidocr")
        self.assertEqual(result.raw_text, "NIK\n3175010101900001")
        self.assertEqual(result.tokens[1].confidence, 0.89)

    def test_normalize_rapid_output_tolerates_tuple_rows_and_bad_scores(self):
        raw = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "NAMA", None),
            ("BUDI", "not-a-number"),
            ([[0, 12], [10, 12], [10, 22], [0, 22]], None, 0.92),
        ]

        result = normalize_rapid_output(raw)

        self.assertEqual(result.provider, "rapidocr")
        self.assertEqual(result.raw_text, "NAMA\nBUDI")
        self.assertEqual([token.confidence for token in result.tokens], [0.0, 0.0])

    def test_rapid_provider_raises_actionable_error_when_dependency_missing(self):
        provider = RapidOcrProvider(engine_factory=lambda: (_ for _ in ()).throw(ImportError("missing")))

        with self.assertRaises(OcrDependencyError) as raised:
            provider.extract_text("sample.jpg")

        self.assertIn("RapidOCR", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
