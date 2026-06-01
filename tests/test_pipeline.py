import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_engine.ocr.base import OcrResult, OcrToken
from ocr_engine.pipeline import KTP_FAST_BOTTOM_RATIO, _first_attempt_max_side, run_ocr_pipeline
from ocr_engine.schemas import FieldResult


class FailingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def extract_text(self, image_path: str) -> OcrResult:
        self.calls += 1
        raise AssertionError("OCR should not run for preflight-rejected input")


class KtpProvider:
    def extract_text(self, image_path: str) -> OcrResult:
        raw_text = "\n".join(
            [
                "PROVINSI DKI JAKARTA",
                "NIK : 3175010101900001",
                "Nama : BUDI SANTOSO",
                "Tempat/Tgl Lahir : JAKARTA, 01-01-1990",
                "Jenis Kelamin : LAKI-LAKI",
                "Alamat : JL MERDEKA",
                "RT/RW : 001/002",
                "Kel/Desa : GAMBIR",
                "Kecamatan : GAMBIR",
                "Pekerjaan : KARYAWAN SWASTA",
                "Kewarganegaraan : WNI",
                "Berlaku Hingga : SEUMUR HIDUP",
            ]
        )
        return OcrResult(raw_text=raw_text, tokens=[OcrToken(token, 0.99) for token in raw_text.split()])


class SequentialProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def extract_text(self, image_path: str) -> OcrResult:
        self.calls += 1
        raw_text = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        return OcrResult(raw_text=raw_text, tokens=[OcrToken(token, 0.99) for token in raw_text.split()])


class PipelineTests(unittest.TestCase):
    def test_ktp_fast_rejects_tiny_image_before_ocr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "tiny.jpg"
            Image.new("RGB", (220, 120), "white").save(image_path)
            provider = FailingProvider()

            result = run_ocr_pipeline(provider, image_path, "KTP", tmp_path, processing_mode="fast")

        self.assertEqual(provider.calls, 0)
        self.assertEqual(result.assessment["decision"], "rejected_input")
        self.assertIn("document_too_small", result.assessment["reason_codes"])
        self.assertIn("pre_ocr_rejected", result.assessment["reason_codes"])
        self.assertEqual(result.ocr_result.provider, "preflight")
        self.assertEqual(result.preprocess["attempts"], [])
        self.assertTrue(result.preprocess["pre_ocr_rejected"])

    def test_stnk_fast_rejects_tiny_image_before_ocr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "tiny-stnk.jpg"
            Image.new("RGB", (420, 180), "white").save(image_path)
            provider = FailingProvider()

            result = run_ocr_pipeline(provider, image_path, "STNK", tmp_path, processing_mode="fast")

        self.assertEqual(provider.calls, 0)
        self.assertEqual(result.assessment["decision"], "rejected_input")
        self.assertIn("document_too_small", result.assessment["reason_codes"])
        self.assertIn("pre_ocr_rejected", result.assessment["reason_codes"])
        self.assertEqual(result.ocr_result.provider, "preflight")

    def test_fast_max_side_can_be_tuned_from_environment(self):
        with patch.dict("os.environ", {"OCR_STNK_FAST_MAX_SIDE": "640", "OCR_KTP_FAST_MAX_SIDE": "496"}):
            stnk_size = _first_attempt_max_side("STNK", "fast", "stnk_fast_roi")
            ktp_size = _first_attempt_max_side("KTP", "fast", "ktp_fast")

        self.assertEqual(stnk_size, 640)
        self.assertEqual(ktp_size, 496)

    def test_ktp_fast_default_uses_benchmarked_smaller_image(self):
        with patch.dict("os.environ", {}, clear=True):
            ktp_size = _first_attempt_max_side("KTP", "fast", "ktp_fast")

        self.assertEqual(ktp_size, 496)
        self.assertEqual(KTP_FAST_BOTTOM_RATIO, 1.0)

    def test_pipeline_reuses_preflight_quality_for_post_ocr_assessment(self):
        preflight = {
            "image": {"width": 900, "height": 600},
            "flags": [],
            "metrics": {"blur_score": 20.0},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "ktp.jpg"
            Image.new("RGB", (900, 600), "white").save(image_path)
            provider = KtpProvider()

            with patch("ocr_engine.pipeline.analyze_image_preflight", return_value=preflight) as pipeline_preflight:
                with patch(
                    "ocr_engine.quality.analyze_image_preflight",
                    side_effect=AssertionError("preflight should not be recomputed after OCR"),
                ):
                    result = run_ocr_pipeline(provider, image_path, "KTP", tmp_path, processing_mode="fast")

        self.assertEqual(pipeline_preflight.call_count, 1)
        self.assertEqual(result.quality["image"], preflight["image"])
        self.assertEqual(result.quality["metrics"]["blur_score"], 20.0)

    def test_fast_ktp_retries_full_page_for_missing_name_and_picks_better_attempt(self):
        first_raw = "\n".join(
            [
                "PROVINSI DKI JAKARTA",
                "NIK : 3175010101900001",
                "Tempat/Tgl Lahir : JAKARTA, 01-01-1990",
                "Jenis Kelamin : LAKI-LAKI",
                "Alamat : JL MERDEKA",
                "RT/RW : 001/002",
                "Kel/Desa : GAMBIR",
                "Kecamatan : GAMBIR",
                "Pekerjaan : KARYAWAN SWASTA",
                "Kewarganegaraan : WNI",
                "Berlaku Hingga : SEUMUR HIDUP",
            ]
        )
        second_raw = "\n".join(
            [
                "PROVINSI DKI JAKARTA",
                "NIK : 3175010101900001",
                "Nama : BUDI SANTOSO",
                "Tempat/Tgl Lahir : JAKARTA, 01-01-1990",
                "Jenis Kelamin : LAKI-LAKI",
                "Alamat : JL MERDEKA",
                "RT/RW : 001/002",
                "Kel/Desa : GAMBIR",
                "Kecamatan : GAMBIR",
                "Pekerjaan : KARYAWAN SWASTA",
                "Kewarganegaraan : WNI",
                "Berlaku Hingga : SEUMUR HIDUP",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "ktp.jpg"
            Image.new("RGB", (900, 600), "white").save(image_path)
            provider = SequentialProvider([first_raw, second_raw])
            preflight = {
                "image": {"width": 900, "height": 600},
                "flags": [],
                "metrics": {"blur_score": 20.0},
            }
            with patch("ocr_engine.pipeline.analyze_image_preflight", return_value=preflight):
                result = run_ocr_pipeline(provider, image_path, "KTP", tmp_path, processing_mode="fast")

        self.assertEqual(provider.calls, 2)
        self.assertEqual(result.assessment["decision"], "approved_for_auto")
        self.assertEqual(result.parsed.fields["nama"].value, "BUDI SANTOSO")
        self.assertEqual(result.preprocess["retry_count"], 1)
        self.assertEqual(result.preprocess["attempts"][1]["strategy"], "full_page")

    def test_pipeline_refreshes_assessment_after_nik_image_fallback(self):
        raw_text = "\n".join(
            [
                "PROVINSI DKI JAKARTA",
                "Nama : BUDI SANTOSO",
                "Tempat/Tgl Lahir : JAKARTA, 01-01-1990",
                "Jenis Kelamin : LAKI-LAKI",
                "Alamat : JL MERDEKA",
                "RT/RW : 001/002",
                "Kel/Desa : GAMBIR",
                "Kecamatan : GAMBIR",
                "Pekerjaan : KARYAWAN SWASTA",
                "Kewarganegaraan : WNI",
                "Berlaku Hingga : SEUMUR HIDUP",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "ktp.jpg"
            Image.new("RGB", (900, 600), "white").save(image_path)
            provider = SequentialProvider([raw_text])
            preflight = {
                "image": {"width": 900, "height": 600},
                "flags": [],
                "metrics": {"blur_score": 20.0},
            }
            fake_fallback = {"attempted": True, "passes": 1, "value": "3175010101900001"}

            def apply_nik(provider_arg, image_arg, parsed_arg, workdir_arg):
                parsed_arg.fields["nik"] = FieldResult(
                    value="3175010101900001",
                    confidence=0.92,
                    status="ok",
                    evidence=["3175010101900001"],
                    raw="image_nik_fallback",
                )
                parsed_arg.warnings = [warning for warning in parsed_arg.warnings if warning != "missing_required:nik"]
                return fake_fallback

            with patch("ocr_engine.pipeline.analyze_image_preflight", return_value=preflight):
                with patch("ocr_engine.pipeline.repair_ktp_nik_from_image", side_effect=apply_nik):
                    result = run_ocr_pipeline(provider, image_path, "KTP", tmp_path, processing_mode="fast")

        self.assertEqual(result.parsed.fields["nik"].value, "3175010101900001")
        self.assertNotIn("missing_required:nik", result.assessment["reason_codes"])
        self.assertNotIn("ktp_auto_missing:nik", result.assessment["reason_codes"])


if __name__ == "__main__":
    unittest.main()
