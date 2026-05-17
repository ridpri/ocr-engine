import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_engine.cli_eval import _process_file
from ocr_engine.ocr.base import OcrResult, OcrToken


class FakeProvider:
    def extract_text(self, image_path: str) -> OcrResult:
        return OcrResult(
            raw_text="PROVINSI DKI JAKARTA\nNIK : 3175010101900001\nNama : BUDI\nAlamat : JL MERDEKA",
            tokens=[
                OcrToken("PROVINSI", 0.99),
                OcrToken("DKI", 0.99),
                OcrToken("JAKARTA", 0.99),
                OcrToken("NIK", 0.99),
                OcrToken("3175010101900001", 0.99),
                OcrToken("BUDI", 0.99),
                OcrToken("JL", 0.99),
                OcrToken("MERDEKA", 0.99),
            ],
            provider="fake",
        )


class MismatchedKtpProvider:
    def __init__(self) -> None:
        self.calls = 0

    def extract_text(self, image_path: str) -> OcrResult:
        self.calls += 1
        return OcrResult(
            raw_text="PROVINSI DKI JAKARTA\nNIK : 12345\nNama : BUDI\nAlamat : JL MERDEKA",
            tokens=[
                OcrToken("PROVINSI", 0.99),
                OcrToken("DKI", 0.99),
                OcrToken("JAKARTA", 0.99),
                OcrToken("NIK", 0.99),
                OcrToken("12345", 0.99),
                OcrToken("BUDI", 0.99),
                OcrToken("JL", 0.99),
                OcrToken("MERDEKA", 0.99),
            ],
            provider="fake",
        )


class SizeRecordingProvider(FakeProvider):
    def __init__(self) -> None:
        self.seen_sizes: list[tuple[int, int]] = []

    def extract_text(self, image_path: str) -> OcrResult:
        with Image.open(image_path) as image:
            self.seen_sizes.append(image.size)
        return super().extract_text(image_path)


class AdaptiveStnkProvider:
    def __init__(self) -> None:
        self.seen_sizes: list[tuple[int, int]] = []

    def extract_text(self, image_path: str) -> OcrResult:
        with Image.open(image_path) as image:
            self.seen_sizes.append(image.size)
            max_side = max(image.size)

        is_initial_prepared = Path(image_path).name == "prepared.jpg"
        if is_initial_prepared or max_side <= 512:
            raw_text = "\n".join(
                [
                    "SURAT TANDA NOMOR KENDARAAN",
                    "NO POLISI : B 1234 ABC",
                    "NAMA PEMILIK : BUDI",
                ]
            )
        else:
            raw_text = "\n".join(
                [
                    "SURAT TANDA NOMOR KENDARAAN",
                    "NO POLISI : B 1234 ABC",
                    "NAMA PEMILIK : BUDI SANTOSO",
                    "TAHUN PEMBUATAN : 2020",
                    "NO RANGKA : MHRRU1860KJ302319",
                    "NO MESIN : L15Z61219016",
                ]
            )

        return OcrResult(
            raw_text=raw_text,
            tokens=[OcrToken(text, 0.99) for text in raw_text.split()],
            provider="fake",
        )


class CompleteStnkProvider(AdaptiveStnkProvider):
    def extract_text(self, image_path: str) -> OcrResult:
        with Image.open(image_path) as image:
            self.seen_sizes.append(image.size)
        raw_text = "\n".join(
            [
                "SURAT TANDA NOMOR KENDARAAN",
                "NO POLISI : B 1234 ABC",
                "NAMA PEMILIK : BUDI SANTOSO",
                "TAHUN PEMBUATAN : 2020",
                "NO RANGKA : MHRRU1860KJ302319",
                "NO MESIN : L15Z61219016",
            ]
        )
        return OcrResult(
            raw_text=raw_text,
            tokens=[OcrToken(text, 0.99) for text in raw_text.split()],
            provider="fake",
        )


class KtpLayoutProvider:
    def extract_text(self, image_path: str) -> OcrResult:
        return OcrResult(
            raw_text="\n".join(
                [
                    "PROVINSI DKI JAKARTA",
                    "NIK : 3216064704060020",
                    "Nama : SALSABILA PUTRI DEWANTI",
                    "Pekerjaan",
                    "PELAJAR/MAHASISWA",
                ]
            ),
            tokens=[
                OcrToken("NIK", 0.99, bbox=[[100, 100], [150, 100], [150, 125], [100, 125]]),
                OcrToken("3216064704060020", 0.99, bbox=[[260, 100], [520, 100], [520, 125], [260, 125]]),
                OcrToken("Nama", 0.99, bbox=[[100, 170], [180, 170], [180, 195], [100, 195]]),
                OcrToken("SALSABILA PUTRI DEWANTI", 0.99, bbox=[[260, 170], [560, 170], [560, 195], [260, 195]]),
                OcrToken("Pekerjaan", 0.99, bbox=[[100, 720], [230, 720], [230, 745], [100, 745]]),
                OcrToken("PELAJAR/MAHASISWA", 0.99, bbox=[[260, 720], [520, 720], [520, 745], [260, 745]]),
                OcrToken("Kewargane", 0.82, bbox=[[100, 780], [240, 780], [240, 805], [100, 805]]),
                OcrToken("VNI", 0.75, bbox=[[260, 780], [310, 780], [310, 805], [260, 805]]),
            ],
            provider="fake",
        )


def _write_textured_image(image_path: Path, size: tuple[int, int] = (900, 600)) -> None:
    width, height = size
    image = Image.new("RGB", size, "white")
    pixels = image.load()
    for y in range(0, height, 16):
        for x in range(0, width, 16):
            color = "black" if (x // 16 + y // 16) % 2 == 0 else "white"
            for yy in range(y, min(y + 8, height)):
                for xx in range(x, min(x + 8, width)):
                    pixels[xx, yy] = (0, 0, 0) if color == "black" else (255, 255, 255)
    image.save(image_path)


class CliEvalTests(unittest.TestCase):
    def test_process_file_includes_quality_and_processing_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "ktp.jpg"
            _write_textured_image(image_path)

            record = _process_file(FakeProvider(), image_path, "KTP")

        self.assertIn("quality", record)
        self.assertIn("processing_time_ms", record)
        self.assertGreaterEqual(record["processing_time_ms"], 0)
        self.assertIn("timings", record["ocr"])
        self.assertGreaterEqual(record["ocr"]["timings"]["total_ms"], 0)
        self.assertEqual(len(record["ocr"]["timings"]["attempts"]), 1)
        attempt_timings = record["ocr"]["timings"]["attempts"][0]
        for stage in ["prepare_ms", "ocr_ms", "parse_ms", "quality_ms", "assessment_ms", "total_ms"]:
            self.assertIn(stage, attempt_timings)
            self.assertGreaterEqual(attempt_timings[stage], 0)
        self.assertEqual(record["input_assessment"]["decision"], "approved_for_auto")

    def test_process_file_accepts_pdf_by_rendering_first_page(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "ktp.pdf"
            pdf_path.write_bytes(b"%PDF-1.7")

            def fake_render_pdf_first_page(input_path, output_path, dpi=200):
                Image.new("RGB", (900, 600), "white").save(output_path, format="PNG")
                return Path(output_path)

            with unittest.mock.patch("ocr_engine.cli_eval.render_pdf_first_page", side_effect=fake_render_pdf_first_page):
                record = _process_file(FakeProvider(), pdf_path, "KTP")

        self.assertEqual(record["document_type"], "KTP")
        self.assertEqual(record["input_assessment"]["expected_document_type"], "KTP")
        self.assertEqual(record["ocr"]["processing_mode"], "accurate")

    def test_stnk_mismatch_does_not_run_ktp_nik_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "stnk.jpg"
            _write_textured_image(image_path)
            provider = MismatchedKtpProvider()

            record = _process_file(provider, image_path, "STNK")

        self.assertEqual(record["input_assessment"]["decision"], "rejected_input")
        self.assertFalse(record["ocr"]["nik_fallback"]["attempted"])
        self.assertEqual(provider.calls, 1)

    def test_stnk_uses_smaller_prepared_image_for_speed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "stnk.jpg"
            Image.new("RGB", (1800, 1200), "white").save(image_path)
            provider = SizeRecordingProvider()

            _process_file(provider, image_path, "STNK")

        self.assertEqual(max(provider.seen_sizes[0]), 1200)

    def test_stnk_fast_mode_uses_smaller_roi_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "stnk.jpg"
            Image.new("RGB", (1800, 1200), "white").save(image_path)
            provider = SizeRecordingProvider()

            record = _process_file(provider, image_path, "STNK", mode="fast")

        self.assertEqual(max(provider.seen_sizes[0]), 512)
        self.assertLess(provider.seen_sizes[0][1], 460)
        self.assertEqual(record["ocr"]["processing_mode"], "fast")
        self.assertEqual(record["ocr"]["preprocess"]["attempts"][0]["strategy"], "stnk_fast_roi")

    def test_ktp_fast_mode_uses_smaller_prepared_image_than_accurate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "ktp.jpg"
            Image.new("RGB", (1800, 1200), "white").save(image_path)
            fast_provider = SizeRecordingProvider()
            accurate_provider = SizeRecordingProvider()

            fast_record = _process_file(fast_provider, image_path, "KTP", mode="fast")
            accurate_record = _process_file(accurate_provider, image_path, "KTP", mode="accurate")

        self.assertEqual(max(fast_provider.seen_sizes[0]), 960)
        self.assertEqual(max(accurate_provider.seen_sizes[0]), 1280)
        self.assertEqual(fast_record["ocr"]["processing_mode"], "fast")
        self.assertEqual(accurate_record["ocr"]["processing_mode"], "accurate")

    def test_ktp_pipeline_applies_layout_hints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "ktp.jpg"
            _write_textured_image(image_path)

            record = _process_file(KtpLayoutProvider(), image_path, "KTP", mode="fast")

        self.assertEqual(record["fields"]["kewarganegaraan"]["value"], "WNI")
        self.assertEqual(record["fields"]["kewarganegaraan"]["status"], "ok")

    def test_stnk_fast_mode_returns_initial_roi_without_sync_full_page_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "stnk.jpg"
            _write_textured_image(image_path, size=(1800, 1200))
            provider = AdaptiveStnkProvider()

            record = _process_file(provider, image_path, "STNK", mode="fast")

        self.assertEqual([max(size) for size in provider.seen_sizes], [512])
        self.assertEqual(record["ocr"]["preprocess"]["attempts"][0]["strategy"], "stnk_fast_roi")
        self.assertEqual(record["ocr"]["preprocess"]["selected_max_side"], 512)
        self.assertEqual(record["ocr"]["preprocess"]["retry_count"], 0)
        self.assertEqual(record["input_assessment"]["decision"], "needs_review")
        self.assertTrue(record["needs_review"])

    def test_stnk_retries_highres_when_fast_pass_is_missing_required_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "stnk.jpg"
            _write_textured_image(image_path, size=(1800, 1200))
            provider = AdaptiveStnkProvider()

            record = _process_file(provider, image_path, "STNK")

        self.assertEqual([max(size) for size in provider.seen_sizes], [1200, 1280])
        self.assertEqual(record["fields"]["tahun_pembuatan"]["value"], "2020")
        self.assertEqual(record["fields"]["nomor_rangka"]["value"], "MHRRU1860KJ302319")
        self.assertEqual(record["fields"]["nomor_mesin"]["value"], "L15Z61219016")
        self.assertEqual(record["input_assessment"]["decision"], "approved_for_auto")
        self.assertEqual(record["ocr"]["preprocess"]["selected_max_side"], 1280)
        self.assertEqual(record["ocr"]["preprocess"]["retry_count"], 1)

    def test_stnk_does_not_retry_when_fast_pass_is_already_approved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "stnk.jpg"
            _write_textured_image(image_path, size=(1800, 1200))
            provider = CompleteStnkProvider()

            record = _process_file(provider, image_path, "STNK")

        self.assertEqual([max(size) for size in provider.seen_sizes], [1200])
        self.assertEqual(record["input_assessment"]["decision"], "approved_for_auto")
        self.assertEqual(record["ocr"]["preprocess"]["selected_max_side"], 1200)
        self.assertEqual(record["ocr"]["preprocess"]["retry_count"], 0)

    def test_stnk_does_not_retry_highres_when_source_has_minimal_resolution_headroom(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "stnk.jpg"
            _write_textured_image(image_path, size=(1280, 720))
            provider = AdaptiveStnkProvider()

            record = _process_file(provider, image_path, "STNK")

        self.assertEqual([max(size) for size in provider.seen_sizes], [1200])
        self.assertEqual(record["ocr"]["preprocess"]["retry_count"], 0)


if __name__ == "__main__":
    unittest.main()
