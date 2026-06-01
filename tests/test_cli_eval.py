import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_engine.cli_eval import _collect_paths, _create_provider, _process_file
from ocr_engine.ocr.base import OcrResult, OcrToken


class FakeProvider:
    def extract_text(self, image_path: str) -> OcrResult:
        return OcrResult(
            raw_text=(
                "PROVINSI DKI JAKARTA\n"
                "NIK : 3175010101900001\n"
                "Nama : BUDI SANTOSO\n"
                "Tempat/Tgl Lahir : JAKARTA, 01-01-1990\n"
                "Jenis Kelamin : LAKI-LAKI\n"
                "Alamat : JL MERDEKA\n"
                "RT/RW : 001/002\n"
                "Kel/Desa : GAMBIR\n"
                "Kecamatan : GAMBIR\n"
                "Pekerjaan : KARYAWAN SWASTA\n"
                "Kewarganegaraan : WNI\n"
                "Berlaku Hingga : SEUMUR HIDUP"
            ),
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


class EmptyTextProvider:
    def __init__(self) -> None:
        self.calls = 0

    def extract_text(self, image_path: str) -> OcrResult:
        self.calls += 1
        return OcrResult(raw_text="", tokens=[], provider="fake")


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
    def test_collect_paths_supports_recursive_seeded_sampling_and_skip_pdf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "nested"
            nested.mkdir()
            for filename in ["a.jpg", "b.jpeg", "c.pdf", "notes.txt"]:
                (root / filename).write_bytes(b"x")
            (nested / "d.png").write_bytes(b"x")

            first = _collect_paths(root, 2, recursive=True, include_pdf=False, random_seed=17)
            second = _collect_paths(root, 2, recursive=True, include_pdf=False, random_seed=17)

        self.assertEqual([path.name for path in first], [path.name for path in second])
        self.assertEqual(len(first), 2)
        self.assertTrue(all(path.suffix.lower() != ".pdf" for path in first))
        self.assertTrue(all(path.name in {"a.jpg", "b.jpeg", "d.png"} for path in first))

    def test_collect_paths_can_skip_single_pdf_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.7")

            paths = _collect_paths(pdf_path, 5, include_pdf=False)

        self.assertEqual(paths, [])

    def test_create_provider_normalizes_provider_name(self):
        with unittest.mock.patch("ocr_engine.cli_eval.RapidOcrProvider", return_value="rapid") as rapid_provider:
            provider = _create_provider(" Rapid ")

        self.assertEqual(provider, "rapid")
        rapid_provider.assert_called_once_with()

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

    def test_stnk_accurate_uses_official_section_roi_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "stnk.jpg"
            Image.new("RGB", (1800, 1200), "white").save(image_path)
            provider = SizeRecordingProvider()

            record = _process_file(provider, image_path, "STNK")

        self.assertEqual(max(provider.seen_sizes[0]), 1200)
        self.assertLess(provider.seen_sizes[0][1], 400)
        self.assertEqual(record["ocr"]["preprocess"]["attempts"][0]["strategy"], "stnk_official_roi")

    def test_stnk_fast_mode_uses_smaller_roi_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "stnk.jpg"
            _write_textured_image(image_path, size=(1800, 1200))
            provider = SizeRecordingProvider()

            record = _process_file(provider, image_path, "STNK", mode="fast")

        self.assertEqual(provider.seen_sizes[0], (561, 422))
        self.assertEqual(record["ocr"]["processing_mode"], "fast")
        self.assertEqual(record["ocr"]["preprocess"]["attempts"][0]["strategy"], "stnk_fast_roi")

    def test_ktp_fast_mode_uses_smaller_prepared_image_than_accurate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "ktp.jpg"
            _write_textured_image(image_path, size=(1800, 1200))
            fast_provider = SizeRecordingProvider()
            accurate_provider = SizeRecordingProvider()

            fast_record = _process_file(fast_provider, image_path, "KTP", mode="fast")
            accurate_record = _process_file(accurate_provider, image_path, "KTP", mode="accurate")

        self.assertEqual(max(fast_provider.seen_sizes[0]), 496)
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

    def test_ktp_empty_ocr_does_not_run_expensive_nik_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "ktp.jpg"
            _write_textured_image(image_path)
            provider = EmptyTextProvider()

            record = _process_file(provider, image_path, "KTP", mode="fast")

        self.assertEqual(record["input_assessment"]["decision"], "rejected_input")
        self.assertFalse(record["ocr"]["nik_fallback"]["attempted"])
        self.assertEqual(provider.calls, 1)

    def test_stnk_fast_mode_returns_initial_roi_without_sync_full_page_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "stnk.jpg"
            _write_textured_image(image_path, size=(1800, 1200))
            provider = AdaptiveStnkProvider()

            record = _process_file(provider, image_path, "STNK", mode="fast")

        self.assertEqual(provider.seen_sizes, [(561, 422)])
        self.assertEqual(record["ocr"]["preprocess"]["attempts"][0]["strategy"], "stnk_fast_roi")
        self.assertEqual(record["ocr"]["preprocess"]["selected_max_side"], 720)
        self.assertEqual(record["ocr"]["preprocess"]["retry_count"], 0)
        self.assertEqual(record["input_assessment"]["decision"], "needs_review")
        self.assertTrue(record["needs_review"])

    def test_stnk_retries_highres_when_fast_pass_is_missing_required_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "stnk.jpg"
            _write_textured_image(image_path, size=(1800, 1200))
            provider = AdaptiveStnkProvider()

            record = _process_file(provider, image_path, "STNK")

        self.assertEqual([max(size) for size in provider.seen_sizes], [1200, 1600])
        self.assertEqual(record["fields"]["tahun_pembuatan"]["value"], "2020")
        self.assertEqual(record["fields"]["nomor_rangka"]["value"], "MHRRU1860KJ302319")
        self.assertEqual(record["fields"]["nomor_mesin"]["value"], "L15Z61219016")
        self.assertEqual(record["input_assessment"]["decision"], "approved_for_auto")
        self.assertEqual(record["ocr"]["preprocess"]["selected_max_side"], 1600)
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

    def test_stnk_retries_enhanced_full_page_even_when_source_has_minimal_resolution_headroom(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "stnk.jpg"
            _write_textured_image(image_path, size=(1280, 720))
            provider = AdaptiveStnkProvider()

            record = _process_file(provider, image_path, "STNK")

        self.assertEqual([max(size) for size in provider.seen_sizes], [1200, 1600])
        self.assertEqual(record["ocr"]["preprocess"]["attempts"][1]["strategy"], "stnk_full_page")
        self.assertEqual(record["ocr"]["preprocess"]["retry_count"], 1)

    def test_process_file_adds_stnk_structure_score_and_usage_class(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "stnk.jpg"
            _write_textured_image(image_path, size=(1800, 1200))

            record = _process_file(CompleteStnkProvider(), image_path, "STNK")

        self.assertGreaterEqual(record["stnk_structure_score"], 0.7)
        self.assertEqual(record["stnk_usage_class"], "web_usable")
        self.assertEqual(record["stnk_usage_reasons"], [])

    def test_process_file_demotes_internal_only_stnk_from_auto_publish(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "stnk.jpg"
            _write_textured_image(image_path, size=(1800, 1200))

            with unittest.mock.patch(
                "ocr_engine.cli_eval.classify_stnk_record",
                return_value=("internal_only", ["processing_time_over_20s"]),
            ):
                record = _process_file(CompleteStnkProvider(), image_path, "STNK")

        self.assertTrue(record["needs_review"])
        self.assertEqual(record["input_assessment"]["decision"], "needs_review")
        self.assertFalse(record["input_assessment"]["can_auto_publish"])
        self.assertIn("stnk_web_usage:processing_time_over_20s", record["input_assessment"]["reason_codes"])


if __name__ == "__main__":
    unittest.main()
