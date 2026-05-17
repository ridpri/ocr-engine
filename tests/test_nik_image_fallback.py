import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_engine.nik_image_fallback import repair_ktp_nik_from_image
from ocr_engine.ocr.base import OcrResult, OcrToken
from ocr_engine.parsers.ktp import parse_ktp_text


class FakeVariantProvider:
    def __init__(self, responses: dict[str, OcrResult]) -> None:
        self.responses = responses
        self.paths: list[str] = []

    def extract_text(self, image_path: str) -> OcrResult:
        name = Path(image_path).name
        self.paths.append(name)
        for key, response in self.responses.items():
            if key in name:
                return response
        return OcrResult(raw_text="", tokens=[], provider="fake")


class NikImageFallbackTests(unittest.TestCase):
    def test_repair_ktp_nik_from_rotated_variant(self):
        parsed = parse_ktp_text(
            """
            NTK
            117305500386001
            Nama : YOKHEBED SETIOWATISANTOSO
            SLEMAN, 10-03-1986
            Jenis Kelamin : PEREMPUAN
            Alamat : JL KEDOYA
            """
        )
        provider = FakeVariantProvider(
            {
                "rot90_top45": OcrResult(
                    raw_text="PROVINSI DKI JAKARTA\nNIK\n3173055003860011",
                    tokens=[
                        OcrToken("NIK", 0.99),
                        OcrToken("3173055003860011", 0.96),
                    ],
                    provider="fake",
                )
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = _make_sample_image(Path(tmpdir) / "input.jpg")
            meta = repair_ktp_nik_from_image(provider, image_path, parsed, Path(tmpdir) / "fallback")

        self.assertTrue(meta["attempted"])
        self.assertEqual(parsed.fields["nik"].value, "3173055003860011")
        self.assertEqual(parsed.fields["nik"].status, "ok")
        self.assertNotIn("invalid:nik", parsed.warnings)
        self.assertTrue(any("rot90_top45" in path for path in provider.paths))

    def test_repair_ktp_nik_rejects_unrelated_16_digit_candidate(self):
        parsed = parse_ktp_text(
            """
            NIK
            117305500386001
            Nama : YOKHEBED SETIOWATISANTOSO
            SLEMAN, 10-03-1986
            Jenis Kelamin : PEREMPUAN
            Alamat : JL KEDOYA
            """
        )
        provider = FakeVariantProvider(
            {
                "rot90_top45": OcrResult(
                    raw_text="VALID UNTUK NOMOR LAIN\n9999999999999999",
                    tokens=[OcrToken("9999999999999999", 0.99)],
                    provider="fake",
                )
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = _make_sample_image(Path(tmpdir) / "input.jpg")
            meta = repair_ktp_nik_from_image(provider, image_path, parsed, Path(tmpdir) / "fallback")

        self.assertTrue(meta["attempted"])
        self.assertEqual(parsed.fields["nik"].status, "invalid")
        self.assertIn("invalid:nik", parsed.warnings)


def _make_sample_image(path: Path) -> Path:
    Image.new("RGB", (320, 200), "white").save(path)
    return path


if __name__ == "__main__":
    unittest.main()
