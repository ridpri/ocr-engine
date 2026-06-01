import io
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_engine.ocr.base import OcrResult, OcrToken
from ocr_engine.postal_code import PostalCodeMatch


class FakeProvider:
    calls: list[str] = []

    def extract_text(self, image_path: str) -> OcrResult:
        self.calls.append(image_path)
        return OcrResult(
            raw_text="\n".join(
                [
                    "PROVINSI DKI JAKARTA",
                    "NIK : 3175010101900001",
                    "Nama : BUDI SANTOSO",
                    "Tempat/Tgl Lahir : JAKARTA, 01-01-1990",
                    "Alamat : JL MERDEKA",
                    "RT/RW : 001/002",
                    "Kel/Desa : GAMBIR",
                    "Kecamatan : GAMBIR",
                    "Jenis Kelamin : LAKI-LAKI",
                    "Agama : ISLAM",
                    "Status Perkawinan : KAWIN",
                    "Pekerjaan : KARYAWAN SWASTA",
                    "Kewarganegaraan : WNI",
                    "Berlaku Hingga : SEUMUR HIDUP",
                ]
            ),
            tokens=[OcrToken("KTP", 0.99)],
            provider="fake",
        )


class StnkProvider:
    def extract_text(self, image_path: str) -> OcrResult:
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


class SlowStnkProvider:
    calls = 0

    def extract_text(self, image_path: str) -> OcrResult:
        self.calls += 1
        time.sleep(0.05)
        return OcrResult(raw_text="SURAT TANDA NOMOR KENDARAAN", tokens=[OcrToken("STNK", 0.99)], provider="fake")


class SlowKtpProvider:
    def extract_text(self, image_path: str) -> OcrResult:
        time.sleep(0.05)
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
        return OcrResult(raw_text=raw_text, tokens=[OcrToken(text, 0.99) for text in raw_text.split()], provider="fake")


class BlockingKtpProvider(SlowKtpProvider):
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def extract_text(self, image_path: str) -> OcrResult:
        self.entered.set()
        self.release.wait(timeout=2)
        return super().extract_text(image_path)


class PartialStnkProvider:
    def extract_text(self, image_path: str) -> OcrResult:
        return OcrResult(
            raw_text="\n".join(
                [
                    "SURAT TANDA NOMOR KENDARAAN",
                    "NO RANGKA : JT7X2RB80J7008922",
                    "NO MESIN : 0660U41260418",
                ]
            ),
            tokens=[OcrToken("STNK", 0.99)],
            provider="fake",
        )


class MissingNikKtpProvider:
    def extract_text(self, image_path: str) -> OcrResult:
        return OcrResult(
            raw_text="\n".join(
                [
                    "PROVINSI DKI JAKARTA",
                    "Nama : BUDI SANTOSO",
                    "Alamat : JL MERDEKA",
                    "RT/RW : 001/002",
                    "Kel/Desa : GAMBIR",
                    "Kecamatan : GAMBIR",
                ]
            ),
            tokens=[OcrToken("KTP", 0.99)],
            provider="fake",
        )


class MissingTtlKtpProvider:
    def extract_text(self, image_path: str) -> OcrResult:
        return OcrResult(
            raw_text="\n".join(
                [
                    "PROVINSI DKI JAKARTA",
                    "NIK : 3175010101900001",
                    "Nama : BUDI SANTOSO",
                    "Jenis Kelamin : LAKI-LAKI",
                    "Alamat : JL MERDEKA",
                    "RT/RW : 001/002",
                    "Kel/Desa : GAMBIR",
                    "Kecamatan : GAMBIR",
                ]
            ),
            tokens=[OcrToken("KTP", 0.99)],
            provider="fake",
        )


class MedanKtpProvider:
    def extract_text(self, image_path: str) -> OcrResult:
        return OcrResult(
            raw_text="\n".join(
                [
                    "PROVINSI SUMATERA UTARA",
                    "KOTA MEDAN",
                    "NIK : 1271184101900001",
                    "Nama : GRESCILIA SIANTURI, SE",
                    "Tempat/Tgl Lahir : MEDAN, 07-01-1990",
                    "Jenis Kelamin : PEREMPUAN",
                    "Alamat : JLN. BUNGA SEDAP MALAMIX KOMP.PERUM DIAMON RESORT NO.C-",
                    "RT/RW : 000/000",
                    "Kel/Desa : SEMPAKATA",
                    "Kecamatan : MEDAN SELAYANG",
                    "Agama : KRISTEN",
                    "Status Perkawinan : KAWIN",
                    "Pekerjaan : MENGURUS RUMAH TANGGA",
                    "Kewarganegaraan : WNI",
                    "Berlaku Hingga : SEUMUR HIDUP",
                ]
            ),
            tokens=[OcrToken("KTP", 0.99)],
            provider="fake",
        )


class ApiEndpointTests(unittest.TestCase):
    def test_stnk_fast_default_checkout_budget_is_twelve_seconds(self):
        from ocr_engine.api import STNK_FAST_RESPONSE_TIMEOUT_SECONDS

        self.assertEqual(STNK_FAST_RESPONSE_TIMEOUT_SECONDS, 12)

    def test_ocr_endpoint_requires_api_key_when_configured(self):
        client = _client_with_fake_provider()

        with patch.dict("os.environ", {"OCR_API_KEY": "secret"}, clear=False):
            response = client.post("/ocr/ktp", files={"file": ("ktp.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 401)
        self.assertIn("API key", response.json()["detail"])

    def test_ocr_endpoint_accepts_configured_api_key(self):
        client = _client_with_fake_provider()

        with patch.dict("os.environ", {"OCR_API_KEY": "secret"}, clear=False):
            response = client.post(
                "/ocr/ktp",
                files={"file": ("ktp.jpg", _jpeg_bytes(), "image/jpeg")},
                headers={"X-API-Key": "secret"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["document_type"], "KTP")

    def test_vps_non_json_http_error_returns_json_detail(self):
        from ocr_engine.api import _post_file_to_vps

        class FakeHttpError(urllib.error.HTTPError):
            def read(self):
                return b"The page could not be found"

        def fake_urlopen(request, timeout):
            raise FakeHttpError(request.full_url, 404, "Not Found", {}, None)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            status, payload = _post_file_to_vps("/ocr/ktp?mode=fast", "ktp.jpg", "image/jpeg", b"image", "secret")

        self.assertEqual(status, 404)
        self.assertIn("non-JSON", payload["detail"])
        self.assertIn("/ocr/ktp?mode=fast", payload["detail"])

    def test_health_endpoint_does_not_require_api_key(self):
        client = _client_with_fake_provider()

        with patch.dict("os.environ", {"OCR_API_KEY": "secret"}, clear=False):
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_fixed_ktp_endpoint_returns_expected_document_type(self):
        client = _client_with_fake_provider()

        response = client.post("/ocr/ktp", files={"file": ("ktp.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["input_assessment"]["expected_document_type"], "KTP")
        self.assertEqual(payload["document_type"], "KTP")
        self.assertIn("timings", payload["ocr"])
        self.assertGreaterEqual(payload["ocr"]["timings"]["total_ms"], 0)
        self.assertEqual(len(payload["ocr"]["timings"]["attempts"]), 1)
        self.assertEqual(payload["ocr"]["processing_mode"], "fast")

    def test_fixed_ktp_endpoint_still_accepts_explicit_accurate_mode(self):
        client = _client_with_fake_provider()

        response = client.post("/ocr/ktp?mode=accurate", files={"file": ("ktp.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ocr"]["processing_mode"], "accurate")

    def test_fixed_ktp_endpoint_accepts_pdf_by_rendering_first_page(self):
        client = _client_with_fake_provider()

        def fake_render_pdf_first_page(input_path, output_path, dpi=200):
            Image.new("RGB", (900, 600), "white").save(output_path, format="PNG")
            return Path(output_path)

        with patch("ocr_engine.api.render_pdf_first_page", side_effect=fake_render_pdf_first_page):
            response = client.post("/ocr/ktp", files={"file": ("ktp.pdf", b"%PDF-1.7", "application/pdf")})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["document_type"], "KTP")
        self.assertEqual(payload["input_assessment"]["expected_document_type"], "KTP")

    def test_fixed_stnk_endpoint_rejects_when_uploaded_content_is_ktp(self):
        client = _client_with_fake_provider()

        response = client.post("/ocr/stnk", files={"file": ("stnk.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["input_assessment"]["expected_document_type"], "STNK")
        self.assertEqual(payload["input_assessment"]["decision"], "rejected_input")
        self.assertIn("document_type_mismatch", payload["input_assessment"]["reason_codes"])
        self.assertEqual(payload["ocr"]["processing_mode"], "accurate")
        self.assertEqual(payload["enrichment"]["status"], "not_requested")

    def test_fixed_stnk_endpoint_queues_background_enrichment_when_fast_result_needs_review(self):
        client = _client_with_provider(PartialStnkProvider())

        response = client.post("/ocr/stnk?mode=fast", files={"file": ("stnk.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["ocr"]["processing_mode"], "fast")
        self.assertTrue(payload["needs_review"])
        self.assertEqual(payload["enrichment"]["status"], "queued")
        self.assertEqual(payload["enrichment"]["mode"], "accurate_background")

    def test_fixed_stnk_endpoint_queues_background_enrichment_for_valid_stnk_fast_response(self):
        client = _client_with_provider(StnkProvider())

        response = client.post("/ocr/stnk?mode=fast&enrich=true", files={"file": ("stnk.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertNotEqual(payload["input_assessment"]["decision"], "rejected_input")
        self.assertEqual(payload["ocr"]["processing_mode"], "fast")
        self.assertEqual(payload["enrichment"]["status"], "queued")
        self.assertEqual(payload["enrichment"]["mode"], "accurate_background")

    def test_enrichment_status_endpoint_returns_completed_job_result(self):
        client = _client_with_provider(StnkProvider())
        with patch("ocr_engine.api.BACKGROUND_OCR_START_DELAY_SECONDS", 0):
            response = client.post("/ocr/stnk?mode=fast&enrich=true", files={"file": ("stnk.jpg", _jpeg_bytes(), "image/jpeg")})
        job_id = response.json()["enrichment"]["job_id"]

        payload = None
        for _ in range(20):
            status_response = client.get(f"/ocr/enrichment/{job_id}")
            payload = status_response.json()
            if payload["status"] == "completed":
                break
            time.sleep(0.05)

        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["result"]["status"], "ok")
        self.assertEqual(payload["result"]["ocr"]["processing_mode"], "accurate")
        self.assertGreaterEqual(payload["result"]["stnk_structure_score"], 0.7)
        self.assertIn(payload["result"]["stnk_usage_class"], {"web_usable", "internal_only", "bad_input"})

    def test_enrichment_status_endpoint_handles_corrupt_job_json(self):
        client = _client_with_provider(StnkProvider())
        job_id = "deadbeefdeadbeefdeadbeefdeadbeef"
        jobs_dir = Path("tmp") / "stnk_enrichment"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        output_path = jobs_dir / f"{job_id}.json"
        output_path.write_text("{not-json", encoding="utf-8")

        try:
            response = client.get(f"/ocr/enrichment/{job_id}")
        finally:
            output_path.unlink(missing_ok=True)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["job_id"], job_id)
        self.assertEqual(payload["error"], "corrupt_enrichment_result")

    def test_stnk_fast_endpoint_returns_timeout_payload_when_checkout_budget_is_exceeded(self):
        client = _client_with_provider(SlowStnkProvider())

        with patch("ocr_engine.api.STNK_FAST_RESPONSE_TIMEOUT_SECONDS", 0.001):
            response = client.post("/ocr/stnk?mode=fast", files={"file": ("stnk.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["needs_review"])
        self.assertEqual(payload["input_assessment"]["reason_codes"], ["processing_timeout"])
        self.assertEqual(payload["ocr"]["processing_mode"], "fast")
        self.assertEqual(payload["enrichment"]["status"], "queued")
        self.assertEqual(payload["enrichment"]["mode"], "fast_background")
        self.assertRegex(payload["enrichment"]["job_id"], r"^[a-f0-9]{32}$")

    def test_ktp_fast_endpoint_waits_without_response_timeout(self):
        client = _client_with_provider(SlowKtpProvider())

        response = client.post("/ocr/ktp?mode=fast", files={"file": ("ktp.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["document_type"], "KTP")
        self.assertNotIn("processing_timeout", payload["input_assessment"]["reason_codes"])
        self.assertEqual(payload["ocr"]["processing_mode"], "fast")
        self.assertEqual(payload["enrichment"]["status"], "not_requested")

    def test_ocr_endpoint_returns_busy_when_request_queue_is_full(self):
        provider = BlockingKtpProvider()
        with patch("ocr_engine.api.OCR_REQUEST_MAX_WORKERS", 1, create=True):
            with patch("ocr_engine.api.OCR_REQUEST_MAX_PENDING", 1, create=True):
                client = _client_with_provider(provider)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(client.post, "/ocr/ktp", files={"file": ("ktp.jpg", _jpeg_bytes(), "image/jpeg")})
            self.assertTrue(provider.entered.wait(timeout=1))
            second_response = client.post("/ocr/ktp", files={"file": ("ktp.jpg", _jpeg_bytes(), "image/jpeg")})
            provider.release.set()
            first_response = first.result(timeout=3)

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 503)
        self.assertEqual(second_response.json()["detail"], "OCR engine is busy. Please retry shortly.")

    def test_timed_out_stnk_fast_request_can_be_polled_without_reupload(self):
        provider = SlowStnkProvider()
        client = _client_with_provider(provider)

        with patch("ocr_engine.api.STNK_FAST_RESPONSE_TIMEOUT_SECONDS", 0.001):
            response = client.post("/ocr/stnk?mode=fast", files={"file": ("stnk.jpg", _jpeg_bytes(), "image/jpeg")})
        job_id = response.json()["enrichment"]["job_id"]

        payload = None
        for _ in range(20):
            status_response = client.get(f"/ocr/enrichment/{job_id}")
            payload = status_response.json()
            if payload["status"] == "completed":
                break
            time.sleep(0.05)

        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["result"]["status"], "ok")
        self.assertEqual(payload["result"]["ocr"]["processing_mode"], "fast")
        self.assertEqual(provider.calls, 1)

    def test_purchase_ktp_endpoint_returns_only_checkout_fields(self):
        client = _client_with_fake_provider()

        response = client.post("/ocr/purchase/ktp", files={"file": ("ktp.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["purpose"], "purchase_checkout")
        self.assertEqual(payload["document_type"], "KTP")
        self.assertEqual(set(payload["fields"].keys()), {"nik", "nama", "alamat", "kode_pos"})
        self.assertEqual(payload["fields"]["nik"]["value"], "3175010101900001")
        self.assertEqual(payload["fields"]["nama"]["value"], "BUDI SANTOSO")
        self.assertIn(payload["fields"]["kode_pos"]["status"], {"ok", "missing"})
        self.assertEqual(payload["ocr"]["processing_mode"], "fast")
        self.assertNotIn("raw_text", payload)
        self.assertEqual(payload["background_full_ocr"]["status"], "not_requested")

    def test_purchase_ktp_endpoint_does_not_wait_for_nik_image_fallback(self):
        client = _client_with_provider(MissingNikKtpProvider())

        response = client.post("/ocr/purchase/ktp", files={"file": ("ktp.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["purpose"], "purchase_checkout")
        self.assertEqual(payload["fields"]["nik"]["status"], "missing")
        self.assertFalse(payload["ocr"]["nik_fallback"]["attempted"])
        self.assertEqual(payload["background_full_ocr"]["status"], "not_requested")

    def test_purchase_ktp_endpoint_waits_without_response_timeout(self):
        client = _client_with_provider(SlowKtpProvider())

        response = client.post("/ocr/purchase/ktp", files={"file": ("ktp.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["purpose"], "purchase_checkout")
        self.assertEqual(payload["document_type"], "KTP")
        self.assertNotIn("processing_timeout", payload["input_assessment"]["reason_codes"])
        self.assertEqual(payload["ocr"]["processing_mode"], "fast")
        self.assertEqual(payload["background_full_ocr"]["status"], "not_requested")

    def test_purchase_ktp_endpoint_is_not_ready_when_auto_contract_needs_review(self):
        client = _client_with_provider(MissingTtlKtpProvider())

        response = client.post("/ocr/purchase/ktp", files={"file": ("ktp.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["ready_for_checkout"])
        self.assertTrue(payload["needs_review"])
        self.assertEqual(payload["input_assessment"]["decision"], "needs_review")
        self.assertIn("ktp_auto_missing:tempat_tanggal_lahir", payload["input_assessment"]["reason_codes"])

    def test_ktp_endpoint_includes_ti_compatible_postal_code_payload(self):
        client = _client_with_provider(MedanKtpProvider())
        postal_match = PostalCodeMatch(
            kode_pos="20132",
            confidence=0.95,
            evidence=["kelurahan_desa:Sempakata"],
            kelurahan="Sempakata",
            kecamatan="MEDAN SELAYANG",
            kode_kecamatan="6750",
            kode_kota="22663",
            nama_kota="MEDAN",
            kode_provinsi="33",
            nama_provinsi="SUMATERA UTARA",
            alamat_lengkap="Sempakata, Medan Selayang, Kota Medan, Sumatera Utara",
            total_options=1,
            match_status="exact_match",
        )

        with patch("ocr_engine.parsers.ktp.lookup_postal_code", return_value=postal_match):
            response = client.post("/ocr/ktp", files={"file": ("ktp.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        compatible = payload["ti_compatible"]
        self.assertEqual(compatible["status"], "success")
        self.assertEqual(compatible["message"], "Kode pos successfully retrieved from KTP")
        self.assertEqual(compatible["ocr_data"]["nama"], "GRESCILIA SIANTURI, SE")
        self.assertEqual(compatible["ocr_data"]["no_ktp"], "1271184101900001")
        self.assertEqual(compatible["ocr_data"]["tempat_lahir"], "MEDAN")
        self.assertEqual(compatible["ocr_data"]["tanggal_lahir"], "1990-01-07T00:00:00.000Z")
        self.assertEqual(compatible["ocr_data"]["kodeKota"], "22663")
        self.assertEqual(compatible["ocr_data"]["kodeKecamatan"], "6750")
        self.assertEqual(compatible["ocr_data"]["kodeProvinsi"], "33")
        self.assertEqual(compatible["ocr_data"]["status_perkawinan"], "K")
        self.assertEqual(compatible["ocr_data"]["jenis_kelamin"], "P")
        self.assertEqual(compatible["kodepos_data"]["kode_pos"], "20132")
        self.assertEqual(compatible["kodepos_data"]["kode_kecamatan"], 6750)
        self.assertEqual(compatible["kodepos_data"]["kode_kota"], 22663)
        self.assertEqual(compatible["kodepos_data"]["total_options"], 1)
        self.assertEqual(compatible["kodepos_data"]["match_status"], "exact_match")

    def test_purchase_stnk_endpoint_queues_ocr_without_sync_wait(self):
        client = _client_with_provider(StnkProvider())

        with patch("ocr_engine.api.STNK_PURCHASE_BACKGROUND_START_DELAY_SECONDS", 999):
            response = client.post("/ocr/purchase/stnk", files={"file": ("stnk.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["purpose"], "purchase_checkout")
        self.assertEqual(payload["document_type"], "STNK")
        self.assertEqual(
            set(payload["fields"].keys()),
            {"nomor_polisi", "nama_pemilik", "nomor_rangka", "nomor_mesin"},
        )
        self.assertEqual(payload["fields"]["nomor_polisi"]["status"], "missing")
        self.assertEqual(payload["ocr"]["processing_mode"], "fast")
        self.assertEqual(payload["ocr"]["provider"], "queued")
        self.assertEqual(payload["input_assessment"]["reason_codes"], ["ocr_queued"])
        self.assertEqual(payload["background_full_ocr"]["status"], "queued")
        self.assertEqual(payload["background_full_ocr"]["mode"], "fast_background")
        self.assertNotIn("tahun_pembuatan", payload["fields"])
        self.assertNotIn("raw_text", payload)

    def test_purchase_stnk_endpoint_is_not_ready_while_ocr_is_queued(self):
        client = _client_with_provider(StnkProvider())

        with patch("ocr_engine.api.STNK_PURCHASE_BACKGROUND_START_DELAY_SECONDS", 999):
            response = client.post("/ocr/purchase/stnk", files={"file": ("stnk.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["ready_for_checkout"])
        self.assertTrue(payload["needs_review"])
        self.assertIsNone(payload["stnk_usage_class"])
        self.assertEqual(payload["stnk_usage_reasons"], [])
        self.assertEqual(payload["input_assessment"]["decision"], "needs_review")
        self.assertFalse(payload["input_assessment"]["can_auto_publish"])
        self.assertIn("ocr_queued", payload["input_assessment"]["reason_codes"])

    def test_purchase_stnk_endpoint_queues_fast_background_for_field_extraction(self):
        client = _client_with_provider(PartialStnkProvider())

        with patch("ocr_engine.api.STNK_PURCHASE_BACKGROUND_START_DELAY_SECONDS", 999):
            response = client.post("/ocr/purchase/stnk", files={"file": ("stnk.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["ocr"]["processing_mode"], "fast")
        self.assertEqual(payload["fields"]["nomor_rangka"]["status"], "missing")
        self.assertEqual(payload["fields"]["nomor_polisi"]["status"], "missing")
        self.assertEqual(payload["background_full_ocr"]["status"], "queued")
        self.assertEqual(payload["background_full_ocr"]["mode"], "fast_background")
        self.assertRegex(payload["background_full_ocr"]["job_id"], r"^[a-f0-9]{32}$")

    def test_purchase_stnk_endpoint_does_not_call_slow_ocr_before_responding(self):
        provider = SlowStnkProvider()
        client = _client_with_provider(provider)

        with patch("ocr_engine.api.STNK_PURCHASE_BACKGROUND_START_DELAY_SECONDS", 999):
            response = client.post("/ocr/purchase/stnk", files={"file": ("stnk.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["purpose"], "purchase_checkout")
        self.assertFalse(payload["ready_for_checkout"])
        self.assertTrue(payload["needs_review"])
        self.assertEqual(payload["input_assessment"]["reason_codes"], ["ocr_queued"])
        self.assertEqual(payload["ocr"]["processing_mode"], "fast")
        self.assertEqual(payload["background_full_ocr"]["status"], "queued")
        self.assertEqual(payload["background_full_ocr"]["mode"], "fast_background")
        self.assertEqual(provider.calls, 0)

    def test_purchase_stnk_endpoint_reports_busy_when_background_queue_is_full(self):
        with patch("ocr_engine.api.BACKGROUND_OCR_MAX_WORKERS", 1, create=True):
            with patch("ocr_engine.api.BACKGROUND_OCR_MAX_PENDING", 1, create=True):
                client = _client_with_provider(StnkProvider())

        with patch("ocr_engine.api.STNK_PURCHASE_BACKGROUND_START_DELAY_SECONDS", 999):
            first_response = client.post("/ocr/purchase/stnk", files={"file": ("stnk.jpg", _jpeg_bytes(), "image/jpeg")})
            second_response = client.post("/ocr/purchase/stnk", files={"file": ("stnk.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(first_response.json()["background_full_ocr"]["status"], "queued")
        self.assertEqual(second_response.status_code, 200)
        self.assertEqual(second_response.json()["background_full_ocr"]["status"], "busy")
        self.assertEqual(second_response.json()["background_full_ocr"]["reason"], "background_queue_full")

    def test_legacy_ocr_endpoint_still_accepts_document_type_query(self):
        client = _client_with_fake_provider()

        response = client.post("/ocr?document_type=KTP", files={"file": ("ktp.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["input_assessment"]["expected_document_type"], "KTP")

    def test_agent_ocr_endpoint_returns_503_when_bridge_is_not_configured(self):
        client = _client_with_fake_provider()

        with patch.dict("os.environ", {}, clear=True):
            response = client.post("/ocr/agent/ktp", files={"file": ("ktp.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 503)
        self.assertIn("not configured", response.json()["detail"])

    def test_agent_ocr_endpoint_runs_configured_command_bridge(self):
        client = _client_with_fake_provider()

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "agent_bridge.py"
            script_path.write_text(
                "import json, sys\n"
                "json.load(sys.stdin)\n"
                "print(json.dumps({"
                "'document_type':'KTP',"
                "'raw_text':'NIK 3175010101900001 Nama BUDI SANTOSO Tempat/Tgl Lahir JAKARTA, 01-01-1990 Alamat JL MERDEKA Pekerjaan KARYAWAN SWASTA Kewarganegaraan WNI Berlaku Hingga SEUMUR HIDUP',"
                "'warnings':['Soft model note that should not block auto approval'],"
                "'fields':{"
                "'nik':{'value':'3175010101900001','confidence':0.99,'status':'confirmed','evidence':'NIK 3175010101900001'},"
                "'nama':{'value':'BUDI SANTOSO','confidence':0.95,'status':'confirmed'},"
                "'tempat_tanggal_lahir':{'value':'JAKARTA, 01-01-1990','confidence':0.95,'status':'confirmed'},"
                "'jenis_kelamin':{'value':'LAKI-LAKI','confidence':0.95,'status':'confirmed'},"
                "'alamat':{'value':'JL MERDEKA','confidence':0.95,'status':'confirmed'},"
                "'rt_rw':{'value':'001/002','confidence':0.95,'status':'confirmed'},"
                "'kelurahan_desa':{'value':'GAMBIR','confidence':0.95,'status':'confirmed'},"
                "'kecamatan':{'value':'GAMBIR','confidence':0.95,'status':'confirmed'},"
                "'pekerjaan':{'value':'KARYAWAN SWASTA','confidence':0.95,'status':'confirmed'},"
                "'kewarganegaraan':{'value':'WNI','confidence':0.95,'status':'confirmed'},"
                "'berlaku_hingga':{'value':'SEUMUR HIDUP','confidence':0.95,'status':'confirmed'}"
                "}}))\n",
                encoding="utf-8",
            )
            command = f'"{sys.executable}" "{script_path}"'

            with patch.dict("os.environ", {"OCR_AGENT_COMMAND": command}):
                response = client.post("/ocr/agent/ktp", files={"file": ("ktp.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["document_type"], "KTP")
        self.assertEqual(payload["fields"]["nik"]["value"], "3175010101900001")
        self.assertEqual(payload["fields"]["nik"]["status"], "ok")
        self.assertEqual(payload["fields"]["nik"]["evidence"], ["NIK 3175010101900001"])
        self.assertEqual(payload["warnings"], [])
        self.assertEqual(payload["agent"]["provider"], "command")
        self.assertEqual(payload["input_assessment"]["decision"], "approved_for_auto")

    def test_agent_stnk_endpoint_revalidates_invalid_plate_from_command_bridge(self):
        client = _client_with_fake_provider()

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "agent_bridge.py"
            script_path.write_text(
                "import json, sys\n"
                "json.load(sys.stdin)\n"
                "fields = {name:{'value':'OK','confidence':0.95,'status':'ok'} for name in ["
                "'nama_pemilik','alamat','merek','tipe','jenis','warna','bahan_bakar']}\n"
                "fields.update({"
                "'nomor_polisi':{'value':'N 0 P','confidence':0.99,'status':'ok'},"
                "'tahun_pembuatan':{'value':'2018','confidence':0.99,'status':'ok'},"
                "'nomor_rangka':{'value':'MHH8A9609JK957076','confidence':0.99,'status':'ok'},"
                "'nomor_mesin':{'value':'F0653123','confidence':0.99,'status':'ok'},"
                "'berlaku_sampai':{'value':'10-11-2025','confidence':0.99,'status':'ok'}"
                "})\n"
                "print(json.dumps({'document_type':'STNK','raw_text':'NO POLISI N 0 P', 'fields': fields}))\n",
                encoding="utf-8",
            )
            command = f'"{sys.executable}" "{script_path}"'

            with patch.dict("os.environ", {"OCR_AGENT_COMMAND": command}):
                response = client.post("/ocr/agent/stnk", files={"file": ("stnk.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["fields"]["nomor_polisi"]["value"], "N 0 P")
        self.assertEqual(payload["fields"]["nomor_polisi"]["status"], "invalid")
        self.assertIn("invalid:nomor_polisi", payload["warnings"])
        self.assertEqual(payload["input_assessment"]["decision"], "needs_review")
        self.assertFalse(payload["input_assessment"]["can_auto_publish"])


def _client_with_fake_provider() -> TestClient:
    FakeProvider.calls = []
    return _client_with_provider(FakeProvider())


def _client_with_provider(provider) -> TestClient:
    with patch("ocr_engine.api.PaddleOcrProvider", return_value=provider):
        from ocr_engine.api import create_app

        return TestClient(create_app())


def _jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    image = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(image)
    for x in range(40, 860, 80):
        draw.line((x, 40, x, 560), fill=(190, 190, 190), width=2)
    for y in range(40, 560, 55):
        draw.line((40, y, 860, y), fill=(120, 120, 120), width=2)
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
