import io
import json
import sys
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

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
        return OcrResult(
            raw_text="\n".join(
                [
                    "SURAT TANDA NOMOR KENDARAAN",
                    "NO POLISI : B 1234 ABC",
                    "NAMA PEMILIK : BUDI SANTOSO",
                    "TAHUN PEMBUATAN : 2020",
                    "NO RANGKA : MHRRU1860KJ302319",
                    "NO MESIN : L15Z61219016",
                ]
            ),
            tokens=[OcrToken("STNK", 0.99)],
            provider="fake",
        )


class SlowStnkProvider:
    calls = 0

    def extract_text(self, image_path: str) -> OcrResult:
        self.calls += 1
        time.sleep(0.05)
        return OcrResult(raw_text="SURAT TANDA NOMOR KENDARAAN", tokens=[OcrToken("STNK", 0.99)], provider="fake")


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
                ]
            ),
            tokens=[OcrToken("KTP", 0.99)],
            provider="fake",
        )


class ApiEndpointTests(unittest.TestCase):
    def test_stnk_fast_default_checkout_budget_is_twenty_seconds(self):
        from ocr_engine.api import STNK_FAST_RESPONSE_TIMEOUT_SECONDS

        self.assertEqual(STNK_FAST_RESPONSE_TIMEOUT_SECONDS, 20)

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
        client = _client_with_provider(StnkProvider())

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

    def test_purchase_stnk_endpoint_returns_checkout_fields_from_accurate_ocr(self):
        client = _client_with_provider(StnkProvider())

        response = client.post("/ocr/purchase/stnk", files={"file": ("stnk.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["purpose"], "purchase_checkout")
        self.assertEqual(payload["document_type"], "STNK")
        self.assertEqual(
            set(payload["fields"].keys()),
            {"nomor_polisi", "nama_pemilik", "nomor_rangka", "nomor_mesin"},
        )
        self.assertEqual(payload["fields"]["nomor_polisi"]["value"], "B 1234 ABC")
        self.assertEqual(payload["ocr"]["processing_mode"], "accurate")
        self.assertEqual(payload["background_full_ocr"]["status"], "not_requested")
        self.assertNotIn("tahun_pembuatan", payload["fields"])
        self.assertNotIn("raw_text", payload)

    def test_purchase_stnk_endpoint_queues_full_page_background_when_required_fields_are_missing(self):
        client = _client_with_provider(PartialStnkProvider())

        response = client.post("/ocr/purchase/stnk", files={"file": ("stnk.jpg", _jpeg_bytes(), "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["ocr"]["processing_mode"], "accurate")
        self.assertEqual(payload["fields"]["nomor_rangka"]["status"], "ok")
        self.assertEqual(payload["fields"]["nomor_polisi"]["status"], "missing")
        self.assertEqual(payload["background_full_ocr"]["status"], "queued")
        self.assertEqual(payload["background_full_ocr"]["mode"], "full_page_background")
        self.assertRegex(payload["background_full_ocr"]["job_id"], r"^[a-f0-9]{32}$")

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
                "'raw_text':'NIK 3175010101900001 Nama BUDI SANTOSO Alamat JL MERDEKA',"
                "'warnings':['Soft model note that should not block auto approval'],"
                "'fields':{"
                "'nik':{'value':'3175010101900001','confidence':0.99,'status':'confirmed','evidence':'NIK 3175010101900001'},"
                "'nama':{'value':'BUDI SANTOSO','confidence':0.95,'status':'confirmed'},"
                "'alamat':{'value':'JL MERDEKA','confidence':0.95,'status':'confirmed'}"
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
    Image.new("RGB", (900, 600), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
