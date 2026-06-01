import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_engine.service import build_input_assessment, choose_parse_document_type, parse_document_text
from ocr_engine.schemas import DocumentResult, FieldResult


class InputAssessmentTests(unittest.TestCase):
    def test_clean_ktp_can_be_approved_for_auto(self):
        raw_text = """
        PROVINSI DKI JAKARTA
        NIK : 3175010101900001
        Nama : BUDI SANTOSO
        Tempat/Tgl Lahir : JAKARTA, 01-01-1990
        Jenis Kelamin : LAKI-LAKI
        Alamat : JL MERDEKA NO 10
        RT/RW : 001/002
        Kel/Desa : GAMBIR
        Kecamatan : GAMBIR
        Pekerjaan : KARYAWAN SWASTA
        Kewarganegaraan : WNI
        Berlaku Hingga : SEUMUR HIDUP
        """

        parse_hint, detected_type = choose_parse_document_type(raw_text, "KTP")
        parsed = parse_document_text(raw_text, document_type_hint=parse_hint)
        assessment = build_input_assessment(raw_text, parsed, "KTP", detected_type)

        self.assertEqual(assessment["decision"], "approved_for_auto")
        self.assertTrue(assessment["can_auto_publish"])
        self.assertEqual(assessment["reason_codes"], [])

    def test_ktp_missing_lower_identity_fields_needs_review(self):
        raw_text = """
        PROVINSI DKI JAKARTA
        NIK : 3175010101900001
        Nama : BUDI SANTOSO
        Tempat/Tgl Lahir : JAKARTA, 01-01-1990
        Jenis Kelamin : LAKI-LAKI
        Alamat : JL MERDEKA NO 10
        RT/RW : 001/002
        Kel/Desa : GAMBIR
        Kecamatan : GAMBIR
        Agama : ISLAM
        Status Perkawinan : KAWIN
        """

        parse_hint, detected_type = choose_parse_document_type(raw_text, "KTP")
        parsed = parse_document_text(raw_text, document_type_hint=parse_hint)
        assessment = build_input_assessment(raw_text, parsed, "KTP", detected_type)

        self.assertEqual(assessment["decision"], "needs_review")
        self.assertFalse(assessment["can_auto_publish"])
        self.assertIn("ktp_auto_missing:pekerjaan", assessment["reason_codes"])
        self.assertIn("ktp_auto_missing:kewarganegaraan", assessment["reason_codes"])
        self.assertIn("ktp_auto_missing:berlaku_hingga", assessment["reason_codes"])

    def test_rapidocr_result_needs_review_by_default(self):
        raw_text = """
        PROVINSI DKI JAKARTA
        NIK : 3175010101900001
        Nama : BUDI SANTOSO
        Tempat/Tgl Lahir : JAKARTA, 01-01-1990
        Jenis Kelamin : LAKI-LAKI
        Alamat : JL MERDEKA NO 10
        RT/RW : 001/002
        Kel/Desa : GAMBIR
        Kecamatan : GAMBIR
        Pekerjaan : KARYAWAN SWASTA
        Kewarganegaraan : WNI
        Berlaku Hingga : SEUMUR HIDUP
        """

        parse_hint, detected_type = choose_parse_document_type(raw_text, "KTP")
        parsed = parse_document_text(raw_text, document_type_hint=parse_hint)
        with patch.dict(os.environ, {"OCR_RAPID_AUTO_PUBLISH": ""}, clear=False):
            assessment = build_input_assessment(raw_text, parsed, "KTP", detected_type, ocr_provider="rapidocr")

        self.assertEqual(assessment["decision"], "needs_review")
        self.assertFalse(assessment["can_auto_publish"])
        self.assertIn("ocr_provider_needs_review:rapidocr", assessment["reason_codes"])

    def test_rapidocr_auto_publish_override_allows_clean_result(self):
        raw_text = """
        PROVINSI DKI JAKARTA
        NIK : 3175010101900001
        Nama : BUDI SANTOSO
        Tempat/Tgl Lahir : JAKARTA, 01-01-1990
        Jenis Kelamin : LAKI-LAKI
        Alamat : JL MERDEKA NO 10
        RT/RW : 001/002
        Kel/Desa : GAMBIR
        Kecamatan : GAMBIR
        Pekerjaan : KARYAWAN SWASTA
        Kewarganegaraan : WNI
        Berlaku Hingga : SEUMUR HIDUP
        """

        parse_hint, detected_type = choose_parse_document_type(raw_text, "KTP")
        parsed = parse_document_text(raw_text, document_type_hint=parse_hint)
        with patch.dict(os.environ, {"OCR_RAPID_AUTO_PUBLISH": "1 "}, clear=False):
            assessment = build_input_assessment(raw_text, parsed, "KTP", detected_type, ocr_provider="rapidocr")

        self.assertEqual(assessment["decision"], "approved_for_auto")
        self.assertTrue(assessment["can_auto_publish"])
        self.assertNotIn("ocr_provider_needs_review:rapidocr", assessment["reason_codes"])

    def test_ktp_missing_auto_approval_fields_needs_review(self):
        raw_text = """
        PROVINSI DKI JAKARTA
        NIK : 3175010101900001
        Nama : BUDI SANTOSO
        Alamat : JL MERDEKA NO 10
        """

        parse_hint, detected_type = choose_parse_document_type(raw_text, "KTP")
        parsed = parse_document_text(raw_text, document_type_hint=parse_hint)
        assessment = build_input_assessment(raw_text, parsed, "KTP", detected_type)

        self.assertEqual(assessment["decision"], "needs_review")
        self.assertFalse(assessment["can_auto_publish"])
        self.assertIn("ktp_auto_missing:tempat_tanggal_lahir", assessment["reason_codes"])
        self.assertIn("ktp_auto_missing:rt_rw", assessment["reason_codes"])

    def test_ktp_low_confidence_name_needs_review(self):
        raw_text = """
        PROVINSI DKI JAKARTA
        NIK : 3175010101900001
        Nama : BUDI SANTOSO
        Tempat/Tgl Lahir : JAKARTA, 01-01-1990
        Jenis Kelamin : LAKI-LAKI
        Alamat : JL MERDEKA NO 10
        RT/RW : 001/002
        Pekerjaan : KARYAWAN SWASTA
        Kewarganegaraan : WNI
        Berlaku Hingga : SEUMUR HIDUP
        """

        parse_hint, detected_type = choose_parse_document_type(raw_text, "KTP")
        parsed = parse_document_text(raw_text, document_type_hint=parse_hint)
        parsed.fields["nama"].confidence = 0.72
        assessment = build_input_assessment(raw_text, parsed, "KTP", detected_type)

        self.assertEqual(assessment["decision"], "needs_review")
        self.assertFalse(assessment["can_auto_publish"])
        self.assertIn("ktp_auto_low_confidence:nama", assessment["reason_codes"])

    def test_ktp_suspicious_region_label_value_needs_review(self):
        parsed = _ktp_result(kelurahan_desa="KOWARGANEGARAAN", kecamatan="TAMAN BALOI")

        assessment = build_input_assessment("PROVINSI KEPULAUAN RIAU\nNIK", parsed, "KTP", "KTP")

        self.assertEqual(assessment["decision"], "needs_review")
        self.assertFalse(assessment["can_auto_publish"])
        self.assertIn("ktp_suspicious_field:kelurahan_desa", assessment["reason_codes"])

    def test_ktp_suspicious_kecamatan_with_digits_needs_review(self):
        parsed = _ktp_result(
            nik="3275126405920003",
            tempat_tanggal_lahir="BEKASI, 24-05-1992",
            jenis_kelamin="PEREMPUAN",
            kecamatan="AGAN 3",
        )

        assessment = build_input_assessment("PROVINSI JAWA BARAT\nNIK", parsed, "KTP", "KTP")

        self.assertEqual(assessment["decision"], "needs_review")
        self.assertFalse(assessment["can_auto_publish"])
        self.assertIn("ktp_suspicious_field:kecamatan", assessment["reason_codes"])

    def test_ktp_suspicious_joined_name_needs_review(self):
        parsed = _ktp_result(nama="IPTHYAKSARAGAT")

        assessment = build_input_assessment("PROVINSI JAWA BARAT\nNIK", parsed, "KTP", "KTP")

        self.assertEqual(assessment["decision"], "needs_review")
        self.assertFalse(assessment["can_auto_publish"])
        self.assertIn("ktp_suspicious_field:nama", assessment["reason_codes"])

    def test_ktp_suspicious_birth_place_label_fragment_needs_review(self):
        parsed = _ktp_result(
            nik="3275126405920003",
            tempat_tanggal_lahir="IGLLAHIR BEKASI, 24-05-1992",
            jenis_kelamin="PEREMPUAN",
        )

        assessment = build_input_assessment("PROVINSI JAWA BARAT\nNIK", parsed, "KTP", "KTP")

        self.assertEqual(assessment["decision"], "needs_review")
        self.assertFalse(assessment["can_auto_publish"])
        self.assertIn("ktp_suspicious_field:tempat_tanggal_lahir", assessment["reason_codes"])

    def test_ktp_nik_birth_date_mismatch_needs_review(self):
        raw_text = """
        PROVINSI DKI JAKARTA
        NIK : 3175010101900001
        Nama : BUDI SANTOSO
        Tempat/Tgl Lahir : JAKARTA, 02-01-1990
        Jenis Kelamin : LAKI-LAKI
        Alamat : JL MERDEKA NO 10
        RT/RW : 001/002
        Kel/Desa : GAMBIR
        Kecamatan : GAMBIR
        Pekerjaan : KARYAWAN SWASTA
        Kewarganegaraan : WNI
        Berlaku Hingga : SEUMUR HIDUP
        """

        parse_hint, detected_type = choose_parse_document_type(raw_text, "KTP")
        parsed = parse_document_text(raw_text, document_type_hint=parse_hint)
        assessment = build_input_assessment(raw_text, parsed, "KTP", detected_type)

        self.assertEqual(assessment["decision"], "needs_review")
        self.assertFalse(assessment["can_auto_publish"])
        self.assertIn("ktp_nik_ttl_mismatch", assessment["reason_codes"])

    def test_ktp_invalid_nik_birth_code_cannot_auto_publish(self):
        raw_text = """
        PROVINSI DKI JAKARTA
        NIK : 3175013501900001
        Nama : BUDI SANTOSO
        Tempat/Tgl Lahir : JAKARTA, 01-01-1990
        Jenis Kelamin : LAKI-LAKI
        Alamat : JL MERDEKA NO 10
        RT/RW : 001/002
        Kel/Desa : GAMBIR
        Kecamatan : GAMBIR
        """

        parse_hint, detected_type = choose_parse_document_type(raw_text, "KTP")
        parsed = parse_document_text(raw_text, document_type_hint=parse_hint)
        assessment = build_input_assessment(raw_text, parsed, "KTP", detected_type)

        self.assertEqual(assessment["decision"], "needs_review")
        self.assertFalse(assessment["can_auto_publish"])
        self.assertIn("ktp_invalid_nik_birth_date", assessment["reason_codes"])

    def test_document_type_mismatch_rejects_structurally(self):
        raw_text = """
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        NO POLISI : B 1234 ABC
        NAMA PEMILIK : BUDI SANTOSO
        MERK : TOYOTA
        NO RANGKA : MHKA1234567890123
        NO MESIN : 1NR1234567
        """

        parse_hint, detected_type = choose_parse_document_type(raw_text, "KTP")
        parsed = parse_document_text(raw_text, document_type_hint=parse_hint)
        assessment = build_input_assessment(raw_text, parsed, "KTP", detected_type)

        self.assertEqual(parse_hint, "STNK")
        self.assertEqual(parsed.document_type, "STNK")
        self.assertEqual(assessment["decision"], "rejected_input")
        self.assertFalse(assessment["can_auto_publish"])
        self.assertIn("document_type_mismatch", assessment["reason_codes"])

    def test_complete_stnk_can_be_approved_for_auto(self):
        parsed = _stnk_result(
            nomor_polisi="B 1234 ABC",
            nama_pemilik="BUDI SANTOSO",
            tahun_pembuatan="2021",
            nomor_rangka="MHRGN5880MJ207222",
            nomor_mesin="L15ZF1008375",
        )

        assessment = build_input_assessment("SURAT TANDA NOMOR KENDARAAN", parsed, "STNK", "STNK")

        self.assertEqual(assessment["decision"], "approved_for_auto")
        self.assertEqual(assessment["reason_codes"], [])

    def test_stnk_suspicious_low_confidence_owner_needs_review(self):
        parsed = _stnk_result(
            nomor_polisi="B 2759 KZP",
            nama_pemilik="LAMAT",
            tahun_pembuatan="2021",
            nomor_rangka="MHRGN5880MJ207222",
            nomor_mesin="L15ZF1008375",
            nama_pemilik_confidence=0.76,
        )

        assessment = build_input_assessment("SURAT TANDA NOMOR KENDARAAN", parsed, "STNK", "STNK")

        self.assertEqual(assessment["decision"], "needs_review")
        self.assertFalse(assessment["can_auto_publish"])
        self.assertIn("stnk_auto_low_confidence:nama_pemilik", assessment["reason_codes"])
        self.assertIn("stnk_suspicious_field:nama_pemilik", assessment["reason_codes"])

    def test_stnk_owner_label_noise_needs_review(self):
        parsed = _stnk_result(
            nomor_polisi="B 5192 BAG",
            nama_pemilik="A ALLAMAT",
            tahun_pembuatan="2020",
            nomor_rangka="MH3SEF510L3100981",
            nomor_mesin="E31WE0108877",
        )

        assessment = build_input_assessment("SURAT TANDA NOMOR KENDARAAN", parsed, "STNK", "STNK")

        self.assertEqual(assessment["decision"], "needs_review")
        self.assertFalse(assessment["can_auto_publish"])
        self.assertIn("stnk_suspicious_field:nama_pemilik", assessment["reason_codes"])

    def test_screen_photo_returns_json_but_cannot_auto_publish(self):
        raw_text = """
        Manage
        KTP SETYO.jpg
        PROVINSI DKI JAKARTA
        NIK
        3175030906710017
        TOSHIBA
        Type here
        100%
        """

        parse_hint, detected_type = choose_parse_document_type(raw_text, "KTP")
        parsed = parse_document_text(raw_text, document_type_hint=parse_hint)
        assessment = build_input_assessment(raw_text, parsed, "KTP", detected_type)

        self.assertEqual(assessment["decision"], "rejected_input")
        self.assertFalse(assessment["can_auto_publish"])
        self.assertIn("screen_or_desktop_capture", assessment["reason_codes"])
        self.assertIn("quality:possible_non_ktp_crop", assessment["reason_codes"])

    def test_screen_photo_plain_jpg_marker_rejects(self):
        raw_text = """
        TOSHIBA
        ICTP SETYO jpg
        PROVINSI DKI JAKARTA
        JAKARTA TIMUR
        NIK
        3175030906710017
        """

        parse_hint, detected_type = choose_parse_document_type(raw_text, "KTP")
        parsed = parse_document_text(raw_text, document_type_hint=parse_hint)
        assessment = build_input_assessment(raw_text, parsed, "KTP", detected_type)

        self.assertEqual(assessment["decision"], "rejected_input")
        self.assertIn("screen_or_desktop_capture", assessment["reason_codes"])

    def test_missing_required_fields_needs_review_without_rejecting(self):
        raw_text = """
        PROVINSI DKI JAKARTA
        Nama : BUDI SANTOSO
        """

        parse_hint, detected_type = choose_parse_document_type(raw_text, "KTP")
        parsed = parse_document_text(raw_text, document_type_hint=parse_hint)
        assessment = build_input_assessment(raw_text, parsed, "KTP", detected_type)

        self.assertEqual(assessment["decision"], "needs_review")
        self.assertFalse(assessment["can_auto_publish"])
        self.assertIn("missing_required:nik", assessment["reason_codes"])

    def test_suspicious_ktp_output_with_missing_ttl_cannot_auto_publish(self):
        raw_text = """
        PROVINSI DKI JAKARTA
        JAKARTA PUSAT
        NIK
        3171075904930002
        Nama
        SEUSUR HOUP
        Jenis Kelamin
        PEREMPUAN
        Alamat
        AIGMAT
        RT/RW
        160/993
        Kel/Desa
        BENDLINGANHE
        Kecamatan
        TANAH ABANG
        """

        parse_hint, detected_type = choose_parse_document_type(raw_text, "KTP")
        parsed = parse_document_text(raw_text, document_type_hint=parse_hint)
        assessment = build_input_assessment(raw_text, parsed, "KTP", detected_type)

        self.assertEqual(assessment["decision"], "rejected_input")
        self.assertFalse(assessment["can_auto_publish"])
        self.assertIn("suspicious_ktp_output", assessment["reason_codes"])


def _ktp_result(
    *,
    nik: str = "3175010101900001",
    nama: str = "BUDI SANTOSO",
    tempat_tanggal_lahir: str = "JAKARTA, 01-01-1990",
    jenis_kelamin: str = "LAKI-LAKI",
    alamat: str = "JL MERDEKA NO 10",
    rt_rw: str = "001/002",
    kelurahan_desa: str = "GAMBIR",
    kecamatan: str = "GAMBIR",
    pekerjaan: str = "KARYAWAN SWASTA",
    kewarganegaraan: str = "WNI",
    berlaku_hingga: str = "SEUMUR HIDUP",
) -> DocumentResult:
    return DocumentResult(
        document_type="KTP",
        schema_version="ktp/v1",
        fields={
            "nik": FieldResult(nik, 0.98, "ok"),
            "nama": FieldResult(nama, 0.88, "ok"),
            "tempat_tanggal_lahir": FieldResult(tempat_tanggal_lahir, 0.88, "ok"),
            "jenis_kelamin": FieldResult(jenis_kelamin, 0.84, "ok"),
            "alamat": FieldResult(alamat, 0.88, "ok"),
            "rt_rw": FieldResult(rt_rw, 0.84, "ok"),
            "kelurahan_desa": FieldResult(kelurahan_desa, 0.84, "ok"),
            "kecamatan": FieldResult(kecamatan, 0.84, "ok"),
            "pekerjaan": FieldResult(pekerjaan, 0.84, "ok"),
            "kewarganegaraan": FieldResult(kewarganegaraan, 0.84, "ok"),
            "berlaku_hingga": FieldResult(berlaku_hingga, 0.84, "ok"),
        },
        raw_text="PROVINSI DKI JAKARTA\nNIK",
    )


def _stnk_result(
    *,
    nomor_polisi: str,
    nama_pemilik: str,
    tahun_pembuatan: str,
    nomor_rangka: str,
    nomor_mesin: str,
    nama_pemilik_confidence: float = 0.9,
) -> DocumentResult:
    return DocumentResult(
        document_type="STNK",
        schema_version="stnk/v1",
        fields={
            "nomor_polisi": FieldResult(nomor_polisi, 0.9, "ok"),
            "nama_pemilik": FieldResult(nama_pemilik, nama_pemilik_confidence, "ok"),
            "tahun_pembuatan": FieldResult(tahun_pembuatan, 0.9, "ok"),
            "nomor_rangka": FieldResult(nomor_rangka, 0.9, "ok"),
            "nomor_mesin": FieldResult(nomor_mesin, 0.9, "ok"),
        },
        raw_text="SURAT TANDA NOMOR KENDARAAN",
    )


if __name__ == "__main__":
    unittest.main()
