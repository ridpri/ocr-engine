import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_engine.service import (
    build_input_assessment,
    detect_document_type,
    parse_document_text,
    select_prepare_max_side,
    should_retry_stnk_highres,
)


class ServiceTests(unittest.TestCase):
    def test_detect_document_type_from_ktp_text(self):
        raw_text = "PROVINSI DKI JAKARTA\nNIK : 3175010101900001\nNama : BUDI"

        self.assertEqual(detect_document_type(raw_text), "KTP")

    def test_detect_document_type_does_not_treat_nik_only_as_ktp(self):
        raw_text = "TANDA BUKTI PEMBAYARAN\nNIK\n3173081508720009\nNOMOR POLISI\nB 1234 ABC"

        self.assertEqual(detect_document_type(raw_text), "UNKNOWN")

    def test_detect_document_type_from_stnk_text(self):
        raw_text = "SURAT TANDA NOMOR KENDARAAN BERMOTOR\nNO POLISI : B 1234 ABC"

        self.assertEqual(detect_document_type(raw_text), "STNK")

    def test_detect_document_type_from_noisy_stnk_tax_sheet(self):
        raw_text = "\n".join(
            [
                "NOMOR BPKB",
                "NOMOR MESIN",
                "NO.RANGKA/NIK",
                "NAMA PEMILIK",
                "TNKB",
                "PKB. BBN-KB. SWDKLLJ, BIAYA ADM.",
            ]
        )

        self.assertEqual(detect_document_type(raw_text), "STNK")

    def test_detect_document_type_from_noisy_stnk_tax_receipt(self):
        raw_text = "\n".join(
            [
                "TANDA BUKTI PELUNASAN KEWAJIBAN PEMBAYARAN",
                "SAMSAT PROVINSI DKI JAKARTA",
                "NOMOR POUSR",
                "B 9241 TYX",
                "NAMA PEMIUK",
                "PT PP PRESISI TBK",
                "TAHUN PEMBUAIAN",
                "2018",
                "NONCR UESN",
                "J088UFJ99935",
                "BERLAKU SAMPALL 08-10-2023",
            ]
        )

        self.assertEqual(detect_document_type(raw_text), "STNK")

    def test_parse_document_text_respects_valid_hint(self):
        raw_text = "NO POLISI : B 1234 ABC\nNAMA PEMILIK : BUDI\nNO RANGKA : MHKA1234567890123\nNO MESIN : 1NR1234567"

        result = parse_document_text(raw_text, document_type_hint="STNK")

        self.assertEqual(result.document_type, "STNK")
        self.assertEqual(result.fields["nomor_polisi"].value, "B 1234 ABC")

    def test_select_prepare_max_side_uses_smaller_stnk_fast_pass(self):
        self.assertEqual(select_prepare_max_side("STNK"), 1200)
        self.assertEqual(select_prepare_max_side("KTP"), 1280)
        self.assertEqual(select_prepare_max_side("AUTO"), 1280)

    def test_should_retry_stnk_highres_when_required_fields_are_missing(self):
        raw_text = "SURAT TANDA NOMOR KENDARAAN\nNO POLISI : B 1234 ABC\nNAMA PEMILIK : BUDI"
        parsed = parse_document_text(raw_text, document_type_hint="STNK")
        assessment = build_input_assessment(raw_text, parsed, "STNK", detect_document_type(raw_text))

        self.assertTrue(should_retry_stnk_highres("STNK", parsed, assessment))

    def test_should_not_retry_stnk_highres_for_ktp_or_mismatched_document(self):
        ktp_text = "PROVINSI DKI JAKARTA\nNIK : 3175010101900001\nNama : BUDI"
        parsed = parse_document_text(ktp_text, document_type_hint="KTP")
        assessment = build_input_assessment(ktp_text, parsed, "STNK", detect_document_type(ktp_text))

        self.assertFalse(should_retry_stnk_highres("STNK", parsed, assessment))
        self.assertFalse(should_retry_stnk_highres("KTP", parsed, assessment))

    def test_ktp_borderline_but_plausible_name_can_auto_pass(self):
        raw_text = "\n".join(
            [
                "PROVINSI JAWA BARAT",
                "KABUPATEN BEKASI",
                "NIK : 3216064704060020",
                "Nama : SALSABILA PUTRI DEWANTI",
                "Tempat/Tgl Lahir : BEKASI, 07-04-2006",
                "Jenis Kelamin : PEREMPUAN",
                "Alamat : JL BIMA ASRI X NO.35",
                "RT/RW : 003/008",
                "Kel/Desa : LAMBANGSARI",
                "Kecamatan : TAMBUN SELATAN",
                "Agama : ISLAM",
                "Status Perkawinan : BELUM KAWIN",
                "Pekerjaan : PELAJAR/MAHASISWA",
                "Kewarganegaraan : WNI",
                "Berlaku Hingga : SEUMUR HIDUP",
            ]
        )
        parsed = parse_document_text(raw_text, document_type_hint="KTP")
        parsed.fields["nama"].confidence = 0.78

        assessment = build_input_assessment(raw_text, parsed, "KTP", detect_document_type(raw_text))

        self.assertEqual(assessment["decision"], "approved_for_auto")
        self.assertNotIn("ktp_auto_low_confidence:nama", assessment["reason_codes"])

    def test_ktp_implausible_borderline_name_stays_review(self):
        raw_text = "\n".join(
            [
                "PROVINSI DKI JAKARTA",
                "KOTA JAKARTA BARAT",
                "NIK : 3173055003860011",
                "Nama : YOKHEBED SETIOWATI SANTOSO",
                "Tempat/Tgl Lahir : SLEMAN, 10-03-1986",
                "Jenis Kelamin : PEREMPUAN",
                "Alamat : JL KEDOYA AGAVE III CG/14",
                "RT/RW : 010/004",
                "Kel/Desa : KEDOYA SELATAN",
                "Kecamatan : KEBON JERUK",
                "Agama : KRISTEN",
                "Status Perkawinan : KAWIN",
                "Pekerjaan : MENGURUS RUMAH TANGGA",
                "Kewarganegaraan : WNI",
                "Berlaku Hingga : SEUMUR HIDUP",
            ]
        )
        parsed = parse_document_text(raw_text, document_type_hint="KTP")
        parsed.fields["nama"].value = "AKARTA BARA"
        parsed.fields["nama"].confidence = 0.72

        assessment = build_input_assessment(raw_text, parsed, "KTP", detect_document_type(raw_text))

        self.assertEqual(assessment["decision"], "needs_review")
        self.assertIn("ktp_auto_low_confidence:nama", assessment["reason_codes"])


if __name__ == "__main__":
    unittest.main()
