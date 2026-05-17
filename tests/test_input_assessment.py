import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_engine.service import build_input_assessment, choose_parse_document_type, parse_document_text


class InputAssessmentTests(unittest.TestCase):
    def test_clean_ktp_can_be_approved_for_auto(self):
        raw_text = """
        PROVINSI DKI JAKARTA
        NIK : 3175010101900001
        Nama : BUDI SANTOSO
        Alamat : JL MERDEKA NO 10
        """

        parse_hint, detected_type = choose_parse_document_type(raw_text, "KTP")
        parsed = parse_document_text(raw_text, document_type_hint=parse_hint)
        assessment = build_input_assessment(raw_text, parsed, "KTP", detected_type)

        self.assertEqual(assessment["decision"], "approved_for_auto")
        self.assertTrue(assessment["can_auto_publish"])
        self.assertEqual(assessment["reason_codes"], [])

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


if __name__ == "__main__":
    unittest.main()
