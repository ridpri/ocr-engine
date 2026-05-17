import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_engine.frontend import frontend_html


class FrontendTests(unittest.TestCase):
    def test_frontend_html_contains_upload_workflow(self):
        html = frontend_html()

        self.assertIn('id="ocr-form"', html)
        self.assertIn('name="document_type"', html)
        self.assertIn('name="engine_type"', html)
        self.assertIn('name="vps_api_key"', html)
        self.assertNotIn('name="processing_mode"', html)
        self.assertNotIn('id="processing-mode"', html)
        self.assertNotIn("Fast checkout", html)
        self.assertNotIn(">Accurate<", html)
        self.assertIn('name="file"', html)
        self.assertIn("/ocr/ktp", html)
        self.assertIn("/ocr/stnk", html)
        self.assertIn("/ocr?document_type=", html)
        self.assertIn("/ocr/agent/ktp", html)
        self.assertIn("/ocr/agent/stnk", html)
        self.assertIn("/ocr/agent?document_type=", html)
        self.assertIn("/ocr/vps/ktp", html)
        self.assertIn("/ocr/vps/stnk", html)
        self.assertIn("/ocr/vps/agent/ktp", html)
        self.assertIn("/ocr/vps/agent/stnk", html)
        self.assertNotIn("/ocr/stnk?mode=", html)
        self.assertNotIn("/ocr/vps/stnk?mode=", html)
        self.assertIn("X-VPS-API-Key", html)
        self.assertIn("Cara Integrasi API", html)
        self.assertIn("Contoh Vercel API Route Proxy", html)
        self.assertIn("Jangan taruh API key langsung di frontend publik", html)
        self.assertIn("needs_review", html)
        self.assertIn("input_assessment", html)
        self.assertIn("summary-decision", html)
        self.assertIn("summary-engine-used", html)
        self.assertIn("summary-retry", html)
        self.assertIn("summary-resolution", html)
        self.assertIn("summary-pipeline-time", html)
        self.assertIn("summary-ocr-time", html)
        self.assertIn("initial-section", html)
        self.assertIn("enrichment-section", html)
        self.assertIn("enrichment-status", html)
        self.assertIn("/ocr/enrichment/", html)

    def test_frontend_explains_stnk_fast_timeout_as_pending_enrichment(self):
        html = frontend_html()

        self.assertIn("isProcessingTimeout", html)
        self.assertIn("OCR accurate masih berjalan", html)
        self.assertIn("Hasil awal belum berisi field OCR", html)


if __name__ == "__main__":
    unittest.main()
