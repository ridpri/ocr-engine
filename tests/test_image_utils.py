from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ocr_engine.image_utils import prepare_ktp_fast_image, prepare_stnk_fast_roi_image, prepare_stnk_official_roi_image


class ImageUtilsTests(unittest.TestCase):
    def test_ktp_fast_roi_keeps_lower_identity_rows_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "ktp.jpg"
            output = Path(tmpdir) / "prepared.jpg"
            Image.new("RGB", (1800, 1200), "white").save(source)

            prepare_ktp_fast_image(source, output, max_side=560, right_ratio=0.72)

            with Image.open(output) as prepared:
                self.assertEqual(prepared.size, (560, 519))

    def test_stnk_fast_roi_can_crop_right_side_noise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "stnk.jpg"
            output = Path(tmpdir) / "prepared.jpg"
            Image.new("RGB", (1800, 1200), "white").save(source)

            prepare_stnk_fast_roi_image(source, output, max_side=720, right_ratio=0.78)

            with Image.open(output) as prepared:
                self.assertEqual(prepared.size, (561, 422))

    def test_stnk_official_roi_rotates_right_half_for_portrait_side_by_side_scan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "stnk-page.png"
            output = Path(tmpdir) / "prepared.jpg"
            image = Image.new("RGB", (500, 1000), "white")
            image.paste("black", (250, 0, 500, 1000))
            image.save(source)

            prepare_stnk_official_roi_image(source, output, max_side=1000)

            with Image.open(output) as prepared:
                self.assertGreater(prepared.width, prepared.height)
                self.assertGreater(prepared.width, 900)
                self.assertLess(prepared.height, 560)


if __name__ == "__main__":
    unittest.main()
