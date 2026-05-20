import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_engine.parsers.ktp import parse_ktp_text
from ocr_engine.parsers.ktp_layout import apply_ktp_layout_hints
from ocr_engine.postal_code import PostalCodeIndex, PostalCodeMatch
from ocr_engine.ocr.base import OcrToken
from ocr_engine.parsers.stnk import match_stnk_label, parse_stnk_text, stnk_structure_score
from ocr_engine.validators import mask_sensitive_text, normalize_nik, validate_plate_number


class KtpParserTests(unittest.TestCase):
    def test_parse_ktp_core_fields_from_labelled_text(self):
        raw_text = """
        PROVINSI DKI JAKARTA
        KOTA ADMINISTRASI JAKARTA PUSAT
        NIK : 3175010101900001
        Nama : BUDI SANTOSO
        Tempat/Tgl Lahir : JAKARTA, 01-01-1990
        Jenis Kelamin : LAKI-LAKI Gol. Darah : O
        Alamat : JL MERDEKA NO 10
        RT/RW : 001/002
        Kel/Desa : MENTENG
        Kecamatan : MENTENG
        Agama : ISLAM
        Status Perkawinan : KAWIN
        Pekerjaan : KARYAWAN SWASTA
        Kewarganegaraan : WNI
        Berlaku Hingga : SEUMUR HIDUP
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.document_type, "KTP")
        self.assertEqual(result.fields["provinsi"].value, "DKI JAKARTA")
        self.assertEqual(result.fields["kabupaten_kota"].value, "JAKARTA PUSAT")
        self.assertEqual(result.fields["nik"].value, "3175010101900001")
        self.assertEqual(result.fields["nik"].status, "ok")
        self.assertEqual(result.fields["nama"].value, "BUDI SANTOSO")
        self.assertEqual(result.fields["tempat_tanggal_lahir"].value, "JAKARTA, 01-01-1990")
        self.assertEqual(result.fields["alamat"].value, "JL MERDEKA NO 10")
        self.assertEqual(result.fields["rt_rw"].value, "001/002")
        self.assertEqual(result.fields["kelurahan_desa"].value, "MENTENG")
        self.assertEqual(result.fields["kecamatan"].value, "MENTENG")
        self.assertEqual(result.fields["berlaku_hingga"].value, "SEUMUR HIDUP")
        self.assertFalse(result.needs_review)

    def test_parse_ktp_marks_missing_required_fields_for_review(self):
        result = parse_ktp_text("Nama : ANI")

        self.assertEqual(result.fields["nama"].value, "ANI")
        self.assertEqual(result.fields["nik"].status, "missing")
        self.assertTrue(result.needs_review)
        self.assertIn("missing_required:nik", result.warnings)

    def test_parse_ktp_reads_joined_province_header(self):
        raw_text = """
        PROVINSIJAWA BARAT
        KABUPATEN BEKASI
        NIK
        3216064704060020
        Nama
        SALSABILA PUTRI DEWANTI
        Alamat
        JL BIMA ASRI X NO.35
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["provinsi"].value, "JAWA BARAT")
        self.assertEqual(result.fields["provinsi"].status, "ok")
        self.assertEqual(result.fields["kabupaten_kota"].value, "BEKASI")
        self.assertEqual(result.fields["kabupaten_kota"].status, "ok")

    def test_parse_ktp_adds_postal_code_from_region_database_lookup(self):
        raw_text = """
        PROVINSI JAWA BARAT
        KABUPATEN BEKASI
        NIK
        3216064704060020
        Nama
        SALSABILA PUTRI DEWANTI
        Alamat
        JL BIMA ASRI X NO.35
        Kel/Desa
        LAMBANGSARI
        Kecamatan
        TAMBUN SELATAN
        """

        with patch(
            "ocr_engine.parsers.ktp.lookup_postal_code",
            return_value=PostalCodeMatch("17510", 0.95, ["kelurahan_desa:Lambangsari"]),
        ):
            result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["kode_pos"].value, "17510")
        self.assertEqual(result.fields["kode_pos"].status, "ok")
        self.assertEqual(result.fields["kode_pos"].raw, "db_kode_wilayah")

    def test_postal_code_index_matches_ktp_regions_to_kelurahan_code(self):
        index = PostalCodeIndex.from_records(
            [
                {
                    "kode_pos": "17510",
                    "address": "Lambangsari, Tambun Selatan, Kabupaten Bekasi, Jawa Barat 17510",
                    "locality": "Lambangsari",
                    "sifat_pos": "kel.",
                    "city_name": "Bekasi",
                    "province_name": "Jawa Barat",
                },
                {
                    "kode_pos": "17111",
                    "address": "Bekasi Pasar Baru BEKASI 17111",
                    "locality": "Bekasi Pasar Baru",
                    "sifat_pos": "Jln.",
                    "city_name": "Bekasi",
                    "province_name": "Jawa Barat",
                },
            ]
        )
        parsed = parse_ktp_text(
            """
            PROVINSI JAWA BARAT
            KABUPATEN BEKASI
            NIK 3216064704060020
            Nama SALSABILA PUTRI DEWANTI
            Alamat JL BIMA ASRI X NO.35
            Kel/Desa LAMBANGSARI
            Kecamatan TAMBUN SELATAN
            """
        )

        match = index.lookup(parsed.fields)

        self.assertIsNotNone(match)
        self.assertEqual(match.kode_pos, "17510")

    def test_parse_ktp_does_not_accept_blood_type_label_as_address(self):
        raw_text = """
        PROVINSI BALI
        NIK
        5102092505870003
        Nama
        IWAYAN QVA ARANTIKA
        Alamat
        Gol. Darah
        Jenis Kelamin
        LAKI-LAKI
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["alamat"].status, "missing")
        self.assertIn("missing_required:alamat", result.warnings)

    def test_parse_ktp_defaults_missing_marital_status_to_belum_kawin(self):
        raw_text = """
        PROVINSI BALI
        NIK
        5102092505870003
        Nama
        IWAYAN QVA ARANTIKA
        Alamat
        BUKIT DELIMA VIII/B
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["status_perkawinan"].value, "BELUM KAWIN")
        self.assertEqual(result.fields["status_perkawinan"].status, "ok")
        self.assertEqual(result.fields["status_perkawinan"].raw, "fallback:default_marital_status")
        self.assertLess(result.fields["status_perkawinan"].confidence, 0.5)

    def test_parse_ktp_falls_back_to_name_between_nik_and_birth_label(self):
        raw_text = """
        PROVINSI JAWA TENGAH
        NIK
        3374114208810004
        DANI ANGGOROWATI
        Tempat/Tgl Lahir : KAB.SEMARANG, 02-08-1981
        Alamat
        JL ESTETIKA BARAT J-20
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["nama"].value, "DANI ANGGOROWATI")
        self.assertEqual(result.fields["tempat_tanggal_lahir"].value, "KAB.SEMARANG, 02-08-1981")

    def test_parse_ktp_repairs_joined_name_spacing(self):
        raw_text = """
        PROVINSI JAWA BARAT
        KABUPATEN BEKASI
        NIK : 3216064704060020
        Nama : TRISUPRIHATIN
        Tempat/Tgl Lahir : BEKASI, 02-08-1981
        Alamat : JL BIMA ASRI X NO.35
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["nama"].value, "TRI SUPRIHATIN")
        self.assertEqual(result.fields["nama"].status, "ok")

    def test_parse_ktp_falls_back_to_birth_place_and_date_pattern(self):
        raw_text = """
        NIK : 3175010101900001
        Nama : BUDI SANTOSO
        JAKARTA, 01-01-1990
        Jenis Kelamin : LAKI-LAKI
        Alamat : JL MERDEKA NO 10
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["tempat_tanggal_lahir"].value, "JAKARTA, 01-01-1990")

    def test_parse_ktp_birth_place_date_handles_missing_comma(self):
        raw_text = """
        NIK : 3174030101760004
        Nama : MAMAN SURYANA
        TEMPAV/TOL LAHIR
        JAKARTA 30-07-1976
        Alamat : JL MELATI NO 86
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["tempat_tanggal_lahir"].value, "JAKARTA, 30-07-1976")

    def test_parse_ktp_birth_place_date_repairs_joined_month_year(self):
        raw_text = """
        NIK
        3173052804620003
        Name
        KADIM HASIMSIM
        Tompal/TglLabe
        MEDAN.28-061963
        Alamal
        J KEMBANGANUAMA BLOKK
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["tempat_tanggal_lahir"].value, "MEDAN, 28-06-1963")

    def test_parse_ktp_birth_place_date_normalizes_labelled_noisy_date(self):
        raw_text = """
        NIK : 3216064704060020
        Nama : SALSABILA PUTRI DEWANTI
        TempatTglLahir
        BEKAS1,07-04-2006
        Alamat : JL BIMA ASRI X NO.35
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["tempat_tanggal_lahir"].value, "BEKASI, 07-04-2006")

    def test_parse_ktp_birth_place_date_repairs_space_between_day_and_month(self):
        raw_text = """
        NIK : 3276011904710005
        Nama : SOLAHUDIN
        Tempat/Tgl Lahir
        JAKARTA,19 04-1971
        Alamat : PERMATA DEPOK BERLIAN
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["tempat_tanggal_lahir"].value, "JAKARTA, 19-04-1971")

    def test_parse_ktp_birth_place_date_repairs_split_place_and_date_after_label(self):
        raw_text = """
        NIK : 323031501740005
        Nama : HENDRA SISWAN SUPARMI
        TempairglLahir
        JAKARTA
        15-01-1974
        Jenis kelamin
        LAKI LAKI
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["tempat_tanggal_lahir"].value, "JAKARTA, 15-01-1974")

    def test_parse_ktp_birth_place_date_repairs_ocr_noise_in_date_digits(self):
        cases = [
            ("PARSOBURAN, 11-10-199:3", "PARSOBURAN, 11-10-1993"),
            ("BANDUNG.10-O61997", "BANDUNG, 10-06-1997"),
            ("SAMOSIR.06-04-196S", "SAMOSIR, 06-04-1965"),
            ("KOTA CIREBON, 22-04-1979", "KOTA CIREBON, 22-04-1979"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                raw_text = f"""
                NIK : 3274032204790008
                Nama : ANDRI PRASETYANTO
                Tempat/Tgl Lahir
                {value}
                Jenis kelamin
                LAKI-LAKI
                """

                result = parse_ktp_text(raw_text)

                self.assertEqual(result.fields["tempat_tanggal_lahir"].value, expected)

    def test_parse_ktp_birth_place_date_ignores_bad_label_capture_and_uses_nearby_lines(self):
        raw_text = """
        NIK : 3671090305770003
        Nama : LILIK EKO MURSITO
        Tempat/Tgl Lahir
        4
        JAKARTA
        03-05-1977
        Jenis kelamin
        LAKI-LAKI
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["tempat_tanggal_lahir"].value, "JAKARTA, 03-05-1977")

    def test_parse_ktp_falls_back_to_address_before_rt_rw(self):
        raw_text = """
        NIK : 3175010101900001
        Nama : BUDI SANTOSO
        Tempat/Tgl Lahir : JAKARTA, 01-01-1990
        JL MERDEKA NO 10
        RT/RW : 001/002
        Kel/Desa : MENTENG
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["alamat"].value, "JL MERDEKA NO 10")

    def test_parse_ktp_fuzzy_status_perkawinan_from_ocr_typo_and_joined_value(self):
        raw_text = """
        NIK : 3175010101900001
        Nama : BUDI SANTOSO
        Tempat/Tgl Lahir : JAKARTA, 01-01-1990
        Alamat : JL MERDEKA NO 10
        Agama
        ISLAM
        Status Perkawinar
        BELUMKAWIN
        Pekerjaan
        PELAJARMAHASISWA
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["status_perkawinan"].value, "BELUM KAWIN")
        self.assertEqual(result.fields["status_perkawinan"].status, "ok")

    def test_parse_ktp_normalizes_dirty_marital_status_value(self):
        raw_text = """
        NIK : 3276011904710005
        Nama : SOLAHUDIN
        Status Perkawinan.KAWiN
        Pekerjaan
        KARYAWAN SWASTA
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["status_perkawinan"].value, "KAWIN")

    def test_parse_ktp_repairs_noisy_marital_status_label(self):
        raw_text = """
        NIK : 3216064704060020
        Nama : SALSABILA PUTRI DEWANTI
        Status Perkawinar
        BEIUM KAWIN
        Pekerjaan
        PELAJAR/MAHASISWA
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["status_perkawinan"].value, "BELUM KAWIN")

    def test_parse_ktp_repairs_marital_status_ocr_i_l_confusion(self):
        raw_text = """
        NIK : 3201070809040011
        Nama : FAREL SEPTIAN MANOSSOH
        Status Perkawinan
        Belum Kawln
        Pekerjaan
        PELAJAR/MAHASISWA
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["status_perkawinan"].value, "BELUM KAWIN")

    def test_parse_ktp_repairs_inline_noisy_marital_status_label_variants(self):
        cases = [
            "Status Perkawinar: KAWIN",
            "Status Perkawinarc KAWIN",
            "Status Perkawinare KAWiN",
        ]
        for status_line in cases:
            with self.subTest(status_line=status_line):
                raw_text = f"""
                NIK : 3175030906710017
                Nama : SETYO BUDI
                {status_line}
                Pekerjaan
                KARYAWAN SWASTA
                """

                result = parse_ktp_text(raw_text)

                self.assertEqual(result.fields["status_perkawinan"].value, "KAWIN")

    def test_parse_ktp_repairs_marital_status_digit_noise(self):
        raw_text = """
        NIK : 3175030906710017
        Nama : SETYO BUDI
        Status Perkawinan
        KAW1N
        Pekerjaan
        KARYAWAN SWASTA
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["status_perkawinan"].value, "KAWIN")

    def test_parse_ktp_repairs_belum_menikah_ocr_noise(self):
        raw_text = """
        NIK : 3175030906710017
        Nama : SETYO BUDI
        Status Perkawinan
        BEIUM MENIKAH
        Pekerjaan
        KARYAWAN SWASTA
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["status_perkawinan"].value, "BELUM KAWIN")

    def test_ktp_layout_repairs_citizenship_from_lower_position(self):
        result = parse_ktp_text(
            """
            NIK : 3216064704060020
            Nama : SALSABILA PUTRI DEWANTI
            Pekerjaan
            PELAJAR/MAHASISWA
            """
        )
        tokens = [
            OcrToken("Nama", 0.99, bbox=[[100, 120], [200, 120], [200, 145], [100, 145]]),
            OcrToken("Kewargane", 0.82, bbox=[[100, 760], [260, 760], [260, 785], [100, 785]]),
            OcrToken("VNI", 0.75, bbox=[[280, 760], [330, 760], [330, 785], [280, 785]]),
            OcrToken("Berlaku", 0.99, bbox=[[100, 840], [200, 840], [200, 865], [100, 865]]),
        ]

        apply_ktp_layout_hints(result, tokens)

        self.assertEqual(result.fields["kewarganegaraan"].value, "WNI")
        self.assertEqual(result.fields["kewarganegaraan"].status, "ok")

    def test_ktp_layout_does_not_use_citizenship_value_from_upper_position(self):
        result = parse_ktp_text(
            """
            NIK : 3216064704060020
            Nama : SALSABILA PUTRI DEWANTI
            Pekerjaan
            PELAJAR/MAHASISWA
            """
        )
        tokens = [
            OcrToken("VNI", 0.75, bbox=[[100, 120], [150, 120], [150, 145], [100, 145]]),
            OcrToken("Pekerjaan", 0.99, bbox=[[100, 700], [230, 700], [230, 725], [100, 725]]),
            OcrToken("Berlaku", 0.99, bbox=[[100, 840], [200, 840], [200, 865], [100, 865]]),
        ]

        apply_ktp_layout_hints(result, tokens)

        self.assertEqual(result.fields["kewarganegaraan"].status, "missing")

    def test_ktp_layout_normalizes_dirty_status_from_status_position(self):
        result = parse_ktp_text(
            """
            NIK : 3276011904710005
            Nama : SOLAHUDIN
            Pekerjaan
            KARYAWAN SWASTA
            """
        )
        tokens = [
            OcrToken("Agama", 0.99, bbox=[[100, 560], [180, 560], [180, 585], [100, 585]]),
            OcrToken("Status Perkawinarc", 0.78, bbox=[[100, 640], [330, 640], [330, 665], [100, 665]]),
            OcrToken("KAW1N", 0.76, bbox=[[350, 640], [430, 640], [430, 665], [350, 665]]),
            OcrToken("Pekerjaan", 0.99, bbox=[[100, 720], [230, 720], [230, 745], [100, 745]]),
        ]

        apply_ktp_layout_hints(result, tokens)

        self.assertEqual(result.fields["status_perkawinan"].value, "KAWIN")

    def test_parse_ktp_repairs_noisy_kelurahan_and_kecamatan_labels(self):
        raw_text = """
        NIK : 3173020211730006
        Nama : JAP,JOBIE
        Alamat
        PERUM PURI DEWATA INDAH BLOK
        RT/RW
        006/006
        KelDesa
        ANCOL
        Kecamatan_:CIPONDOH
        Agama
        KATHOLIK
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["kelurahan_desa"].value, "ANCOL")
        self.assertEqual(result.fields["kecamatan"].value, "CIPONDOH")

    def test_parse_ktp_repairs_split_and_typo_kecamatan_label(self):
        raw_text = """
        NIK : 3201260402980005
        Nama : MOH. HIFDZI YUSA
        Alamat
        KP. SUKABIRUS
        RT/RW
        002/006
        Kel/Desa
        GADOG
        Kec
        Eatan
        MEGAMENDUNG
        Agama
        ISLAM
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["kecamatan"].value, "MEGAMENDUNG")

    def test_parse_ktp_accepts_region_value_containing_nik_substring(self):
        raw_text = """
        NIK : 3374114208810004
        Nama : DANI ANGGOROWATI
        Kel/Desa
        PEDALANGAN
        Kecamatan
        BANYUMANIK
        Agama
        ISLAM
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["kecamatan"].value, "BANYUMANIK")

    def test_parse_ktp_repairs_joined_kecamatan_label_and_value(self):
        raw_text = """
        NIK : 3173020211730006
        Nama : JAP,JOBIE
        Kel/Desa
        PORIS PLAWAD UTARA
        Kecamatan_CIPONDOH
        Agama
        KATHOLIK
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["kecamatan"].value, "CIPONDOH")

    def test_parse_ktp_repairs_transposed_region_and_expiry_values(self):
        raw_text = """
        Jenis Kelamin
        Temp
        Nama
        NIK
        Berlaky Hingga
        Kewarganegaraan:WNI
        Pekerjaan
        Status Perkawinan: KAWIN
        Agama
        Alamat
        Kecamatan
        Kel/Desa
        RT/RW
        Tgl Lahir
        3173065505780001
        :15-05-2017
        :BUDHA
        :KALIDERES
        :006/ 007
        : JLN.BIMA BLOK C 11/7
        :PEREMPUAN
        :SINKAWANG, 15-05-1978
        : MENGURUS RUMAH TANGGA
        :TEGAL ALUR
        :TJONG FUI SIAN
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["berlaku_hingga"].value, "15-05-2017")
        self.assertEqual(result.fields["kecamatan"].value, "KALIDERES")
        self.assertEqual(result.fields["kelurahan_desa"].value, "TEGAL ALUR")

    def test_parse_ktp_repairs_noisy_berlaku_hingga_labels(self):
        cases = [
            ("BerlakuHingga:SEUMUR HIDUP", "SEUMUR HIDUP"),
            ("Berlaku Hing\nSEUMUR HIDUP", "SEUMUR HIDUP"),
            ("Barlaku Hingga\nSEUMUR HIDUP", "SEUMUR HIDUP"),
            ("Serfaku Hingga\nSEUMUR HIDUP", "SEUMUR HIDUP"),
            ("Bertaku Hingga\nSEUMUR HIDUP", "SEUMUR HIDUP"),
            ("Berlaku: Hingga\n25-09-2017", "25-09-2017"),
        ]
        for expiry_text, expected in cases:
            with self.subTest(expiry_text=expiry_text):
                raw_text = f"""
                NIK : 3175010101900001
                Nama : BUDI SANTOSO
                Kewarganegaraan: WNI
                {expiry_text}
                """

                result = parse_ktp_text(raw_text)

                self.assertEqual(result.fields["berlaku_hingga"].value, expected)

    def test_ktp_layout_repairs_region_and_expiry_from_positions(self):
        result = parse_ktp_text(
            """
            NIK : 3216064704060020
            Nama : SALSABILA PUTRI DEWANTI
            Alamat
            JL BIMA ASRI X NO.35
            """
        )
        tokens = [
            OcrToken("KelDesa", 0.73, bbox=[[100, 520], [210, 520], [210, 545], [100, 545]]),
            OcrToken("LAMBANGSARI", 0.91, bbox=[[250, 520], [430, 520], [430, 545], [250, 545]]),
            OcrToken("Kecamnatan", 0.71, bbox=[[100, 580], [235, 580], [235, 605], [100, 605]]),
            OcrToken("TAMBUNSELATAN", 0.89, bbox=[[250, 580], [500, 580], [500, 605], [250, 605]]),
            OcrToken("Serfaku Hing", 0.72, bbox=[[100, 820], [260, 820], [260, 845], [100, 845]]),
            OcrToken("SEUMUR HIDUP", 0.91, bbox=[[290, 820], [470, 820], [470, 845], [290, 845]]),
        ]

        apply_ktp_layout_hints(result, tokens)

        self.assertEqual(result.fields["kelurahan_desa"].value, "LAMBANGSARI")
        self.assertEqual(result.fields["kecamatan"].value, "TAMBUNSELATAN")
        self.assertEqual(result.fields["berlaku_hingga"].value, "SEUMUR HIDUP")

    def test_parse_ktp_nik_corrects_comma_to_one_near_label(self):
        raw_text = """
        PROVINSI JAWA BARAT
        NIK
        :32760119047,0005
        Nama
        SOLAHUDIN
        Alamat : JL MERDEKA NO 1
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["nik"].value, "3276011904710005")
        self.assertEqual(result.fields["nik"].status, "ok")

    def test_parse_ktp_nik_finds_value_when_layout_separates_labels_and_values(self):
        raw_text = """
        Tempat/Tgl Lahir
        Nama
        NIK
        Kecamatan
        Kel/Desa
        RT/RW
        : 15-08-2016
        : KEMBANGAN
        : JAKARTA, 15-08-1972
        3173085508720009
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["nik"].value, "3173085508720009")
        self.assertEqual(result.fields["nik"].status, "ok")

    def test_parse_ktp_nik_prefers_standalone_16_digit_line_after_dates(self):
        raw_text = "\n".join(
            [
                "NIK",
                ": 15-08-2016",
                ": JAKARTA, 15-08-1972",
                "3173085508720009",
                "RT/RW",
            ]
        )

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["nik"].value, "3173085508720009")
        self.assertEqual(result.fields["nik"].status, "ok")

    def test_parse_ktp_falls_back_to_name_after_standalone_nik_value(self):
        raw_text = """
        PROVINSI DKI JAKARTA
        3174030101760004
        MAMAN SURYANA
        TEMPAV/TOL LAHIR
        JAKARTA 30-07-1976
        ALAMAT
        JL MELATI NO 86
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["nama"].value, "MAMAN SURYANA")

    def test_parse_ktp_transposed_name_skips_job_value_after_nik(self):
        raw_text = """
        Borlaku Hingga
        Kewarganegaraan: WNI
        Pekerjaan
        Status Perkawinan: KAWIN
        Agama
        Alamat
        Jenis Kelamin
        Tempat/Tgl Lahir
        Nama
        NIK
        Kecamatan
        Kel/Desa
        RT/RW
        : 15-08-2016
        :KEMBANGAN
        : JAKARTA, 15-08-1972
        3173081508720009
        KARYAWAN SWASTA
        KRISTEN
        :JOGLO
        :008/ 002
        : JLAL MUBAROK II NO. 32 C
        :LAKI-LAKI
        TAN AGUS SETIADI
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["nama"].value, "TAN AGUS SETIADI")

    def test_parse_ktp_transposed_name_prefers_three_word_person_value(self):
        raw_text = """
        Jenis Kelamin
        Temp
        Nama
        NIK
        Berlaky Hingga
        Kewarganegaraan:WNI
        Pekerjaan
        Status Perkawinan: KAWIN
        Agama
        Alamat
        Kecamatan
        Kel/Desa
        RT/RW
        Tgl Lahir
        3173065505780001
        :15-05-2017
        :BUDHA
        :KALIDERES
        :006/ 007
        : JLN.BIMA BLOK C 11/7
        :PEREMPUAN
        :SINKAWANG, 15-05-1978
        : MENGURUS RUMAH TANGGA
        :TEGAL ALUR
        :TJONG FUI SIAN
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["nama"].value, "TJONG FUI SIAN")

    def test_parse_ktp_name_fallback_rejects_noisy_blok_address(self):
        raw_text = """
        NIK
        3173052804620003
        Tompal/TglLabe
        Name
        MEDAN.28-061963
        KADIM HASIMSIM
        Jenigkelamn
        Alamal
        LAKILAKI
        J KEMBANGANUAMA BLOKK
        AT/RW
        NO.7
        008/009
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["nama"].value, "KADIM HASIMSIM")

    def test_parse_ktp_repairs_name_when_label_capture_reads_birth_label(self):
        raw_text = """
        PROVINSI DKI JAKARTA
        : 3275011602820021
        NIK
        :FARIANDREAMSYAH
        Nama
        Tempat Tol Lahir
        WONOGIRI, 16-02-1982
        Alamal
        JL. TEBET UTARA
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["nama"].value, "FARIANDREAMSYAH")

    def test_parse_ktp_marks_short_nik_invalid_near_noisy_label(self):
        raw_text = """
        NTK
        117305500386001
        SEUMUR HIDUP
        WNI
        MENGURUS RUMAH TANGGA
        KAWIN
        KRISTEN
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["nik"].status, "invalid")
        self.assertEqual(result.fields["nik"].value, "117305500386001")
        self.assertIn("invalid:nik", result.warnings)

    def test_parse_ktp_falls_back_to_name_after_birth_line_in_value_only_layout(self):
        raw_text = """
        NTK
        117305500386001
        SEUMUR HIDUP
        WNI
        MENGURUS RUMAH TANG
        KAWIN
        KRISTEN
        KEBONJERUK
        KEDOYA SELATAN
        JL.KEDOYAAGA
        PEREMPUAN
        SLEMAN,10-03-1986
        YOKHEBED SETIOWATISANTOSC
        110/004
        PROVINSIDKIJAKARTA
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["nik"].status, "invalid")
        self.assertEqual(result.fields["nama"].value, "YOKHEBED SETIOWATISANTOSC")

    def test_parse_ktp_falls_back_to_name_after_birth_and_sex_in_transposed_layout(self):
        raw_text = """
        Berlaku Hingga
        Kewarganegaraan: WNI
        Pekerjaan
        Status Perkawinan: KAWIN
        Agama
        Aumut
        Jenis kelamin
        Tempat/Tgi Lhir
        Nania
        NIK
        Kecamatan
        Kel/Desa
        RT/RW
        SEUMUR HIDUP
        KARYAWAN SWASTA
        ISLAM
        ALAM BARAJO
        KENALI BESAR
        050/000
        KOMP.WISMA BUNGA BLOKC
        :RIAU, 18-09-1379
        LAKI-LAKI
        ZUBRAN HADI
        157107180979006
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["nik"].status, "invalid")
        self.assertEqual(result.fields["nama"].value, "ZUBRAN HADI")
        self.assertEqual(result.fields["agama"].value, "ISLAM")

    def test_parse_ktp_flags_screenshot_like_crop_and_rejects_drive_as_name(self):
        raw_text = """
        Manage
        KTP SETYO.jpg
        PROVINSI DKI JAKARTA
        JAKARTA TIMUR
        NIK
        3175030906710017
        100%
        RW Drive
        TOSHIBA
        Type here
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["nik"].status, "ok")
        self.assertEqual(result.fields["nama"].status, "missing")
        self.assertEqual(result.fields["alamat"].status, "missing")
        self.assertIn("quality:possible_non_ktp_crop", result.warnings)

    def test_parse_ktp_falls_back_to_address_before_noisy_rt_rw(self):
        raw_text = """
        NIK : 3324156103910001
        NAMA : SITI AMINAH
        TEMPAT/TGL LAHIR : PEKALONGAN, 21-03-1991
        JENIS KELAMIN
        PEREMPUAN
        DUKUH KRAJAN
        RT/AW
        003/002
        KEL/DESA : BANYUURIP
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["alamat"].value, "DUKUH KRAJAN")

    def test_parse_ktp_falls_back_to_address_from_transposed_jalan_value(self):
        raw_text = """
        ALAMAT
        KECAMATAN
        KEL/DESA
        RT/RW
        TGL LAHIR
        3173061505170001
        : 15-05-2017
        : TANJUNG PRIOK
        : 006/007
        : JLN. DANAU SUNTER A 11/7
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["alamat"].value, "JLN. DANAU SUNTER A 11/7")

    def test_parse_ktp_transposed_address_does_not_treat_kelapa_as_kel_label(self):
        raw_text = """
        ALAMAT
        RT/RW
        3173061505170001
        : 006/007
        : JLN. KELAPA GADING NO 1
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["alamat"].value, "JLN. KELAPA GADING NO 1")

    def test_parse_ktp_does_not_fill_name_from_birth_line_after_nik(self):
        raw_text = """
        PROVINSI DKI JAKARTA
        3174030101760004
        JAKARTA, 30-07-1976
        ALAMAT
        JL MELATI NO 86
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["nama"].status, "missing")

    def test_parse_ktp_does_not_fill_address_from_name_before_rt_rw(self):
        raw_text = """
        NIK : 3175010101900001
        Nama : BUDI SANTOSO
        Tempat/Tgl Lahir : JAKARTA, 01-01-1990
        BUDI SANTOSO
        RT/AW
        001/002
        Kel/Desa : MENTENG
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["alamat"].status, "missing")

    def test_parse_ktp_normalizes_gender_with_blood_type_noise(self):
        raw_text = """
        NIK : 3175010101900001
        Nama : BUDI SANTOSO
        Jenis Kelamin
        LAKI-LAKI Gol. Darah: AB
        Alamat : JL MERDEKA NO 10
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["jenis_kelamin"].value, "LAKI-LAKI")

    def test_parse_ktp_repairs_gender_when_label_capture_reads_address(self):
        raw_text = """
        NIK : 3171050104840001
        Nama
        SYAFWAN HADY
        Tempat/Tgl Lahir
        PANGKALAN SUSU, 04-04-1984
        Gol. Darah:O
        :LAKI-LAKI
        Jenis kelamin
        J.L.HANG TUAH II NO.5
        Alamat
        RT/RW
        002/004
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["jenis_kelamin"].value, "LAKI-LAKI")

    def test_parse_ktp_normalizes_ocr_gender_typo(self):
        raw_text = """
        NIK : 3174016605830004
        Nama : MEIGA PRANURAINI
        Jenis kelamin
        :PEREMPUIAN
        Alamat : JL TEBET UTARA
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["jenis_kelamin"].value, "PEREMPUAN")

    def test_parse_ktp_repairs_rt_rw_after_noise_line(self):
        raw_text = """
        NIK : 1403097008840002
        Nama : JERNIH DEBORA SINAGA
        Alamat
        PERMATACIMANGGIS CLUSTER ONYX
        RT/RW
        rsno
        003/025
        Kel/Desa
        CIMPAEUN
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["rt_rw"].value, "003/025")

    def test_parse_ktp_normalizes_rt_rw_with_missing_separator(self):
        raw_text = """
        NIK : 3173052301830004
        Nama : ERICK
        Alamat : JL FLAMBOYAN NO 31
        RT/RW
        :0041005
        Kel/Desa : KEBON JERUK
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["rt_rw"].value, "004/005")

    def test_parse_ktp_repairs_job_when_capture_reads_date(self):
        raw_text = """
        NIK : 3175072802780006
        Nama : MUHAMMAD NORMAN
        Status Perkawinan: KAWIN
        Pekerjaan
        31-08-2012
        :KARYAWAN BUMN
        Kewarganegaraan: WNI
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["pekerjaan"].value, "KARYAWAN BUMN")

    def test_parse_ktp_repairs_job_when_capture_reads_city(self):
        raw_text = """
        NIK : 3672055905860005
        Nama : MEILANNIH
        Status Perkawinan
        KAWIN
        Pekerjaan
        KOTA CILEGON
        :MENGURUS RUMAHTANCGA
        Kewarganegaraan
        WNI
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["pekerjaan"].value, "MENGURUS RUMAH TANGGA")

    def test_parse_ktp_repairs_job_from_previous_line_near_label(self):
        raw_text = """
        NIK : 3578170504790011
        Nama : OSBER SITUMORANG
        Status Perkawinan : KAWIN
        KOTA SURABAYA
        : TENTARA NASIONAL INDONESIA (TNI)
        Pekerjaan
        08-04-2012
        Kewarganegaraan:WNI
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["pekerjaan"].value, "TENTARA NASIONAL INDONESIA (TNI)")

    def test_parse_ktp_repairs_noisy_citizenship_value(self):
        raw_text = """
        NIK : 3216064704060020
        Nama : SALSABILA PUTRI DEWANTI
        Pekerjaan
        PELAJARMAHASISWA
        BEKASI
        Kewargane
        VNI
        22-05-2023
        Berlaku Hing
        SEUMUR HIDUP
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["kewarganegaraan"].value, "WNI")

    def test_parse_ktp_normalizes_labelled_citizenship_value(self):
        raw_text = """
        NIK : 3276011904710005
        Nama : SOLAHUDIN
        Kewarganegaraan.WNI
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["kewarganegaraan"].value, "WNI")

    def test_parse_ktp_repairs_truncated_citizenship_value_near_label(self):
        raw_text = """
        NIK : 3171050104850007
        Nama : IBRAHIM FARUK
        Kewarganegaraan: WN
        Berlaku Hingga
        SEUMUR HIDUP
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["kewarganegaraan"].value, "WNI")

    def test_parse_ktp_repairs_noisy_citizenship_label_and_value(self):
        raw_text = """
        NIK : 3174072003630005
        Nama : ISMET SYARIFULA FANE
        Kevrganegaraan:OWNI
        Berlaku Hingga
        SEUMUR HIDUP
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["kewarganegaraan"].value, "WNI")

    def test_parse_ktp_repairs_corrupt_citizenship_value_only_near_label(self):
        raw_text = """
        NIK : 3216064704060020
        Nama : SALSABILA PUTRI DEWANTI
        Kewarganegaraan
        Vke
        KOTA BEKASI
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["kewarganegaraan"].value, "WNI")

    def test_parse_ktp_preserves_pensioner_job(self):
        raw_text = """
        NIK : 3175041506640013
        Nama : SARAFUDDIN
        Jenis Kelamin : LAKI-LAKI
        RT/RW : 007/009
        Pekerjaan : PENSIUNAN
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["pekerjaan"].value, "PENSIUNAN")

    def test_parse_ktp_repairs_noisy_bumn_job_values(self):
        cases = [
            "Pokerjaan\n:KARYAWAN BUUN",
            "Pekerjaan\nKARYAWAN BUN",
        ]
        for job_text in cases:
            with self.subTest(job_text=job_text):
                raw_text = f"""
                NIK : 3274032204790008
                Nama : ANDRI PRASETYANTO
                Status Perkawinan: KAWIN
                {job_text}
                Kewarganegaraan: WNI
                """

                result = parse_ktp_text(raw_text)

                self.assertEqual(result.fields["pekerjaan"].value, "KARYAWAN BUMN")

    def test_parse_ktp_repairs_joined_tni_job_value(self):
        raw_text = """
        NIK : 8271030604650006
        Nama : MASRON SILALAHI
        Status Perkawinan
        KAWIN
        Pekerjaan
        TENTARANASIONAL INDONESIA
        BALAU
        (TNI)
        Kewarganegaraart
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["pekerjaan"].value, "TENTARA NASIONAL INDONESIA (TNI)")

    def test_parse_ktp_repairs_citizenship_value_before_noisy_label(self):
        raw_text = """
        NIK : 8271030604650006
        Nama : MASRON SILALAHI
        Pekerjaan
        TENTARANASIONAL INDONESIA
        OWNI
        24-01-2023
        Kewarganegaraart
        BerlakuHingga
        SEUMURHIDUP
        """

        result = parse_ktp_text(raw_text)

        self.assertEqual(result.fields["kewarganegaraan"].value, "WNI")


class StnkParserTests(unittest.TestCase):
    def test_stnk_fuzzy_label_matcher_handles_common_ocr_typos(self):
        match = match_stnk_label("NOM0R MES1N")

        self.assertIsNotNone(match)
        self.assertEqual(match["field_name"], "nomor_mesin")
        self.assertGreaterEqual(match["score"], 0.9)

    def test_parse_stnk_uses_fuzzy_labels_for_noisy_required_fields(self):
        raw_text = """
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        N0MOR P0L1SI : B 1234 ABC
        NAMA PEM1L1K : BUDI SANTOSO
        TAHUN PEMBUATAM : 2020
        N0M0R RANGKA/NIK/VIN : MHRRU1860KJ302319
        NOM0R MES1N : L15Z61219016
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_polisi"].value, "B 1234 ABC")
        self.assertEqual(result.fields["nama_pemilik"].value, "BUDI SANTOSO")
        self.assertEqual(result.fields["tahun_pembuatan"].value, "2020")
        self.assertEqual(result.fields["nomor_rangka"].value, "MHRRU1860KJ302319")
        self.assertEqual(result.fields["nomor_mesin"].value, "L15Z61219016")
        self.assertNotIn("missing_required:nomor_mesin", result.warnings)

    def test_parse_stnk_repairs_noisy_tax_sheet_year_and_engine(self):
        raw_text = """
        TANDA BUKTI PELUNASAN KEWAJIBAN PEMBAYARAN
        MORREGSTRASI: B 9335 TYY
        PT.PP PRESISI
        HINO
        DUMPER TR TRO
        KENDARAAN KHUSUS
        FMJN1DEGJFM260JDTW
        JAKARN 8 NOP 2017
        TABNREGISTRASI :2017:
        2017
        NOMOR MES
        JOBEUFJ87329
        BERLAKU SAMPA
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["tahun_pembuatan"].value, "2017")
        self.assertEqual(result.fields["nomor_mesin"].value, "J08EUFJ87329")
        self.assertNotIn("missing_required:tahun_pembuatan", result.warnings)
        self.assertNotIn("missing_required:nomor_mesin", result.warnings)

    def test_parse_stnk_repairs_misread_mje_rangka_and_unlabelled_engine(self):
        raw_text = """
        TANDA BUKTI PELUNASAN KEWAJIBAN PEMBAYARAN
        NOMOR REGRSTRAS.: 8 9241 TYX
        TARIN PEMBUAIAN
        2018
        NOMPR PANGKAUNTIK VIN
        M3ETM67N13JE2515OON
        NONCR UESN
        J088UFJ99935
        BERLAKU SAMPALL 08-10-2023
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["tahun_pembuatan"].value, "2018")
        self.assertEqual(result.fields["nomor_rangka"].value, "MJETM67N13JE2515O")
        self.assertEqual(result.fields["nomor_mesin"].value, "J08EUFJ99935")
        self.assertNotIn("missing_required:nomor_mesin", result.warnings)

    def test_stnk_structure_score_separates_structured_stnk_from_non_stnk_text(self):
        noisy_stnk = """
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        N0MOR P0L1SI
        NAMA PEM1L1K
        TAHUN PEMBUATAM
        N0M0R RANGKA/NIK/VIN
        NOM0R MES1N
        """
        ktp_text = """
        PROVINSI DKI JAKARTA
        NIK 3175010101900001
        NAMA BUDI SANTOSO
        ALAMAT JL MERDEKA
        """

        self.assertGreaterEqual(stnk_structure_score(noisy_stnk), 0.75)
        self.assertLess(stnk_structure_score(ktp_text), 0.3)

    def test_parse_stnk_core_fields_from_labelled_text(self):
        raw_text = """
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        NO POLISI : B 1234 ABC
        NAMA PEMILIK : BUDI SANTOSO
        ALAMAT : JL MERDEKA NO 10 JAKARTA
        MERK : TOYOTA
        TYPE : AVANZA 1.3 G
        JENIS : MINIBUS
        TAHUN PEMBUATAN : 2020
        WARNA : HITAM
        NO RANGKA : MHKA1234567890123
        NO MESIN : 1NR1234567
        BAHAN BAKAR : BENSIN
        BERLAKU SAMPAI : 01-01-2027
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.document_type, "STNK")
        self.assertEqual(result.fields["nomor_polisi"].value, "B 1234 ABC")
        self.assertEqual(result.fields["nomor_polisi"].status, "ok")
        self.assertEqual(result.fields["nama_pemilik"].value, "BUDI SANTOSO")
        self.assertEqual(result.fields["merek"].value, "TOYOTA")
        self.assertEqual(result.fields["tipe"].value, "AVANZA 1.3 G")
        self.assertEqual(result.fields["nomor_rangka"].value, "MHKA1234567890123")
        self.assertEqual(result.fields["nomor_mesin"].value, "1NR1234567")
        self.assertFalse(result.needs_review)

    def test_parse_stnk_normalizes_noisy_plate_vehicle_ids_and_year(self):
        raw_text = """
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        NO. POL : B-1234-ABC
        NAMA PEMILIK : BUDI SANTOSO
        ALAMAT : JL MERDEKA NO 10 JAKARTA
        MEREK : TOYOTA
        TIPE : AVANZA
        TAHUN : 2O2O
        NOMOR RANGKA : mhka-1234 5678 90123
        NO. MESIN : 1nr 123-4567
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_polisi"].value, "B 1234 ABC")
        self.assertEqual(result.fields["tahun_pembuatan"].value, "2020")
        self.assertEqual(result.fields["nomor_rangka"].value, "MHKA1234567890123")
        self.assertEqual(result.fields["nomor_mesin"].value, "1NR1234567")
        self.assertNotIn("invalid:nomor_rangka", result.warnings)
        self.assertNotIn("invalid:nomor_mesin", result.warnings)

    def test_parse_stnk_marks_required_vehicle_ids_invalid(self):
        raw_text = """
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        NO POLISI : B 1234 ABC
        NAMA PEMILIK : BUDI SANTOSO
        NO RANGKA : ABC
        NO MESIN : 12
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_rangka"].status, "invalid")
        self.assertEqual(result.fields["nomor_mesin"].status, "invalid")
        self.assertIn("invalid:nomor_rangka", result.warnings)
        self.assertIn("invalid:nomor_mesin", result.warnings)

    def test_parse_stnk_requires_manufacture_year_for_auto_processing(self):
        raw_text = """
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        NO POLISI : B 1234 ABC
        NAMA PEMILIK : BUDI SANTOSO
        NO RANGKA : MHKA1234567890123
        NO MESIN : 1NR1234567
        """

        result = parse_stnk_text(raw_text)

        self.assertIn("missing_required:tahun_pembuatan", result.warnings)
        self.assertTrue(result.needs_review)

    def test_parse_stnk_repairs_tax_sheet_label_block_layout(self):
        raw_text = """
        NOMOR BPKB
        NOMOR MESIN
        BERLA
        NO.RANGKA/NIK
        WARNA KB
        ISI SILINDER/HP
        TAHUN PERAKITAN
        TAHUN PEMBUATAN
        JENIS / MODEL
        MERK/TYPE
        ALAMAT
        NAMA PEMILIK
        TNKB
        PKB. BBN-KB. SWDKLLJ, BIAYA ADM.
        51
        P06479290F
        PE31023038
        JM6DK2W7AH0301954
        MERAH METALIK
        1.998cc
        2017
        2017
        MINIBUS/MP
        MAZDA/CX-3 5WGN RHD
        JAMBI TIMURKOTA JAMBI
        JL. BANDA NO 27 RT.07 KEL. BUDIMAN KEC.
        EMI SUKAMT
        BH 1146 MS
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_polisi"].value, "BH 1146 MS")
        self.assertEqual(result.fields["nama_pemilik"].value, "EMI SUKAMT")
        self.assertEqual(result.fields["tahun_pembuatan"].value, "2017")
        self.assertEqual(result.fields["nomor_rangka"].value, "JM6DK2W7AH0301954")
        self.assertEqual(result.fields["nomor_mesin"].value, "PE31023038")
        self.assertNotIn("missing_required:nomor_polisi", result.warnings)
        self.assertNotIn("invalid:nomor_rangka", result.warnings)

    def test_parse_stnk_repairs_official_layout_when_labels_capture_headings(self):
        raw_text = """
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        NOMOR REGISTRASI
        BH 1146 MS
        NAMA PEMILIK
        EMI SUKAMLT
        NIK/TDP/KITAS/KITAP
        STNK
        ALAMAT
        NAME OF OWNE
        JL. BANDA NO.27 RT O7 KEL BUDIMAN
        MERK
        MAZDA
        WARNA
        T
        MERAHMETALIK
        TYPE
        CX-3 SWGN RHD
        TAHUN REGISTRASI
        2021
        MODEL
        MINIBUS
        TAHUN PEMBUATAN
        NOMOR BPKB
        P06479290F
        2017
        NOMOR RANGKA/NIK/VIN
        JM6DK2W7AH0301954
        NOMOR MESIN
        DATE OF EXPIRE
        PE31023038
        BERLAKU SAMPAI
        11 Januari 2026
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_polisi"].value, "BH 1146 MS")
        self.assertEqual(result.fields["nama_pemilik"].value, "EMI SUKAMLT")
        self.assertEqual(result.fields["tahun_pembuatan"].value, "2017")
        self.assertEqual(result.fields["nomor_rangka"].value, "JM6DK2W7AH0301954")
        self.assertEqual(result.fields["nomor_mesin"].value, "PE31023038")

    def test_parse_stnk_repairs_owner_plate_and_vehicle_id_from_tax_receipt(self):
        raw_text = """
        TANDA BUKTI PELUNASAN KEWAJIBAN PEMBAYARAN
        NIK
        3173081508720009
        8 1293 PLP
        NOMOR POLISI
        KX.XXX.XXX.XXX
        TAN AGUS SETIADI
        NAMA PEMILIK
        5.145.000
        JL AL MUBAROK II/32C RT8/2 JOGLO
        ISUZU
        MERK
        TYPE
        UCR6Y MU-X R2 4X2
        TAHUN REGISTRASI
        2022
        TAHUN PEMBUATAN
        2014
        NOMOR RANGONIOVIN
        MPAUCR86GETO00$10
        NOMOR MESIN
        550 002
        BERLAKU SAMPAI
        21-12-2024
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_polisi"].value, "B 1293 PLP")
        self.assertEqual(result.fields["nama_pemilik"].value, "TAN AGUS SETIADI")
        self.assertEqual(result.fields["tahun_pembuatan"].value, "2014")
        self.assertEqual(result.fields["nomor_rangka"].value, "MPAUCR86GETO0010")
        self.assertEqual(result.fields["nomor_mesin"].value, "550002")

    def test_parse_stnk_does_not_accept_masked_owner_or_date_heading_as_engine(self):
        raw_text = """
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        NOMOR REGISTRASI
        BH 1146 MS
        NOMOR RANGKA/NIK/VIN
        JM6DK2W7AH0301954
        NO URUT PENDAFTARAN
        ENTITY NUMBER
        PE31023038
        BERLAKU SAMPAI
        11 Januari 2026
        NOMOR MESIN
        DATE OF EXPIRE
        NOMOR POLISI
        KX.XXX.XXX.XXX
        Xxx.xxx.xxx xx.xxx.xxx
        TAN AGUS SETIADI
        NAMA PEMILIK
        5.145.000
        TAHUN PEMBUATAN
        2017
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_mesin"].value, "PE31023038")
        self.assertEqual(result.fields["nama_pemilik"].value, "TAN AGUS SETIADI")

    def test_parse_stnk_rejects_color_as_owner_and_date_as_vehicle_id(self):
        raw_text = """
        NOMOR POLISI
        B 2321 SZL
        NAMA PEMILIK
        SILVER METALIK
        NOMOR RANGKA/NIK/VIN
        LFAKKWT15NOV2023
        NOMOR MESIN
        4JAN2024
        TAHUN PEMBUATAN
        2017
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nama_pemilik"].status, "missing")
        self.assertNotEqual(result.fields["nomor_rangka"].status, "ok")
        self.assertNotEqual(result.fields["nomor_mesin"].status, "ok")

    def test_parse_stnk_rejects_short_rangka_and_receipt_heading_as_owner(self):
        raw_text = """
        NOMOR POLISI
        B 2321 SZL
        NAMA PEMILIK
        TANDA BUKTI PELUNASAN KEWAJIBAN PEMBAYARAN
        NOMOR RANGKA/NIK/VIN
        HR15742189T
        NOMOR MESIN
        HR15742189T
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nama_pemilik"].status, "missing")
        self.assertNotEqual(result.fields["nomor_rangka"].status, "ok")

    def test_parse_stnk_rejects_address_fragment_as_rangka(self):
        raw_text = """
        TAHUN PEMBUATANPERAKITAN
        2019
        RT006020KELBINONGK
        NO.MESIN
        L15Z61219016
        """

        result = parse_stnk_text(raw_text)

        self.assertNotEqual(result.fields["nomor_rangka"].status, "ok")

    def test_parse_stnk_repairs_rangka_when_value_appears_before_noisy_label(self):
        raw_text = """
        IDENT.
        2D4918DK772ND
        DIRLANT
        MPAUCR86GETO00$10
        4
        NOMOR RANGONIOVIN
        1G0398
        NOMOR MESIN
        550 002
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_rangka"].value, "MPAUCR86GETO0010")

    def test_parse_stnk_repairs_truncated_year_and_nomor_rango_label(self):
        raw_text = """
        TAHUN REGISTRASI
        TAHUN PEMBUATA
        2014
        AN RIBU RWPIAH
        NOMOR BPKB
        COKLAT
        AN KAPOLDA
        LANTAS
        JAYA
        IDENT
        2D4918DK772ND
        MPAUCR86GETO00510
        2
        NOMOR RANGO
        NOMOR MESIN
        1G0398
        BERLAKU SAMPAI
        21-12-2024
        550 002
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["tahun_pembuatan"].value, "2014")
        self.assertEqual(result.fields["nomor_rangka"].value, "MPAUCR86GETO00510")

    def test_parse_stnk_repairs_merged_tahun_pembuatanperakitan_label(self):
        raw_text = """
        TAHUN PEMBUATANPERAKITAN
        PERUM TAMAN PARAHIYANGAN
        MHRRU1860KJ302319
        2019
        NO.MESIN
        L15Z61219016
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["tahun_pembuatan"].value, "2019")
        self.assertEqual(result.fields["nomor_rangka"].value, "MHRRU1860KJ302319")

    def test_parse_stnk_extracts_embedded_rangka_after_noisy_rwngka_marker(self):
        raw_text = """
        RWNGKA/NKMHMFM65GAPK000349/912020******2000
        DIEETAPKAN TOL
        POPB C01526569
        .MESIN
        6M60-299237
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_rangka"].value, "MHMFM65GAPK000349")

    def test_parse_stnk_repairs_truncated_owner_and_plate_labels(self):
        raw_text = """
        NAMA PEMILI
        HELENA NAIBA O SH
        NOMOR POLIS
        3-1616-JFA
        TAHUN PEMBUATAN
        2021
        NO RANGKA
        MHRDD1850LJ123456
        NO MESIN
        L12B31234567
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nama_pemilik"].value, "HELENA NAIBA O SH")
        self.assertEqual(result.fields["nomor_polisi"].value, "B 1616 JFA")

    def test_parse_stnk_repairs_rangka_split_across_nearby_ocr_lines(self):
        raw_text = """
        NO MESIN
        NO RANGIA/NIK
        ISI SILINDER/AP
        WARNAKB
        10319804817
        L12832393474
        MHRDD1850L
        1198 CC
        PUTIH
        NAMA PEMILI
        NO MESIN
        L12B31234567
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_rangka"].value, "MHRDD1850LL128323")
        self.assertNotIn("missing_required:nomor_rangka", result.warnings)

    def test_parse_stnk_repairs_truncated_tax_sheet_label_block_with_delayed_values(self):
        raw_text = """
        NAMA PEMILI
        NOMOR POLIS
        BERLAKU S/D
        S
        Q01091527
        .10319804817
        L12832393474
        .MHRDD1850L
        1198 CC
        PUTH
        MINIBUS
        HONDA/BRIOS
        BOJONG NANGKA
        DASANA INDAH
        HELENA NAIBA O SH
        3-1616-JFA
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nama_pemilik"].value, "HELENA NAIBA O SH")
        self.assertEqual(result.fields["nomor_polisi"].value, "B 1616 JFA")

    def test_parse_stnk_accepts_company_owner_after_register_marker(self):
        raw_text = """
        REGISTER
        PT. MALUKU INDAH
        SANKSCAOM
        PEMILIK
        POKOK
        NOMOR POLISI
        N 6 ZNB
        TAHUN PEMBUATAN
        2023
        NO RANGKA
        MHMFM65GAPK000349
        NO MESIN
        6M60299237
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nama_pemilik"].value, "PT. MALUKU INDAH")

    def test_parse_stnk_prefers_official_stnk_section_over_tax_receipt_noise(self):
        raw_text = """
        SAMSAT Provies
        NIKNO. HP
        327508170560000X/081,0000000000
        NOMOR REGISTRAS
        B.3470 1NR
        NAMA PEMLIK
        SYUNT SE AK
        ALAMAT
        PERN BNI SATTVRINGIN BK J NO 13 6 RT
        PINIBUS LISTRIK NO BPG
        W4TNA KB
        NOMORMESIN
        1Z200XY1N018555 XC0S
        13,un 2025
        BERLAKU SID
        13 unT 2026
        KEPOLISIAN NEGARA REPUBLIK INDONESIA
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        NRKB
        B 1470 KNR
        NIKNPWTNIB/KITAS/KITAP
        327
        TOW MMER
        naa
        NAMA PEMILIK
        SYUKRI, SE AK
        AARUSIN, B.L M
        STNK
        ALAMAT
        PERUM BUMI JATIWARINGIN BLK J/13 6 RT 03 RW
        3107
        JATIWARINGIN
        Kenderaan Baru
        MERK
        BYD
        DA ENA
        HITAM
        TYPE
        UKE-RWD-M (4X2) AT
        JENIS
        MB. PENUMPANG
        MODEL
        MINIBUS LISTRIK
        WARNAT
        HITAM
        TAHUN PEMBUATAN
        2025
        NOMOR RANGKA/NIKNIN :
        LGXCH4CD3S2107503
        MIMIOBMESINMOTOR PENGGLRAK
        TZ200XYT3M5018555
        BERLAKU SAMPALDATE OF COPIRE
        13 Juni 2030.
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_polisi"].value, "B 1470 KNR")
        self.assertEqual(result.fields["nama_pemilik"].value, "SYUKRI, SE AK")
        self.assertEqual(result.fields["merek"].value, "BYD")
        self.assertEqual(result.fields["jenis"].value, "MB. PENUMPANG")
        self.assertEqual(result.fields["warna"].value, "HITAM")
        self.assertEqual(result.fields["tahun_pembuatan"].value, "2025")
        self.assertEqual(result.fields["nomor_rangka"].value, "LGXCH4CD3S2107503")
        self.assertEqual(result.fields["nomor_mesin"].value, "TZ200XYT3M5018555")
        self.assertEqual(result.fields["berlaku_sampai"].value, "13 Juni 2030")

    def test_parse_stnk_repairs_official_layout_with_untagged_noise_fields(self):
        raw_text = """
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        NRKB
        B 1470 KNR
        NAMA PEMILIK
        SYUKRI, SE AK
        BYD
        WARNA HITAM
        TIPE & TIPE DAGANG
        UKE-RWD-M (4X2) AT
        JENIS
        MB. PENUMPANG
        MODEL MINIBUS LISTRIK
        ISISILINDERDAYALISTRIK 230000 WATT
        NOMORRANGKANIKVIN LGXCH4CD3S2107503
        NOMOBMESINMOP PENGGLRAK TZ200XYT3M5018555
        13 Junt 2030.
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["merek"].value, "BYD")
        self.assertEqual(result.fields["tipe"].value, "UKE-RWD-M (4X2) AT")
        self.assertEqual(result.fields["warna"].value, "HITAM")
        self.assertEqual(result.fields["bahan_bakar"].value, "LISTRIK")

    def test_parse_stnk_prefers_vehicle_brand_near_spec_labels_over_address_noise(self):
        raw_text = """
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        NRKB
        B 1470 KNR
        NAMA PEMILIK
        SYUKRI, SE AK
        STNK
        ALAMAY
        PERUM BUMI JATIWARINGIN BLK J/13 6 RT 03 RW
        JATIWARINGIN
        Kendaraan Baru
        PONGESAIEAN/NALIDATION
        BYD
        WARNA
        HITAM
        TIPE & TIPE DAGANG
        UKE-RWD-M (4X2) AT
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["merek"].value, "BYD")

    def test_parse_stnk_skips_short_noise_between_type_label_and_value(self):
        raw_text = """
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        NRKB
        B 1470 KNR
        NAMA PEMILIK
        SYUKRI, SE AK
        BYD
        WARNA
        HITAM
        TIPE & TIPE DAGANG
        ren
        UKE-RWD-M (4X2) AT
        JENIS
        MB. PENUMPANG
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["tipe"].value, "UKE-RWD-M (4X2) AT")

    def test_parse_stnk_skips_color_between_type_label_and_value(self):
        raw_text = """
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        NRKB
        B 9207 TYZ
        TYPE E
        KUNING
        KENDARAAN KHUSUS
        WARNA TNKB
        JENISRY Y
        2018
        DUMPER TR TRO
        FM8JN1D-EGJ/FM26OJAHAN BAKAR
        SOLAR
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["tipe"].value, "FM8JN1D-EGJ/FM26OJAHAN BAKAR")

    def test_parse_stnk_skips_color_label_between_type_label_and_value(self):
        raw_text = """
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        NRKB
        B 9238 TYZ
        TYPE
        WARNA
        HIJAU
        FM8JN1D-EGJ/FM26OJAHANBAKAR
        GENISY Y
        KENDARAAN KHUSUS
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["tipe"].value, "FM8JN1D-EGJ/FM26OJAHANBAKAR")

    def test_parse_stnk_prefers_engine_before_noisy_official_engine_label(self):
        raw_text = """
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        NRKB
        B 9207 TYZ
        NOMOR RANGKA/NIK/VIN
        MJEFM8JN1JJE13133
        JO8EUFJ99909
        NOMOR MESIN
        DATE OF EXPIRE
        B 2039135
        BERLAKU SAMPAI
        13 Juli 2028
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_mesin"].value, "JO8EUFJ99909")

    def test_parse_stnk_repairs_type_when_value_appears_before_type_label(self):
        raw_text = """
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        NRKB
        B 9239 TYZ
        MERK
        HINO
        FM8JN1D-EGJ/FM26OJ0SUNER
        BIAYA ADM STNK
        TYPE
        JENIS
        KENDARAAN KHUSUS
        BAHAN BAKAR
        SOLAR
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["tipe"].value, "FM8JN1D-EGJ/FM26OJ0SUNER")

    def test_parse_stnk_repairs_tax_receipt_type_when_value_appears_before_type_label(self):
        raw_text = """
        NOMOR POLISI
        8 9239
        TYZ
        MERK
        HINO
        FM8JN1D-EGJ/FM26OJ0SUNER
        BIAYA ADM STNK
        TYPE
        JENIS
        KENDARAAN KHUSUS
        BAHAN BAKAR
        SOLAR
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["tipe"].value, "FM8JN1D-EGJ/FM26OJ0SUNER")

    def test_parse_stnk_repairs_type_from_typee_label(self):
        raw_text = """
        TYPEE
        CX-3 SWGNRIID(CL200OPAIANPAKR2ERA
        RENSIN
        JENIS
        MB. PENUMPANG
        TAHUN PEMBUATAN
        2017
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["tipe"].value, "CX-3 SWGNRIID(CL200OPAIANPAKR2ERA")

    def test_parse_stnk_repairs_type_from_model_block_textual_value(self):
        raw_text = """
        MOBIL PENUMPA
        BAHAN BAM
        JENIS
        WARNA TNKB
        HITAM
        JEEP L.C.HDTP
        2022
        MODEL
        TAHUN REGISTRASI
        TAHUN PEMBUATAN
        2014
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["tipe"].value, "JEEP L.C.HDTP")

    def test_parse_stnk_strips_merged_fuel_label_from_type_value(self):
        raw_text = """
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        NOMOR REGISTRASI
        B 9757 TYZ
        TYPE
        FM8JN1D-EGJ/FM260JBAHAN BAKAR
        SOLAR
        TAHUN PEMBUATAN
        2018
        NOMOR MESIN
        J08EUFR03596
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["tipe"].value, "FM8JN1D-EGJ/FM260J")

    def test_parse_stnk_strips_merged_fuel_energy_label_from_type_value(self):
        raw_text = """
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        NOMOR REGISTRASI
        B 9337 TYY
        TIPE
        FM8JN20-EGJ/FM26OJBAHAN BAKARSUMBER ENERG
        SOLAR
        TAHUN PEMBUATAN
        2017
        NOMOR MESIN
        J08EUP187047
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["tipe"].value, "FM8JN20-EGJ/FM26OJ")

    def test_parse_stnk_repairs_year_before_noisy_manufacture_year_value(self):
        raw_text = """
        MODEL
        MBL TANGKI
        2019
        REGISTR
        P0350367
        TAHUN PEMBUATAN
        04009
        NOMOR RANGKA/NIK/VIN
        MJEC1J643K5177618
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["tahun_pembuatan"].value, "2019")

    def test_parse_stnk_repairs_year_after_fuzzy_manufacture_year_label(self):
        raw_text = """
        MODEL
        TAHUN PENBUATIN
        TAHIN REGSTRASI
        2017
        2017
        2024
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["tahun_pembuatan"].value, "2017")

    def test_parse_stnk_repairs_year_from_perakitan_slash_pair(self):
        raw_text = """
        TAHUN PEMBUATANPERAKITAN:
        2024/2024
        NOMOR RANGKANIK:
        MFJ831540RJ003682
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["tahun_pembuatan"].value, "2024")

    def test_parse_stnk_repairs_owner_from_noisy_mapemilik_label(self):
        raw_text = """
        MOR POLISI
        B 2008 BRG
        MAPEMILIK
        SARAFUDDIN
        AMAT
        RELA NO2 RT7/9 MENTENG ATAS
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nama_pemilik"].value, "SARAFUDDIN")

    def test_parse_stnk_does_not_use_vehicle_brand_as_owner_after_noisy_owner_line(self):
        raw_text = """
        NAMA PEMILIK
        1.
        PUSPITA FEBYRIZKI NUGRONMTTAKEA J324WYONUGROI0, 1
        A ALAMAT
        KP PLUMBUNGAN RT/RW 003/002
        MERK
        TOYOTA
        """

        result = parse_stnk_text(raw_text)

        self.assertNotEqual(result.fields["nama_pemilik"].value, "TOYOTA")

    def test_parse_stnk_repairs_rangka_from_noisy_official_nik_line(self):
        raw_text = """
        SAMSAT PROVINSI
        NIK
        NM0RW00MINMJEFM8JN1.3JE237:49CEMT
        NOMORMESN
        J08EUFJ97272
        BERLAKU SAMPAI
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_rangka"].value, "MJEFM8JN13JE23749")

    def test_parse_stnk_repairs_engine_from_noisy_mein_label(self):
        raw_text = """
        NOMOR BANGKANIKVIN
        MJEC1J643K5177618
        NOMOR MEIN
        WO4DTRR67339
        BERLAKU SAMPAI:02-04-2024
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_mesin"].value, "WO4DTRR67339")

    def test_parse_stnk_prefers_hino_engine_over_short_noise_fragment(self):
        raw_text = """
        MMAVIN
        4ON00
        HOMPR MESIN
        J08EUF399904
        BERLAKU SAMPAI
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_mesin"].value, "J08EUF399904")

    def test_parse_stnk_repairs_engine_before_fuzzy_mesin_label(self):
        raw_text = """
        NOMORBANGKANIKAVIN
        MJEFH8JN1JJE27720
        J08EUFR03596
        NOMOR MESIY
        SERLAKUSAMPAI27-03 2024
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_mesin"].value, "J08EUFR03596")

    def test_parse_stnk_repairs_engine_from_truncated_esin_label(self):
        raw_text = """
        WGKANKVNMMLAA4261LG013433
        ESIN
        15E4EAFTL3040017
        BERLAKU SAMPAI
        13-11-2025
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_mesin"].value, "15E4EAFTL3040017")

    def test_parse_stnk_keeps_labelled_engine_over_kode_lokasi_fragment(self):
        raw_text = """
        KODE LOKASI
        0064/U3/190919
        NOMOR RANGKANIK/VIN
        MHFJB8GS7K1575086
        BERLAKU SAMPAI:
        19-09-024
        NOMOR MESIN
        2GDC623732
        DATE OF EXPIRE
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_mesin"].value, "2GDC623732")

    def test_parse_stnk_does_not_use_mfj_rangka_as_engine(self):
        raw_text = """
        NOMOR RANGKANIK:
        MFJ831540RJ003682
        NOMOR MESIN
        400959D0164403
        X/ARPOLIS1NRP.75061073
        [13:51:53#18-12-2025#Rp.
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_mesin"].value, "400959D0164403")

    def test_parse_stnk_prefers_stronger_engine_candidate_over_short_fragment(self):
        raw_text = """
        NOMOR RANGKA/NIK/VIN
        MPAUCR86GETO0010
        NOMOR MESIN
        G0398
        LatfUsman, S.L.K, M.Ham.
        550 002
        BERLAKU SAMPAI
        21-12-2024
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_mesin"].value, "550002")

    def test_parse_stnk_prefers_engine_after_label_over_registration_sequence_before_label(self):
        raw_text = """
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        NOMOR REGISTRASI
        B 9241 TYZ
        NOMOR RANGKA NIKVIN:
        MJEM8JNJE25150
        NO URUT PENDAFTARAN
        /U35/0810
        NOMOR MESIN
        J08EUFJ99935
        BERLAKU SAMPAI: 08-10-2023
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_mesin"].value, "J08EUFJ99935")

    def test_parse_stnk_ignores_tnkb_color_when_vehicle_color_appears_later(self):
        raw_text = """
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        NRKB
        B 9170 TYY
        MERK
        HINO
        WARNA TNKB
        KUNING
        MODEL
        DUMPER TR TRO
        WARNA
        MJEFM8JN1HJE16631
        HIJAU
        IDENT
        5G4913LL552NY
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["warna"].value, "HIJAU")

    def test_parse_stnk_tax_sheet_ignores_tnkb_color_when_vehicle_color_appears_later(self):
        raw_text = """
        NOMOR POLISI
        B 9170 TYY
        MERK
        HINO
        WARNA TNKB
        KUNING
        MODEL
        DUMPER TR TRO
        WARNA
        MJEFM8JN1HJE16631
        HIJAU
        IDENT
        5G4913LL552NY
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["warna"].value, "HIJAU")

    def test_parse_stnk_does_not_overwrite_labelled_values_with_noisy_official_tail(self):
        raw_text = """
        NO POLISI : B 1234 ABC
        NAMA PEMILIK : BUDI SANTOSO
        TAHUN PEMBUATAN : 2020
        NO RANGKA : MHRRU1860KJ302319
        NO MESIN : L15Z61219016
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        NRKB
        W 4 TNA
        NAMA PEMILIK
        TOW MMER
        MESIN
        ABCDE
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_polisi"].value, "B 1234 ABC")
        self.assertEqual(result.fields["nama_pemilik"].value, "BUDI SANTOSO")
        self.assertEqual(result.fields["nomor_mesin"].value, "L15Z61219016")

    def test_parse_stnk_repairs_year_from_standalone_value_before_noisy_registration_label(self):
        raw_text = """
        SURAT TANDA NOMOR KENDARAAN BERMOTOR
        NRKB
        B 1470 KNR
        NAMA PEMILIK
        SYUKRI, SE AK
        MODEL
        MINIBUS LISTRIK
        WAARNATNM
        HITAM
        2025
        TAHUN REGISTRASH
        2025
        NOMORRANGKAUNIKAIN
        LGXCH4CD3S2107503
        NIMOBMESINMOTPOR PENGAGLRLAK
        TZ200XYT3M5018555
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["tahun_pembuatan"].value, "2025")
        self.assertEqual(result.fields["tahun_pembuatan"].status, "ok")

    def test_parse_stnk_repairs_expiry_date_after_noisy_expiry_label_and_engine_label(self):
        for month_ocr in ["Jund", "Junt"]:
            with self.subTest(month_ocr=month_ocr):
                raw_text = f"""
                SURAT TANDA NOMOR KENDARAAN BERMOTOR
                NRKB
                B 1470 KNR
                NAMA PEMILIK
                SYUKRI, SE AK
                TAHUN REGISTRASH
                2025
                NOMORRANGKAUNIKAIN
                LGXCH4CD3S2107503
                BERLAKU SAMPALDATE OF CRPI
                NIMOBMESINMOTPOR PENGAGLRLAK
                TZ200XYT3M5018555
                13 {month_ocr} 2030.
                """

                result = parse_stnk_text(raw_text)

                self.assertEqual(result.fields["berlaku_sampai"].value, "13 Juni 2030")
                self.assertEqual(result.fields["berlaku_sampai"].raw, "official_section:berlaku_sampai")

    def test_parse_stnk_repairs_year_from_noisy_manufacture_pair_line(self):
        raw_text = """
        UERNITYPE
        KADA/BIANTE 2.0L 6A/T
        JENISMCOEL
        NINIBUS
        AMNPERATNAMIN 2013/2013
        NOMOR MESIN
        PE30607980
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["tahun_pembuatan"].value, "2013")

    def test_parse_stnk_keeps_engine_after_label_over_kode_fragment(self):
        raw_text = """
        KODE LKASC06GZ111NI
        NOMOR RANGKA/NIK/VIN
        MHDH3BA1S6J123456
        NOMOR MESIN
        L15714731905
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_mesin"].value, "L15714731905")

    def test_parse_stnk_reads_engine_value_with_leading_colon_after_label(self):
        raw_text = """
        Nomor mesin
        :64FL5Q55|002
        10) Warna kendaraan
        HITAM METALIK
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_mesin"].value, "64FL5Q55002")

    def test_parse_stnk_repairs_plate_suffix_digit_before_type_block(self):
        raw_text = """
        NOMOR POLISI
        B 2073 BB5
        TYPE
        3201
        N20 CKD AT
        NOMOR MESIN
        A4270791
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_polisi"].value, "B 2073 BBS")

    def test_parse_stnk_repairs_split_registration_plate_and_ignores_address_rt_rw(self):
        raw_text = """
        NOMOR REGISTRASI
        F
        1159 AE
        ALAMAT
        JL RAYA TAMAN CIMANGGU NO 59 RT OO1 RW O-
        NO REGISTRASI LAMA:
        B
        2937 KFL
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_polisi"].value, "F 1159 AE")

    def test_parse_stnk_prefers_stronger_plate_candidate_over_short_noise(self):
        raw_text = """
        .11
        AETRO JAY
        NAMA PEMILIK
        VENICA
        B 1686 03
        NUNER
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nomor_polisi"].value, "B 1686 OJ")
        self.assertEqual(result.fields["nama_pemilik"].value, "VENICA")

    def test_parse_stnk_prefers_bilingual_owner_over_role_noise(self):
        raw_text = """
        NOMOR REGISTRASI
        B 1484 UNP
        NAMA PEMILIK
        ARD
        KEPALA
        2
        STNK
        NAME OF OWNER
        LAURA SANTOSO
        ALAMAT
        RUKO GDG BUKIT INDAH
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nama_pemilik"].value, "LAURA SANTOSO")

    def test_parse_stnk_prefers_company_owner_over_region_noise(self):
        raw_text = """
        NAMA PEMILIK
        NO. KOHIR
        BANDUNG I PDJDJRAN
        Jawa Barat
        FT JASUKA BANGUN PRATANA
        NIK/NO. HP
        91203089XXXXX
        """

        result = parse_stnk_text(raw_text)

        self.assertEqual(result.fields["nama_pemilik"].value, "PT JASUKA BANGUN PRATANA")


class ValidatorTests(unittest.TestCase):
    def test_normalize_nik_accepts_only_16_digits(self):
        self.assertEqual(normalize_nik("3175 0101-0190 0001"), "3175010101900001")
        self.assertIsNone(normalize_nik("3175"))

    def test_validate_plate_number_accepts_indonesian_plate_shape(self):
        self.assertTrue(validate_plate_number("B 1234 ABC"))
        self.assertTrue(validate_plate_number("AB1234CD"))
        self.assertFalse(validate_plate_number("N 0 P"))
        self.assertFalse(validate_plate_number("1234567890"))

    def test_mask_sensitive_text_redacts_nik_like_numbers(self):
        masked = mask_sensitive_text("NIK 3175010101900001 NO POLISI B 1234 ABC")

        self.assertIn("317501******0001", masked)
        self.assertNotIn("3175010101900001", masked)


if __name__ == "__main__":
    unittest.main()
