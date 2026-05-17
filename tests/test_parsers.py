import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_engine.parsers.ktp import parse_ktp_text
from ocr_engine.parsers.ktp_layout import apply_ktp_layout_hints
from ocr_engine.ocr.base import OcrToken
from ocr_engine.parsers.stnk import parse_stnk_text
from ocr_engine.validators import mask_sensitive_text, normalize_nik, validate_plate_number


class KtpParserTests(unittest.TestCase):
    def test_parse_ktp_core_fields_from_labelled_text(self):
        raw_text = """
        PROVINSI DKI JAKARTA
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
            OcrToken("Berlaku Hing", 0.72, bbox=[[100, 820], [260, 820], [260, 845], [100, 845]]),
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
