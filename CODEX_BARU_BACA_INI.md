# Codex Baru: Lanjutkan Project OCR Engine Ini

Instruksi untuk Codex baru:

```text
Baca file CODEX_BARU_BACA_INI.md ini sampai selesai, lalu lanjutkan project OCR Engine KTP/STNK dari folder workspace ini.
Jangan mulai dari nol. Ikuti status, keputusan teknis, dan perintah di bawah.
```

## Workspace Baru

Di device baru, user akan memakai folder:

```text
D:\OD SMEA\OneDrive - Asuransi Jasa Indonesia\Automation\OCR Engine\ocr-engine-app
```

Catatan: di device lama folder asalnya:

```text
D:\OD\SMEA\OneDrive - Asuransi Jasa Indonesia\Automation\OCR Engine\ocr-engine-app
```

Jadi semua referensi path lama harus dianggap relatif ke workspace baru. Jangan hardcode path lama.

## Tujuan Project

Project ini adalah service OCR lokal untuk:

- KTP
- STNK

Target utama:

- API service, bukan frontend production.
- Frontend di project ini hanya untuk testing manual.
- Engine lokal utama: PaddleOCR.
- Data diproses lokal, tidak dikirim ke vendor/API luar.
- Output JSON harus berisi field extraction, status field, confidence, warnings, quality, decision, dan telemetry.

## Status Terakhir

Server lokal terakhir berjalan di:

```text
http://127.0.0.1:8000/
```

Full regression test terakhir:

```text
116 tests OK
```

Perubahan parser KTP terakhir sudah memperbaiki field:

- `nik`
- `nama`
- `alamat`
- `tempat_tanggal_lahir`
- `jenis_kelamin`
- `agama`
- `status_perkawinan`
- `pekerjaan`
- `kewarganegaraan`
- `kelurahan_desa`
- `kecamatan`
- `berlaku_hingga`

Field yang baru diperbaiki di sesi terakhir:

- `tempat_tanggal_lahir`
  - menangani tempat dan tanggal terpisah baris, misalnya `JAKARTA` lalu `15-01-1974`
  - repair OCR tanggal seperti `10-O61997`, `199:3`, `196S`
  - menerima tempat lahir seperti `KOTA CIREBON`
- `pekerjaan`
  - repair `KARYAWAN BUUN` / `KARYAWAN BUN` menjadi `KARYAWAN BUMN`
  - repair `TENTARANASIONAL INDONESIA` menjadi `TENTARA NASIONAL INDONESIA (TNI)`
- `kewarganegaraan`
  - repair `OWNI` di dekat/sebelum label rusak seperti `Kewarganegaraart`

Perubahan sebelumnya:

- `kelurahan_desa`, `kecamatan`, `berlaku_hingga` sudah diperbaiki untuk label OCR rusak.
- Layout-aware hint memakai bbox/token PaddleOCR sudah dipakai untuk beberapa field KTP.
- KTP fast mode memakai ukuran lebih kecil dari accurate.
- STNK punya fast ROI dan background enrichment, tetapi fokus terakhir user adalah KTP dulu.

## File Penting

Parser dan pipeline:

```text
src\ocr_engine\pipeline.py
src\ocr_engine\parsers\ktp.py
src\ocr_engine\parsers\ktp_layout.py
src\ocr_engine\parsers\stnk.py
src\ocr_engine\ocr\paddle_provider.py
src\ocr_engine\api.py
src\ocr_engine\frontend.py
```

Tests:

```text
tests\test_parsers.py
tests\test_cli_eval.py
tests\test_api.py
tests\test_frontend.py
tests\test_ocr_provider.py
```

Sample data di sibling folder:

```text
D:\OD SMEA\OneDrive - Asuransi Jasa Indonesia\Automation\OCR Engine\Sample KTP
D:\OD SMEA\OneDrive - Asuransi Jasa Indonesia\Automation\OCR Engine\Sample STNK
```

Jika path itu tidak ada di device baru, cari sibling folder `Sample KTP` dan `Sample STNK` di folder OCR Engine.

## Setup di Device Baru

Jalankan dari PowerShell:

```powershell
cd "D:\OD SMEA\OneDrive - Asuransi Jasa Indonesia\Automation\OCR Engine\ocr-engine-app"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -e .
```

Kalau `py -3.12` tidak ada, cek Python:

```powershell
py -0p
python --version
```

Project butuh Python modern yang kompatibel dengan dependency PaddleOCR.

## Start Server

Local device saja:

```powershell
cd "D:\OD SMEA\OneDrive - Asuransi Jasa Indonesia\Automation\OCR Engine\ocr-engine-app"
.\.venv\Scripts\python.exe -m uvicorn ocr_engine.api:app --host 127.0.0.1 --port 8000
```

Buka:

```text
http://127.0.0.1:8000/
```

Kalau mau diakses device lain satu jaringan:

```powershell
.\.venv\Scripts\python.exe -m uvicorn ocr_engine.api:app --host 0.0.0.0 --port 8000
```

Lalu buka dari device lain:

```text
http://IP-PC:8000/
```

## Test Wajib

Jalankan:

```powershell
cd "D:\OD SMEA\OneDrive - Asuransi Jasa Indonesia\Automation\OCR Engine\ocr-engine-app"
.\.venv\Scripts\python.exe -m unittest tests.test_parsers tests.test_cli_eval tests.test_api tests.test_frontend tests.test_ocr_provider -v
```

Expected terakhir:

```text
116 tests OK
```

Kalau jumlah test berubah karena ada improvement baru, pastikan semua hijau.

## Benchmark KTP

```powershell
.\.venv\Scripts\python.exe -m ocr_engine.cli_eval --input "D:\OD SMEA\OneDrive - Asuransi Jasa Indonesia\Automation\OCR Engine\Sample KTP" --limit 10 --document-type KTP --mode fast --summary-json tmp\ktp-fast-summary-new-device.json --jsonl tmp\ktp-fast-new-device.jsonl
```

## Benchmark STNK

```powershell
.\.venv\Scripts\python.exe -m ocr_engine.cli_eval --input "D:\OD SMEA\OneDrive - Asuransi Jasa Indonesia\Automation\OCR Engine\Sample STNK" --limit 10 --document-type STNK --mode fast --summary-json tmp\stnk-fast-summary-new-device.json --jsonl tmp\stnk-fast-new-device.jsonl
```

## Cara Kerja yang Harus Dilanjutkan

Untuk improvement parser:

1. Cari failure dari `tmp\*.jsonl` atau dari sample upload user.
2. Tambah regression test dulu di `tests\test_parsers.py` atau `tests\test_cli_eval.py`.
3. Jalankan test dan pastikan gagal karena behavior belum ada.
4. Patch parser/pipeline minimal.
5. Jalankan target test.
6. Jalankan full test.
7. Restart server lokal.
8. Beri user statistik ringkas.

Jangan menambah OCR pass baru untuk semua dokumen kecuali benar-benar perlu. Improve parser/layout hint hampir tidak menambah waktu, sedangkan OCR retry/high-res akan memperlambat.

## Keputusan Teknis

- Gunakan PaddleOCR sebagai engine lokal utama.
- Local-only processing boleh.
- Untuk KTP, prioritaskan akurasi field wajib dan field underwriting.
- Untuk STNK, fast ROI + enrichment tetap ada, tetapi jangan korbankan akurasi nomor rangka/mesin/nomor polisi.
- API adalah produk utama; frontend hanya testing.
- Jangan taruh API key permanen di frontend static.
- Jangan commit `.venv` atau `node_modules`.

## Catatan Performa

Yang lambat biasanya OCR PaddleOCR, bukan parser.

Parser/layout hint biasanya hanya beberapa ms. Jadi improvement regex/normalisasi/layout-aware aman untuk speed. Yang memperlambat:

- OCR ulang
- high-res retry
- mode accurate untuk semua file
- AI/LLM fallback
- crop banyak area lalu OCR satu-satu

## Prompt Singkat untuk User

User bisa buka Codex baru lalu kirim:

```text
Baca file CODEX_BARU_BACA_INI.md di workspace ini, lalu lanjutkan project OCR Engine KTP/STNK persis dari status terakhir. Folder device ini adalah D:\OD SMEA\OneDrive - Asuransi Jasa Indonesia\Automation\OCR Engine\ocr-engine-app. Setelah baca, cek dependency, jalankan test, lalu start server lokal.
```
