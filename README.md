# OCR Engine POC KTP/STNK

POC lokal untuk membaca image KTP/STNK, menjalankan OCR dengan PaddleOCR, lalu mengekstrak field inti ke JSON.

## Scope

- Input: image `jpg`, `jpeg`, `png`, `bmp`, `webp`.
- Output: JSON field KTP/STNK, confidence, status, warnings, dan raw OCR yang sudah dimasking.
- Engine utama: PaddleOCR.
- Pemrosesan data: lokal, tanpa vendor/API eksternal.
- Parser: rule-based untuk POC, belum production-ready.

PDF diproses dengan merender halaman pertama ke image lokal sebelum OCR. DOC belum didukung.

## Install Lokal

Gunakan Python 3.12.

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Jika instalasi PaddleOCR gagal, cek kompatibilitas `paddlepaddle` untuk versi Python/OS yang dipakai. Parser dan test inti tetap bisa dijalankan tanpa PaddleOCR karena dependency OCR di-load saat runtime.

## Jalankan Test

```powershell
python -m unittest discover -s tests -v
python -m compileall src tests
```

## Jalankan API

Untuk Windows CPU, adapter sudah mematikan MKL-DNN/oneDNN default karena ada bug runtime PaddlePaddle pada beberapa versi. Jika model PaddleOCR belum pernah terunduh dan jaringan menolak SSL certificate model host, gunakan flag berikut hanya untuk bootstrap lokal:

```powershell
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True'
$env:OCR_ENGINE_ALLOW_INSECURE_MODEL_DOWNLOADS='1'
```

Jangan gunakan `OCR_ENGINE_ALLOW_INSECURE_MODEL_DOWNLOADS=1` untuk production. Setelah model tersimpan di cache lokal, flag itu bisa dilepas.

```powershell
uvicorn ocr_engine.api:app --host 0.0.0.0 --port 8000
```

Health check:

```powershell
curl http://localhost:8000/health
```

OCR upload:

```powershell
curl -X POST "http://localhost:8000/ocr?document_type=AUTO" ^
  -F "file=@D:\\OD\\SMEA\\OneDrive - Asuransi Jasa Indonesia\\Automation\\OCR Engine\\Sample KTP\\KTP (10).jpg"
```

## Service Kedua: Agent OpenClaw/Codex

Service kedua dipisah dari engine lokal. Endpoint ini menerima format upload yang sama, tetapi meneruskan image ke bridge agent yang bisa diisi OpenClaw/Codex.

Endpoint:

- `POST /ocr/agent?document_type=AUTO|KTP|STNK`
- `POST /ocr/agent/ktp`
- `POST /ocr/agent/stnk`

Konfigurasi command bridge:

```powershell
$env:OCR_AGENT_COMMAND='openclaw ...'
```

Command akan menerima JSON request dari stdin dan harus mengembalikan JSON ke stdout dengan bentuk minimal:

```json
{
  "document_type": "KTP",
  "detected_document_type": "KTP",
  "raw_text": "teks OCR atau ringkasan evidence",
  "fields": {
    "nik": {"value": "3175010101900001", "confidence": 0.99, "status": "ok"}
  },
  "warnings": []
}
```

Konfigurasi webhook OpenClaw:

```powershell
$env:OPENCLAW_OCR_WEBHOOK_URL='http://127.0.0.1:18789/path/to/ocr-workflow'
$env:OPENCLAW_OCR_TOKEN='optional-token'
```

Jika belum dikonfigurasi, endpoint agent akan mengembalikan HTTP `503` supaya service lokal tetap aman dan jelas statusnya.

## Jalankan CLI Eval Sample

```powershell
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True'
$env:OCR_ENGINE_ALLOW_INSECURE_MODEL_DOWNLOADS='1'
python -m ocr_engine.cli_eval --input "D:\\OD\\SMEA\\OneDrive - Asuransi Jasa Indonesia\\Automation\\OCR Engine\\Sample KTP" --limit 5 --document-type AUTO
```

Output CLI satu JSON per file. Raw OCR dimasking untuk angka seperti NIK.

## Docker

```powershell
docker build -t ocr-engine-poc .
docker run --rm -p 8000:8000 ocr-engine-poc
```

## Deploy Frontend ke Vercel

Project ini menyertakan deploy frontend static untuk Vercel. Build akan mengekspor HTML tester dari `ocr_engine.frontend` ke folder `public`, lalu route Vercel `/health` dan `/ocr/*` diproxy ke backend OCR.

Environment variable Vercel:

- `OCR_API_BASE_URL`: base URL backend OCR, default `http://203.194.113.161`
- `OCR_API_KEY`: API key backend OCR, optional tetapi direkomendasikan supaya key tidak diketik di browser

Frontend tester tidak lagi menampilkan pilihan mode `fast`/`accurate`. Untuk menghindari STNK sering terlihat kosong karena timeout fast checkout, endpoint STNK default ke proses `accurate`; query `mode=fast|accurate` tetap tersedia hanya untuk integrasi teknis yang membutuhkan kompatibilitas lama.

Build command:

```powershell
python scripts/export_frontend.py
```

Output directory:

```text
public
```

## Field Output

KTP:

- `nik`
- `nama`
- `tempat_tanggal_lahir`
- `jenis_kelamin`
- `alamat`
- `rt_rw`
- `kelurahan_desa`
- `kecamatan`
- `agama`
- `status_perkawinan`
- `pekerjaan`
- `kewarganegaraan`
- `berlaku_hingga`

STNK:

- `nomor_polisi`
- `nama_pemilik`
- `alamat`
- `merek`
- `tipe`
- `jenis`
- `tahun_pembuatan`
- `warna`
- `nomor_rangka`
- `nomor_mesin`
- `bahan_bakar`
- `berlaku_sampai`

Setiap field punya:

- `value`
- `confidence`
- `status`: `ok`, `missing`, atau `invalid`
- `evidence`
- `raw`

## Batasan POC

- Belum ada training model khusus KTP/STNK.
- Parser masih bergantung pada label OCR yang terbaca.
- Akurasi sangat bergantung pada kualitas foto.
- Tidak ada penyimpanan permanen hasil OCR.
- Belum ada queue worker, auth, audit log lengkap, atau review UI.

Untuk demo besok, positioning yang aman: POC local OCR pipeline dengan output JSON dan fallback review, bukan production-ready identity verification engine.
