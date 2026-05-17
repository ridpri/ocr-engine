# Lanjut OCR Engine di Device Baru

Folder app:

```powershell
D:\OD\SMEA\OneDrive - Asuransi Jasa Indonesia\Automation\OCR Engine\ocr-engine-app
```

Jalankan sekali di device baru:

```powershell
cd "D:\OD\SMEA\OneDrive - Asuransi Jasa Indonesia\Automation\OCR Engine\ocr-engine-app"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -e .
```

Jalankan server:

```powershell
cd "D:\OD\SMEA\OneDrive - Asuransi Jasa Indonesia\Automation\OCR Engine\ocr-engine-app"
.\.venv\Scripts\python.exe -m uvicorn ocr_engine.api:app --host 127.0.0.1 --port 8000
```

Buka browser:

```text
http://127.0.0.1:8000/
```

Tes:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_parsers tests.test_cli_eval tests.test_api tests.test_frontend tests.test_ocr_provider -v
```
