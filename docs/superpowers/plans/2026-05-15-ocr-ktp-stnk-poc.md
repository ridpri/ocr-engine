# OCR KTP/STNK POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-only OCR POC that accepts KTP/STNK images, runs OCR through a lazy PaddleOCR provider, parses core fields, and returns structured JSON with confidence and warnings.

**Architecture:** Keep the POC modular: OCR provider produces normalized text/tokens, parsers handle KTP/STNK field extraction, validators assign field status/confidence, and FastAPI exposes a thin upload endpoint. The service must run locally and avoid vendor API calls.

**Tech Stack:** Python 3.12, FastAPI, Pillow, optional PaddleOCR, Docker, stdlib `unittest` for parser tests.

---

### Task 1: Parser And Validator Core

**Files:**
- Create: `src/ocr_engine/schemas.py`
- Create: `src/ocr_engine/validators.py`
- Create: `src/ocr_engine/parsers/ktp.py`
- Create: `src/ocr_engine/parsers/stnk.py`
- Test: `tests/test_parsers.py`

- [ ] **Step 1: Write failing tests**

Run: `python -m unittest tests.test_parsers -v`
Expected: import failure because `ocr_engine` does not exist yet.

- [ ] **Step 2: Implement dataclasses and parser functions**

Create field result, document result, NIK/plate validators, KTP parser, and STNK parser.

- [ ] **Step 3: Run parser tests**

Run: `python -m unittest tests.test_parsers -v`
Expected: all parser tests pass.

### Task 2: OCR Provider Interface

**Files:**
- Create: `src/ocr_engine/ocr/base.py`
- Create: `src/ocr_engine/ocr/paddle_provider.py`
- Test: `tests/test_ocr_provider.py`

- [ ] **Step 1: Write failing tests for OCR result normalization**

Run: `python -m unittest tests.test_ocr_provider -v`
Expected: import failure before provider files exist.

- [ ] **Step 2: Implement provider interface**

Lazy import PaddleOCR so tests and API imports do not fail when PaddleOCR is not installed.

- [ ] **Step 3: Run OCR provider tests**

Run: `python -m unittest tests.test_ocr_provider -v`
Expected: normalization and missing dependency error behavior pass.

### Task 3: API And CLI

**Files:**
- Create: `src/ocr_engine/api.py`
- Create: `src/ocr_engine/cli_eval.py`
- Create: `run_api.py`

- [ ] **Step 1: Implement FastAPI endpoint**

`POST /ocr` accepts a JPEG/PNG file and optional `document_type` hint, runs OCR, selects parser, and returns structured result.

- [ ] **Step 2: Implement CLI eval**

`python -m ocr_engine.cli_eval --input <folder> --limit 5` runs OCR over local samples and writes redacted console summaries.

### Task 4: Packaging And Docs

**Files:**
- Create: `requirements.txt`
- Create: `Dockerfile`
- Create: `.gitignore`
- Create: `README.md`

- [ ] **Step 1: Add install/run instructions**

Document local install, optional PaddleOCR dependency behavior, sample folder usage, and data privacy limitations.

- [ ] **Step 2: Verify**

Run:
- `python -m unittest discover -v`
- `python -m compileall src tests`

Expected: tests and compile pass. PaddleOCR sample execution is attempted only after dependencies are installed.
