import asyncio
import json
import os
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

from fastapi.responses import HTMLResponse, JSONResponse

from ocr_engine.agent_bridge import AgentBridgeError, AgentBridgeNotConfigured, run_agent_ocr_bridge
from ocr_engine.frontend import frontend_html
from ocr_engine.ocr.base import OcrDependencyError
from ocr_engine.ocr.paddle_provider import PaddleOcrProvider
from ocr_engine.pdf_utils import render_pdf_first_page
from ocr_engine.pipeline import run_ocr_pipeline
from ocr_engine.service import parse_document_text
from ocr_engine.validators import mask_sensitive_text


STNK_FAST_RESPONSE_TIMEOUT_SECONDS = 8.5


class OcrResponseTimeout(Exception):
    def __init__(self, job_id: str) -> None:
        super().__init__("OCR response timeout")
        self.job_id = job_id


def create_app():
    from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile

    app = FastAPI(
        title="Local OCR Engine POC",
        version="0.1.0",
        description="Local-only KTP/STNK OCR POC using PaddleOCR and rule-based parsers.",
    )
    provider = PaddleOcrProvider()
    stnk_fast_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stnk-fast-ocr")

    @app.middleware("http")
    async def require_api_key(request: Request, call_next):
        api_key = os.getenv("OCR_API_KEY")
        if api_key and request.url.path.startswith("/ocr"):
            supplied = request.headers.get("x-api-key")
            authorization = request.headers.get("authorization", "")
            if authorization.lower().startswith("bearer "):
                supplied = authorization[7:].strip()
            if supplied != api_key:
                return JSONResponse(status_code=401, content={"detail": "Invalid or missing OCR API key."})
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    def frontend() -> str:
        return frontend_html()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ocr/enrichment/{job_id}")
    def enrichment_status(job_id: str) -> dict:
        if not job_id.isalnum():
            raise HTTPException(status_code=404, detail="Enrichment job not found.")
        output_path = Path("tmp") / "stnk_enrichment" / f"{job_id}.json"
        if not output_path.exists():
            return {"status": "pending", "job_id": job_id}
        return {
            "status": "completed",
            "job_id": job_id,
            "result": json.loads(output_path.read_text(encoding="utf-8")),
        }

    @app.post("/ocr")
    async def ocr_document(
        file: UploadFile = File(...),
        document_type: str = Query("AUTO", pattern="^(AUTO|KTP|STNK)$"),
        mode: str = Query("accurate", pattern="^(fast|accurate)$"),
        enrich: bool = Query(False),
    ) -> dict:
        return await _ocr_document(file, document_type, mode, enrich)

    @app.post("/ocr/ktp")
    async def ocr_ktp(
        file: UploadFile = File(...),
        mode: str = Query("accurate", pattern="^(fast|accurate)$"),
        enrich: bool = Query(False),
    ) -> dict:
        return await _ocr_document(file, "KTP", mode, enrich)

    @app.post("/ocr/stnk")
    async def ocr_stnk(
        file: UploadFile = File(...),
        mode: str = Query("fast", pattern="^(fast|accurate)$"),
        enrich: bool = Query(False),
    ) -> dict:
        return await _ocr_document(file, "STNK", mode, enrich)

    @app.post("/ocr/agent")
    async def ocr_agent_document(
        file: UploadFile = File(...),
        document_type: str = Query("AUTO", pattern="^(AUTO|KTP|STNK)$"),
    ) -> dict:
        return await _ocr_agent_document(file, document_type)

    @app.post("/ocr/agent/ktp")
    async def ocr_agent_ktp(file: UploadFile = File(...)) -> dict:
        return await _ocr_agent_document(file, "KTP")

    @app.post("/ocr/agent/stnk")
    async def ocr_agent_stnk(file: UploadFile = File(...)) -> dict:
        return await _ocr_agent_document(file, "STNK")

    @app.post("/ocr/vps")
    async def ocr_vps_document(
        request: Request,
        file: UploadFile = File(...),
        document_type: str = Query("AUTO", pattern="^(AUTO|KTP|STNK)$"),
        mode: str = Query("accurate", pattern="^(fast|accurate)$"),
    ) -> JSONResponse:
        return await _ocr_vps_document(request, file, f"/ocr?document_type={document_type}&mode={mode}")

    @app.post("/ocr/vps/ktp")
    async def ocr_vps_ktp(
        request: Request,
        file: UploadFile = File(...),
        mode: str = Query("accurate", pattern="^(fast|accurate)$"),
    ) -> JSONResponse:
        return await _ocr_vps_document(request, file, f"/ocr/ktp?mode={mode}")

    @app.post("/ocr/vps/stnk")
    async def ocr_vps_stnk(
        request: Request,
        file: UploadFile = File(...),
        mode: str = Query("fast", pattern="^(fast|accurate)$"),
    ) -> JSONResponse:
        return await _ocr_vps_document(request, file, f"/ocr/stnk?mode={mode}")

    @app.post("/ocr/vps/agent")
    async def ocr_vps_agent_document(
        request: Request,
        file: UploadFile = File(...),
        document_type: str = Query("AUTO", pattern="^(AUTO|KTP|STNK)$"),
    ) -> JSONResponse:
        return await _ocr_vps_document(request, file, f"/ocr/agent?document_type={document_type}")

    @app.post("/ocr/vps/agent/ktp")
    async def ocr_vps_agent_ktp(request: Request, file: UploadFile = File(...)) -> JSONResponse:
        return await _ocr_vps_document(request, file, "/ocr/agent/ktp")

    @app.post("/ocr/vps/agent/stnk")
    async def ocr_vps_agent_stnk(request: Request, file: UploadFile = File(...)) -> JSONResponse:
        return await _ocr_vps_document(request, file, "/ocr/agent/stnk")

    async def _ocr_document(file: UploadFile, document_type: str, mode: str, enrich: bool) -> dict:
        started = time.perf_counter()
        suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
        if suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".pdf"}:
            raise HTTPException(status_code=415, detail="Only image and PDF files are supported in this POC.")

        tmpdir = Path(tempfile.mkdtemp(prefix="ocr-engine-"))
        upload_path = tmpdir / f"upload{suffix}"
        upload_bytes = await file.read()
        upload_path.write_bytes(upload_bytes)
        raw_path = upload_path
        ocr_suffix = suffix
        ocr_upload_bytes = upload_bytes
        if suffix.lower() == ".pdf":
            raw_path = render_pdf_first_page(upload_path, tmpdir / "raw.png")
            ocr_suffix = ".png"
            ocr_upload_bytes = raw_path.read_bytes()

        try:
            result = await _run_pipeline_for_request(
                provider,
                stnk_fast_executor,
                raw_path,
                document_type,
                tmpdir,
                mode,
            )
        except OcrResponseTimeout as exc:
            return _stnk_fast_timeout_payload(started, exc.job_id)
        except OcrDependencyError as exc:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise

        shutil.rmtree(tmpdir, ignore_errors=True)
        parsed = result.parsed
        assessment = result.assessment
        payload = parsed.to_dict()
        payload["needs_review"] = parsed.needs_review or assessment["decision"] != "approved_for_auto"
        payload["input_assessment"] = assessment
        payload["quality"] = result.quality
        payload["processing_time_ms"] = round((time.perf_counter() - started) * 1000, 2)
        payload["ocr"] = {
            "provider": result.ocr_result.provider,
            "token_count": len(result.ocr_result.tokens),
            "processing_mode": result.processing_mode,
            "raw_text_masked": mask_sensitive_text(result.ocr_result.raw_text),
            "nik_fallback": result.nik_fallback,
            "preprocess": result.preprocess,
            "timings": result.timings,
        }
        should_enrich = enrich or (
            document_type.upper() == "STNK"
            and result.processing_mode == "fast"
            and payload["needs_review"]
            and assessment.get("decision") != "rejected_input"
        )
        payload["enrichment"] = _schedule_stnk_enrichment(
            provider,
            ocr_upload_bytes,
            ocr_suffix,
            document_type,
            result.processing_mode,
            assessment,
            should_enrich,
        )
        return payload

    async def _ocr_vps_document(request: Request, file: UploadFile, remote_path: str) -> JSONResponse:
        suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
        if suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".pdf"}:
            raise HTTPException(status_code=415, detail="Only image and PDF files are supported in this POC.")
        api_key = request.headers.get("x-vps-api-key") or os.getenv("OCR_VPS_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="VPS API key is required for VPS engine testing.")
        status_code, payload = await asyncio.to_thread(
            _post_file_to_vps,
            remote_path,
            file.filename or f"upload{suffix}",
            file.content_type or "application/octet-stream",
            await file.read(),
            api_key,
        )
        return JSONResponse(status_code=status_code, content=payload)

    async def _ocr_agent_document(file: UploadFile, document_type: str) -> dict:
        suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
        if suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".pdf"}:
            raise HTTPException(status_code=415, detail="Only image and PDF files are supported in this POC.")

        tmpdir = Path(tempfile.mkdtemp(prefix="ocr-agent-"))
        upload_path = tmpdir / f"upload{suffix}"
        upload_path.write_bytes(await file.read())
        image_path = upload_path
        if suffix.lower() == ".pdf":
            image_path = render_pdf_first_page(upload_path, tmpdir / "raw.png")

        try:
            result = await asyncio.to_thread(run_agent_ocr_bridge, image_path, document_type, file.filename)
        except AgentBridgeNotConfigured as exc:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except AgentBridgeError as exc:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise

        shutil.rmtree(tmpdir, ignore_errors=True)
        parsed = result.parsed
        payload = parsed.to_dict()
        payload["needs_review"] = parsed.needs_review or result.assessment["decision"] != "approved_for_auto"
        payload["input_assessment"] = result.assessment
        payload["quality"] = result.raw_response.get("quality") or {
            "image": {},
            "flags": result.raw_response.get("quality_flags") or [],
            "metrics": {"overall_score": result.raw_response.get("quality_score", 1.0)},
        }
        payload["processing_time_ms"] = result.processing_time_ms
        payload["agent"] = {
            "provider": result.provider,
            "model": result.model,
            "bridge": "openclaw_codex",
        }
        return payload

    return app


app = create_app()


def _post_file_to_vps(
    remote_path: str,
    filename: str,
    content_type: str,
    upload_bytes: bytes,
    api_key: str,
) -> tuple[int, dict]:
    base_url = os.getenv("OCR_VPS_BASE_URL", "http://203.194.113.161").rstrip("/")
    timeout = float(os.getenv("OCR_VPS_TIMEOUT_SECONDS", "300"))
    boundary = f"----ocr-vps-{uuid.uuid4().hex}"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            upload_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    request = urllib.request.Request(
        f"{base_url}{remote_path}",
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            "X-API-Key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"detail": raw or str(exc)}
        return exc.code, payload
    except (urllib.error.URLError, TimeoutError) as exc:
        return 502, {"detail": f"VPS OCR request failed: {exc}"}
    except json.JSONDecodeError as exc:
        return 502, {"detail": f"VPS OCR returned non-JSON response: {exc}"}


async def _run_pipeline_for_request(
    provider,
    executor: ThreadPoolExecutor,
    raw_path: Path,
    document_type: str,
    tmpdir: Path,
    mode: str,
):
    if document_type.upper() == "STNK" and mode == "fast":
        future = executor.submit(run_ocr_pipeline, provider, raw_path, document_type, tmpdir, mode)
        try:
            return await asyncio.to_thread(future.result, STNK_FAST_RESPONSE_TIMEOUT_SECONDS)
        except FutureTimeoutError as exc:
            job_id = _queue_timed_out_stnk_fast_result(provider, future, raw_path, tmpdir)
            raise OcrResponseTimeout(job_id) from exc

    return run_ocr_pipeline(provider, raw_path, document_type, tmpdir, processing_mode=mode)


def _queue_timed_out_stnk_fast_result(provider, future, raw_path: Path, tmpdir: Path) -> str:
    job_id = uuid.uuid4().hex
    jobs_dir = Path("tmp") / "stnk_enrichment"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    output_path = jobs_dir / f"{job_id}.json"
    pending_path = jobs_dir / f"{job_id}.pending.json"
    pending_path.write_text(
        json.dumps({"status": "pending", "job_id": job_id, "mode": "accurate_background"}, ensure_ascii=False),
        encoding="utf-8",
    )
    future.add_done_callback(
        lambda completed: _write_timed_out_stnk_accurate_result(provider, completed, raw_path, output_path, pending_path, tmpdir)
    )
    return job_id


def _write_timed_out_stnk_accurate_result(provider, future, raw_path: Path, output_path: Path, pending_path: Path, tmpdir: Path) -> None:
    started = time.perf_counter()
    try:
        future.result()
        accurate_workdir = tmpdir / "accurate-background"
        accurate_workdir.mkdir(parents=True, exist_ok=True)
        result = run_ocr_pipeline(provider, raw_path, "STNK", accurate_workdir, processing_mode="accurate")
        record = _pipeline_result_record(
            result,
            float(result.timings.get("total_ms", round((time.perf_counter() - started) * 1000, 2))),
            status="ok",
        )
    except Exception as exc:
        record = {
            "status": "failed",
            "error": str(exc),
            "processing_time_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    pending_path.unlink(missing_ok=True)
    shutil.rmtree(tmpdir, ignore_errors=True)


def _stnk_fast_timeout_payload(started: float, job_id: str) -> dict:
    parsed = parse_document_text("", document_type_hint="STNK")
    processing_time_ms = round((time.perf_counter() - started) * 1000, 2)
    payload = parsed.to_dict()
    payload["needs_review"] = True
    payload["warnings"] = ["processing_timeout"]
    payload["input_assessment"] = {
        "decision": "needs_review",
        "can_auto_publish": False,
        "expected_document_type": "STNK",
        "detected_document_type": "UNKNOWN",
        "reason_codes": ["processing_timeout"],
        "message": "OCR STNK melewati batas waktu checkout. Gunakan review manual atau proses accurate di belakang layar.",
    }
    payload["quality"] = {
        "image": {},
        "flags": ["processing_timeout"],
        "metrics": {"overall_score": 0.0},
    }
    payload["processing_time_ms"] = processing_time_ms
    payload["ocr"] = {
        "provider": "paddleocr",
        "token_count": 0,
        "processing_mode": "fast",
        "raw_text_masked": "",
        "nik_fallback": {"attempted": False, "passes": 0, "value": None},
        "preprocess": {
            "selected_max_side": None,
            "retry_count": 0,
            "attempts": [
                {
                    "index": 0,
                    "max_side": None,
                    "strategy": "stnk_fast_roi",
                    "document_type": "STNK",
                    "detected_document_type": "UNKNOWN",
                    "decision": "needs_review",
                    "warnings": ["processing_timeout"],
                }
            ],
        },
        "timings": {
            "total_ms": processing_time_ms,
            "selected_attempt_index": 0,
            "nik_fallback_ms": 0.0,
            "attempts": [
                {
                    "prepare_ms": 0.0,
                    "ocr_ms": processing_time_ms,
                    "parse_ms": 0.0,
                    "quality_ms": 0.0,
                    "assessment_ms": 0.0,
                    "total_ms": processing_time_ms,
                }
            ],
        },
    }
    payload["enrichment"] = {
        "status": "queued",
        "mode": "accurate_background",
        "job_id": job_id,
        "message": "Upload sudah diterima. OCR accurate berjalan di belakang dan bisa dipoll tanpa upload ulang.",
    }
    return payload


def _pipeline_result_record(result, processing_time_ms: float, status: str = "ok") -> dict:
    parsed = result.parsed
    return {
        "status": status,
        "document_type": parsed.document_type,
        "input_assessment": result.assessment,
        "quality": result.quality,
        "processing_time_ms": processing_time_ms,
        "fields": {key: field.to_dict() for key, field in parsed.fields.items()},
        "ocr": {
            "provider": result.ocr_result.provider,
            "token_count": len(result.ocr_result.tokens),
            "processing_mode": result.processing_mode,
            "raw_text_masked": mask_sensitive_text(result.ocr_result.raw_text),
            "nik_fallback": result.nik_fallback,
            "preprocess": result.preprocess,
            "timings": result.timings,
        },
    }


def _schedule_stnk_enrichment(
    provider,
    upload_bytes: bytes,
    suffix: str,
    document_type: str,
    processing_mode: str,
    assessment: dict,
    enrich: bool,
) -> dict:
    if not enrich:
        return {"status": "not_requested"}
    if document_type.upper() != "STNK" or processing_mode != "fast":
        return {"status": "not_applicable"}
    if assessment.get("decision") == "rejected_input":
        return {"status": "skipped", "reason": "rejected_input"}

    job_id = uuid.uuid4().hex
    jobs_dir = Path("tmp") / "stnk_enrichment"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    image_path = jobs_dir / f"{job_id}{suffix.lower()}"
    image_path.write_bytes(upload_bytes)
    output_path = jobs_dir / f"{job_id}.json"
    pending_path = jobs_dir / f"{job_id}.pending.json"
    pending_path.write_text(json.dumps({"status": "pending", "job_id": job_id}, ensure_ascii=False), encoding="utf-8")
    thread = threading.Thread(target=_run_stnk_enrichment_job, args=(provider, image_path, output_path, pending_path), daemon=True)
    thread.start()
    return {
        "status": "queued",
        "mode": "accurate_background",
        "job_id": job_id,
        "output_path": str(output_path),
    }


def _run_stnk_enrichment_job(provider, image_path: Path, output_path: Path, pending_path: Path) -> None:
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="ocr-engine-enrich-") as tmpdir:
            result = run_ocr_pipeline(provider, image_path, "STNK", Path(tmpdir), processing_mode="accurate")
            record = _pipeline_result_record(result, round((time.perf_counter() - started) * 1000, 2), status="ok")
    except Exception as exc:
        record = {
            "status": "failed",
            "error": str(exc),
            "processing_time_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    pending_path.unlink(missing_ok=True)
