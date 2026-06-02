import asyncio
import json
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi.responses import HTMLResponse, JSONResponse

from ocr_engine.agent_bridge import AgentBridgeError, AgentBridgeNotConfigured, run_agent_ocr_bridge
from ocr_engine.frontend import frontend_html
from ocr_engine.ocr.base import OcrDependencyError
from ocr_engine.ocr.paddle_provider import PaddleOcrProvider
from ocr_engine.ocr.rapid_provider import RapidOcrProvider
from ocr_engine.pdf_utils import render_pdf_first_page
from ocr_engine.pipeline import run_ocr_pipeline
from ocr_engine.postal_code import get_default_postal_code_index
from ocr_engine.quality import analyze_image_preflight
from ocr_engine.service import build_input_assessment, parse_document_text
from ocr_engine.parsers.stnk import stnk_structure_score
from ocr_engine.stnk_usage import apply_stnk_web_usage_gate, classify_stnk_record
from ocr_engine.validators import mask_sensitive_text


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


STNK_FAST_RESPONSE_TIMEOUT_SECONDS = _env_float("STNK_FAST_RESPONSE_TIMEOUT_SECONDS", 12)
STNK_PURCHASE_RESPONSE_TIMEOUT_SECONDS = _env_float("STNK_PURCHASE_RESPONSE_TIMEOUT_SECONDS", 6)
BACKGROUND_OCR_START_DELAY_SECONDS = _env_float("BACKGROUND_OCR_START_DELAY_SECONDS", 10)
KTP_PURCHASE_BACKGROUND_START_DELAY_SECONDS = _env_float("KTP_PURCHASE_BACKGROUND_START_DELAY_SECONDS", 45)
STNK_PURCHASE_BACKGROUND_START_DELAY_SECONDS = _env_float("STNK_PURCHASE_BACKGROUND_START_DELAY_SECONDS", 20)
OCR_REQUEST_MAX_WORKERS = _env_int("OCR_REQUEST_MAX_WORKERS", 1)
OCR_REQUEST_MAX_PENDING = _env_int("OCR_REQUEST_MAX_PENDING", max(1, OCR_REQUEST_MAX_WORKERS))
BACKGROUND_OCR_MAX_WORKERS = _env_int("BACKGROUND_OCR_MAX_WORKERS", 1)
BACKGROUND_OCR_MAX_PENDING = _env_int("BACKGROUND_OCR_MAX_PENDING", 16)
ENRICHMENT_JOB_TTL_SECONDS = _env_float("ENRICHMENT_JOB_TTL_SECONDS", 24 * 60 * 60)


class OcrResponseTimeout(Exception):
    def __init__(self, job_id: str) -> None:
        super().__init__("OCR response timeout")
        self.job_id = job_id


class OcrCapacityExceeded(Exception):
    pass


def create_app():
    from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile

    provider = _create_ocr_provider()
    request_executor = ThreadPoolExecutor(max_workers=OCR_REQUEST_MAX_WORKERS, thread_name_prefix="ocr-request")
    request_slots = threading.BoundedSemaphore(OCR_REQUEST_MAX_PENDING)
    background_executor = ThreadPoolExecutor(max_workers=BACKGROUND_OCR_MAX_WORKERS, thread_name_prefix="ocr-background")
    background_slots = threading.BoundedSemaphore(BACKGROUND_OCR_MAX_PENDING)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        _cleanup_enrichment_jobs()
        _warm_postal_code_index()
        warm_up = getattr(provider, "warm_up", None)
        if callable(warm_up):
            warm_up()
        try:
            yield
        finally:
            request_executor.shutdown(wait=False, cancel_futures=True)
            background_executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="Local OCR Engine POC",
        version="0.1.0",
        description="Local-only KTP/STNK OCR POC using PaddleOCR and rule-based parsers.",
        lifespan=lifespan,
    )

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

    @app.post("/ui/ocr")
    async def ui_ocr_document(
        file: UploadFile = File(...),
        document_type: str = Query("AUTO", pattern="^(AUTO|KTP|STNK)$"),
        mode: str = Query("fast", pattern="^(fast|accurate)$"),
    ) -> dict:
        return await _ocr_document(
            file,
            document_type,
            mode,
            False,
            run_nik_fallback=False,
            force_strategy=_ui_force_strategy(document_type, mode),
        )

    @app.get("/ocr/enrichment/{job_id}")
    def enrichment_status(job_id: str) -> dict:
        if not job_id.isalnum():
            raise HTTPException(status_code=404, detail="Enrichment job not found.")
        output_path = Path("tmp") / "stnk_enrichment" / f"{job_id}.json"
        if not output_path.exists():
            return {"status": "pending", "job_id": job_id}
        try:
            result = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"status": "failed", "job_id": job_id, "error": "corrupt_enrichment_result"}
        return {
            "status": "completed",
            "job_id": job_id,
            "result": result,
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
        mode: str = Query("fast", pattern="^(fast|accurate)$"),
        enrich: bool = Query(False),
    ) -> dict:
        return await _ocr_document(file, "KTP", mode, enrich)

    @app.post("/ocr/stnk")
    async def ocr_stnk(
        file: UploadFile = File(...),
        mode: str = Query("accurate", pattern="^(fast|accurate)$"),
        enrich: bool = Query(False),
    ) -> dict:
        return await _ocr_document(file, "STNK", mode, enrich)

    @app.post("/ocr/purchase/ktp")
    async def ocr_purchase_ktp(file: UploadFile = File(...)) -> dict:
        payload = await _ocr_document(
            file,
            "KTP",
            "fast",
            False,
            run_nik_fallback=False,
        )
        return _purchase_checkout_payload(payload, ["nik", "nama", "alamat", "kode_pos"], background_key="background_full_ocr")

    @app.post("/ocr/purchase/stnk")
    async def ocr_purchase_stnk(file: UploadFile = File(...)) -> dict:
        payload = await _ocr_purchase_stnk_queued(file)
        return _purchase_checkout_payload(
            payload,
            ["nomor_polisi", "nama_pemilik", "nomor_rangka", "nomor_mesin"],
            background_key="background_full_ocr",
        )

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
        mode: str = Query("fast", pattern="^(fast|accurate)$"),
    ) -> JSONResponse:
        return await _ocr_vps_document(request, file, f"/ocr/ktp?mode={mode}")

    @app.post("/ocr/vps/stnk")
    async def ocr_vps_stnk(
        request: Request,
        file: UploadFile = File(...),
        mode: str = Query("accurate", pattern="^(fast|accurate)$"),
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

    async def _ocr_document(
        file: UploadFile,
        document_type: str,
        mode: str,
        enrich: bool,
        run_nik_fallback: bool = True,
        purchase_background: bool = False,
        response_timeout_seconds: float | None = None,
        force_strategy: str | None = None,
    ) -> dict:
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
                request_executor,
                request_slots,
                raw_path,
                document_type,
                tmpdir,
                mode,
                run_nik_fallback=run_nik_fallback,
                response_timeout_seconds=response_timeout_seconds,
                force_strategy=force_strategy,
            )
        except OcrResponseTimeout as exc:
            return _ocr_timeout_payload(started, exc.job_id, document_type, mode)
        except OcrCapacityExceeded as exc:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
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
        if parsed.document_type == "KTP":
            payload["ti_compatible"] = _ti_compatible_ktp_payload(payload)
        if document_type.upper() == "STNK" or parsed.document_type == "STNK":
            payload["stnk_structure_score"] = stnk_structure_score(result.ocr_result.raw_text)
            usage_class, usage_reasons = classify_stnk_record(payload)
            payload["stnk_usage_class"] = usage_class
            payload["stnk_usage_reasons"] = usage_reasons
            apply_stnk_web_usage_gate(payload)
            assessment = payload["input_assessment"]
        should_enrich = enrich or (
            document_type.upper() == "STNK"
            and result.processing_mode == "fast"
            and payload["needs_review"]
            and assessment.get("decision") != "rejected_input"
        )
        if purchase_background:
            payload["enrichment"] = _schedule_background_ocr(
                background_executor,
                background_slots,
                provider,
                ocr_upload_bytes,
                ocr_suffix,
                document_type,
                assessment,
                processing_mode="fast",
                delay_seconds=KTP_PURCHASE_BACKGROUND_START_DELAY_SECONDS,
            )
        elif _should_run_stnk_full_page_background(document_type, parsed, assessment, result.processing_mode):
            payload["enrichment"] = _schedule_background_ocr(
                background_executor,
                background_slots,
                provider,
                ocr_upload_bytes,
                ocr_suffix,
                document_type,
                assessment,
                processing_mode="accurate",
                delay_seconds=0,
                force_strategy="stnk_full_page",
                mode_label="full_page_background",
            )
        else:
            payload["enrichment"] = _schedule_stnk_enrichment(
                provider,
                background_executor,
                background_slots,
                ocr_upload_bytes,
                ocr_suffix,
                document_type,
                result.processing_mode,
                assessment,
                should_enrich,
            )
        return payload

    async def _ocr_purchase_stnk_queued(file: UploadFile) -> dict:
        started = time.perf_counter()
        suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
        if suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".pdf"}:
            raise HTTPException(status_code=415, detail="Only image and PDF files are supported in this POC.")

        tmpdir = Path(tempfile.mkdtemp(prefix="ocr-engine-purchase-stnk-"))
        try:
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

            preflight_quality = analyze_image_preflight(raw_path)
            parsed = parse_document_text("", document_type_hint="STNK")
            payload = parsed.to_dict()
            reject_before_ocr = bool(
                set(preflight_quality.get("flags") or []).intersection({"document_too_small", "blur_detected"})
            )

            if reject_before_ocr:
                quality = {
                    "image": dict(preflight_quality.get("image") or {}),
                    "flags": [*preflight_quality.get("flags", []), "pre_ocr_rejected"],
                    "metrics": dict(preflight_quality.get("metrics") or {}),
                }
                assessment = build_input_assessment("", parsed, "STNK", "UNKNOWN", quality=quality)
                background = {"status": "skipped", "reason": "rejected_input"}
                ocr_provider = "preflight"
                reason_warnings = list(parsed.warnings)
            else:
                quality = preflight_quality
                assessment = {
                    "decision": "needs_review",
                    "can_auto_publish": False,
                    "expected_document_type": "STNK",
                    "detected_document_type": "PENDING",
                    "reason_codes": ["ocr_queued"],
                    "message": "Upload diterima. OCR STNK berjalan di belakang dan hasil dapat dipoll tanpa upload ulang.",
                }
                background = _schedule_background_ocr(
                    background_executor,
                    background_slots,
                    provider,
                    ocr_upload_bytes,
                    ocr_suffix,
                    "STNK",
                    assessment,
                    processing_mode="fast",
                    delay_seconds=STNK_PURCHASE_BACKGROUND_START_DELAY_SECONDS,
                    mode_label="fast_background",
                )
                ocr_provider = "queued"
                reason_warnings = ["ocr_queued"]

            processing_time_ms = round((time.perf_counter() - started) * 1000, 2)
            payload["needs_review"] = True
            payload["warnings"] = reason_warnings
            payload["input_assessment"] = assessment
            payload["quality"] = quality
            payload["processing_time_ms"] = processing_time_ms
            payload["ocr"] = {
                "provider": ocr_provider,
                "token_count": 0,
                "processing_mode": "fast",
                "raw_text_masked": "",
                "nik_fallback": {"attempted": False, "passes": 0, "value": None},
                "preprocess": {
                    "selected_max_side": None,
                    "retry_count": 0,
                    "attempts": [],
                    "pre_ocr_rejected": reject_before_ocr,
                },
                "timings": {
                    "total_ms": processing_time_ms,
                    "preflight_ms": processing_time_ms,
                    "selected_attempt_index": None,
                    "nik_fallback_ms": 0.0,
                    "attempts": [],
                },
            }
            payload["enrichment"] = background
            return payload
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

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
        if parsed.document_type == "KTP":
            payload["ti_compatible"] = _ti_compatible_ktp_payload(payload)
        return payload

    return app


def _create_ocr_provider():
    provider_name = os.getenv("OCR_PROVIDER", "paddle").strip().lower()
    if provider_name == "rapid":
        return RapidOcrProvider()
    return PaddleOcrProvider()


def _ui_force_strategy(document_type: str, mode: str) -> str | None:
    if mode != "fast":
        return None
    document = document_type.upper()
    if document == "KTP":
        return "ktp_fast"
    if document == "STNK":
        return "stnk_fast_roi"
    return None


def _warm_postal_code_index() -> None:
    try:
        get_default_postal_code_index()
    except Exception:
        return


def _cleanup_enrichment_jobs() -> None:
    if ENRICHMENT_JOB_TTL_SECONDS <= 0:
        return
    jobs_dir = Path("tmp") / "stnk_enrichment"
    if not jobs_dir.exists():
        return
    cutoff = time.time() - ENRICHMENT_JOB_TTL_SECONDS
    for path in jobs_dir.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            continue


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


app = create_app()


def _post_file_to_vps(
    remote_path: str,
    filename: str,
    content_type: str,
    upload_bytes: bytes,
    api_key: str,
) -> tuple[int, dict]:
    base_url = os.getenv("OCR_VPS_BASE_URL", "http://203.194.113.161").rstrip("/")
    target_url = f"{base_url}{remote_path}"
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
        target_url,
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
            payload = {"detail": _non_json_vps_detail(target_url, exc.code, raw or str(exc))}
        return exc.code, payload
    except (urllib.error.URLError, TimeoutError) as exc:
        return 502, {"detail": f"VPS OCR request failed: {exc}"}
    except json.JSONDecodeError as exc:
        return 502, {"detail": _non_json_vps_detail(target_url, 502, str(exc))}


def _non_json_vps_detail(target_url: str, status_code: int, raw: str) -> str:
    snippet = re.sub(r"\s+", " ", raw).strip()[:220]
    return f"VPS OCR returned non-JSON response from {target_url} (HTTP {status_code}). {snippet}"


async def _run_pipeline_for_request(
    provider,
    executor: ThreadPoolExecutor,
    slots: threading.BoundedSemaphore,
    raw_path: Path,
    document_type: str,
    tmpdir: Path,
    mode: str,
    run_nik_fallback: bool = True,
    response_timeout_seconds: float | None = None,
    force_strategy: str | None = None,
):
    timeout_seconds = response_timeout_seconds
    if timeout_seconds is None and document_type.upper() == "STNK" and mode == "fast":
        timeout_seconds = STNK_FAST_RESPONSE_TIMEOUT_SECONDS

    future = _submit_bounded(
        executor,
        slots,
        run_ocr_pipeline,
        provider,
        raw_path,
        document_type,
        tmpdir,
        mode,
        run_nik_fallback,
        force_strategy,
    )
    try:
        if timeout_seconds is None:
            return await asyncio.wrap_future(future)
        return await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        job_id = _queue_timed_out_ocr_result(future, tmpdir, document_type, mode)
        raise OcrResponseTimeout(job_id) from exc


def _submit_bounded(executor: ThreadPoolExecutor, slots: threading.BoundedSemaphore, func, *args):
    if not slots.acquire(blocking=False):
        raise OcrCapacityExceeded("OCR engine is busy. Please retry shortly.")
    try:
        future = executor.submit(func, *args)
    except Exception:
        slots.release()
        raise
    future.add_done_callback(lambda _completed: slots.release())
    return future


def _queue_timed_out_ocr_result(future, tmpdir: Path, document_type: str, processing_mode: str) -> str:
    job_id = uuid.uuid4().hex
    jobs_dir = Path("tmp") / "stnk_enrichment"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    output_path = jobs_dir / f"{job_id}.json"
    pending_path = jobs_dir / f"{job_id}.pending.json"
    _write_json_atomic(
        pending_path,
        {
            "status": "pending",
            "job_id": job_id,
            "document_type": document_type,
            "mode": f"{processing_mode}_background",
        },
    )
    future.add_done_callback(
        lambda completed: _write_timed_out_ocr_result(completed, output_path, pending_path, tmpdir)
    )
    return job_id


def _write_timed_out_ocr_result(future, output_path: Path, pending_path: Path, tmpdir: Path) -> None:
    started = time.perf_counter()
    try:
        result = future.result()
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
    _write_json_atomic(output_path, record)
    pending_path.unlink(missing_ok=True)
    shutil.rmtree(tmpdir, ignore_errors=True)


def _ocr_timeout_payload(started: float, job_id: str, document_type: str, processing_mode: str) -> dict:
    expected = document_type.upper()
    parsed = parse_document_text("", document_type_hint=expected if expected in {"KTP", "STNK"} else "AUTO")
    processing_time_ms = round((time.perf_counter() - started) * 1000, 2)
    payload = parsed.to_dict()
    payload["needs_review"] = True
    payload["warnings"] = ["processing_timeout"]
    payload["input_assessment"] = {
        "decision": "needs_review",
        "can_auto_publish": False,
        "expected_document_type": expected,
        "detected_document_type": "UNKNOWN",
        "reason_codes": ["processing_timeout"],
        "message": f"OCR {expected} masih berjalan. Hasil dapat dipoll tanpa upload ulang.",
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
        "processing_mode": processing_mode,
        "raw_text_masked": "",
        "nik_fallback": {"attempted": False, "passes": 0, "value": None},
        "preprocess": {
            "selected_max_side": None,
            "retry_count": 0,
            "attempts": [
                {
                    "index": 0,
                    "max_side": None,
                    "strategy": _timeout_strategy(expected, processing_mode),
                    "document_type": expected,
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
        "mode": f"{processing_mode}_background",
        "job_id": job_id,
        "message": "Upload sudah diterima. OCR yang sama tetap berjalan di belakang dan bisa dipoll tanpa upload ulang.",
    }
    return payload


def _timeout_strategy(document_type: str, processing_mode: str) -> str:
    if document_type == "KTP" and processing_mode == "fast":
        return "ktp_fast"
    if document_type == "STNK" and processing_mode == "fast":
        return "stnk_fast_roi"
    return "full_page"


def _purchase_checkout_payload(payload: dict, field_names: list[str], background_key: str) -> dict:
    fields = {
        field_name: payload.get("fields", {}).get(field_name, {"value": None, "status": "missing"})
        for field_name in field_names
    }
    missing_required = [
        field_name
        for field_name, field in fields.items()
        if field.get("status") != "ok" or field.get("value") in {None, ""}
    ]
    assessment = payload.get("input_assessment", {})
    enrichment = payload.get("enrichment", {"status": "not_applicable"})
    document_type = str(payload.get("document_type") or "").upper()
    stnk_web_ready = document_type != "STNK" or payload.get("stnk_usage_class") == "web_usable"
    result = {
        "purpose": "purchase_checkout",
        "document_type": payload.get("document_type"),
        "fields": fields,
        "ti_compatible": payload.get("ti_compatible"),
        "ready_for_checkout": (
            not missing_required and stnk_web_ready and assessment.get("decision") == "approved_for_auto"
        ),
        "missing_required": missing_required,
        "needs_review": bool(payload.get("needs_review")) or bool(missing_required) or not stnk_web_ready,
        "input_assessment": assessment,
        "quality": payload.get("quality", {}),
        "processing_time_ms": payload.get("processing_time_ms"),
        "ocr": _purchase_ocr_summary(payload.get("ocr", {})),
        background_key: enrichment,
    }
    if document_type == "STNK":
        result["stnk_usage_class"] = payload.get("stnk_usage_class")
        result["stnk_usage_reasons"] = payload.get("stnk_usage_reasons", [])
    return result


def _purchase_ocr_summary(ocr: dict) -> dict:
    return {
        "provider": ocr.get("provider"),
        "token_count": ocr.get("token_count", 0),
        "processing_mode": ocr.get("processing_mode"),
        "nik_fallback": ocr.get("nik_fallback", {"attempted": False, "passes": 0, "value": None}),
        "preprocess": ocr.get("preprocess", {}),
        "timings": ocr.get("timings", {}),
    }


def _pipeline_result_record(result, processing_time_ms: float, status: str = "ok") -> dict:
    parsed = result.parsed
    record = {
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
    if parsed.document_type == "KTP":
        record["ti_compatible"] = _ti_compatible_ktp_payload(record)
    if parsed.document_type == "STNK":
        record["stnk_structure_score"] = stnk_structure_score(result.ocr_result.raw_text)
        usage_class, usage_reasons = classify_stnk_record(record)
        record["stnk_usage_class"] = usage_class
        record["stnk_usage_reasons"] = usage_reasons
        apply_stnk_web_usage_gate(record)
    return record


def _ti_compatible_ktp_payload(record: dict) -> dict:
    fields = record.get("fields", {})
    postal = fields.get("kode_pos", {})
    postal_meta = postal.get("metadata") or {}
    success = postal.get("status") == "ok" and bool(postal.get("value"))
    return {
        "status": "success" if success else "not_found",
        "message": "Kode pos successfully retrieved from KTP" if success else "Kode pos not found from KTP",
        "session_id": str(uuid.uuid4()),
        "ocr_data": {
            "nama": _field_value(fields, "nama"),
            "no_ktp": _field_value(fields, "nik"),
            "tempat_lahir": _birth_place(_field_value(fields, "tempat_tanggal_lahir")),
            "tanggal_lahir": _birth_date_iso(_field_value(fields, "tempat_tanggal_lahir")),
            "alamat": _formatted_ktp_address(fields),
            "kota": _field_value(fields, "kabupaten_kota"),
            "kodeKota": _string_or_none(postal_meta.get("kode_kota")),
            "kodeKecamatan": _string_or_none(postal_meta.get("kode_kecamatan")),
            "kodeProvinsi": _string_or_none(postal_meta.get("kode_provinsi")),
            "agama": _field_value(fields, "agama"),
            "status_perkawinan": _marital_code(_field_value(fields, "status_perkawinan")),
            "pekerjaan": _field_value(fields, "pekerjaan"),
            "kewarganegaraan": _field_value(fields, "kewarganegaraan"),
            "jenis_kelamin": _gender_code(_field_value(fields, "jenis_kelamin")),
        },
        "kodepos_data": {
            "kode_pos": _field_value(fields, "kode_pos"),
            "kelurahan": postal_meta.get("kelurahan") or _field_value(fields, "kelurahan_desa"),
            "kecamatan": postal_meta.get("kecamatan") or _field_value(fields, "kecamatan"),
            "kode_kecamatan": _int_or_none(postal_meta.get("kode_kecamatan")),
            "kode_kota": _int_or_none(postal_meta.get("kode_kota")),
            "nama_kota": postal_meta.get("nama_kota") or _field_value(fields, "kabupaten_kota"),
            "alamat_lengkap": postal_meta.get("alamat_lengkap"),
            "total_options": postal_meta.get("total_options", 0),
            "match_status": postal_meta.get("match_status", "not_found"),
        },
        "processed_at": datetime.now().isoformat(),
    }


def _field_value(fields: dict, field_name: str) -> str | None:
    value = (fields.get(field_name) or {}).get("value")
    return str(value) if value not in {None, ""} else None


def _formatted_ktp_address(fields: dict) -> str | None:
    address = _field_value(fields, "alamat")
    if not address:
        return None
    parts = [address]
    rt_rw = _field_value(fields, "rt_rw")
    kelurahan = _field_value(fields, "kelurahan_desa")
    kecamatan = _field_value(fields, "kecamatan")
    if rt_rw:
        parts.append(f"RT/RW {rt_rw}")
    if kelurahan:
        parts.append(f"Kel/Desa {kelurahan}")
    if kecamatan:
        parts.append(f"Kecamatan {kecamatan}")
    return ", ".join(parts)


def _birth_place(value: str | None) -> str | None:
    if not value:
        return None
    return value.split(",", 1)[0].strip() or None


def _birth_date_iso(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", value)
    if not match:
        return None
    day, month, year = [int(part) for part in match.groups()]
    try:
        return datetime(year, month, day).strftime("%Y-%m-%dT00:00:00.000Z")
    except ValueError:
        return None


def _marital_code(value: str | None) -> str | None:
    if not value:
        return None
    upper = value.upper()
    if "BELUM" in upper:
        return "TK"
    if "KAWIN" in upper:
        return "K"
    if "CERAI" in upper:
        return "C"
    return value


def _gender_code(value: str | None) -> str | None:
    if not value:
        return None
    upper = value.upper()
    if "PEREMPUAN" in upper:
        return "P"
    if "LAKI" in upper:
        return "L"
    return value


def _int_or_none(value) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_or_none(value) -> str | None:
    return str(value) if value not in {None, ""} else None


def _should_run_stnk_full_page_background(document_type: str, parsed, assessment: dict, processing_mode: str) -> bool:
    if document_type.upper() != "STNK" or parsed.document_type != "STNK":
        return False
    if processing_mode != "accurate":
        return False
    if assessment.get("decision") == "rejected_input":
        return False
    required = ("nomor_polisi", "nama_pemilik", "nomor_rangka", "nomor_mesin")
    return any(
        parsed.fields.get(field_name) and parsed.fields[field_name].status in {"missing", "invalid"}
        for field_name in required
    )


def _schedule_stnk_enrichment(
    provider,
    background_executor: ThreadPoolExecutor,
    background_slots: threading.BoundedSemaphore,
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
    if not background_slots.acquire(blocking=False):
        return {"status": "busy", "reason": "background_queue_full"}

    try:
        job_id = uuid.uuid4().hex
        jobs_dir = Path("tmp") / "stnk_enrichment"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        image_path = jobs_dir / f"{job_id}{suffix.lower()}"
        image_path.write_bytes(upload_bytes)
        output_path = jobs_dir / f"{job_id}.json"
        pending_path = jobs_dir / f"{job_id}.pending.json"
        _write_json_atomic(pending_path, {"status": "pending", "job_id": job_id})
        _start_background_job_timer(
            background_executor,
            background_slots,
            BACKGROUND_OCR_START_DELAY_SECONDS,
            _run_stnk_enrichment_job,
            (provider, image_path, output_path, pending_path),
            output_path,
            pending_path,
        )
    except Exception:
        background_slots.release()
        raise
    return {
        "status": "queued",
        "mode": "accurate_background",
        "job_id": job_id,
        "output_path": str(output_path),
    }


def _schedule_background_ocr(
    background_executor: ThreadPoolExecutor,
    background_slots: threading.BoundedSemaphore,
    provider,
    upload_bytes: bytes,
    suffix: str,
    document_type: str,
    assessment: dict,
    processing_mode: str = "accurate",
    delay_seconds: float = BACKGROUND_OCR_START_DELAY_SECONDS,
    force_strategy: str | None = None,
    mode_label: str | None = None,
) -> dict:
    if assessment.get("decision") == "rejected_input":
        return {"status": "skipped", "reason": "rejected_input"}
    if not background_slots.acquire(blocking=False):
        return {"status": "busy", "reason": "background_queue_full"}

    try:
        job_id = uuid.uuid4().hex
        jobs_dir = Path("tmp") / "stnk_enrichment"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        image_path = jobs_dir / f"{job_id}{suffix.lower()}"
        image_path.write_bytes(upload_bytes)
        output_path = jobs_dir / f"{job_id}.json"
        pending_path = jobs_dir / f"{job_id}.pending.json"
        _write_json_atomic(pending_path, {"status": "pending", "job_id": job_id, "document_type": document_type})
        _start_background_job_timer(
            background_executor,
            background_slots,
            delay_seconds,
            _run_background_ocr_job,
            (provider, image_path, document_type, processing_mode, output_path, pending_path, force_strategy),
            output_path,
            pending_path,
        )
    except Exception:
        background_slots.release()
        raise
    return {
        "status": "queued",
        "mode": mode_label or f"{processing_mode}_background",
        "job_id": job_id,
        "output_path": str(output_path),
    }


def _start_background_job_timer(
    executor: ThreadPoolExecutor,
    slots: threading.BoundedSemaphore,
    delay_seconds: float,
    job_func,
    job_args: tuple,
    output_path: Path,
    pending_path: Path,
) -> None:
    timer = threading.Timer(
        delay_seconds,
        _submit_background_job,
        args=(executor, slots, job_func, job_args, output_path, pending_path),
    )
    timer.daemon = True
    timer.start()


def _submit_background_job(
    executor: ThreadPoolExecutor,
    slots: threading.BoundedSemaphore,
    job_func,
    job_args: tuple,
    output_path: Path,
    pending_path: Path,
) -> None:
    try:
        future = executor.submit(job_func, *job_args)
    except Exception as exc:
        _write_background_failure(output_path, pending_path, exc)
        slots.release()
        return
    future.add_done_callback(lambda _completed: slots.release())


def _write_background_failure(output_path: Path, pending_path: Path, exc: Exception) -> None:
    _write_json_atomic(
        output_path,
        {
            "status": "failed",
            "error": str(exc),
            "processing_time_ms": 0.0,
        },
    )
    pending_path.unlink(missing_ok=True)


def _run_background_ocr_job(
    provider,
    image_path: Path,
    document_type: str,
    processing_mode: str,
    output_path: Path,
    pending_path: Path,
    force_strategy: str | None = None,
) -> None:
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="ocr-engine-background-") as tmpdir:
            result = run_ocr_pipeline(
                provider,
                image_path,
                document_type,
                Path(tmpdir),
                processing_mode=processing_mode,
                force_strategy=force_strategy,
            )
            record = _pipeline_result_record(result, round((time.perf_counter() - started) * 1000, 2), status="ok")
    except Exception as exc:
        record = {
            "status": "failed",
            "error": str(exc),
            "processing_time_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    _write_json_atomic(output_path, record)
    pending_path.unlink(missing_ok=True)


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

    _write_json_atomic(output_path, record)
    pending_path.unlink(missing_ok=True)
