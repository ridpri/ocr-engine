from __future__ import annotations

import base64
import json
import mimetypes
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ocr_engine.parsers.ktp import KTP_LABELS
from ocr_engine.parsers.stnk import STNK_LABELS
from ocr_engine.schemas import DocumentResult, FieldResult
from ocr_engine.service import build_input_assessment, detect_document_type
from ocr_engine.validators import normalize_plate_number, validate_plate_number


class AgentBridgeNotConfigured(RuntimeError):
    """Raised when no OpenClaw/Codex bridge is configured."""


class AgentBridgeError(RuntimeError):
    """Raised when the OpenClaw/Codex bridge returns an invalid result."""


@dataclass(slots=True)
class AgentOcrResult:
    parsed: DocumentResult
    assessment: dict
    provider: str
    model: str | None
    processing_time_ms: float
    raw_response: dict


def run_agent_ocr_bridge(image_path: Path, document_type: str, filename: str | None = None) -> AgentOcrResult:
    started = time.perf_counter()
    requested = _normalize_document_type(document_type)
    request = _build_agent_request(image_path, requested, filename)

    if os.getenv("OPENCLAW_OCR_WEBHOOK_URL"):
        provider = "openclaw_webhook"
        response = _run_webhook_bridge(request)
    elif os.getenv("OCR_AGENT_COMMAND"):
        provider = "command"
        response = _run_command_bridge(request)
    else:
        raise AgentBridgeNotConfigured(
            "Agent OCR service is not configured. Set OCR_AGENT_COMMAND or OPENCLAW_OCR_WEBHOOK_URL."
        )

    parsed = _document_result_from_agent_response(response, requested)
    detected = response.get("detected_document_type") or detect_document_type(parsed.raw_text)
    if detected == "UNKNOWN" and parsed.document_type in {"KTP", "STNK"}:
        detected = parsed.document_type
    assessment = build_input_assessment(
        parsed.raw_text,
        parsed,
        requested,
        detected,
        quality=_agent_quality_stub(response),
    )
    return AgentOcrResult(
        parsed=parsed,
        assessment=assessment,
        provider=provider,
        model=response.get("model") or os.getenv("OCR_AGENT_MODEL"),
        processing_time_ms=round((time.perf_counter() - started) * 1000, 2),
        raw_response=response,
    )


def _build_agent_request(image_path: Path, document_type: str, filename: str | None) -> dict:
    mime_type = mimetypes.guess_type(filename or image_path.name)[0] or "image/jpeg"
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "task": "extract_structured_document_fields",
        "document_type": document_type,
        "filename": filename or image_path.name,
        "mime_type": mime_type,
        "image_base64": image_base64,
        "schema": _field_names_for_document_type(document_type),
        "instructions": _agent_instructions(document_type),
    }


def _run_command_bridge(request: dict) -> dict:
    command = os.environ["OCR_AGENT_COMMAND"]
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(request, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=float(os.getenv("OCR_AGENT_TIMEOUT_SECONDS", "60")),
            check=False,
            shell=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise AgentBridgeError("Agent command timed out.") from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise AgentBridgeError(f"Agent command failed: {detail}")
    return _parse_json_response(completed.stdout)


def _run_webhook_bridge(request: dict) -> dict:
    url = os.environ["OPENCLAW_OCR_WEBHOOK_URL"]
    payload = json.dumps(request, ensure_ascii=False).encode("utf-8")
    http_request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENCLAW_OCR_TOKEN', '')}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=float(os.getenv("OCR_AGENT_TIMEOUT_SECONDS", "60"))) as response:
            return _parse_json_response(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise AgentBridgeError(f"OpenClaw webhook failed: {exc}") from exc


def _parse_json_response(text: str) -> dict:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AgentBridgeError("Agent bridge must return JSON.") from exc
    if not isinstance(payload, dict):
        raise AgentBridgeError("Agent bridge JSON response must be an object.")
    return payload


def _document_result_from_agent_response(response: dict, requested_document_type: str) -> DocumentResult:
    document_type = _normalize_document_type(response.get("document_type") or requested_document_type)
    field_names = _field_names_for_document_type(document_type)
    response_fields = response.get("fields") or {}
    if not isinstance(response_fields, dict):
        raise AgentBridgeError("Agent response 'fields' must be an object.")

    fields = {
        field_name: _field_result_from_agent_value(response_fields.get(field_name))
        for field_name in field_names
    }
    warnings = _normalize_agent_warnings(response.get("warnings") or [])
    _validate_agent_fields(document_type, fields, warnings)
    raw_text = response.get("raw_text") or ""
    return DocumentResult(
        document_type=document_type,
        schema_version=f"{document_type.lower()}.agent.v1",
        fields=fields,
        warnings=warnings,
        raw_text=str(raw_text),
        engine_version="ocr-engine-agent/0.1.0",
    )


def _field_result_from_agent_value(value) -> FieldResult:
    if isinstance(value, dict):
        status = _normalize_agent_status(value.get("status"), value.get("value"))
        return FieldResult(
            value=value.get("value"),
            confidence=float(value.get("confidence", 0.85 if value.get("value") else 0.0)),
            status=status,
            evidence=_normalize_agent_evidence(value.get("evidence", [])),
            raw=value.get("raw"),
        )
    if value:
        return FieldResult(value=str(value), confidence=0.85, status="ok", evidence=[str(value)])
    return FieldResult(value=None, confidence=0.0, status="missing")


def _normalize_agent_status(status, field_value) -> str:
    if not field_value:
        return "missing"
    normalized = str(status or "ok").strip().lower()
    if normalized in {"ok", "present", "valid", "found", "read", "detected", "confirmed"}:
        return "ok"
    if normalized in {"missing", "absent", "unreadable", "not_found", "not found", "empty"}:
        return "missing"
    return normalized


def _normalize_agent_evidence(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _normalize_agent_warnings(values) -> list[str]:
    warnings: list[str] = []
    for value in values:
        warning = str(value).strip()
        if not warning:
            continue
        if warning.startswith(("missing_required:", "invalid:", "quality:")) or warning in {
            "document_type:auto_guess",
            "document_type_mismatch",
            "document_type_unknown",
            "screen_or_desktop_capture",
            "blur_detected",
            "document_too_small",
            "low_text_density",
        }:
            warnings.append(warning)
    return warnings


def _validate_agent_fields(document_type: str, fields: dict[str, FieldResult], warnings: list[str]) -> None:
    if document_type != "STNK":
        return

    plate = fields.get("nomor_polisi")
    if plate and plate.value:
        if validate_plate_number(str(plate.value)):
            plate.value = normalize_plate_number(str(plate.value)) or plate.value
        else:
            plate.status = "invalid"
            plate.confidence = min(plate.confidence, 0.35)
            _append_warning_once(warnings, "invalid:nomor_polisi")


def _append_warning_once(warnings: list[str], warning: str) -> None:
    if warning not in warnings:
        warnings.append(warning)


def _field_names_for_document_type(document_type: str) -> list[str]:
    normalized = _normalize_document_type(document_type)
    if normalized == "STNK":
        return list(STNK_LABELS)
    return list(KTP_LABELS)


def _agent_quality_stub(response: dict) -> dict:
    flags = response.get("quality_flags") or []
    return {
        "image": {},
        "flags": [str(flag) for flag in flags],
        "metrics": {"overall_score": float(response.get("quality_score", 1.0))},
    }


def _normalize_document_type(document_type: str | None) -> str:
    value = (document_type or "AUTO").upper()
    if value == "AUTO":
        return "KTP"
    if value not in {"KTP", "STNK"}:
        raise AgentBridgeError("document_type must be KTP, STNK, or AUTO.")
    return value


def _agent_instructions(document_type: str) -> str:
    fields = ", ".join(_field_names_for_document_type(document_type))
    return (
        "Extract Indonesian document fields from the image. Return strict JSON only with keys: "
        "document_type, detected_document_type, raw_text, fields, warnings, quality_flags. "
        f"Expected document_type is {document_type}. Field names: {fields}. "
        "Each field value must be an object with value, confidence, status, evidence, raw. "
        "Use status missing when unreadable; do not invent values."
    )
