from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


KTP_FIELDS = [
    "nik",
    "nama",
    "tempat_tanggal_lahir",
    "jenis_kelamin",
    "alamat",
    "rt_rw",
    "kelurahan_desa",
    "kecamatan",
    "agama",
    "status_perkawinan",
    "pekerjaan",
    "kewarganegaraan",
    "berlaku_hingga",
]
STNK_FIELDS = [
    "nomor_polisi",
    "nama_pemilik",
    "alamat",
    "merek",
    "tipe",
    "jenis",
    "tahun_pembuatan",
    "warna",
    "nomor_rangka",
    "nomor_mesin",
    "bahan_bakar",
    "berlaku_sampai",
]


def main() -> int:
    request = json.load(sys.stdin)
    document_type = _normalize_document_type(request.get("document_type"))
    fields = STNK_FIELDS if document_type == "STNK" else KTP_FIELDS
    suffix = _suffix_from_mime_type(request.get("mime_type"))
    model = os.getenv("OCR_AGENT_CODEX_MODEL", "gpt-5.4-mini")

    with tempfile.TemporaryDirectory(prefix="ocr-codex-agent-") as tmp:
        tmpdir = Path(tmp)
        image_path = tmpdir / f"input{suffix}"
        output_path = tmpdir / "codex-output.json"
        image_path.write_bytes(base64.b64decode(request["image_base64"]))
        prompt = _prompt(document_type, fields)

        completed = subprocess.run(
            [
                _codex_executable(),
                "exec",
                "--ephemeral",
                "--skip-git-repo-check",
                "-m",
                model,
                "--image",
                str(image_path),
                "-o",
                str(output_path),
                prompt,
            ],
            text=True,
            capture_output=True,
            timeout=float(os.getenv("OCR_AGENT_CODEX_TIMEOUT_SECONDS", "120")),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout).strip())
        if not output_path.exists():
            raise RuntimeError("Codex did not write an output file.")

        payload = _extract_json_object(output_path.read_text(encoding="utf-8"))
        payload.setdefault("document_type", document_type)
        payload.setdefault("detected_document_type", payload["document_type"])
        payload.setdefault("raw_text", "")
        payload.setdefault("fields", {})
        payload["model"] = model
        payload["warnings"] = list(payload.get("warnings") or [])
        print(json.dumps(payload, ensure_ascii=False))
        return 0


def _prompt(document_type: str, fields: list[str]) -> str:
    return (
        "You are an OCR extraction engine for Indonesian KTP/STNK documents. "
        "Read the attached image directly and return strict JSON only, no markdown, no explanation. "
        "The JSON object must contain document_type, detected_document_type, raw_text, fields, warnings. "
        f"Expected document_type: {document_type}. "
        f"Field names: {', '.join(fields)}. "
        "Every field must be an object with value, confidence, status, evidence, raw. "
        "Use status missing and value null when unreadable. Do not invent values. "
        "For STNK, nomor_polisi is the police plate/registration number, not a receipt number."
    )


def _extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Codex output must be a JSON object.")
    return value


def _normalize_document_type(value: str | None) -> str:
    upper = (value or "AUTO").upper()
    return upper if upper in {"KTP", "STNK"} else "KTP"


def _suffix_from_mime_type(value: str | None) -> str:
    mime = (value or "").lower()
    if "png" in mime:
        return ".png"
    if "webp" in mime:
        return ".webp"
    return ".jpg"


def _codex_executable() -> str:
    if configured := os.getenv("OCR_AGENT_CODEX_EXECUTABLE"):
        return configured
    if detected := shutil.which("codex"):
        return detected
    default = Path.home() / "AppData" / "Local" / "OpenAI" / "Codex" / "bin" / "codex.exe"
    return str(default)


if __name__ == "__main__":
    raise SystemExit(main())
