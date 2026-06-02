from __future__ import annotations


def frontend_html() -> str:
    return """<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OCR KTP/STNK</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f7fb;
      --surface: #ffffff;
      --ink: #061a32;
      --muted: #516985;
      --border: #cdd9e8;
      --soft: #e7fbfd;
      --accent: #006f7c;
      --ok: #127a4a;
      --warn: #a35d00;
      --bad: #b42318;
    }
    * { box-sizing: border-box; }
    html {
      min-height: 100%;
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
    }
    body {
      margin: 0;
      min-height: 100vh;
      overflow-x: hidden;
      overflow-y: auto;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    main {
      width: min(760px, calc(100% - 32px));
      margin: 0 auto;
      padding: 30px 0 48px;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.2;
      font-weight: 750;
    }
    .choose-label {
      color: #2e4662;
      font-size: 13px;
      line-height: 1.2;
      cursor: pointer;
    }
    .upload-card,
    .result-card {
      border: 1px solid var(--border);
      border-radius: 7px;
      background: var(--surface);
    }
    .upload-card {
      padding: 17px 18px;
      margin-bottom: 18px;
    }
    .document-tabs {
      display: inline-grid;
      grid-template-columns: repeat(2, minmax(82px, 1fr));
      gap: 4px;
      margin-bottom: 12px;
      padding: 4px;
      border: 1px solid var(--border);
      border-radius: 7px;
      background: #f5f8fb;
    }
    .document-tab {
      min-height: 34px;
      border: 0;
      border-radius: 5px;
      padding: 7px 14px;
      background: transparent;
      color: #405671;
      font: inherit;
      font-size: 13px;
      font-weight: 750;
      cursor: pointer;
    }
    .document-tab[aria-selected="true"] {
      background: var(--soft);
      color: var(--accent);
    }
    input[type="file"] {
      width: 100%;
      padding: 9px;
      border: 1px solid var(--border);
      border-radius: 7px;
      background: #fbfdff;
      color: #38506c;
      font: inherit;
      font-size: 15px;
    }
    .result-card {
      padding: 17px 18px 18px;
    }
    .checkerboard {
      min-height: 340px;
      display: grid;
      place-items: center;
      overflow: hidden;
      border-radius: 5px;
      background:
        linear-gradient(45deg, #182636 25%, transparent 25%) 0 0 / 24px 24px,
        linear-gradient(45deg, transparent 75%, #182636 75%) 0 0 / 24px 24px,
        linear-gradient(45deg, transparent 75%, #182636 75%) 12px 12px / 24px 24px,
        linear-gradient(45deg, #182636 25%, #0f1b29 25%) 12px 12px / 24px 24px;
    }
    .checkerboard img {
      display: none;
      max-width: 100%;
      max-height: 62vh;
      object-fit: contain;
    }
    .summary {
      display: grid;
      gap: 10px;
      margin-top: 15px;
      max-width: 480px;
    }
    .summary-row {
      display: grid;
      grid-template-columns: minmax(96px, 140px) minmax(120px, max-content);
      gap: 12px;
      align-items: center;
      min-height: 37px;
    }
    .summary-row span,
    .field-name {
      color: var(--muted);
      font-size: 13px;
    }
    .value-pill {
      min-width: 96px;
      min-height: 36px;
      display: inline-flex;
      align-items: center;
      padding: 8px 15px;
      border-radius: 7px;
      background: var(--soft);
      color: var(--accent);
      font-weight: 750;
      overflow-wrap: anywhere;
    }
    .value-pill.ok {
      background: #e9f8ef;
      color: var(--ok);
    }
    .value-pill.warn {
      background: #fff8e8;
      color: var(--warn);
    }
    .value-pill.bad {
      background: #fff1f0;
      color: var(--bad);
    }
    .notice {
      display: none;
      margin-top: 14px;
      padding: 11px 12px;
      border: 1px solid var(--border);
      border-radius: 7px;
      background: #f8fbfe;
      color: #354251;
      font-size: 13px;
      line-height: 1.45;
    }
    .notice.warn {
      border-color: #f2c572;
      background: #fff8e8;
      color: #6b3d00;
    }
    .notice.bad {
      border-color: #f0aaa4;
      background: #fff1f0;
      color: #7a1b12;
    }
    .fields-wrap {
      display: none;
      margin-top: 14px;
    }
    .fields-title {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 5px;
    }
    .fields-list { display: grid; }
    .field-row {
      display: grid;
      grid-template-columns: minmax(108px, 180px) minmax(0, 1fr);
      gap: 12px;
      padding: 11px 0;
      border-bottom: 1px solid #edf1f6;
    }
    .field-row:last-child { border-bottom: 0; }
    .field-value {
      color: #162236;
      font-size: 15px;
      font-weight: 700;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .field-note {
      margin-top: 4px;
      color: var(--warn);
      font-size: 12px;
      font-weight: 650;
    }
    details {
      display: none;
      margin-top: 14px;
      border: 1px solid var(--border);
      border-radius: 7px;
      background: #fafcff;
    }
    summary {
      cursor: pointer;
      padding: 10px 12px;
      font-weight: 650;
      color: #354251;
    }
    pre {
      margin: 0;
      padding: 0 12px 12px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.5;
      color: #26313f;
    }
    .copy-row {
      display: none;
      justify-content: flex-end;
      margin-top: 14px;
    }
    button {
      border: 0;
      border-radius: 7px;
      padding: 10px 13px;
      background: #e6ebf1;
      color: #243141;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    .visually-hidden {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }
    @media (max-width: 640px) {
      main { width: min(100%, calc(100% - 24px)); padding-top: 24px; }
      header { align-items: flex-start; }
      .checkerboard { min-height: 250px; }
      .summary-row { grid-template-columns: minmax(94px, 128px) minmax(96px, 1fr); }
      .value-pill { width: 100%; }
      .field-row {
        grid-template-columns: 1fr;
        gap: 3px;
        padding: 12px 0;
      }
      .field-value { font-size: 15px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>OCR KTP/STNK</h1>
      <label class="choose-label" for="file-input">Choose a photo</label>
    </header>

    <section class="upload-card">
      <form id="ocr-form">
        <div class="document-tabs" role="tablist" aria-label="Jenis dokumen">
          <button class="document-tab" id="tab-ktp" type="button" role="tab" aria-selected="true" data-document-type="KTP">KTP</button>
          <button class="document-tab" id="tab-stnk" type="button" role="tab" aria-selected="false" data-document-type="STNK">STNK</button>
        </div>
        <input id="file-input" name="file" type="file" accept="image/*,application/pdf,.pdf" required />
      </form>
    </section>

    <section class="result-card">
      <div class="checkerboard">
        <img id="result-preview-img" alt="Preview dokumen" />
      </div>

      <div class="summary" aria-live="polite">
        <div class="summary-row"><span>Dokumen</span><strong class="value-pill" id="summary-type">-</strong></div>
        <div class="summary-row"><span>Status</span><strong class="value-pill" id="summary-status">-</strong></div>
        <div class="summary-row"><span>Waktu proses</span><strong class="value-pill" id="summary-processing-time">-</strong></div>
      </div>

      <div id="assessment" class="notice"></div>
      <div id="fields-wrap" class="fields-wrap">
        <div class="fields-title">Hasil OCR</div>
        <div class="fields-list" id="fields-body"></div>
      </div>
      <details id="raw-json-details">
        <summary>Raw JSON</summary>
        <pre id="raw-json"></pre>
      </details>
      <div class="copy-row" id="copy-row">
        <button id="copy-json" type="button">Copy JSON</button>
      </div>
      <div id="live-status" class="visually-hidden" aria-live="polite"></div>
    </section>
  </main>

  <script>
    const fileInput = document.getElementById("file-input");
    const resultPreviewImg = document.getElementById("result-preview-img");
    const fieldsWrap = document.getElementById("fields-wrap");
    const fieldsBody = document.getElementById("fields-body");
    const assessmentEl = document.getElementById("assessment");
    const rawJsonDetails = document.getElementById("raw-json-details");
    const rawJson = document.getElementById("raw-json");
    const copyRow = document.getElementById("copy-row");
    const copyJson = document.getElementById("copy-json");
    const liveStatus = document.getElementById("live-status");
    const documentTabs = Array.from(document.querySelectorAll(".document-tab"));

    let lastJson = null;
    let activePreviewUrl = "";
    let activeRequestId = 0;
    let selectedDocumentType = "KTP";

    const FIELD_ORDER = {
      KTP: [
        "provinsi", "kabupaten_kota", "nik", "nama", "tempat_tanggal_lahir", "jenis_kelamin",
        "alamat", "rt_rw", "kelurahan_desa", "kecamatan", "kode_pos", "agama",
        "status_perkawinan", "pekerjaan", "kewarganegaraan", "berlaku_hingga"
      ],
      STNK: [
        "nomor_polisi", "nama_pemilik", "nomor_rangka", "nomor_mesin", "merek", "tipe",
        "jenis", "tahun_pembuatan", "warna", "bahan_bakar", "berlaku_sampai", "alamat"
      ]
    };
    const FIELD_LABELS = {
      provinsi: "Provinsi",
      kabupaten_kota: "Kabupaten/Kota",
      nik: "NIK",
      nama: "Nama",
      tempat_tanggal_lahir: "Tempat/Tgl Lahir",
      jenis_kelamin: "Jenis Kelamin",
      alamat: "Alamat",
      rt_rw: "RT/RW",
      kelurahan_desa: "Kelurahan/Desa",
      kecamatan: "Kecamatan",
      kode_pos: "Kode Pos",
      agama: "Agama",
      status_perkawinan: "Status Perkawinan",
      pekerjaan: "Pekerjaan",
      kewarganegaraan: "Kewarganegaraan",
      berlaku_hingga: "Berlaku Hingga",
      nomor_polisi: "Nomor Polisi",
      nama_pemilik: "Nama Pemilik",
      nomor_rangka: "Nomor Rangka",
      nomor_mesin: "Nomor Mesin",
      merek: "Merek",
      tipe: "Tipe",
      jenis: "Jenis",
      tahun_pembuatan: "Tahun Pembuatan",
      warna: "Warna",
      bahan_bakar: "Bahan Bakar",
      berlaku_sampai: "Berlaku Sampai"
    };

    fileInput.addEventListener("change", handleFileSelection);
    documentTabs.forEach((tab) => tab.addEventListener("click", handleDocumentTabClick));

    async function handleFileSelection() {
      const file = fileInput.files[0];
      if (!file) return;
      setPreview(file);
      await runOcr(file);
    }

    async function handleDocumentTabClick(event) {
      const nextType = event.currentTarget.dataset.documentType;
      if (!nextType || nextType === selectedDocumentType) return;
      selectedDocumentType = nextType;
      renderDocumentTabs();
      const file = fileInput.files[0];
      if (file) await runOcr(file);
    }

    function renderDocumentTabs() {
      documentTabs.forEach((tab) => {
        tab.setAttribute("aria-selected", tab.dataset.documentType === selectedDocumentType ? "true" : "false");
      });
    }

    function ocrEndpoint() {
      return `/ui/ocr?document_type=${encodeURIComponent(selectedDocumentType)}&mode=fast`;
    }

    function setPreview(file) {
      const url = URL.createObjectURL(file);
      if (activePreviewUrl) URL.revokeObjectURL(activePreviewUrl);
      activePreviewUrl = url;
      resultPreviewImg.src = url;
      resultPreviewImg.style.display = "block";
    }

    async function runOcr(file) {
      const requestId = ++activeRequestId;
      clearRenderedResult();
      liveStatus.textContent = "OCR sedang berjalan.";
      setSummaryStatus("processing");
      document.getElementById("summary-processing-time").textContent = "-";

      try {
        const body = new FormData();
        body.append("file", file);

        const requestStarted = performance.now();
        const response = await fetch(ocrEndpoint(), { method: "POST", body });
        const roundtripMs = performance.now() - requestStarted;
        const data = await parseJsonResponse(response, "OCR gagal");
        if (!response.ok) throw new Error(data.detail || "OCR gagal.");
        if (requestId !== activeRequestId) return;

        renderResult(data, roundtripMs);
        liveStatus.textContent = "OCR selesai.";
      } catch (error) {
        if (requestId !== activeRequestId) return;
        renderError(error.message);
      }
    }

    async function parseJsonResponse(response, fallbackMessage) {
      const text = await response.text();
      if (!text) return {};
      try {
        return JSON.parse(text);
      } catch {
        const contentType = response.headers.get("content-type") || "unknown content-type";
        const snippet = text.replace(/\\s+/g, " ").trim().slice(0, 180);
        return {
          detail: `${fallbackMessage}: server mengembalikan non-JSON (${response.status}, ${contentType}). ${snippet || "Response kosong."}`
        };
      }
    }

    function clearRenderedResult() {
      fieldsWrap.style.display = "none";
      fieldsBody.innerHTML = "";
      assessmentEl.style.display = "none";
      assessmentEl.textContent = "";
      assessmentEl.className = "notice";
      rawJsonDetails.style.display = "none";
      rawJson.textContent = "";
      copyRow.style.display = "none";
      document.getElementById("summary-type").textContent = "-";
      setSummaryStatus("");
      document.getElementById("summary-processing-time").textContent = "-";
      lastJson = null;
    }

    function renderError(message) {
      clearRenderedResult();
      setSummaryStatus("failed");
      assessmentEl.style.display = "block";
      assessmentEl.className = "notice bad";
      assessmentEl.textContent = message;
      liveStatus.textContent = message;
    }

    function renderResult(data, roundtripMs) {
      lastJson = data;
      const inputAssessment = data.input_assessment || {};
      const decision = inputAssessment.decision || (data.needs_review ? "needs_review" : "approved_for_auto");

      document.getElementById("summary-type").textContent = data.document_type || "-";
      setSummaryStatus(decision);
      document.getElementById("summary-processing-time").textContent = formatMs(data.processing_time_ms ?? roundtripMs);

      const reasonCodes = inputAssessment.reason_codes || [];
      const warnings = data.warnings || [];
      const showAssessment = decision !== "approved_for_auto" || reasonCodes.length || warnings.length;
      const messages = showAssessment
        ? [inputAssessment.message || "", reasonCodes.length ? `Reasons: ${reasonCodes.join(", ")}` : "", warnings.length ? `Warnings: ${warnings.join(", ")}` : ""].filter(Boolean)
        : [];
      if (messages.length) {
        assessmentEl.style.display = "block";
        assessmentEl.className = decision === "rejected_input" ? "notice bad" : decision === "needs_review" ? "notice warn" : "notice";
        assessmentEl.textContent = messages.join(" ");
      }

      fieldsBody.innerHTML = "";
      orderedFieldEntries(data).forEach(([key, field]) => {
        const row = document.createElement("div");
        row.className = "field-row";
        const note = fieldNote(field);
        row.innerHTML = `
          <div class="field-name">${escapeHtml(fieldLabel(key))}</div>
          <div>
            <div class="field-value">${escapeHtml(fieldValue(field))}</div>
            ${note ? `<div class="field-note">${escapeHtml(note)}</div>` : ""}
          </div>
        `;
        fieldsBody.appendChild(row);
      });
      fieldsWrap.style.display = fieldsBody.children.length ? "block" : "none";

      rawJsonDetails.style.display = "block";
      copyRow.style.display = "flex";
      rawJson.textContent = JSON.stringify(data, null, 2);
    }

    function orderedFieldEntries(data) {
      const fields = data.fields || {};
      const preferred = FIELD_ORDER[data.document_type] || [];
      const seen = new Set();
      const entries = [];
      preferred.forEach((key) => {
        if (Object.prototype.hasOwnProperty.call(fields, key)) {
          seen.add(key);
          entries.push([key, fields[key]]);
        }
      });
      Object.entries(fields).forEach(([key, field]) => {
        if (!seen.has(key)) entries.push([key, field]);
      });
      return entries;
    }

    function setSummaryStatus(decision) {
      const statusEl = document.getElementById("summary-status");
      statusEl.className = `value-pill ${decisionClass(decision)}`;
      statusEl.textContent = decisionLabel(decision);
    }

    function decisionLabel(decision) {
      if (decision === "approved_for_auto") return "Lolos";
      if (decision === "needs_review") return "Perlu review";
      if (decision === "rejected_input") return "Ditolak";
      if (decision === "processing") return "Memproses";
      if (decision === "failed") return "Gagal";
      return "-";
    }

    function decisionClass(decision) {
      if (decision === "approved_for_auto") return "ok";
      if (decision === "needs_review" || decision === "processing") return "warn";
      if (decision === "rejected_input" || decision === "failed") return "bad";
      return "";
    }

    function fieldLabel(key) {
      return FIELD_LABELS[key] || key.replaceAll("_", " ").replace(/\\b\\w/g, (char) => char.toUpperCase());
    }

    function fieldValue(field) {
      const value = field?.value;
      if (value === null || value === undefined || value === "") return "-";
      return String(value);
    }

    function fieldNote(field) {
      const status = field?.status || "";
      if (!status || status === "ok") return "";
      return status === "missing" ? "Belum terbaca" : status === "invalid" ? "Perlu dicek" : status;
    }

    function formatMs(value) {
      if (typeof value !== "number") return "-";
      if (value >= 1000) return `${(value / 1000).toFixed(2)}s`;
      return `${value.toFixed(0)}ms`;
    }

    copyJson.addEventListener("click", async () => {
      if (!lastJson) return;
      await navigator.clipboard.writeText(JSON.stringify(lastJson, null, 2));
      liveStatus.textContent = "JSON disalin ke clipboard.";
    });

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }
  </script>
</body>
</html>"""
