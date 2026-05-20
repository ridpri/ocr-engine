const DEFAULT_BACKEND = "http://203.194.113.161";

module.exports.config = {
  api: {
    bodyParser: false,
  },
};

module.exports = async function handler(req, res) {
  const backend = (process.env.OCR_API_BASE_URL || DEFAULT_BACKEND).replace(/\/+$/, "");
  const upstreamPath = buildUpstreamPath(req);
  const headers = {};

  for (const [key, value] of Object.entries(req.headers)) {
    const lower = key.toLowerCase();
    if (["content-type", "accept"].includes(lower) && value) {
      headers[key] = value;
    }
  }

  const configuredKey = process.env.OCR_API_KEY;
  const browserKey = req.headers["x-api-key"] || req.headers["x-vps-api-key"];
  if (configuredKey || browserKey) {
    headers["X-API-Key"] = configuredKey || browserKey;
  }

  try {
    const body = ["GET", "HEAD"].includes(req.method) ? undefined : await readBody(req);
    const response = await fetch(`${backend}${upstreamPath}`, {
      method: req.method,
      headers,
      body,
    });
    const responseBody = Buffer.from(await response.arrayBuffer());

    res.status(response.status);
    res.setHeader("content-type", response.headers.get("content-type") || "application/json");
    res.send(responseBody);
  } catch (error) {
    res.status(502).json({ detail: `OCR proxy failed: ${error.message}` });
  }
};

function buildUpstreamPath(req) {
  let pathParts = Array.isArray(req.query.path)
    ? req.query.path
    : String(req.query.path || "").split("/").filter(Boolean);
  if (pathParts[0] === "vps") {
    pathParts = pathParts.slice(1);
  }
  const searchParams = new URLSearchParams();

  for (const [key, value] of Object.entries(req.query || {})) {
    if (key === "path") continue;
    if (Array.isArray(value)) {
      for (const item of value) searchParams.append(key, item);
    } else if (value !== undefined) {
      searchParams.append(key, value);
    }
  }

  const suffix = searchParams.toString();
  const path = `/ocr/${pathParts.map(encodeURIComponent).join("/")}`;
  return suffix ? `${path}?${suffix}` : path;
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("error", reject);
    req.on("end", () => resolve(Buffer.concat(chunks)));
  });
}
