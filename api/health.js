const DEFAULT_BACKEND = "http://203.194.113.161";

module.exports = async function handler(req, res) {
  const backend = (process.env.OCR_API_BASE_URL || DEFAULT_BACKEND).replace(/\/+$/, "");

  try {
    const response = await fetch(`${backend}/health`);
    const text = await response.text();
    res.status(response.status);
    res.setHeader("content-type", response.headers.get("content-type") || "application/json");
    res.send(text);
  } catch (error) {
    res.status(502).json({ status: "offline", detail: error.message });
  }
};
