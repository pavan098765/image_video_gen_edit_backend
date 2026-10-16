/** Pure request policy for the Gemini director worker. No secrets, no I/O. */

const DEFAULT_MAX_BODY = 120_000;
const DEFAULT_MAX_TOKENS = 8192;

export function sanitizeRequest(body, options = {}) {
  const allow = new Set(options.allowlist || []);
  const maxBody = options.maxBodyChars || DEFAULT_MAX_BODY;
  if (!body || typeof body !== "object") {
    return { ok: false, status: 400, error: "JSON body required" };
  }
  const raw = JSON.stringify(body);
  if (raw.length > maxBody) {
    return { ok: false, status: 413, error: "Body is too large" };
  }
  const step = typeof body.step === "string" ? body.step.trim() : "";
  if (!step) {
    return { ok: false, status: 400, error: "step is required" };
  }
  const model = typeof body.model === "string" ? body.model.trim() : "";
  if (!model || (allow.size > 0 && !allow.has(model))) {
    return { ok: false, status: 400, error: "model is not allowlisted" };
  }
  if (!Array.isArray(body.contents) || body.contents.length === 0) {
    return { ok: false, status: 400, error: "contents is required" };
  }
  const maxTokens = Math.min(
    Number(body.maxOutputTokens) || DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOKENS,
  );
  const upstream = {
    contents: body.contents,
    generationConfig: { maxOutputTokens: maxTokens },
  };
  if (body.responseSchema && typeof body.responseSchema === "object") {
    upstream.generationConfig.responseMimeType = "application/json";
    upstream.generationConfig.responseSchema = body.responseSchema;
  }
  if (step === "research" && body.tools) {
    upstream.tools = body.tools;
  }
  return { ok: true, step, model, upstream };
}

export function responseHasKey(payload, key) {
  if (!key) return false;
  return JSON.stringify(payload).includes(key);
}
