/** Pure request policy for the Gemini director worker. No secrets, no I/O. */

const DEFAULT_MAX_BODY = 120_000;
const DEFAULT_MAX_IMAGE_BODY = 8_000_000;
const DEFAULT_MAX_TOKENS = 8192;

function hasInlineImage(contents) {
  if (!Array.isArray(contents)) return false;
  for (const item of contents) {
    const parts = item && item.parts;
    if (!Array.isArray(parts)) continue;
    for (const part of parts) {
      const blob = part && (part.inline_data || part.inlineData);
      if (blob && typeof blob.data === "string" && blob.data.length > 0) return true;
    }
  }
  return false;
}

export function sanitizeRequest(body, options = {}) {
  const allow = new Set(options.allowlist || []);
  const maxBody = options.maxBodyChars || DEFAULT_MAX_BODY;
  if (!body || typeof body !== "object") {
    return { ok: false, status: 400, error: "JSON body required" };
  }
  const raw = JSON.stringify(body);
  const limit = hasInlineImage(body.contents)
    ? options.maxImageBodyChars || DEFAULT_MAX_IMAGE_BODY
    : maxBody;
  if (raw.length > limit) {
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
