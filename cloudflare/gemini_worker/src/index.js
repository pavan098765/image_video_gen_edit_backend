import { createVerify, createPublicKey } from "node:crypto";
import { sanitizeRequest, responseHasKey } from "./policy.js";

const CERTS_URL =
  "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com";

function decodeJson(part) {
  const pad = part.length % 4 === 0 ? "" : "=".repeat(4 - (part.length % 4));
  const b64 = part.replace(/-/g, "+").replace(/_/g, "/") + pad;
  return JSON.parse(atob(b64));
}

export async function verifyFirebaseToken(token, projectId, fetchCerts = fetch) {
  if (!token || !projectId) return null;
  const parts = token.split(".");
  if (parts.length !== 3) return null;
  let header;
  let payload;
  try {
    header = decodeJson(parts[0]);
    payload = decodeJson(parts[1]);
  } catch {
    return null;
  }
  if (header.alg !== "RS256" || !header.kid) return null;
  const now = Math.floor(Date.now() / 1000);
  if (!payload.exp || payload.exp < now) return null;
  if (payload.aud !== projectId) return null;
  if (payload.iss !== `https://securetoken.google.com/${projectId}`) return null;
  if (!payload.sub) return null;

  const certsResp = await fetchCerts(CERTS_URL);
  if (!certsResp.ok) return null;
  const certs = await certsResp.json();
  const pem = certs[header.kid];
  if (!pem) return null;
  const verifier = createVerify("RSA-SHA256");
  verifier.update(`${parts[0]}.${parts[1]}`);
  const signature = Buffer.from(parts[2], "base64url");
  const ok = verifier.verify(createPublicKey(pem), signature);
  return ok ? String(payload.sub) : null;
}

async function allowCall(env, uid) {
  const limit = Number(env.HOURLY_LIMIT || 60);
  if (!env.RATE_LIMIT) return true;
  const hour = new Date().toISOString().slice(0, 13);
  const key = `uid:${uid}:${hour}`;
  const current = Number((await env.RATE_LIMIT.get(key)) || "0");
  if (current >= limit) return false;
  await env.RATE_LIMIT.put(key, String(current + 1), { expirationTtl: 7200 });
  return true;
}

async function handle(request, env) {
  if (request.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: cors() });
  }
  if (request.method !== "POST" || new URL(request.url).pathname !== "/v1/generate") {
    return json({ error: "Not found" }, 404);
  }
  const auth = request.headers.get("Authorization") || "";
  const token = auth.startsWith("Bearer ") ? auth.slice(7).trim() : "";
  if (!token) return json({ error: "Sign in required" }, 401);
  const uid = await verifyFirebaseToken(token, env.FIREBASE_PROJECT_ID);
  if (!uid) return json({ error: "Sign in required" }, 401);

  let body;
  try {
    body = await request.json();
  } catch {
    return json({ error: "JSON body required" }, 400);
  }
  const allowlist = String(env.GEMINI_MODEL || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const checked = sanitizeRequest(body, { allowlist });
  if (!checked.ok) return json({ error: checked.error }, checked.status);
  if (!(await allowCall(env, uid))) {
    return json({ error: "Hourly limit reached" }, 429);
  }

  const url =
    "https://generativelanguage.googleapis.com/v1beta/models/" +
    `${encodeURIComponent(checked.model)}:generateContent`;
  const upstream = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": env.GEMINI_API_KEY,
    },
    body: JSON.stringify(checked.upstream),
  });
  const text = await upstream.text();
  console.log(JSON.stringify({ uid, step: checked.step, status: upstream.status }));
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    parsed = { error: "Upstream failed" };
  }
  if (responseHasKey(parsed, env.GEMINI_API_KEY)) {
    return json({ error: "Upstream failed" }, 502);
  }
  return json(parsed, upstream.ok ? 200 : upstream.status);
}

function cors() {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Authorization, Content-Type",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
  };
}

function json(obj, status) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json", ...cors() },
  });
}

export default { fetch: handle };
