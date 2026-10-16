import test from "node:test";
import assert from "node:assert/strict";
import { sanitizeRequest, responseHasKey } from "../src/policy.js";

const allow = ["gemini-2.5-flash"];

test("a request without a model is rejected", () => {
  const out = sanitizeRequest(
    { step: "script", contents: [{ parts: [{ text: "hi" }] }] },
    { allowlist: allow },
  );
  assert.equal(out.ok, false);
  assert.equal(out.status, 400);
});

test("a model that is not allowlisted is rejected", () => {
  const out = sanitizeRequest(
    {
      step: "script",
      model: "other-model",
      contents: [{ parts: [{ text: "hi" }] }],
    },
    { allowlist: allow },
  );
  assert.equal(out.ok, false);
  assert.match(out.error, /allowlisted/);
});

test("google_search is removed unless step is research", () => {
  const tools = [{ google_search: {} }];
  const script = sanitizeRequest(
    {
      step: "script",
      model: "gemini-2.5-flash",
      contents: [{ parts: [{ text: "hi" }] }],
      tools,
    },
    { allowlist: allow },
  );
  assert.equal(script.ok, true);
  assert.equal(script.upstream.tools, undefined);

  const research = sanitizeRequest(
    {
      step: "research",
      model: "gemini-2.5-flash",
      contents: [{ parts: [{ text: "hi" }] }],
      tools,
    },
    { allowlist: allow },
  );
  assert.deepEqual(research.upstream.tools, tools);
});

test("the response check catches a leaked key", () => {
  assert.equal(responseHasKey({ text: "secret-key-value" }, "secret-key-value"), true);
  assert.equal(responseHasKey({ text: "ok" }, "secret-key-value"), false);
});
