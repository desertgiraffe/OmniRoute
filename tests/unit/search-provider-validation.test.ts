import test from "node:test";
import assert from "node:assert/strict";

const { validateProviderApiKey } = await import("../../src/lib/providers/validation.ts");

test("serper validation accepts authenticated non-auth upstream errors", async () => {
  const originalFetch = globalThis.fetch;

  globalThis.fetch = async () =>
    new Response(JSON.stringify({ error: "credits_exhausted" }), {
      status: 402,
      headers: { "content-type": "application/json" },
    });

  try {
    const result = await validateProviderApiKey({
      provider: "serper-search",
      apiKey: "valid-serper-key",
    });

    assert.equal(result.valid, true);
    assert.equal(result.error, null);
    assert.equal(result.unsupported, false);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("serper validation still rejects unauthorized keys", async () => {
  const originalFetch = globalThis.fetch;

  globalThis.fetch = async () =>
    new Response(JSON.stringify({ error: "Unauthorized" }), {
      status: 403,
      headers: { "content-type": "application/json" },
    });

  try {
    const result = await validateProviderApiKey({
      provider: "serper-search",
      apiKey: "bad-serper-key",
    });

    assert.equal(result.valid, false);
    assert.equal(result.error, "Invalid API key");
    assert.equal(result.unsupported, false);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("Kimi Code API-key validation uses the messages endpoint for both provider ids", async () => {
  const originalFetch = globalThis.fetch;
  let calls = [];

  globalThis.fetch = async (url, init = {}) => {
    calls.push({
      url: String(url),
      method: init.method || "GET",
      headers: init.headers || {},
    });

    return new Response(JSON.stringify({ ok: true }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  };

  try {
    for (const provider of ["kimi-coding", "kimi-coding-apikey"]) {
      calls = [];
      const result = await validateProviderApiKey({
        provider,
        apiKey: "sk-kimi-test",
      });

      assert.equal(result.valid, true);
      assert.equal(result.error, null);
      // The Anthropic-like validator first probes /models then falls back to the messages endpoint.
      assert.equal(calls.length, 2);

      // calls[0] is the models probe; calls[1] is the POST to the messages endpoint.
      assert.equal(calls[1].url, "https://api.kimi.com/coding/v1/messages?beta=true");
      assert.equal(calls[1].method, "POST");
      assert.equal(calls[1].headers["x-api-key"], "sk-kimi-test");
      assert.equal(calls[1].headers["Anthropic-Version"], "2023-06-01");

      for (const call of calls) {
        assert.equal(call.url.includes("?beta=true/messages"), false);
        assert.equal(call.url.includes("?beta=true/models"), false);
      }
    }
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("bailian-coding-plan validation accepts 400 as valid auth path", async () => {
  const originalFetch = globalThis.fetch;

  globalThis.fetch = async () =>
    new Response(JSON.stringify({ error: "invalid request" }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });

  try {
    const result = await validateProviderApiKey({
      provider: "bailian-coding-plan",
      apiKey: "valid-bailian-key",
    });

    assert.equal(result.valid, true);
    assert.equal(result.error, null);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("bailian-coding-plan validation rejects 401 as invalid key", async () => {
  const originalFetch = globalThis.fetch;

  globalThis.fetch = async () =>
    new Response(JSON.stringify({ error: "Unauthorized" }), {
      status: 401,
      headers: { "content-type": "application/json" },
    });

  try {
    const result = await validateProviderApiKey({
      provider: "bailian-coding-plan",
      apiKey: "bad-bailian-key",
    });

    assert.equal(result.valid, false);
    assert.equal(result.error, "Invalid API key");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("bailian-coding-plan validation rejects 403 as invalid key", async () => {
  const originalFetch = globalThis.fetch;

  globalThis.fetch = async () =>
    new Response(JSON.stringify({ error: "Forbidden" }), {
      status: 403,
      headers: { "content-type": "application/json" },
    });

  try {
    const result = await validateProviderApiKey({
      provider: "bailian-coding-plan",
      apiKey: "bad-bailian-key",
    });

    assert.equal(result.valid, false);
    assert.equal(result.error, "Invalid API key");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

// ── zai-paas-search (Z.AI PAAS v4 Web Search) ──────────────────────────────
// The validator re-resolves the region/baseUrl (mirroring the handler) and
// probes POST /api/paas/v4/web_search with a minimal body. These tests guard
// the region→endpoint mapping, the probe body shape, the baseUrl override, and
// the status→valid mapping (401/403 invalid, 400/402 valid).

test("zai-paas-search validation resolves china region to open.bigmodel.cn and probes the PAAS v4 body", async () => {
  const originalFetch = globalThis.fetch;
  let captured;

  globalThis.fetch = async (url, init = {}) => {
    captured = {
      url: String(url),
      headers: init.headers,
      body: JSON.parse(String(init.body || "{}")),
    };
    return new Response(JSON.stringify({ id: "t", created: 1, search_result: [] }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  };

  try {
    const result = await validateProviderApiKey({
      provider: "zai-paas-search",
      apiKey: "zai-paas-key",
      providerSpecificData: { apiRegion: "china" },
    });

    assert.equal(result.valid, true);
    assert.equal(result.error, null);
    assert.equal(captured.url, "https://open.bigmodel.cn/api/paas/v4/web_search");
    assert.equal(captured.headers.Authorization, "Bearer zai-paas-key");
    assert.equal(captured.headers["Content-Type"], "application/json");
    assert.deepEqual(captured.body, {
      search_engine: "search-prime",
      search_query: "test",
      count: 1,
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("zai-paas-search validation defaults to api.z.ai when no region is set", async () => {
  const originalFetch = globalThis.fetch;
  let captured;

  globalThis.fetch = async (url) => {
    captured = { url: String(url) };
    return new Response(JSON.stringify({ id: "t", created: 1, search_result: [] }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  };

  try {
    const result = await validateProviderApiKey({
      provider: "zai-paas-search",
      apiKey: "zai-paas-key",
    });

    assert.equal(result.valid, true);
    assert.equal(captured.url, "https://api.z.ai/api/paas/v4/web_search");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("zai-paas-search validation honors an explicit baseUrl override over region", async () => {
  const originalFetch = globalThis.fetch;
  let captured;

  globalThis.fetch = async (url) => {
    captured = { url: String(url) };
    return new Response(JSON.stringify({ id: "t", created: 1, search_result: [] }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  };

  try {
    const result = await validateProviderApiKey({
      provider: "zai-paas-search",
      apiKey: "zai-paas-key",
      providerSpecificData: {
        apiRegion: "china",
        baseUrl: "https://my-proxy.example.com/web_search/",
      },
    });

    assert.equal(result.valid, true);
    // Trailing slash stripped; override wins over region.
    assert.equal(captured.url, "https://my-proxy.example.com/web_search");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("zai-paas-search validation rejects 401 as invalid key", async () => {
  const originalFetch = globalThis.fetch;

  globalThis.fetch = async () =>
    new Response(JSON.stringify({ error: "Unauthorized" }), {
      status: 401,
      headers: { "content-type": "application/json" },
    });

  try {
    const result = await validateProviderApiKey({
      provider: "zai-paas-search",
      apiKey: "bad-key",
    });

    assert.equal(result.valid, false);
    assert.equal(result.error, "Invalid API key");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("zai-paas-search validation accepts 402 (credits-exhausted) as a valid key", async () => {
  const originalFetch = globalThis.fetch;

  globalThis.fetch = async () =>
    new Response(JSON.stringify({ error: "credits_exhausted" }), {
      status: 402,
      headers: { "content-type": "application/json" },
    });

  try {
    const result = await validateProviderApiKey({
      provider: "zai-paas-search",
      apiKey: "valid-but-broke-key",
    });

    assert.equal(result.valid, true);
    assert.equal(result.error, null);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
