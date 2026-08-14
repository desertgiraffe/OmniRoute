import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

// Integration test for the WebSearch sub-request synthesis short-circuit on
// /v1/messages. Claude Code's WebSearchTool sends a sub-request with the native
// web_search_20250305 server tool + a "Perform a web search for the query: <q>"
// user message. With interception enabled for the provider, OmniRoute must
// short-circuit it: run executeWebSearch (→ zai-paas-search) and return a
// synthetic Anthropic SSE response with server_tool_use + web_search_tool_result
// blocks — NO upstream chat call to GLM.

const TEST_DATA_DIR = fs.mkdtempSync(path.join(os.tmpdir(), "omniroute-ws-synth-"));
process.env.DATA_DIR = TEST_DATA_DIR;

const core = await import("../../src/lib/db/core.ts");
const providersDb = await import("../../src/lib/db/providers.ts");
const settingsDb = await import("../../src/lib/db/settings.ts");
const messagesRoute = await import("../../src/app/api/v1/messages/route.ts");

async function resetStorage() {
  core.resetDbInstance();
  fs.rmSync(TEST_DATA_DIR, { recursive: true, force: true });
  fs.mkdirSync(TEST_DATA_DIR, { recursive: true });
}

async function seedConnection(
  provider: string,
  overrides: { apiKey?: string | null; providerSpecificData?: Record<string, unknown> } = {}
) {
  return providersDb.createProviderConnection({
    provider,
    authType: "apikey",
    name: `${provider}-${Math.random().toString(16).slice(2, 8)}`,
    apiKey: overrides.apiKey ?? "test-key",
    isActive: true,
    testStatus: "active",
    providerSpecificData: overrides.providerSpecificData || {},
  });
}

// Set the per-provider interception rule (mirrors the operator's DB row:
// interception_rules / glm = {"interceptSearch":true}).
async function enableGlmIntercept() {
  const db = core.getDbInstance();
  db.prepare("INSERT OR REPLACE INTO key_value (namespace, key, value) VALUES (?, ?, ?)").run(
    "interception_rules",
    "glm",
    JSON.stringify({ interceptSearch: true })
  );
}

test.beforeEach(async () => {
  await resetStorage();
});

test.after(() => {
  core.resetDbInstance();
  fs.rmSync(TEST_DATA_DIR, { recursive: true, force: true });
});

test("WebSearch sub-request on /v1/messages is short-circuited to a synthetic server-tool SSE response", async () => {
  await seedConnection("zai-paas-search", {
    apiKey: "zai-paas-key",
    providerSpecificData: { region: "china" },
  });
  await enableGlmIntercept();

  const originalFetch = globalThis.fetch;
  const fetchCalls: string[] = [];
  globalThis.fetch = async (url, init = {}) => {
    const urlStr = String(url);
    fetchCalls.push(urlStr);
    // Mock the zai-paas-search PAAS v4 upstream (open.bigmodel.cn/api/paas/v4/web_search).
    if (urlStr.includes("/api/paas/v4/web_search")) {
      return new Response(
        JSON.stringify({
          id: "t",
          created: 1,
          search_result: [
            {
              title: "Elon Musk - Wikipedia",
              link: "https://en.wikipedia.org/wiki/Elon_Musk",
              content: "Entrepreneur",
              publish_date: "",
              icon: "",
              media: "wikipedia",
            },
            {
              title: "SpaceX",
              link: "https://www.spacex.com",
              content: "Aerospace",
              publish_date: "",
              icon: "",
              media: "spacex",
            },
          ],
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      );
    }
    // Any other fetch (e.g. an upstream GLM chat call) would be a bug — the
    // sub-request must be short-circuited. Return a 500 to make it obvious.
    return new Response(JSON.stringify({ error: "unexpected upstream call" }), {
      status: 500,
      headers: { "content-type": "application/json" },
    });
  };

  try {
    const response = await messagesRoute.POST(
      new Request("http://localhost/v1/messages", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          accept: "text/event-stream",
        },
        body: JSON.stringify({
          model: "glm/glm-4.7",
          stream: true,
          messages: [{ role: "user", content: "Perform a web search for the query: Elon Musk" }],
          tools: [{ type: "web_search_20250305", name: "web_search", max_uses: 8 }],
        }),
      })
    );

    assert.equal(response.status, 200);
    const sseText = await response.text();
    assert.ok(sseText.includes("server_tool_use"), "SSE must contain a server_tool_use block");
    assert.ok(
      sseText.includes("web_search_tool_result"),
      "SSE must contain a web_search_tool_result block"
    );
    assert.ok(sseText.includes("Elon Musk - Wikipedia"), "result hit title present");
    assert.ok(sseText.includes("https://www.spacex.com"), "result hit url present");
    assert.ok(sseText.includes("end_turn"), "stop_reason end_turn present");

    // The sub-request must NOT have triggered an upstream GLM chat call — only the
    // zai-paas-search fetch should have happened.
    const upstreamChatCalls = fetchCalls.filter((u) => !u.includes("/api/paas/v4/web_search"));
    assert.equal(
      upstreamChatCalls.length,
      0,
      `expected no upstream chat call, got: ${JSON.stringify(upstreamChatCalls)}`
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("a normal chat turn with the server tool but WITHOUT the WebSearch prefix is NOT short-circuited", async () => {
  await seedConnection("zai-paas-search", { apiKey: "zai-paas-key" });
  await enableGlmIntercept();

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () =>
    new Response(JSON.stringify({ error: "should not be called by detector" }), {
      status: 500,
    });

  try {
    // No "Perform a web search for the query:" prefix → not a sub-request → must NOT
    // synthesize. It will fall through to the normal chat pipeline (which will try
    // to route to glm; here we only assert it did NOT return a synthetic 200 SSE
    // with web_search_tool_result, i.e. the short-circuit did not fire).
    const response = await messagesRoute.POST(
      new Request("http://localhost/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json", accept: "text/event-stream" },
        body: JSON.stringify({
          model: "glm/glm-5.2",
          stream: true,
          messages: [{ role: "user", content: "What is the capital of France?" }],
          tools: [{ type: "web_search_20250305", name: "web_search" }],
        }),
      })
    );
    const text = await response.text();
    assert.ok(
      !text.includes("web_search_tool_result"),
      "non-sub-request must not be short-circuited into a synthetic web_search_tool_result"
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});
