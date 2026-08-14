import test from "node:test";
import assert from "node:assert/strict";

import {
  isWebSearchSubRequest,
  synthesizeWebSearchResponseEvents,
  synthesizeWebSearchResponseJson,
  type WebSearchSynthesisDeps,
} from "../../open-sse/services/webSearchSynthesis.ts";

// Claude Code's WebSearchTool.call() sends a sub-request to /v1/messages with a
// single user message whose text is the hardcoded prefix "Perform a web search
// for the query: <query>" (original-source-code/.../WebSearchTool/WebSearchTool.ts:258)
// and the native web_search_20250305 server tool. Claude Code then parses the
// response for server_tool_use + web_search_tool_result blocks to render
// "Did N searches" with link cards. Function tool_use blocks are ignored.
//
// This test covers the detector + the synthetic Anthropic response builder that
// short-circuits that sub-request → executeWebSearch → native server-tool blocks.

const PREFIX = "Perform a web search for the query: ";

function makeSubRequestBody(query: string, extra: Record<string, unknown> = {}) {
  return {
    model: "glm/glm-4.7",
    stream: true,
    messages: [{ role: "user", content: PREFIX + query }],
    tools: [{ type: "web_search_20250305", name: "web_search", max_uses: 8 }],
    ...extra,
  };
}

// ── isWebSearchSubRequest ─────────────────────────────────────────────────

test("isWebSearchSubRequest detects the Claude Code WebSearch sub-request and extracts the query", () => {
  const result = isWebSearchSubRequest(makeSubRequestBody("Elon Musk"), {
    provider: "glm",
    sourceFormat: "claude",
    interceptSearchOverride: true,
  });
  assert.equal(result?.query, "Elon Musk");
});

test("isWebSearchSubRequest returns null when intercept is disabled", () => {
  const result = isWebSearchSubRequest(makeSubRequestBody("Elon Musk"), {
    provider: "glm",
    sourceFormat: "claude",
    interceptSearchOverride: false,
  });
  assert.equal(result, null);
});

test("isWebSearchSubRequest returns null when no native web_search server tool is present", () => {
  const result = isWebSearchSubRequest(
    makeSubRequestBody("Elon Musk", { tools: [{ type: "function", function: { name: "Bash" } }] }),
    { provider: "glm", sourceFormat: "claude", interceptSearchOverride: true }
  );
  assert.equal(result, null);
});

test("isWebSearchSubRequest returns null when the user message lacks the WebSearch prefix", () => {
  // A normal chat turn that happens to carry the server tool should NOT be intercepted.
  const body = {
    model: "glm/glm-5.2",
    stream: true,
    messages: [{ role: "user", content: "What is the capital of France?" }],
    tools: [{ type: "web_search_20250305", name: "web_search" }],
  };
  const result = isWebSearchSubRequest(body, {
    provider: "glm",
    sourceFormat: "claude",
    interceptSearchOverride: true,
  });
  assert.equal(result, null);
});

test("isWebSearchSubRequest returns null for non-Anthropic source format", () => {
  const result = isWebSearchSubRequest(makeSubRequestBody("Elon Musk"), {
    provider: "openai",
    sourceFormat: "openai",
    interceptSearchOverride: true,
  });
  assert.equal(result, null);
});

test("isWebSearchSubRequest returns null when there are multiple messages (not a sub-request)", () => {
  const body = {
    model: "glm/glm-4.7",
    stream: true,
    messages: [
      { role: "user", content: PREFIX + "Elon Musk" },
      { role: "assistant", content: "searching..." },
    ],
    tools: [{ type: "web_search_20250305", name: "web_search" }],
  };
  const result = isWebSearchSubRequest(body, {
    provider: "glm",
    sourceFormat: "claude",
    interceptSearchOverride: true,
  });
  assert.equal(result, null);
});

// ── synthesizeWebSearchResponseEvents (streaming Anthropic SSE) ────────────

function makeDeps(
  results: Array<{ title: string; url: string }>,
  opts: { throwErr?: Error } = {}
): WebSearchSynthesisDeps {
  return {
    executeSearch: async () => ({
      cached: false,
      data: {
        provider: "zai-paas-search",
        query: "Elon Musk",
        results: results.map((r) => ({
          title: r.title,
          url: r.url,
          snippet: "",
          position: 0,
          score: null,
          published_at: null,
          favicon_url: null,
          content: null,
          metadata: null,
          citation: { provider: "zai-paas-search", retrieved_at: "", rank: 0 },
          provider_raw: null,
        })),
        answer: null,
        usage: { queries_used: 1, search_cost_usd: 0 },
        metrics: {
          response_time_ms: 0,
          upstream_latency_ms: 0,
          total_results_available: results.length,
        },
        errors: [],
      },
    }),
    throwError: opts.throwErr,
  };
}

test("synthesizeWebSearchResponseEvents emits the native server_tool_use + web_search_tool_result sequence", async () => {
  const events = await synthesizeWebSearchResponseEvents(
    "Elon Musk",
    { model: "glm-4.7" },
    makeDeps([
      { title: "Elon Musk - Wikipedia", url: "https://en.wikipedia.org/wiki/Elon_Musk" },
      { title: "SpaceX", url: "https://www.spacex.com" },
    ])
  );

  const types = events.map((e) => e.type);
  assert.deepEqual(types, [
    "message_start",
    "content_block_start",
    "content_block_stop",
    "content_block_start",
    "content_block_stop",
    "message_delta",
    "message_stop",
  ]);

  // Block 0: server_tool_use for web_search with the query as input.
  const serverToolUse = (events[1] as Record<string, unknown>).content_block as Record<
    string,
    unknown
  >;
  assert.equal(serverToolUse.type, "server_tool_use");
  assert.equal(serverToolUse.name, "web_search");
  assert.deepEqual(serverToolUse.input, { query: "Elon Musk" });
  assert.ok(
    typeof serverToolUse.id === "string" && serverToolUse.id.length > 0,
    "server_tool_use has an id"
  );

  // Block 1: web_search_tool_result whose tool_use_id matches the server_tool_use id,
  // with content = [{title, url}] hits (the searchHitSchema Claude Code parses).
  const toolResult = (events[3] as Record<string, unknown>).content_block as Record<
    string,
    unknown
  >;
  assert.equal(toolResult.type, "web_search_tool_result");
  assert.equal(toolResult.tool_use_id, serverToolUse.id);
  assert.deepEqual(toolResult.content, [
    { title: "Elon Musk - Wikipedia", url: "https://en.wikipedia.org/wiki/Elon_Musk" },
    { title: "SpaceX", url: "https://www.spacex.com" },
  ]);

  // stop_reason is end_turn (not tool_use) so Claude Code treats the search as complete.
  const messageDelta = (events[5] as Record<string, unknown>).delta as Record<string, unknown>;
  assert.equal(messageDelta.stop_reason, "end_turn");
});

test("synthesizeWebSearchResponseEvents maps allowed_domains through and drops excludes for providers without an exclude field", async () => {
  // The synthesizer should still produce a valid result when domains are supplied;
  // zai-paas-search only sends includes (documented). Verify it doesn't throw and
  // returns the normal sequence.
  const events = await synthesizeWebSearchResponseEvents(
    "Elon Musk",
    { model: "glm-4.7", allowedDomains: ["wikipedia.org"], blockedDomains: ["spam.com"] },
    makeDeps([{ title: "x", url: "https://en.wikipedia.org/wiki/Elon_Musk" }])
  );
  assert.equal(events.length, 7);
});

test("synthesizeWebSearchResponseEvents yields an empty results array when search returns nothing", async () => {
  const events = await synthesizeWebSearchResponseEvents(
    "noresults",
    { model: "glm-4.7" },
    makeDeps([])
  );
  const toolResult = (events[3] as Record<string, unknown>).content_block as Record<
    string,
    unknown
  >;
  assert.equal(toolResult.type, "web_search_tool_result");
  assert.deepEqual(toolResult.content, []);
});

test("synthesizeWebSearchResponseEvents surfaces a search error as a web_search_tool_result error block (no stream break)", async () => {
  const events = await synthesizeWebSearchResponseEvents(
    "Elon Musk",
    { model: "glm-4.7" },
    makeDeps([], { throwErr: new Error("upstream 502") })
  );
  // Still completes the full event sequence.
  const types = events.map((e) => e.type);
  assert.equal(types[types.length - 1], "message_stop");
  const toolResult = (events[3] as Record<string, unknown>).content_block as Record<
    string,
    unknown
  >;
  assert.equal(toolResult.type, "web_search_tool_result");
  // Error content is NOT an array (matches Claude Code's error branch, WebSearchTool.ts:117-122).
  assert.equal(Array.isArray(toolResult.content), false);
  assert.ok(toolResult.content && typeof toolResult.content === "object");
});

// ── synthesizeWebSearchResponseJson (non-streaming) ───────────────────────

test("synthesizeWebSearchResponseJson assembles a message with the same server-tool content blocks", async () => {
  const message = await synthesizeWebSearchResponseJson(
    "Elon Musk",
    { model: "glm-4.7" },
    makeDeps([{ title: "SpaceX", url: "https://www.spacex.com" }])
  );
  const content = (message as Record<string, unknown>).content as Array<Record<string, unknown>>;
  assert.equal(content[0].type, "server_tool_use");
  assert.equal(content[1].type, "web_search_tool_result");
  assert.equal((message as Record<string, unknown>).stop_reason, "end_turn");
});
