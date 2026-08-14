/**
 * WebSearch sub-request synthesis (Claude Code interop).
 *
 * Claude Code's built-in WebSearch tool does NOT execute a search locally.
 * WebSearchTool.call() (original-source-code/.../WebSearchTool/WebSearchTool.ts:254)
 * makes a SEPARATE streaming /v1/messages request to ANTHROPIC_BASE_URL (OmniRoute)
 * carrying the native `web_search_20250305` server tool, routed to the small-fast
 * model, with the query hardcoded into the single user message:
 *   "Perform a web search for the query: <query>"  (WebSearchTool.ts:258)
 *
 * Claude Code then parses the response for `server_tool_use` + `web_search_tool_result`
 * blocks (makeOutputFromSearchResponse, WebSearchTool.ts:86) to render "Did N
 * searches" with link cards. Plain function `tool_use` blocks are ignored — which
 * is why OmniRoute's existing rewrite (web_search → omniroute_web_search function
 * tool) yields "Did 0 searches": the model emits a function tool_use Claude Code
 * never counts.
 *
 * This module short-circuits that sub-request: detect it, run executeWebSearch
 * against the configured search provider (e.g. zai-paas-search), and return a
 * synthetic Anthropic response with native `server_tool_use` +
 * `web_search_tool_result` blocks so Claude Code renders real results. No upstream
 * chat call is made for the sub-request.
 *
 * Works for both streaming (SSE event array) and non-streaming (assembled message)
 * responses, and for remote clients (lives on /v1/messages — not the loopback-gated
 * /api/mcp/ route, routeGuard.ts:30).
 */

import { randomUUID } from "crypto";
import { hasNativeWebSearchTool } from "./webSearchRouting.ts";
import { sanitizeErrorMessage } from "../utils/error.ts";

const WEB_SEARCH_SUBREQUEST_PREFIX = "Perform a web search for the query: ";
const WEB_SEARCH_TOOL_NAME = "web_search";

export interface WebSearchSynthesisContext {
  /** Resolved provider id (e.g. "glm"). */
  provider?: string | null;
  /** Request source format — only "claude" (Anthropic /v1/messages) is eligible. */
  sourceFormat?: string | null;
  /** Per-(provider,model) interception override from interceptionRules. */
  interceptSearchOverride?: boolean;
}

export interface WebSearchSynthesisCallOptions {
  model: string;
  /** allowed_domains from Claude Code's server tool (WebSearchTool.ts:80). */
  allowedDomains?: string[];
  /** blocked_domains from Claude Code's server tool (WebSearchTool.ts:81). */
  blockedDomains?: string[];
}

export interface WebSearchSynthesisDeps {
  /** Injected executeWebSearch so tests can stub it. Production wires the real one. */
  executeSearch: (input: {
    query: string;
    search_type?: "web" | "news";
    max_results?: number;
    filters?: { include_domains?: string[]; exclude_domains?: string[] };
  }) => Promise<{
    cached: boolean;
    data: {
      results: Array<{ title?: string; url?: string }>;
    };
  }>;
  /** If set, executeSearch is skipped and this error is surfaced. */
  throwError?: Error;
}

type JsonRecord = Record<string, unknown>;

/**
 * Detect Claude Code's WebSearch sub-request and extract the query.
 *
 * Returns {query} when ALL hold: Anthropic source format, a native web_search
 * server tool present, interception enabled for the provider/model, and a single
 * user message whose text starts with the hardcoded WebSearch prefix. Otherwise
 * null (fall through to existing behavior — do not break other web_search callers).
 */
export function isWebSearchSubRequest(
  body: unknown,
  context: WebSearchSynthesisContext
): { query: string } | null {
  if (context.sourceFormat !== "claude") return null;
  if (context.interceptSearchOverride !== true) return null;
  if (!hasNativeWebSearchTool(body)) return null;

  const record = (body && typeof body === "object" ? body : {}) as JsonRecord;
  const messages = record.messages;
  if (!Array.isArray(messages) || messages.length !== 1) return null;

  const msg = messages[0] as JsonRecord | null;
  if (!msg || msg.role !== "user") return null;

  // User message content can be a string or an array of content blocks. The
  // WebSearch sub-request sends a plain string.
  const content = msg.content;
  if (typeof content !== "string") return null;
  if (!content.startsWith(WEB_SEARCH_SUBREQUEST_PREFIX)) return null;

  const query = content.slice(WEB_SEARCH_SUBREQUEST_PREFIX.length).trim();
  if (!query) return null;

  return { query };
}

/**
 * Build the native Anthropic content blocks for a completed web search:
 * server_tool_use (web_search, input {query}) + web_search_tool_result (hits).
 * Mirrors the block sequence Claude Code parses (WebSearchTool.ts:103-129).
 */
function buildWebSearchContentBlocks(
  query: string,
  hits: Array<{ title: string; url: string }>,
  errorBlock: JsonRecord | null
): JsonRecord[] {
  const toolUseId = `srvtoolu_${randomUUID()}`;
  const blocks: JsonRecord[] = [
    {
      type: "server_tool_use",
      id: toolUseId,
      name: WEB_SEARCH_TOOL_NAME,
      input: { query },
    },
  ];

  if (errorBlock) {
    // Claude Code error branch expects non-array content (WebSearchTool.ts:117).
    blocks.push({ type: "web_search_tool_result", tool_use_id: toolUseId, content: errorBlock });
  } else {
    blocks.push({ type: "web_search_tool_result", tool_use_id: toolUseId, content: hits });
  }
  return blocks;
}

function toHits(
  results: Array<{ title?: string; url?: string }>
): Array<{ title: string; url: string }> {
  const hits: Array<{ title: string; url: string }> = [];
  for (const r of results) {
    const title = typeof r?.title === "string" ? r.title : "";
    const url = typeof r?.url === "string" ? r.url : "";
    if (!url) continue; // a hit without a URL is useless to Claude Code's link cards
    hits.push({ title, url });
  }
  return hits;
}

async function runSearch(
  query: string,
  options: WebSearchSynthesisCallOptions,
  deps: WebSearchSynthesisDeps
): Promise<{ hits: Array<{ title: string; url: string }>; errorBlock: JsonRecord | null }> {
  if (deps.throwError) {
    return {
      hits: [],
      errorBlock: { error_code: "search_failed", message: sanitizeErrorMessage(deps.throwError) },
    };
  }
  try {
    const result = await deps.executeSearch({
      query,
      search_type: "web",
      max_results: 8, // Claude Code's max_uses (WebSearchTool.ts:82)
      filters: {
        include_domains: options.allowedDomains,
        exclude_domains: options.blockedDomains,
      },
    });
    return { hits: toHits(result?.data?.results ?? []), errorBlock: null };
  } catch (err) {
    return {
      hits: [],
      errorBlock: { error_code: "search_failed", message: sanitizeErrorMessage(err) },
    };
  }
}

/**
 * Build the synthetic Anthropic SSE event sequence for the WebSearch sub-request.
 * Returns the full event array: message_start → content_block_start/stop (×2) →
 * message_delta (end_turn) → message_stop. Event shape mirrors
 * buildSyntheticClaudeEmptyResponseEvents (open-sse/utils/stream.ts:501).
 */
export async function synthesizeWebSearchResponseEvents(
  query: string,
  options: WebSearchSynthesisCallOptions,
  deps: WebSearchSynthesisDeps
): Promise<JsonRecord[]> {
  const { hits, errorBlock } = await runSearch(query, options, deps);
  const model = options.model || "unknown";
  const blocks = buildWebSearchContentBlocks(query, hits, errorBlock);

  const events: JsonRecord[] = [
    {
      type: "message_start",
      message: {
        id: `msg_synthetic_search_${randomUUID()}`,
        type: "message",
        role: "assistant",
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    },
  ];

  blocks.forEach((block, index) => {
    events.push({ type: "content_block_start", index, content_block: block });
    events.push({ type: "content_block_stop", index });
  });

  events.push({
    type: "message_delta",
    delta: { stop_reason: "end_turn", stop_sequence: null },
    usage: { input_tokens: 0, output_tokens: 0 },
  });
  events.push({ type: "message_stop" });
  return events;
}

/**
 * Build the assembled Anthropic message for a non-streaming WebSearch sub-request.
 * Same content blocks as the streaming variant.
 */
export async function synthesizeWebSearchResponseJson(
  query: string,
  options: WebSearchSynthesisCallOptions,
  deps: WebSearchSynthesisDeps
): Promise<JsonRecord> {
  const { hits, errorBlock } = await runSearch(query, options, deps);
  const blocks = buildWebSearchContentBlocks(query, hits, errorBlock);
  return {
    id: `msg_synthetic_search_${randomUUID()}`,
    type: "message",
    role: "assistant",
    model: options.model || "unknown",
    content: blocks,
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: { input_tokens: 0, output_tokens: 0 },
  };
}
