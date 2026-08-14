/**
 * tests/unit/reasoning-placeholder-strip.test.ts
 *
 * Live incident: streamed assistant text was losing the spaces BETWEEN words
 * (e.g. "Bilden är en riktig JPEG nu" -> "Bildenärenriktig JPEG nu") on the
 * Responses-API / Claude streaming paths. Root cause: stripInternalReasoning
 * Placeholder() (open-sse/utils/reasoningPlaceholder.ts, added by #8081/#8162)
 * is called on every individual `delta.content` chunk, and it unconditionally
 * called .trim() even when the placeholder sentinel was never present in that
 * chunk. Tokenizers commonly emit sub-word tokens with a LEADING space as
 * part of the token (e.g. " en", " riktig") — each such chunk got its only
 * whitespace character (the inter-word space) silently trimmed away before
 * being appended to the accumulated message, while the words themselves
 * stayed intact. Punctuation-only chunks were largely unaffected, matching
 * what was observed live.
 */
import test from "node:test";
import assert from "node:assert/strict";
import {
  NON_ANTHROPIC_THINKING_PLACEHOLDER,
  stripInternalReasoningPlaceholder,
} from "../../open-sse/utils/reasoningPlaceholder.ts";

test("a plain word-token chunk with a leading space is NOT trimmed (no placeholder present)", () => {
  assert.equal(stripInternalReasoningPlaceholder(" en"), " en");
  assert.equal(stripInternalReasoningPlaceholder(" riktig"), " riktig");
});

test("a plain word-token chunk with a trailing space is NOT trimmed (no placeholder present)", () => {
  assert.equal(stripInternalReasoningPlaceholder("Bilden "), "Bilden ");
});

test("reassembling word-token chunks preserves inter-word spaces", () => {
  const chunks = ["Bilden", " är", " en", " riktig", " JPEG", " nu"];
  const rebuilt = chunks.map(stripInternalReasoningPlaceholder).join("");
  assert.equal(rebuilt, "Bilden är en riktig JPEG nu");
});

test("a chunk that IS exactly the placeholder collapses to empty string", () => {
  assert.equal(stripInternalReasoningPlaceholder(NON_ANTHROPIC_THINKING_PLACEHOLDER), "");
  assert.equal(stripInternalReasoningPlaceholder(`  ${NON_ANTHROPIC_THINKING_PLACEHOLDER}  `), "");
});

test("a chunk with the placeholder mixed into real text strips it (trim only affects the outer ends)", () => {
  // replaceAll leaves a double space where the placeholder was removed; trim()
  // only strips the string's own leading/trailing whitespace, not internal gaps.
  assert.equal(
    stripInternalReasoningPlaceholder(`foo ${NON_ANTHROPIC_THINKING_PLACEHOLDER} bar`),
    "foo  bar"
  );
});

test("an empty string stays empty", () => {
  assert.equal(stripInternalReasoningPlaceholder(""), "");
});

test("a pure-whitespace delta is NOT collapsed when no placeholder is present (#8341 regression)", () => {
  // GLM's tokenizer emits the space before a digit as its own delta ("Season"
  // + " " + "2"). The unconditional .trim()===`""` collapse (reintroduced by
  // #8254 during a conflict resolution, reverting the #8341 gate) eats that
  // pure-space delta, gluing "Season" to "2" -> "Season2". Real-world symptom:
  // "January7,2026", "first3 episodes", "Player167" — spaces dropped at every
  // letter→digit boundary. A whitespace-only chunk with no placeholder must be
  // preserved verbatim so the assembled text keeps its spaces.
  assert.equal(stripInternalReasoningPlaceholder(" "), " ");
  assert.equal(stripInternalReasoningPlaceholder("  "), "  ");
  assert.equal(stripInternalReasoningPlaceholder("\n"), "\n");
});

test("reassembling chunks with a pure-space delta before a digit preserves the space", () => {
  // The exact GLM→Anthropic failure: deltas ["Season", " ", "2"] must rebuild
  // to "Season 2", not "Season2".
  const chunks = ["Season", " ", "2"];
  const rebuilt = chunks.map(stripInternalReasoningPlaceholder).join("");
  assert.equal(rebuilt, "Season 2");
});
