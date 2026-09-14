import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import test from "node:test";

import { ELEMENT_IDS } from "../js/render.js";

const root = resolve(import.meta.dirname, "..");
const markup = await readFile(resolve(root, "index.html"), "utf8");
const styles = await readFile(resolve(root, "styles.css"), "utf8");

test("every element the renderer needs exists in the markup", () => {
  const missing = ELEMENT_IDS.filter((id) => !new RegExp(`id="${id}"`).test(markup));
  assert.deepEqual(missing, [], `index.html is missing: ${missing.join(", ")}`);
});

test("the stylesheet keeps the hidden attribute ahead of every class that sets display", () => {
  const rule = styles.match(/\[hidden\]\s*\{([^}]*)\}/);

  assert.ok(rule, "styles.css has no [hidden] rule");
  assert.match(rule[1], /display:\s*none\s*!important/);
});

test("the page loads the application module and its stylesheet", () => {
  assert.match(markup, /<script type="module" src="js\/app\.js">/);
  assert.match(markup, /<link rel="stylesheet" href="styles\.css"/);
});

test("the markup carries the panels the timeline of a run depends on", () => {
  for (const state of ["idle", "loading", "error", "success"]) {
    assert.match(markup, new RegExp(`data-state-panel="${state}"`));
  }
});
