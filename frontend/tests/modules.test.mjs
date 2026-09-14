import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import test from "node:test";

const root = resolve(import.meta.dirname, "..");

const MODULES = [
  "api",
  "config",
  "emotions",
  "format",
  "history",
  "parse",
  "render",
  "thumbnail",
];

test("every module loads without touching the DOM at import time", async () => {
  for (const name of MODULES) {
    const namespace = await import(`../js/${name}.js`);
    assert.ok(Object.keys(namespace).length > 0, `js/${name}.js exports nothing`);
  }
});

test("the modules the app wires together still expose the names it imports", async () => {
  const api = await import("../js/api.js");
  for (const name of ["checkHealth", "runInference", "resolvePredictUrl", "InferenceError"]) {
    assert.ok(name in api, `js/api.js no longer exports ${name}`);
  }

  const render = await import("../js/render.js");
  for (const name of ["getElements", "showState", "renderPrediction", "renderHistory", "renderPreview"]) {
    assert.ok(name in render, `js/render.js no longer exports ${name}`);
  }

  const parse = await import("../js/parse.js");
  assert.ok("parsePrediction" in parse && "ModelOutputError" in parse);
});

test("the element ids the renderer asks for match the ids app.js drives", async () => {
  const { ELEMENT_IDS } = await import("../js/render.js");
  const source = await readFile(resolve(root, "js/app.js"), "utf8");

  const referenced = [...source.matchAll(/ui\.(\w+)\b/g)].map((match) => match[1]);
  const unknown = [...new Set(referenced)].filter((name) => !ELEMENT_IDS.includes(name));

  assert.deepEqual(unknown, [], `app.js drives ui.${unknown.join(", ui.")} which getElements never fetches`);
});

test("the module graph is relative and stays inside js/", async () => {
  for (const name of MODULES) {
    const source = await readFile(resolve(root, "js", `${name}.js`), "utf8");
    for (const match of source.matchAll(/from\s+"([^"]+)"/g)) {
      assert.match(match[1], /^\.\/[\w.-]+\.js$/, `js/${name}.js imports outside the folder: ${match[1]}`);
    }
  }
});
