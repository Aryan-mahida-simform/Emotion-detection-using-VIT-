import assert from "node:assert/strict";
import test from "node:test";

import { carriesFiles, firstFile, isImageFile } from "../js/files.js";

function dragEvent(types, files = []) {
  return { dataTransfer: { types, files } };
}

test("a drag carrying files is recognised", () => {
  assert.equal(carriesFiles(dragEvent(["Files"])), true);
  assert.equal(carriesFiles(dragEvent(["text/plain", "Files"])), true);
});

test("a drag carrying text or a URL is left to the browser", () => {
  assert.equal(carriesFiles(dragEvent(["text/plain"])), false);
  assert.equal(carriesFiles(dragEvent(["text/uri-list"])), false);
  assert.equal(carriesFiles(dragEvent([])), false);
  assert.equal(carriesFiles({}), false);
  assert.equal(carriesFiles(undefined), false);
});

test("firstFile returns the file the drop carried, or null", () => {
  const file = { name: "face.png" };
  assert.equal(firstFile(dragEvent(["Files"], [file])), file);
  assert.equal(firstFile(dragEvent(["Files"])), null);
  assert.equal(firstFile({}), null);
});

test("a typed image file is an image", () => {
  assert.equal(isImageFile({ type: "image/png", name: "face.png" }), true);
  assert.equal(isImageFile({ type: "image/jpeg", name: "face" }), true);
  assert.equal(isImageFile({ type: "image/heic", name: "face" }), true);
});

test("a typed non-image file is refused", () => {
  assert.equal(isImageFile({ type: "text/plain", name: "face.png" }), false);
  assert.equal(isImageFile({ type: "application/pdf", name: "report.pdf" }), false);
});

test("a file dragged in with an empty MIME type is read from its name", () => {
  for (const name of ["face.png", "face.JPG", "face.jpeg", "face.webp", "face.avif", "face.tiff"]) {
    assert.equal(isImageFile({ type: "", name }), true, name);
  }
});

test("an untyped file whose name is not an image is refused", () => {
  assert.equal(isImageFile({ type: "", name: "notes.txt" }), false);
  assert.equal(isImageFile({ type: "", name: "archive.zip" }), false);
  assert.equal(isImageFile({ type: "", name: "" }), false);
  assert.equal(isImageFile(undefined), false);
});
