#!/usr/bin/env node
import { cp, mkdir, readdir, rm, stat } from "node:fs/promises";
import { join, resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const outDir = join(root, "dist");
const ENTRIES = ["index.html", "styles.css", "js"];

await rm(outDir, { recursive: true, force: true });
await mkdir(outDir, { recursive: true });

for (const entry of ENTRIES) {
  await cp(join(root, entry), join(outDir, entry), { recursive: true });
}

const files = await listFiles(outDir);
let total = 0;
for (const file of files) {
  total += (await stat(file)).size;
}

process.stdout.write(`built ${files.length} files into ${outDir} (${(total / 1024).toFixed(1)} KB)\n`);
for (const file of files) {
  process.stdout.write(`  ${file.slice(outDir.length + 1)}\n`);
}

async function listFiles(directory) {
  const found = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      found.push(...(await listFiles(path)));
      continue;
    }
    found.push(path);
  }
  return found;
}
