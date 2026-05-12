import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, "..");
const srcDir = join(root, "node_modules", "onnxruntime-web", "dist");
const outDir = join(root, "public", "ort");

if (!existsSync(srcDir)) {
  console.warn("[copy-ort-wasm] onnxruntime-web/dist not found yet. Skipping copy.");
  process.exit(0);
}

mkdirSync(outDir, { recursive: true });

// ONNX Runtime Web 1.19+ dynamically imports both .mjs helper modules and .wasm binaries.
// Copying only .wasm causes runtime errors such as:
// Failed to fetch dynamically imported module: /ort/ort-wasm-simd-threaded.jsep.mjs
const allowed = new Set([".wasm", ".mjs"]);
const files = readdirSync(srcDir).filter((name) => {
  if (!name.startsWith("ort")) return false;
  return [...allowed].some((ext) => name.endsWith(ext));
});

if (files.length === 0) {
  console.warn("[copy-ort-wasm] No ONNX Runtime Web .wasm/.mjs files found in onnxruntime-web/dist.");
  process.exit(0);
}

let copied = 0;
let totalBytes = 0;

for (const file of files) {
  const src = join(srcDir, file);
  const dst = join(outDir, file);
  copyFileSync(src, dst);
  copied += 1;
  totalBytes += statSync(dst).size;
  console.log(`[copy-ort-wasm] copied ${file} -> public/ort/${file}`);
}

console.log(`[copy-ort-wasm] copied ${copied} files (${(totalBytes / 1024 / 1024).toFixed(2)} MB) to public/ort`);
