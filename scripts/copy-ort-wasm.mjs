import { copyFileSync, existsSync, mkdirSync, readdirSync } from "node:fs";
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

const files = readdirSync(srcDir).filter((name) => name.endsWith(".wasm"));
if (files.length === 0) {
  console.warn("[copy-ort-wasm] No .wasm files found in onnxruntime-web/dist.");
  process.exit(0);
}

for (const file of files) {
  copyFileSync(join(srcDir, file), join(outDir, file));
  console.log(`[copy-ort-wasm] copied ${file} -> public/ort/${file}`);
}
