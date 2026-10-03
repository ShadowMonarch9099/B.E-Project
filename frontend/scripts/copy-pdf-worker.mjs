// Copies pdf.js's worker into public/ so the viewer can load it from /pdf.worker.min.mjs.
// Runs before `npm run dev` and `npm run build`; the copy is gitignored.
import { copyFileSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const src = join(dirname(require.resolve("pdfjs-dist/package.json")), "build", "pdf.worker.min.mjs");
const dest = join(dirname(fileURLToPath(import.meta.url)), "..", "public", "pdf.worker.min.mjs");
mkdirSync(dirname(dest), { recursive: true });
copyFileSync(src, dest);
