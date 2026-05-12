# Runtime fix for `/ort/ort-wasm-simd-threaded.jsep.mjs`

This project is a one-repo Vercel app:

- Frontend: `app/page.tsx`
- Backend API routes: `app/api/*/route.ts`
- ONNX inference: browser-side through `onnxruntime-web`

The browser-side design is used because the uploaded MiniLM ONNX bundle is too large for normal Vercel Function request bodies.

## What was fixed

The earlier copy script copied only `.wasm` files. ONNX Runtime Web 1.19+ also dynamically loads `.mjs` helper files, for example:

```txt
/ort/ort-wasm-simd-threaded.jsep.mjs
```

So the corrected `scripts/copy-ort-wasm.mjs` copies both:

```txt
*.wasm
*.mjs
```

from:

```txt
node_modules/onnxruntime-web/dist
```

to:

```txt
public/ort
```

## Clean local reinstall

```bash
rm -rf node_modules package-lock.json .next public/ort
npm install
npm run build
npm run dev
```

Then verify these URLs work:

```txt
http://localhost:3000/ort/ort-wasm-simd-threaded.jsep.mjs
http://localhost:3000/ort/ort-wasm-simd-threaded.jsep.wasm
```

On Vercel, verify:

```txt
https://YOUR_SITE.vercel.app/ort/ort-wasm-simd-threaded.jsep.mjs
```

If this URL is 404, the `postinstall` script did not run or `public/ort` was not included in the deployment.
