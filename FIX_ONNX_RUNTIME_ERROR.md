# Fix for `i._OrtGetInputOutputMetadata is not a function`

This error is caused by an ONNX Runtime Web version mismatch:

- the JavaScript package is one version, but
- the `.wasm` runtime files are from another version.

The project is now fixed by:

1. Pinning `onnxruntime-web` exactly to `1.21.0`.
2. Copying the matching `.wasm` files from `node_modules/onnxruntime-web/dist` into `public/ort` during `postinstall`.
3. Loading WASM from `/ort/` instead of a hard-coded CDN version.
4. Disabling proxy workers and multi-threading for easier Vercel/browser compatibility.

## Important local reset

After applying this fix, do not reuse the old install. Run:

```bash
rm -rf node_modules package-lock.json .next
npm install
npm run dev
```

Then open:

```text
http://localhost:3000
```

## Important browser reset

Clear browser cache or open the app in an incognito window, because old WASM files may be cached.

## Vercel

Commit and push the fixed project. Vercel will run `npm install`, which triggers `postinstall` and copies the correct WASM runtime files into `public/ort` before building.
