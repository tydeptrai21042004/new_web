# Deploy instructions for Vercel

## 1. Prepare the project

```bash
cd fraud-vercel-single
npm install
npm run build
```

## 2. Push to GitHub

```bash
git init
git add .
git commit -m "single vercel onnx fraud demo"
git branch -M main
git remote add origin YOUR_REPO_URL
git push -u origin main
```

## 3. Import in Vercel

- Framework preset: Next.js
- Build command: `npm run build`
- Output directory: `.next`
- Install command: `npm install`

## 4. Use the app

After deployment:

1. Open your Vercel URL.
2. Upload `web_model_bundle.zip`.
3. Paste filing text or upload CSV.
4. Run prediction.

## Why no FastAPI backend?

For a single Vercel app, large ONNX upload through a serverless function is not reliable. This project loads the ONNX file directly in the browser using ONNX Runtime Web.
