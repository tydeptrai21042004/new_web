import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    app: "fraud-onnx-vercel-one-repo",
    mode: "single-repo-vercel",
    backend: "Next.js App Router API routes",
    inference: "client-side ONNX Runtime Web",
    reason: "Large ONNX model uploads should stay in the browser; Vercel Functions have small request body limits."
  });
}
