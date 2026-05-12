export const runtime = "edge";

export async function GET() {
  return Response.json({
    ok: true,
    app: "fraud-onnx-vercel-single",
    inference: "client-side ONNX Runtime Web",
    note: "Large ONNX uploads are handled in the browser, not through Vercel Functions."
  });
}
