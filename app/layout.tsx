import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Financial Fraud ONNX Demo",
  description: "Single Vercel deployment for ONNX financial statement fraud detection"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
