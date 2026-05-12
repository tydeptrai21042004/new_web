"use client";

import { useMemo, useRef, useState } from "react";
import Papa from "papaparse";
import {
  FraudOnnxEngine,
  LoadedFiles,
  PredictionResult,
  loadBundleZip,
  loadIndividualFiles
} from "@/lib/minilm";

type BatchRow = Record<string, string | number | boolean | null | undefined>;

type EngineStatus = ReturnType<FraudOnnxEngine["getStatus"]>;

function formatPercent(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

function truncate(text: unknown, max = 120) {
  const s = String(text ?? "");
  return s.length > max ? `${s.slice(0, max)}...` : s;
}

function csvDownload(rows: BatchRow[]) {
  const csv = Papa.unparse(rows);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  return URL.createObjectURL(blob);
}

export default function Home() {
  const [engine, setEngine] = useState<FraudOnnxEngine | null>(null);
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [loadingModel, setLoadingModel] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [batchRunning, setBatchRunning] = useState(false);
  const [message, setMessage] = useState<string>("");
  const [error, setError] = useState<string>("");

  const [bundleFile, setBundleFile] = useState<File | null>(null);
  const [onnxFile, setOnnxFile] = useState<File | null>(null);
  const [vocabFile, setVocabFile] = useState<File | null>(null);
  const [weightsFile, setWeightsFile] = useState<File | null>(null);
  const [metadataFile, setMetadataFile] = useState<File | null>(null);

  const [inputText, setInputText] = useState("");
  const [singleResult, setSingleResult] = useState<PredictionResult | null>(null);

  const [csvRows, setCsvRows] = useState<BatchRow[]>([]);
  const [csvColumns, setCsvColumns] = useState<string[]>([]);
  const [selectedTextColumn, setSelectedTextColumn] = useState<string>("");
  const [batchResults, setBatchResults] = useState<BatchRow[]>([]);
  const [batchProgress, setBatchProgress] = useState(0);

  const bundleInputRef = useRef<HTMLInputElement | null>(null);

  const canLoadIndividual = onnxFile && vocabFile && weightsFile;

  async function createEngine(files: LoadedFiles) {
    const instance = await FraudOnnxEngine.create(files);
    setEngine(instance);
    setStatus(instance.getStatus());
    setSingleResult(null);
    setBatchResults([]);
    setMessage(`Loaded model from ${files.sourceName}.`);
  }

  async function handleLoadBundle() {
    if (!bundleFile) {
      setError("Please choose web_model_bundle.zip first.");
      return;
    }

    setLoadingModel(true);
    setError("");
    setMessage("Reading ZIP bundle and creating ONNX session in your browser...");

    try {
      const files = await loadBundleZip(bundleFile);
      await createEngine(files);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load bundle.");
    } finally {
      setLoadingModel(false);
    }
  }

  async function handleLoadIndividual() {
    if (!onnxFile || !vocabFile || !weightsFile) {
      setError("Please upload model.onnx, vocab.txt, and classifier_weights.json.");
      return;
    }

    setLoadingModel(true);
    setError("");
    setMessage("Creating ONNX session from individual files in your browser...");

    try {
      const files = await loadIndividualFiles({
        onnxFile,
        vocabFile,
        weightsFile,
        metadataFile
      });
      await createEngine(files);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load individual files.");
    } finally {
      setLoadingModel(false);
    }
  }

  async function handlePredict() {
    if (!engine) {
      setError("Please load the ONNX model bundle first.");
      return;
    }

    setPredicting(true);
    setError("");
    setMessage("Running ONNX embedding and fraud classifier in your browser...");

    try {
      const result = await engine.predict(inputText);
      setSingleResult(result);
      setMessage("Prediction finished.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed.");
    } finally {
      setPredicting(false);
    }
  }

  function handleCsvUpload(file: File | null) {
    setBatchResults([]);
    setCsvRows([]);
    setCsvColumns([]);
    setSelectedTextColumn("");
    setError("");

    if (!file) return;

    Papa.parse<BatchRow>(file, {
      header: true,
      skipEmptyLines: true,
      complete: (result) => {
        const rows = result.data ?? [];
        const columns = result.meta.fields ?? [];
        setCsvRows(rows);
        setCsvColumns(columns);
        const guess = columns.find((c) => /filing|fillings|text|content|mda|report/i.test(c)) ?? columns[0] ?? "";
        setSelectedTextColumn(guess);
        setMessage(`Loaded CSV with ${rows.length} rows.`);
      },
      error: (err) => setError(err.message)
    });
  }

  async function handleBatchPredict() {
    if (!engine) {
      setError("Please load the ONNX model bundle first.");
      return;
    }
    if (!selectedTextColumn) {
      setError("Please select the text column for prediction.");
      return;
    }

    setBatchRunning(true);
    setError("");
    setBatchResults([]);
    setBatchProgress(0);
    setMessage("Running batch prediction in browser. Large CSV files may take time.");

    try {
      const output: BatchRow[] = [];
      for (let i = 0; i < csvRows.length; i += 1) {
        const row = csvRows[i];
        const text = String(row[selectedTextColumn] ?? "");
        try {
          const pred = await engine.predict(text);
          output.push({
            ...row,
            fraud_probability: Number(pred.probability.toFixed(6)),
            predicted_label: pred.label,
            predicted_class: pred.predictionText,
            chunks_used: pred.chunksUsed
          });
        } catch (err) {
          output.push({
            ...row,
            fraud_probability: "",
            predicted_label: "",
            predicted_class: "ERROR",
            prediction_error: err instanceof Error ? err.message : "Prediction failed"
          });
        }
        setBatchProgress((i + 1) / Math.max(csvRows.length, 1));
        // Let the UI update during long runs.
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
      setBatchResults(output);
      setMessage("Batch prediction finished.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Batch prediction failed.");
    } finally {
      setBatchRunning(false);
    }
  }

  const batchDownloadUrl = useMemo(() => {
    if (batchResults.length === 0) return "";
    return csvDownload(batchResults);
  }, [batchResults]);

  return (
    <main className="page">
      <section className="hero">
        <div className="card">
          <h1>Financial Statement Fraud Detection</h1>
          <p>
            Single Vercel deployment. Upload your exported MiniLM ONNX bundle directly in the browser,
            then predict fraud probability for one filing or a CSV file. The model is not sent to a backend.
          </p>
          <div className="badge-row">
            <span className="badge">Next.js</span>
            <span className="badge">Vercel</span>
            <span className="badge">ONNX Runtime Web</span>
            <span className="badge">Client-side inference</span>
          </div>
        </div>

        <div className="card">
          <h2>Why browser-side?</h2>
          <p>
            MiniLM ONNX files are too large for normal Vercel Function upload bodies. This app keeps upload,
            embedding, and prediction inside the browser so it can deploy as one Vercel project.
          </p>
          <button
            className="secondary"
            onClick={() => bundleInputRef.current?.click()}
            disabled={loadingModel}
          >
            Choose model bundle
          </button>
        </div>
      </section>

      <section className="grid">
        <div className="stack">
          <div className="card">
            <h2>1. Load model</h2>
            <p>
              Recommended: upload <strong>web_model_bundle.zip</strong> generated by the corrected notebook.
              It should contain <code>model.onnx</code>, <code>vocab.txt</code>, and <code>classifier_weights.json</code>.
            </p>

            <div className="upload-box">
              <div className="file-row">
                <label>Upload full bundle ZIP</label>
                <input
                  ref={bundleInputRef}
                  type="file"
                  accept=".zip"
                  onChange={(e) => setBundleFile(e.target.files?.[0] ?? null)}
                />
              </div>
              <button onClick={handleLoadBundle} disabled={!bundleFile || loadingModel}>
                {loadingModel ? "Loading model..." : "Load ZIP bundle"}
              </button>
            </div>

            <p className="notice">
              Alternative: upload files separately if you do not want to use a ZIP bundle.
            </p>

            <div className="upload-box">
              <div className="file-row">
                <label>model.onnx</label>
                <input type="file" accept=".onnx" onChange={(e) => setOnnxFile(e.target.files?.[0] ?? null)} />
              </div>
              <div className="file-row">
                <label>vocab.txt</label>
                <input type="file" accept=".txt" onChange={(e) => setVocabFile(e.target.files?.[0] ?? null)} />
              </div>
              <div className="file-row">
                <label>classifier_weights.json</label>
                <input
                  type="file"
                  accept=".json"
                  onChange={(e) => setWeightsFile(e.target.files?.[0] ?? null)}
                />
              </div>
              <div className="file-row">
                <label>model_metadata.json optional</label>
                <input
                  type="file"
                  accept=".json"
                  onChange={(e) => setMetadataFile(e.target.files?.[0] ?? null)}
                />
              </div>
              <button onClick={handleLoadIndividual} disabled={!canLoadIndividual || loadingModel}>
                Load individual files
              </button>
            </div>
          </div>

          <div className="card">
            <h2>Model status</h2>
            {status ? (
              <div className="status">
                <div className="metric">
                  <span>Source</span>
                  <strong>{status.sourceName}</strong>
                </div>
                <div className="metric">
                  <span>Embedding dim</span>
                  <strong>{status.embeddingDim}</strong>
                </div>
                <div className="metric">
                  <span>Threshold</span>
                  <strong>{status.threshold}</strong>
                </div>
                <div className="metric">
                  <span>Max tokens</span>
                  <strong>{status.maxLength}</strong>
                </div>
                <div className="metric">
                  <span>ONNX inputs</span>
                  <strong>{status.inputNames.join(", ")}</strong>
                </div>
                <div className="metric">
                  <span>ONNX outputs</span>
                  <strong>{status.outputNames.join(", ")}</strong>
                </div>
              </div>
            ) : (
              <p>No model loaded yet.</p>
            )}
          </div>
        </div>

        <div className="stack">
          <div className="card">
            <h2>2. Predict one filing</h2>
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Paste financial filing / MD&A text here..."
            />
            <div className="actions">
              <button onClick={handlePredict} disabled={!engine || predicting}>
                {predicting ? "Predicting..." : "Predict fraud probability"}
              </button>
              <button
                className="secondary"
                onClick={() => {
                  setInputText("");
                  setSingleResult(null);
                }}
              >
                Clear
              </button>
            </div>

            {singleResult && (
              <div className={`result ${singleResult.label === 1 ? "fraud" : "safe"}`}>
                <h3>{singleResult.predictionText}</h3>
                <p>
                  Fraud probability: <strong>{formatPercent(singleResult.probability)}</strong>
                </p>
                <div className="prob-bar">
                  <div className="prob-fill" style={{ width: formatPercent(singleResult.probability) }} />
                </div>
                <p>
                  Chunks used: {singleResult.chunksUsed} · Threshold: {singleResult.threshold} · Dim:{" "}
                  {singleResult.embeddingDim}
                </p>
              </div>
            )}
          </div>

          <div className="card">
            <h2>3. Batch CSV prediction</h2>
            <p>Upload a CSV and choose the column that contains filing text.</p>
            <input type="file" accept=".csv" onChange={(e) => handleCsvUpload(e.target.files?.[0] ?? null)} />

            {csvColumns.length > 0 && (
              <div className="stack" style={{ marginTop: 14 }}>
                <label>Text column</label>
                <select
                  value={selectedTextColumn}
                  onChange={(e) => setSelectedTextColumn(e.target.value)}
                  style={{ padding: 12, borderRadius: 14 }}
                >
                  {csvColumns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
                <button onClick={handleBatchPredict} disabled={!engine || batchRunning || csvRows.length === 0}>
                  {batchRunning ? "Running batch..." : `Predict ${csvRows.length} rows`}
                </button>
                {batchRunning && (
                  <div className="prob-bar">
                    <div className="prob-fill" style={{ width: formatPercent(batchProgress) }} />
                  </div>
                )}
              </div>
            )}

            {batchResults.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <div className="actions">
                  <a href={batchDownloadUrl} download="fraud_predictions.csv">
                    <button>Download predictions CSV</button>
                  </a>
                </div>
                <div className="table-wrap" style={{ marginTop: 14 }}>
                  <table>
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Preview</th>
                        <th>Fraud probability</th>
                        <th>Prediction</th>
                        <th>Chunks</th>
                      </tr>
                    </thead>
                    <tbody>
                      {batchResults.slice(0, 10).map((row, idx) => (
                        <tr key={idx}>
                          <td>{idx + 1}</td>
                          <td>{truncate(row[selectedTextColumn], 160)}</td>
                          <td>{row.fraud_probability}</td>
                          <td>{row.predicted_class}</td>
                          <td>{row.chunks_used}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p>Showing first 10 rows. Download the CSV for all results.</p>
              </div>
            )}
          </div>

          {(message || error) && (
            <div className="card">
              {message && <div className="success">{message}</div>}
              {error && <div className="error" style={{ marginTop: message ? 10 : 0 }}>{error}</div>}
            </div>
          )}
        </div>
      </section>
    </main>
  );
}
