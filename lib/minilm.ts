import JSZip from "jszip";
import * as ort from "onnxruntime-web";

export type ClassifierWeights = {
  model_type?: string;
  embedding_model?: string;
  embedding_dim: number;
  chunk_words?: number;
  max_chunks_per_document?: number;
  max_length?: number;
  threshold: number;
  scaler_mean: number[];
  scaler_scale: number[];
  coef: number[];
  intercept: number;
};

export type LoadedFiles = {
  onnxBuffer: ArrayBuffer;
  vocabText: string;
  weights: ClassifierWeights;
  metadata?: Record<string, unknown>;
  sourceName: string;
};

export type PredictionResult = {
  probability: number;
  label: 0 | 1;
  predictionText: "Fraud" | "Non-fraud";
  chunksUsed: number;
  embeddingDim: number;
  threshold: number;
  modelType: string;
};

type EncodedInput = {
  inputIds: number[];
  attentionMask: number[];
  tokenTypeIds: number[];
};

function pickEntry(zip: JSZip, names: string[], extensions: string[] = []) {
  const files = Object.values(zip.files).filter((entry) => !entry.dir);

  for (const wanted of names) {
    const found = files.find((entry) => entry.name.split("/").pop() === wanted);
    if (found) return found;
  }

  for (const ext of extensions) {
    const found = files.find((entry) => entry.name.toLowerCase().endsWith(ext));
    if (found) return found;
  }

  return null;
}

export async function loadBundleZip(file: File): Promise<LoadedFiles> {
  const zip = await JSZip.loadAsync(file);

  const onnxEntry = pickEntry(zip, ["model.onnx", "model_quantized.onnx"], [".onnx"]);
  const vocabEntry = pickEntry(zip, ["vocab.txt"]);
  const weightsEntry = pickEntry(zip, ["classifier_weights.json", "weights.json"]);
  const metadataEntry = pickEntry(zip, ["model_metadata.json", "metadata.json"]);

  if (!onnxEntry) {
    throw new Error("Bundle does not contain a .onnx file. Expected model.onnx.");
  }
  if (!vocabEntry) {
    throw new Error("Bundle does not contain vocab.txt. Please export tokenizer files from the notebook.");
  }
  if (!weightsEntry) {
    throw new Error("Bundle does not contain classifier_weights.json.");
  }

  const onnxBuffer = await onnxEntry.async("arraybuffer");
  const vocabText = await vocabEntry.async("string");
  const weights = JSON.parse(await weightsEntry.async("string")) as ClassifierWeights;
  const metadata = metadataEntry
    ? (JSON.parse(await metadataEntry.async("string")) as Record<string, unknown>)
    : undefined;

  validateWeights(weights);

  return {
    onnxBuffer,
    vocabText,
    weights,
    metadata,
    sourceName: file.name
  };
}

export async function loadIndividualFiles(args: {
  onnxFile: File;
  vocabFile: File;
  weightsFile: File;
  metadataFile?: File | null;
}): Promise<LoadedFiles> {
  const onnxBuffer = await args.onnxFile.arrayBuffer();
  const vocabText = await args.vocabFile.text();
  const weights = JSON.parse(await args.weightsFile.text()) as ClassifierWeights;
  const metadata = args.metadataFile
    ? (JSON.parse(await args.metadataFile.text()) as Record<string, unknown>)
    : undefined;

  validateWeights(weights);

  return {
    onnxBuffer,
    vocabText,
    weights,
    metadata,
    sourceName: args.onnxFile.name
  };
}

function validateWeights(weights: ClassifierWeights) {
  const dim = weights.embedding_dim;
  if (!Number.isInteger(dim) || dim <= 0) {
    throw new Error("classifier_weights.json must contain a positive embedding_dim.");
  }
  for (const key of ["coef", "scaler_mean", "scaler_scale"] as const) {
    if (!Array.isArray(weights[key]) || weights[key].length !== dim) {
      throw new Error(`classifier_weights.json field ${key} must have length ${dim}.`);
    }
  }
  if (typeof weights.intercept !== "number") {
    throw new Error("classifier_weights.json must contain numeric intercept.");
  }
}

function buildVocab(vocabText: string): Map<string, number> {
  const vocab = new Map<string, number>();
  const lines = vocabText.split(/\r?\n/);
  for (let i = 0; i < lines.length; i += 1) {
    const token = lines[i].trim();
    if (token && !vocab.has(token)) vocab.set(token, i);
  }
  return vocab;
}

function stripAccents(text: string) {
  return text.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
}

function isPunctuation(char: string) {
  return /[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/.test(char);
}

function basicTokenize(text: string) {
  const cleaned = stripAccents(text)
    .replace(/[\u0000-\u001f\u007f]/g, " ")
    .toLowerCase();

  const tokens: string[] = [];
  for (const raw of cleaned.split(/\s+/)) {
    if (!raw) continue;
    let current = "";
    for (const ch of raw) {
      if (isPunctuation(ch)) {
        if (current) tokens.push(current);
        tokens.push(ch);
        current = "";
      } else {
        current += ch;
      }
    }
    if (current) tokens.push(current);
  }
  return tokens;
}

function wordPieceTokenize(token: string, vocab: Map<string, number>) {
  const unkToken = "[UNK]";
  const maxInputCharsPerWord = 100;
  if (token.length > maxInputCharsPerWord) return [unkToken];

  const subTokens: string[] = [];
  let start = 0;
  while (start < token.length) {
    let end = token.length;
    let curSubstr: string | null = null;

    while (start < end) {
      let substr = token.slice(start, end);
      if (start > 0) substr = `##${substr}`;
      if (vocab.has(substr)) {
        curSubstr = substr;
        break;
      }
      end -= 1;
    }

    if (curSubstr === null) return [unkToken];
    subTokens.push(curSubstr);
    start = end;
  }
  return subTokens;
}

function encodeText(text: string, vocab: Map<string, number>, maxLength: number): EncodedInput {
  const clsId = vocab.get("[CLS]") ?? 101;
  const sepId = vocab.get("[SEP]") ?? 102;
  const padId = vocab.get("[PAD]") ?? 0;
  const unkId = vocab.get("[UNK]") ?? 100;

  const pieces: string[] = [];
  for (const token of basicTokenize(text)) {
    pieces.push(...wordPieceTokenize(token, vocab));
  }

  const usable = pieces.slice(0, Math.max(0, maxLength - 2));
  const ids = [clsId, ...usable.map((tok) => vocab.get(tok) ?? unkId), sepId];
  const attention = new Array(ids.length).fill(1);
  const tokenTypeIds = new Array(ids.length).fill(0);

  while (ids.length < maxLength) {
    ids.push(padId);
    attention.push(0);
    tokenTypeIds.push(0);
  }

  return {
    inputIds: ids,
    attentionMask: attention,
    tokenTypeIds
  };
}

function toInt64Tensor(values: number[], dims: number[]) {
  return new ort.Tensor(
    "int64",
    BigInt64Array.from(values.map((v) => BigInt(v))),
    dims
  );
}

function l2Normalize(vector: number[], eps = 1e-12) {
  let sum = 0;
  for (const v of vector) sum += v * v;
  const denom = Math.sqrt(sum) + eps;
  return vector.map((v) => v / denom);
}

function meanPoolTokenEmbeddings(tensor: ort.Tensor, attentionMask: number[]) {
  if (tensor.dims.length !== 3) {
    throw new Error(`Expected 3D token embeddings, got dims ${tensor.dims.join("x")}`);
  }
  const seqLen = tensor.dims[1];
  const hidden = tensor.dims[2];
  const data = Array.from(tensor.data as Float32Array | Float64Array | number[]);
  const pooled = new Array(hidden).fill(0);
  let count = 0;

  for (let i = 0; i < seqLen; i += 1) {
    if (attentionMask[i] !== 1) continue;
    count += 1;
    const offset = i * hidden;
    for (let j = 0; j < hidden; j += 1) {
      pooled[j] += data[offset + j];
    }
  }

  const denom = Math.max(count, 1);
  return pooled.map((v) => v / denom);
}

function extractEmbedding(outputs: Record<string, ort.Tensor>, attentionMask: number[]) {
  const entries = Object.entries(outputs);

  const preferred2d = entries.find(([name, tensor]) =>
    /sentence|embedding|pool/i.test(name) && tensor.dims.length === 2 && tensor.dims[0] === 1
  );
  if (preferred2d) {
    return Array.from(preferred2d[1].data as Float32Array | Float64Array | number[]);
  }

  const any2d = entries.find(([, tensor]) => tensor.dims.length === 2 && tensor.dims[0] === 1);
  if (any2d) {
    return Array.from(any2d[1].data as Float32Array | Float64Array | number[]);
  }

  const preferred3d = entries.find(([name, tensor]) =>
    /last_hidden|token|embedding/i.test(name) && tensor.dims.length === 3
  );
  if (preferred3d) return meanPoolTokenEmbeddings(preferred3d[1], attentionMask);

  const any3d = entries.find(([, tensor]) => tensor.dims.length === 3);
  if (any3d) return meanPoolTokenEmbeddings(any3d[1], attentionMask);

  const shapeList = entries.map(([name, tensor]) => `${name}:${tensor.dims.join("x")}`).join(", ");
  throw new Error(`Could not find a usable embedding output. Outputs: ${shapeList}`);
}

function chunkText(text: string, chunkWords: number, maxChunks: number) {
  const words = (text.toLowerCase().match(/[a-zA-Z]+/g) ?? []);
  if (words.length === 0) return [text];

  const chunks: string[] = [];
  for (let start = 0; start < words.length; start += chunkWords) {
    const part = words.slice(start, start + chunkWords);
    if (part.length >= 5) chunks.push(part.join(" "));
  }

  if (chunks.length === 0) chunks.push(words.slice(0, chunkWords).join(" "));
  if (chunks.length <= maxChunks) return chunks;

  const firstCount = Math.floor(maxChunks / 2);
  const lastCount = maxChunks - firstCount;
  return [...chunks.slice(0, firstCount), ...chunks.slice(-lastCount)];
}

export class FraudOnnxEngine {
  private session: ort.InferenceSession;
  private vocab: Map<string, number>;
  private weights: ClassifierWeights;
  private sourceName: string;

  private constructor(args: {
    session: ort.InferenceSession;
    vocab: Map<string, number>;
    weights: ClassifierWeights;
    sourceName: string;
  }) {
    this.session = args.session;
    this.vocab = args.vocab;
    this.weights = args.weights;
    this.sourceName = args.sourceName;
  }

  static async create(files: LoadedFiles) {
    // IMPORTANT:
    // ONNX Runtime Web has a JS bundle + .wasm runtime pair.
    // They MUST come from the exact same package version, otherwise errors like
    // "_OrtGetInputOutputMetadata is not a function" appear.
    // The package.json pins onnxruntime-web to 1.21.0 and postinstall copies the
    // matching .wasm files from node_modules to public/ort/.
    ort.env.wasm.wasmPaths = "/ort/";
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;

    const modelBytes = new Uint8Array(files.onnxBuffer.slice(0));
    const session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all"
    });

    return new FraudOnnxEngine({
      session,
      vocab: buildVocab(files.vocabText),
      weights: files.weights,
      sourceName: files.sourceName
    });
  }

  getStatus() {
    return {
      sourceName: this.sourceName,
      inputNames: this.session.inputNames,
      outputNames: this.session.outputNames,
      embeddingDim: this.weights.embedding_dim,
      maxLength: this.weights.max_length ?? 256,
      threshold: this.weights.threshold,
      modelType: this.weights.model_type ?? "MiniLM ONNX + Logistic Regression"
    };
  }

  private async embedChunk(text: string) {
    const maxLength = this.weights.max_length ?? 256;
    const encoded = encodeText(text, this.vocab, maxLength);
    const dims = [1, maxLength];

    const feeds: Record<string, ort.Tensor> = {};
    for (const name of this.session.inputNames) {
      if (name === "input_ids" || /input.*ids/i.test(name)) {
        feeds[name] = toInt64Tensor(encoded.inputIds, dims);
      } else if (name === "attention_mask" || /attention.*mask/i.test(name)) {
        feeds[name] = toInt64Tensor(encoded.attentionMask, dims);
      } else if (name === "token_type_ids" || /token.*type/i.test(name)) {
        feeds[name] = toInt64Tensor(encoded.tokenTypeIds, dims);
      }
    }

    if (Object.keys(feeds).length === 0) {
      throw new Error(`Could not map ONNX inputs: ${this.session.inputNames.join(", ")}`);
    }

    const outputs = await this.session.run(feeds);
    const embedding = extractEmbedding(outputs, encoded.attentionMask);
    return l2Normalize(embedding);
  }

  async embedDocument(text: string) {
    const chunkWords = this.weights.chunk_words ?? 180;
    const maxChunks = this.weights.max_chunks_per_document ?? 80;
    const chunks = chunkText(text, chunkWords, maxChunks);

    const sum = new Array(this.weights.embedding_dim).fill(0);
    for (const chunk of chunks) {
      const emb = await this.embedChunk(chunk);
      if (emb.length !== this.weights.embedding_dim) {
        throw new Error(
          `Embedding dimension mismatch. Expected ${this.weights.embedding_dim}, got ${emb.length}.`
        );
      }
      for (let i = 0; i < sum.length; i += 1) sum[i] += emb[i];
    }

    const mean = sum.map((v) => v / chunks.length);
    return {
      embedding: l2Normalize(mean),
      chunksUsed: chunks.length
    };
  }

  async predict(text: string): Promise<PredictionResult> {
    if (text.trim().length < 20) {
      throw new Error("Please provide more filing text. Minimum length is 20 characters.");
    }

    const { embedding, chunksUsed } = await this.embedDocument(text);
    let score = this.weights.intercept;

    for (let i = 0; i < this.weights.embedding_dim; i += 1) {
      const scale = this.weights.scaler_scale[i] || 1.0;
      const standardized = (embedding[i] - this.weights.scaler_mean[i]) / scale;
      score += standardized * this.weights.coef[i];
    }

    const probability = 1 / (1 + Math.exp(-score));
    const label = probability >= this.weights.threshold ? 1 : 0;

    return {
      probability,
      label: label as 0 | 1,
      predictionText: label === 1 ? "Fraud" : "Non-fraud",
      chunksUsed,
      embeddingDim: this.weights.embedding_dim,
      threshold: this.weights.threshold,
      modelType: this.weights.model_type ?? "MiniLM ONNX + Logistic Regression"
    };
  }
}
