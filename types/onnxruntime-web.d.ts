// Local type declaration for onnxruntime-web.
//
// Why this file exists:
// Some Vercel/Next.js TypeScript builds cannot resolve the package's bundled
// types because of the package.json "exports" field. Runtime import works, but
// type checking fails with:
// "Could not find a declaration file for module 'onnxruntime-web'".
//
// This declaration covers only the ONNX Runtime Web APIs used by this app.

declare module "onnxruntime-web" {
  export type TensorElementType =
    | "float32"
    | "float64"
    | "int32"
    | "int64"
    | "uint8"
    | "bool"
    | string;

  export type TensorData =
    | Float32Array
    | Float64Array
    | Int32Array
    | BigInt64Array
    | Uint8Array
    | Uint8ClampedArray
    | Int8Array
    | Uint16Array
    | Int16Array
    | Uint32Array
    | BigUint64Array
    | number[]
    | bigint[]
    | boolean[];

  export class Tensor {
    constructor(type: TensorElementType, data: TensorData, dims?: readonly number[]);
    readonly type: TensorElementType;
    readonly data: TensorData;
    readonly dims: number[];
  }

  export interface InferenceSessionCreateOptions {
    executionProviders?: string[];
    graphOptimizationLevel?: "disabled" | "basic" | "extended" | "all" | string;
    [key: string]: unknown;
  }

  export class InferenceSession {
    static create(
      model: Uint8Array | ArrayBuffer | string,
      options?: InferenceSessionCreateOptions
    ): Promise<InferenceSession>;

    readonly inputNames: string[];
    readonly outputNames: string[];

    run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>;
  }

  export const env: {
    wasm: {
      wasmPaths?: string | Record<string, string>;
      numThreads?: number;
      proxy?: boolean;
      [key: string]: unknown;
    };
    [key: string]: unknown;
  };
}
