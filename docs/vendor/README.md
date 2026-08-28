# Vendored browser dependencies

Committed so the standalone page (`docs/`) has no network/CDN dependency. These
files are served as-is by GitHub Pages; there is no build/copy step.

## onnxruntime-web

`docs/app.js` runs `model.onnx` in the browser with this.

- Source: npm `onnxruntime-web@1.29.0` (`bun add onnxruntime-web`), `dist/`
- Files: `ort.wasm.min.js` (UMD, wasm backend only), plus its glue
  `ort-wasm-simd-threaded.mjs` and binary `ort-wasm-simd-threaded.wasm`
- Runs single-threaded (`ort.env.wasm.numThreads = 1`) so it needs no
  COOP/COEP cross-origin-isolation headers, which GitHub Pages does not send.

To update: bump the package, re-copy these three files here, adjust the version
above.
