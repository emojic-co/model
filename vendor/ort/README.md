# Vendored onnxruntime-web

Committed so CI and the standalone page have no network/CDN dependency.
`build_web.py` copies these into `docs/vendor/`.

- Source: npm `onnxruntime-web@1.29.0` (`bun add onnxruntime-web`), `dist/`
- Files: `ort.wasm.min.js` (UMD, wasm backend only), plus its glue
  `ort-wasm-simd-threaded.mjs` and binary `ort-wasm-simd-threaded.wasm`
- Runs single-threaded (`ort.env.wasm.numThreads = 1`) so it needs no
  COOP/COEP cross-origin-isolation headers, which GitHub Pages does not send.

To update: bump the package, re-copy these three files, adjust the version above.
