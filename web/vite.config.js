import { resolve } from 'node:path'
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  base: './',
  plugins: [react()],
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: {
    chunkSizeWarningLimit: 30000,
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, 'index.html'),
        stylePreview: resolve(import.meta.dirname, 'style-preview.html'),
      },
    },
  },
  test: { environment: 'node' },
})
