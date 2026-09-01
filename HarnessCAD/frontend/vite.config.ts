import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";


const backend = "http://127.0.0.1:8000";

export default defineConfig({
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: backend, changeOrigin: true },
      "/harness-runs": { target: backend, changeOrigin: true },
      "/harness-runs-v2": { target: backend, changeOrigin: true }
    }
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        index: fileURLToPath(new URL("./index.html", import.meta.url)),
        harness: fileURLToPath(new URL("./harness.html", import.meta.url)),
        harnessV2: fileURLToPath(new URL("./harness-v2.html", import.meta.url))
      }
    }
  }
});
