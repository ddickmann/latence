// The standalone app: the same component in a minimal shell of its own. `--mode preview`
// writes one script and one stylesheet, which scripts/make-preview.mjs folds into a single
// self-contained page.
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  base: "./",
  build: {
    outDir: mode === "preview" ? "preview/build" : "dist-standalone",
    emptyOutDir: true,
    cssCodeSplit: false,
    assetsInlineLimit: 100_000_000,
    rollupOptions: { output: { inlineDynamicImports: true } },
  },
  define: { __RW_ROUTER__: JSON.stringify(mode === "preview" ? "memory" : "hash") },
  css: { preprocessorOptions: { scss: { quietDeps: true, silenceDeprecations: ["mixed-decls", "global-builtin", "import"] } } },
}));
