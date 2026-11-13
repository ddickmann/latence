// The library build (what a host imports) and the unit tests. Everything the host already
// has stays external, so a mounted workbench runs on the host's one React and one router.
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    cssCodeSplit: false,
    lib: {
      entry: "src/index.ts",
      formats: ["es"],
      fileName: () => "resolution-workbench.js",
      cssFileName: "resolution-workbench",
    },
    rollupOptions: {
      external: ["react", "react-dom", "react/jsx-runtime", "react-router", "@carbon/react", /^@carbon\/icons-react/],
    },
  },
  css: { preprocessorOptions: { scss: { quietDeps: true, silenceDeprecations: ["mixed-decls", "global-builtin", "import"] } } },
  test: {
    environment: "jsdom",
    setupFiles: ["src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    css: false,
  },
});
