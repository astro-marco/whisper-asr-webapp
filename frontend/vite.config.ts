import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// https://vitejs.dev/config/
export default defineConfig({
  base: "/static/",
  plugins: [svelte()],
  build: {
    outDir: "dist", // Changed from ../static
    emptyOutDir: true
  }
});
