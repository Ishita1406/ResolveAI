/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// The backend's routes the frontend uses. In development Vite forwards them, so the browser
// sees one origin (no CORS); in Docker nginx does the same (frontend/nginx.conf).
const BACKEND_ROUTES = ["/voice/offer", "/admin/conversations", "/admin/evals", "/health"];

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const backend = env.BACKEND_URL || "http://127.0.0.1:8000";
  return {
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      port: 5173,
      proxy: Object.fromEntries(BACKEND_ROUTES.map((route) => [route, backend])),
    },
    test: {
      environment: "jsdom",
    },
  };
});
