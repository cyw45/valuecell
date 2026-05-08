import { reactRouter } from "@react-router/dev/vite";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";
import createSvgSpritePlugin from "vite-plugin-svg-sprite";
import tsconfigPaths from "vite-tsconfig-paths";

const host = process.env.TAURI_DEV_HOST;
const devPort = Number(process.env.VITE_DEV_PORT || "1430");
const hmrPort = Number(process.env.VITE_HMR_PORT || String(devPort + 1));
const backendTarget = process.env.VITE_BACKEND_TARGET || "http://127.0.0.1:8002";

// https://vite.dev/config/
export default defineConfig(async () => ({
  plugins: [
    tailwindcss(),
    reactRouter(),
    tsconfigPaths(),
    createSvgSpritePlugin({
      exportType: "vanilla",
      include: "**/assets/svg/**/*.svg",
      svgo: {
        plugins: [
          {
            name: "preset-default",
            params: {
              overrides: {
                // Keep viewBox attribute, important for icon scaling
                removeViewBox: false,
                // Keep accessibility attributes
                removeUnknownsAndDefaults: {
                  keepDataAttrs: false,
                  keepAriaAttrs: true,
                },
                // Clean up IDs while maintaining uniqueness
                cleanupIds: {
                  minify: true,
                  preserve: [],
                },
                // Preserve currentColor and don't remove useful attributes
                removeUselessStrokeAndFill: false,
              },
            },
          },
          // Only remove data attributes and classes, preserve fill/stroke for currentColor
          {
            name: "removeAttrs",
            params: {
              attrs: "(data-.*|class)",
              elemSeparator: ",",
            },
          },
          // Remove unnecessary metadata and comments
          "removeMetadata",
          "removeComments",
          // Remove empty elements
          "removeEmptyText",
          "removeEmptyContainers",
          // Optimize paths and merge when possible
          "convertPathData",
          "mergePaths",
          // Convert colors but preserve currentColor
          {
            name: "convertColors",
            params: {
              currentColor: true,
            },
          },
        ],
      },
    }),
  ],

  // Vite options tailored for Tauri development and only applied in `tauri dev` or `tauri build`
  //
  // 1. prevent Vite from obscuring rust errors
  clearScreen: false,
  // 2. tauri expects a fixed port, fail if that port is not available
  server: {
    port: devPort,
    strictPort: true,
    host: host || false,
    proxy: {
      "/api/v1": {
        target: backendTarget,
        changeOrigin: true,
      },
    },
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: hmrPort,
        }
      : undefined,
    watch: {
      // 3. tell Vite to ignore watching `src-tauri`
      ignored: ["**/src-tauri/**"],
    },
  },
  resolve:
    process.env.NODE_ENV === "development"
      ? {}
      : {
          alias: {
            "react-dom/server": "react-dom/server.node",
          },
        }
}));
