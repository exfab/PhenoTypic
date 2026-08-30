/**
 * Build the vendored Viv + deck.gl IIFE.
 *
 * Run by hand -- there is no npm in CI (viewer-viv-rebuild spec section 3):
 *
 *     npm ci && node build.mjs
 *
 * Writes ../../src/phenotypic/gui/results_viewer/_assets/viv/viv-bundle.min.js
 * and stamps VERSION's contents into the artifact so a stale vendored file can
 * be detected by comparing the two.
 */
import { build } from "esbuild";
import { readFileSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const version = readFileSync(resolve(here, "VERSION"), "utf8").trim();
const outfile = resolve(
  here,
  "../../src/phenotypic/gui/results_viewer/_assets/viv/viv-bundle.min.js",
);

await build({
  entryPoints: [resolve(here, "entry.mjs")],
  outfile,
  bundle: true,
  minify: true,
  format: "iife",
  platform: "browser",
  target: ["es2022"],
  legalComments: "none",
  define: {
    __PHENOTYPIC_VIV_BUNDLE_VERSION__: JSON.stringify(version),
    "process.env.NODE_ENV": '"production"',
  },
  banner: {
    js:
      `/* PhenoTypic vendored Viv bundle -- ${version}\n` +
      "   Built from tools/viv-bundle (npm ci && node build.mjs). Do not edit.\n" +
      "   Viv and vizarr are MIT; see NOTICE and licenses/. */",
  },
  logLevel: "info",
});

const bytes = statSync(outfile).size;
console.log(`\nversion : ${version}`);
console.log(`outfile : ${outfile}`);
console.log(`size    : ${bytes} B (${(bytes / 1024 / 1024).toFixed(2)} MiB)`);
