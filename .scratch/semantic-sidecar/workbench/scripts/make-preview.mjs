// Folds the preview build (one script, one stylesheet) into a single self-contained page:
// no document tags, everything inline, fonts from Google Fonts. It is what gets shown to someone
// on a phone; the standalone build in dist-standalone/ is the same app as ordinary files.
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const build = "preview/build/assets";
const files = readdirSync(build);
const one = (ext) => {
  const hits = files.filter((f) => f.endsWith(ext));
  if (hits.length !== 1) throw new Error(`expected one ${ext} file in ${build}, found ${hits.length}`);
  return readFileSync(join(build, hits[0]), "utf8");
};
const css = one(".css").replace(/<\/style/gi, "<\\/style");
const js = one(".js").replace(/<\/script/gi, "<\\/script");

// Whatever shows the page may want markup of its own in the head (a viewport hint, say).
const extraHead = process.env.RW_PREVIEW_HEAD ? `${process.env.RW_PREVIEW_HEAD}\n` : "";

const page = `<title>Resolution workbench</title>
${extraHead}<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&family=IBM+Plex+Sans:wght@300;400;600&display=swap">
<style>${css}</style>
<div id="root"></div>
<script type="module">${js}</script>
`;
writeFileSync("preview/resolution-workbench.html", page);
console.log(`preview/resolution-workbench.html: ${(page.length / 1024).toFixed(0)} kB`);
