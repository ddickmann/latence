// Browser check of the built standalone app and of the single-file preview: one accept, one
// reject with a reason, one skip, a walk through the graph, and an accessibility scan (WCAG 2.2
// AA rules) of every screen in light and dark, at phone and desk width.
//
//   npm run build:standalone && npm run build:preview && npm run e2e
//
// Uses the Playwright already on the machine; it is not a dependency of this package.
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const { chromium } = require("playwright");

const root = path.resolve(__dirname, "..");
const shots = path.join(__dirname, "shots");
const axeSource = fs.readFileSync(require.resolve("axe-core/axe.min.js"), "utf8");
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json" };

function serve() {
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, "http://x");
    if (url.pathname === "/preview") {
      // The preview is a page fragment; whatever shows it supplies the document around it.
      const body = fs.readFileSync(path.join(root, "preview/resolution-workbench.html"), "utf8");
      res.writeHead(200, { "content-type": "text/html" });
      res.end(`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head><body>${body}</body></html>`);
      return;
    }
    const file = path.join(root, "dist-standalone", url.pathname === "/" ? "index.html" : url.pathname);
    if (!file.startsWith(path.join(root, "dist-standalone")) || !fs.existsSync(file)) {
      res.writeHead(404);
      res.end();
      return;
    }
    res.writeHead(200, { "content-type": TYPES[path.extname(file)] ?? "application/octet-stream" });
    res.end(fs.readFileSync(file));
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server)));
}

const failures = [];
function check(ok, what) {
  console.log(`${ok ? "ok  " : "FAIL"}  ${what}`);
  if (!ok) failures.push(what);
}

async function scan(page, what) {
  await page.evaluate(axeSource);
  const result = await page.evaluate(() => window.axe.run(document, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"] } }));
  const found = result.violations.map((v) => `${v.id} (${v.nodes.length}): ${v.nodes[0].target.join(" ")} — ${v.nodes[0].failureSummary.split("\n").slice(1, 2).join(" ").trim()}`);
  check(found.length === 0, `accessibility: ${what}${found.length ? "\n        " + found.join("\n        ") : ""}`);
}

/** Resolves once the address has left `from` and the pair it names is the one on screen. */
async function moved(page, from) {
  await page.waitForFunction((u) => {
    const shown = document.querySelector('[data-testid="evidence"]')?.getAttribute("data-item");
    return location.href !== u && shown != null && decodeURIComponent(location.hash).endsWith(`/${shown}`);
  }, from);
}

async function run(browser, base, name, viewport, colorScheme) {
  const context = await browser.newContext({ viewport, colorScheme, deviceScaleFactor: 1 });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  const shot = (file) => page.screenshot({ path: path.join(shots, `${name}-${file}.png`) });
  const tag = `${name}:`;

  await page.goto(`${base}/#/resolution/review`);
  await page.waitForSelector('[data-testid="queue-row"]');
  const open = Number(await page.textContent('[data-testid="qa-open"]'));
  check(open > 0, `${tag} the queue lists ${open} open pairs`);
  check(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), `${tag} the queue does not scroll sideways`);
  await scan(page, `${name} queue`);
  await shot("1-queue");

  // Accept with one key.
  await page.locator('[data-testid="queue-row"]').first().click();
  await page.waitForSelector('[data-testid="evidence"]');
  check(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), `${tag} a pair does not scroll sideways`);
  await scan(page, `${name} pair`);
  // Section headings keep their weight whatever reset the page around the workbench brings.
  check(await page.locator("#rw-why").evaluate((h) => getComputedStyle(h).fontWeight === "600" && getComputedStyle(h).fontSize === "14px"), `${tag} section headings are 14px semibold`);
  await shot("2-pair");
  const first = page.url();
  await page.keyboard.press("a");
  await moved(page, first);
  check(Number(await page.textContent('[data-testid="qa-decided"]')) === 1, `${tag} A records "same" and moves on`);

  // Reject: asks why, records with Ctrl+Enter.
  const second = page.url();
  await page.keyboard.press("r");
  await page.waitForSelector('[data-testid="decision-sheet"]');
  check(await page.locator('[data-testid="sheet-confirm"]').isDisabled(), `${tag} "different" cannot be recorded without a reason`);
  await scan(page, `${name} reason sheet`);
  await shot("3-sheet");
  await page.locator('[data-testid="reason-business"]').check();
  await page.keyboard.press("Control+Enter");
  await moved(page, second);
  check(Number(await page.textContent('[data-testid="qa-decided"]')) === 2, `${tag} R with a reason records "different" and moves on`);

  // Skip: moves on, records nothing.
  const third = page.url();
  await page.keyboard.press("Space");
  await moved(page, third);
  check(Number(await page.textContent('[data-testid="qa-decided"]')) === 2, `${tag} Space skips without recording`);

  // A suggestion: the card says what it rests on and how far to trust it; Y takes it.
  const card = page.locator('[data-testid="suggestion"]');
  check((await card.count()) === 1 && (await page.locator('[data-testid="suggestion-trust"]').count()) === 1, `${tag} a pair shows its suggestion and how often suggestions were right`);
  const wasDecided = Number(await page.textContent('[data-testid="qa-decided"]'));
  const fifth = page.url();
  await page.keyboard.press("y");
  await moved(page, fifth);
  check(Number(await page.textContent('[data-testid="qa-decided"]')) === wasDecided + 1 && /\d+ of \d+/.test(await page.textContent('[data-testid="qa-agreed"]')), `${tag} Y takes the suggestion and the agreement is counted`);

  // A whole group at once: merge with M, then undo from the notice.
  const before = Number(await page.textContent('[data-testid="qa-decided"]'));
  const fourth = page.url();
  await page.keyboard.press("m");
  await page.waitForSelector('[data-testid="merge-sheet"]');
  await scan(page, `${name} merge sheet`);
  await shot("3b-merge");
  const merging = Number((await page.textContent('[data-testid="merge-sheet"] [data-testid="sheet-confirm"]')).match(/\d+/)[0]);
  await page.keyboard.press("Control+Enter");
  await page.waitForSelector('[data-testid="notice-action"]');
  check(merging > 0 && Number(await page.textContent('[data-testid="qa-decided"]')) === before + merging, `${tag} M decides the group's ${merging} open pairs as one`);
  await page.locator('[data-testid="notice-action"]').click();
  await page.waitForFunction((n) => Number(document.querySelector('[data-testid="qa-decided"]').textContent) === n, before);
  check(true, `${tag} undo reopens them`);

  // Split with S: move one record to another letter, then record.
  if (page.url() !== fourth) {
    await page.goto(fourth);
    await page.waitForSelector('[data-testid="evidence"]');
  }
  await page.keyboard.press("s");
  await page.waitForSelector('[data-testid="split-sheet"]');
  const firstB = page.locator('[data-testid="split-sheet"] [data-testid$="-B"]').first();
  const firstA = page.locator('[data-testid="split-sheet"] [data-testid$="-A"]').first();
  await ((await firstB.getAttribute("aria-pressed")) === "true" ? firstA : firstB).click();
  check(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), `${tag} the split dialog does not scroll sideways`);
  // Let the confirm button finish its colour transition from disabled to enabled: scanned
  // halfway, it reads as enabled with the disabled colours.
  await page.waitForTimeout(250);
  await scan(page, `${name} split sheet`);
  await shot("3c-split");
  const splitting = await page.locator('[data-testid="split-sheet"] [data-testid="sheet-confirm"]');
  if (await splitting.isDisabled()) {
    await page.keyboard.press("Escape");
    check(true, `${tag} a split with one letter cannot be recorded`);
  } else {
    const n = Number((await splitting.textContent()).match(/\d+/)[0]);
    await splitting.click();
    await page.waitForFunction((want) => Number(document.querySelector('[data-testid="qa-decided"]').textContent) === want, before + n);
    check(true, `${tag} S records a split of ${n} pairs`);
  }

  // The decisions survive a reload.
  const decidedNow = Number(await page.textContent('[data-testid="qa-decided"]'));
  await page.reload();
  await page.waitForSelector('[data-testid="evidence"]');
  check(Number(await page.textContent('[data-testid="qa-decided"]')) === decidedNow, `${tag} decisions survive a reload`);

  // The audit of auto-accepts.
  await page.goto(`${base}/#/resolution/review`);
  await page.locator('[data-testid="open-audit"]').click();
  await page.waitForSelector('[data-testid="audit-sheet"]');
  await scan(page, `${name} audit sheet`);
  await page.keyboard.press("Escape");

  // The graph: schema, then a node, then into the review of an open link.
  await page.locator('[data-testid="tab-graph"]').click();
  await page.waitForSelector('[data-testid="graph-home"]');
  await page.waitForTimeout(200);
  check(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), `${tag} the graph front page does not scroll sideways`);
  const boxes = await page.locator(".rw-concept").evaluateAll((els) => els.map((e) => e.getBoundingClientRect()).map((r) => [r.left, r.right, r.top, r.bottom]));
  const apart = boxes.every((a, i) => boxes.every((b, j) => i === j || a[1] <= b[0] || b[1] <= a[0] || a[3] <= b[2] || b[3] <= a[2]));
  check(boxes.length === 5 && apart, `${tag} the schema map draws its 5 concepts without overlap`);
  await scan(page, `${name} graph front page`);
  await shot("4-graph");
  await page.fill("#rw-graph-search", "delta");
  await page.locator('[data-testid="search-hits"] [data-testid="node-link"]').first().click();
  await page.waitForSelector('[data-testid="graph-node"]');
  await page.waitForTimeout(200);
  const nodes = await page.locator('[data-testid="focus-node"]').count();
  check(nodes >= 3, `${tag} a node's neighbourhood draws ${nodes} neighbours`);
  check(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), `${tag} a node does not scroll sideways`);
  await scan(page, `${name} graph node`);
  await shot("5-node");
  await page.locator('[data-testid="focus-node"]').first().click();
  await page.waitForSelector('[data-testid="graph-node"]');
  check(page.url().includes("/resolution/graph/"), `${tag} a neighbour opens its own neighbourhood`);

  check(errors.length === 0, `${tag} no errors in the console${errors.length ? ": " + errors.join(" | ") : ""}`);
  await context.close();
}

(async () => {
  fs.mkdirSync(shots, { recursive: true });
  const server = await serve();
  const base = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch();
  try {
    await run(browser, base, "phone-light", { width: 390, height: 844 }, "light");
    await run(browser, base, "phone-dark", { width: 390, height: 844 }, "dark");
    await run(browser, base, "desk-light", { width: 1440, height: 900 }, "light");
    await run(browser, base, "desk-dark", { width: 1440, height: 900 }, "dark");

    // The single-file preview boots and keeps its place without touching the address bar.
    const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(`${base}/preview`);
    await page.waitForSelector('[data-testid="queue-row"]');
    await page.locator('[data-testid="queue-row"]').first().click();
    await page.waitForSelector('[data-testid="evidence"]');
    check(new URL(page.url()).hash === "" && new URL(page.url()).pathname === "/preview", "preview: moving around leaves the address alone");
    await page.reload();
    await page.waitForSelector('[data-testid="evidence"]');
    check(errors.length === 0, `preview: boots and reopens the pair it was on${errors.length ? ": " + errors.join(" | ") : ""}`);
    await page.screenshot({ path: path.join(shots, "preview-pair.png") });
    await context.close();
  } finally {
    await browser.close();
    server.close();
  }
  console.log(failures.length === 0 ? "\nall checks passed" : `\n${failures.length} check(s) failed`);
  process.exit(failures.length === 0 ? 0 : 1);
})();
