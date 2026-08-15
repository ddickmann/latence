# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""The single-page labeling UI — one static HTML document, no external assets (ticket 21).

Served verbatim by :mod:`latence_benchmark.labeling.server` at ``/``. The page asks
``/api/state`` which mode the server runs in and renders either the gold-set flow
(document browser + span marking by text selection) or the calibration flow (one
Hyperedge card at a time, 1/2/3 judgment keys). Everything is inline — a strict
no-external-assets page that works on ``127.0.0.1`` with no network at all.

Span offsets: each chunk's text is rendered as a SINGLE text node (``textContent``) with
``white-space: pre-wrap``, so a browser selection inside one chunk maps 1:1 onto character
offsets into the corpus text — the client slices and compares before submitting, and the
server re-verifies at write time (:func:`latence_benchmark.labeling.goldset.verify_fact`).
"""

from __future__ import annotations

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>latence-label</title>
<style>
  :root {
    --bg: #10141a; --panel: #171d26; --panel2: #1e2632; --text: #dce3ec; --dim: #8b98a9;
    --accent: #5aa0f2; --good: #4fbf8f; --bad: #e2695e; --warn: #e0a84f; --mark: #35507a;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.5 -apple-system, "Segoe UI", sans-serif; }
  header { display: flex; gap: 16px; align-items: baseline; padding: 10px 16px;
           background: var(--panel); border-bottom: 1px solid #000; position: sticky;
           top: 0; z-index: 5; flex-wrap: wrap; }
  header h1 { font-size: 15px; margin: 0; }
  header .stat { color: var(--dim); }
  header .warn { color: var(--warn); }
  main { display: flex; height: calc(100vh - 46px); }
  #sidebar { width: 290px; min-width: 220px; overflow-y: auto; background: var(--panel);
             border-right: 1px solid #000; padding: 8px; }
  #sidebar input { width: 100%; padding: 6px 8px; margin-bottom: 8px; border-radius: 6px;
                   border: 1px solid #2b3646; background: var(--panel2); color: var(--text); }
  .doc { padding: 6px 8px; border-radius: 6px; cursor: pointer; margin-bottom: 2px;
         overflow-wrap: anywhere; }
  .doc:hover { background: var(--panel2); }
  .doc.active { background: var(--mark); }
  .doc .meta { color: var(--dim); font-size: 12px; }
  #content { flex: 1; overflow-y: auto; padding: 14px 18px 60vh; }
  .chunk { margin-bottom: 14px; background: var(--panel); border-radius: 8px;
           border: 1px solid #232d3b; }
  .chunk .cid { font-size: 11px; color: var(--dim); padding: 5px 10px 0;
                font-family: ui-monospace, monospace; overflow-wrap: anywhere; }
  .chunk .text { padding: 6px 10px 10px; white-space: pre-wrap; overflow-wrap: anywhere;
                 font-family: ui-monospace, monospace; font-size: 13px; }
  .chunk .text::selection, .chunk .text ::selection { background: var(--accent); color: #08101c; }
  #form { width: 380px; min-width: 300px; overflow-y: auto; background: var(--panel);
          border-left: 1px solid #000; padding: 12px; }
  #form h2 { font-size: 13px; margin: 0 0 8px; color: var(--dim); text-transform: uppercase;
             letter-spacing: 0.06em; }
  textarea, select, input[type=text] { width: 100%; border-radius: 6px; padding: 7px 9px;
      border: 1px solid #2b3646; background: var(--panel2); color: var(--text); font: inherit; }
  textarea { min-height: 70px; resize: vertical; }
  label { display: block; margin: 10px 0 4px; color: var(--dim); font-size: 12px; }
  .span { background: var(--panel2); border: 1px solid #2b3646; border-radius: 6px;
          padding: 6px 8px; margin: 6px 0; font-size: 12px; }
  .span .t { font-family: ui-monospace, monospace; overflow-wrap: anywhere; }
  .span .m { color: var(--dim); display: flex; justify-content: space-between; gap: 8px; }
  .span button { background: none; border: none; color: var(--bad); cursor: pointer;
                 font-size: 13px; padding: 0; }
  button.primary { width: 100%; margin-top: 12px; padding: 9px; border-radius: 6px;
      border: none; background: var(--accent); color: #08101c; font-weight: 600;
      cursor: pointer; }
  button.primary:disabled { opacity: 0.45; cursor: default; }
  #msg { margin-top: 10px; font-size: 13px; min-height: 20px; }
  #msg.ok { color: var(--good); } #msg.err { color: var(--bad); }
  .hint { color: var(--dim); font-size: 12px; margin-top: 10px; }
  kbd { background: var(--panel2); border: 1px solid #2b3646; border-radius: 4px;
        padding: 0 5px; font-family: ui-monospace, monospace; font-size: 11px; }
  /* calibration */
  #cal { max-width: 860px; margin: 0 auto; padding: 20px; }
  #cal .tau { background: var(--panel); border: 1px solid #232d3b; border-radius: 8px;
              padding: 14px 16px; font-size: 16px; white-space: pre-wrap; }
  #cal h3 { color: var(--dim); font-size: 12px; text-transform: uppercase;
            letter-spacing: 0.06em; margin: 18px 0 6px; }
  #cal pre { background: var(--panel); border: 1px solid #232d3b; border-radius: 8px;
             padding: 10px 12px; overflow-x: auto; font-size: 12px; margin: 0; }
  .ent { display: inline-block; background: var(--panel2); border: 1px solid #2b3646;
         border-radius: 12px; padding: 2px 10px; margin: 2px 4px 2px 0; font-size: 12px;
         font-family: ui-monospace, monospace; }
  #judge { display: flex; gap: 10px; margin-top: 18px; }
  #judge button { flex: 1; padding: 12px; border-radius: 8px; border: 1px solid #2b3646;
                  background: var(--panel2); color: var(--text); cursor: pointer;
                  font: inherit; }
  #judge button.j-correct.sel { background: var(--good); color: #08101c; }
  #judge button.j-incorrect.sel { background: var(--bad); color: #08101c; }
  #judge button.j-partial.sel { background: var(--warn); color: #08101c; }
  #nav { display: flex; gap: 10px; margin-top: 14px; align-items: center;
         color: var(--dim); }
  #nav button { background: var(--panel2); border: 1px solid #2b3646; color: var(--text);
                border-radius: 6px; padding: 6px 14px; cursor: pointer; }
</style>
</head>
<body>
<header>
  <h1 id="title">latence-label</h1>
  <span class="stat" id="progress"></span>
  <span class="stat" id="dist"></span>
  <span class="warn" id="warning"></span>
</header>
<main id="main"></main>
<script>
"use strict";
const $ = (id) => document.getElementById(id);
const api = async (path, opts) => {
  const res = await fetch(path, opts);
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || res.statusText);
  return body;
};
const esc = (s) => {
  const d = document.createElement("div"); d.textContent = s; return d.innerHTML;
};

function renderCoverage(cov) {
  if (!cov) return;
  $("progress").textContent = cov.total + " question" + (cov.total === 1 ? "" : "s") + " labeled";
  const dist = Object.entries(cov.hop_distribution)
    .map(([h, n]) => h + "-hop: " + n).join("  ·  ");
  $("dist").textContent = dist;
  $("warning").textContent = cov.warning || "";
}

/* ---------------- gold-set mode ---------------- */
const gold = { docs: [], activeDoc: null, chunks: {}, spans: [] };

async function goldInit() {
  $("title").textContent = "latence-label — gold set";
  $("main").innerHTML =
    '<div id="sidebar"><input id="filter" placeholder="filter documents (/)">' +
    '<div id="docs"></div></div>' +
    '<div id="content"><p class="hint">Pick a document, select text with the mouse to ' +
    'mark a supporting fact, then fill the form on the right.</p></div>' +
    '<div id="form"><h2>New question</h2>' +
    '<label for="question">Question</label><textarea id="question"></textarea>' +
    '<label for="hops">Hop count</label><select id="hops">' +
    [1,2,3,4,5].map(h =>
      '<option value="'+h+'"'+(h===2?" selected":"")+'>'+h+'</option>').join("") +
    '</select>' +
    '<label>Supporting facts (select text in a chunk)</label><div id="spans"></div>' +
    '<button class="primary" id="save">Save question (Ctrl+Enter)</button>' +
    '<div id="msg"></div>' +
    '<p class="hint"><kbd>/</kbd> filter docs · select text → span is added · ' +
    '<kbd>Ctrl</kbd>+<kbd>Enter</kbd> save</p></div>';
  const docs = await api("/api/documents");
  gold.docs = docs.documents;
  renderDocs("");
  $("filter").addEventListener("input", (e) => renderDocs(e.target.value));
  $("save").addEventListener("click", saveQuestion);
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement.tagName !== "TEXTAREA"
        && document.activeElement.tagName !== "INPUT") { e.preventDefault(); $("filter").focus(); }
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); saveQuestion(); }
  });
  document.addEventListener("mouseup", captureSelection);
  const state = await api("/api/state");
  renderCoverage(state.coverage);
}

function renderDocs(filter) {
  const box = $("docs");
  box.innerHTML = "";
  const needle = filter.toLowerCase();
  for (const doc of gold.docs) {
    const label = doc.file_name || doc.document_id;
    if (needle && !label.toLowerCase().includes(needle)
        && !doc.document_id.toLowerCase().includes(needle)) continue;
    const div = document.createElement("div");
    div.className = "doc" + (doc.document_id === gold.activeDoc ? " active" : "");
    div.innerHTML = "<div>" + esc(label) + "</div><div class='meta'>" +
      doc.chunk_count + " chunks · " + doc.labeled_fact_count + " facts marked</div>";
    div.addEventListener("click", () => openDoc(doc.document_id));
    box.appendChild(div);
  }
}

async function openDoc(documentId) {
  const data = await api("/api/document?document_id=" + encodeURIComponent(documentId));
  gold.activeDoc = documentId;
  gold.chunks = {};
  const content = $("content");
  content.innerHTML = "";
  for (const chunk of data.chunks) {
    gold.chunks[chunk.chunk_record_id] = chunk;
    const div = document.createElement("div");
    div.className = "chunk";
    const cid = document.createElement("div");
    cid.className = "cid";
    cid.textContent = chunk.chunk_record_id;
    const text = document.createElement("div");
    text.className = "text";
    text.dataset.chunkId = chunk.chunk_record_id;
    text.textContent = chunk.text;   /* ONE text node — offsets map 1:1 */
    div.appendChild(cid); div.appendChild(text);
    content.appendChild(div);
  }
  renderDocs($("filter").value);
  content.scrollTop = 0;
}

function captureSelection() {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed || sel.rangeCount !== 1) return;
  const range = sel.getRangeAt(0);
  if (range.startContainer !== range.endContainer) return;
  const node = range.startContainer;
  const holder = node.parentElement && node.parentElement.closest(".chunk .text");
  if (!holder) return;
  const chunkId = holder.dataset.chunkId;
  const chunk = gold.chunks[chunkId];
  if (!chunk) return;
  const start = range.startOffset, end = range.endOffset;
  if (end <= start) return;
  const text = chunk.text.slice(start, end);
  if (text !== sel.toString()) { message("selection did not round-trip; try again", true); return; }
  gold.spans.push({ document_id: chunk.document_id, chunk_record_id: chunkId,
                    char_start: start, char_end: end, text });
  sel.removeAllRanges();
  renderSpans();
}

function renderSpans() {
  const box = $("spans");
  box.innerHTML = "";
  gold.spans.forEach((span, i) => {
    const div = document.createElement("div");
    div.className = "span";
    div.innerHTML = "<div class='t'>" + esc(span.text) + "</div>" +
      "<div class='m'><span>" + esc(span.chunk_record_id) + " [" + span.char_start + ":" +
      span.char_end + "]</span><button title='remove'>&times;</button></div>";
    div.querySelector("button").addEventListener("click", () => {
      gold.spans.splice(i, 1); renderSpans();
    });
    box.appendChild(div);
  });
  if (!gold.spans.length) box.innerHTML = "<p class='hint'>no spans marked yet</p>";
}

async function saveQuestion() {
  const question = $("question").value.trim();
  const hops = parseInt($("hops").value, 10);
  try {
    const body = JSON.stringify({ question, hop_count: hops, supporting_facts: gold.spans });
    const saved = await api("/api/questions", { method: "POST",
      headers: { "Content-Type": "application/json" }, body });
    gold.spans = [];
    $("question").value = "";
    renderSpans();
    renderCoverage(saved.coverage);
    message("saved " + saved.question.question_id, false);
    const docs = await api("/api/documents");
    gold.docs = docs.documents;
    renderDocs($("filter").value);
  } catch (err) { message(err.message, true); }
}

function message(text, isErr) {
  const box = $("msg");
  box.textContent = text;
  box.className = isErr ? "err" : "ok";
}

/* ---------------- calibration mode ---------------- */
const cal = { rows: [], judgments: {}, index: 0 };

async function calInit() {
  $("title").textContent = "latence-label — γ calibration";
  $("main").innerHTML = '<div id="content"><div id="cal">' +
    '<div id="pos" class="hint"></div><div class="tau" id="tau"></div>' +
    '<h3>Entities</h3><div id="ents"></div>' +
    '<h3>Relations</h3><pre id="rels"></pre>' +
    '<div id="judge">' +
    '<button class="j-correct" data-j="correct">1 · correct</button>' +
    '<button class="j-incorrect" data-j="incorrect">2 · incorrect</button>' +
    '<button class="j-partial" data-j="partial">3 · partial</button></div>' +
    '<label for="note">Note (optional, <kbd>n</kbd>)</label><input type="text" id="note">' +
    '<div id="nav"><button id="prev">&larr; prev</button>' +
    '<button id="next">next &rarr;</button><span id="msg"></span></div>' +
    '<p class="hint"><kbd>1</kbd>/<kbd>2</kbd>/<kbd>3</kbd> judge + advance · ' +
    '<kbd>&larr;</kbd>/<kbd>&rarr;</kbd> navigate</p></div></div>';
  const data = await api("/api/hyperedges");
  cal.rows = data.hyperedges;
  cal.judgments = data.judgments;
  cal.index = cal.rows.findIndex(r => !(r.hyperedge_id in cal.judgments));
  if (cal.index < 0) cal.index = 0;
  document.querySelectorAll("#judge button").forEach(btn =>
    btn.addEventListener("click", () => judge(btn.dataset.j)));
  $("prev").addEventListener("click", () => move(-1));
  $("next").addEventListener("click", () => move(1));
  document.addEventListener("keydown", (e) => {
    if (document.activeElement === $("note")) {
      if (e.key === "Enter") { e.preventDefault(); $("note").blur(); }
      return;
    }
    if (e.key === "1") judge("correct");
    else if (e.key === "2") judge("incorrect");
    else if (e.key === "3") judge("partial");
    else if (e.key === "ArrowLeft") move(-1);
    else if (e.key === "ArrowRight") move(1);
    else if (e.key === "n") { e.preventDefault(); $("note").focus(); }
  });
  renderCal();
}

function calProgress() {
  const judged = Object.keys(cal.judgments).length;
  $("progress").textContent = judged + " / " + cal.rows.length + " judged";
}

function renderCal() {
  calProgress();
  const row = cal.rows[cal.index];
  if (!row) { $("tau").textContent = "no hyperedges loaded"; return; }
  const extras = [];
  if (row.gamma !== null) extras.push("γ=" + row.gamma.toFixed(3));
  if (row.support !== null) extras.push("support=" + row.support);
  if (row.arity !== null) extras.push("arity=" + row.arity);
  $("pos").textContent = (cal.index + 1) + " / " + cal.rows.length + " — " +
    row.hyperedge_id + (extras.length ? "  ·  " + extras.join("  ") : "");
  $("tau").textContent = row.tau_text;
  $("ents").innerHTML = row.entity_node_ids.map(e => "<span class='ent'>" + esc(e) + "</span>")
    .join("") || "<span class='hint'>none</span>";
  let rels = row.relations_json;
  try { rels = JSON.stringify(JSON.parse(rels), null, 2); } catch (err) { /* show raw */ }
  $("rels").textContent = rels;
  const current = cal.judgments[row.hyperedge_id];
  document.querySelectorAll("#judge button").forEach(btn =>
    btn.classList.toggle("sel", !!current && current.judgment === btn.dataset.j));
  $("note").value = current && current.note ? current.note : "";
}

async function judge(value) {
  const row = cal.rows[cal.index];
  if (!row) return;
  try {
    const body = JSON.stringify({ hyperedge_id: row.hyperedge_id, judgment: value,
                                  note: $("note").value.trim() });
    const saved = await api("/api/judgments", { method: "POST",
      headers: { "Content-Type": "application/json" }, body });
    cal.judgments[row.hyperedge_id] = saved.judgment;
    message("saved", false);
    move(1);
  } catch (err) { message(err.message, true); }
}

function move(delta) {
  cal.index = Math.min(Math.max(cal.index + delta, 0), Math.max(cal.rows.length - 1, 0));
  renderCal();
}

/* ---------------- boot ---------------- */
api("/api/state").then((state) => {
  if (state.mode === "gold") return goldInit();
  return calInit();
}).catch((err) => { $("main").textContent = "failed to load state: " + err.message; });
</script>
</body>
</html>
"""
