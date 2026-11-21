// Enzino Video2Prompt — node card (enzinoai.com look)
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "EnzinoVideo2Prompt";
const CARD_MIN = 250; // empty card; it grows with the prompt so there is never a scrollbar
const PREVIEW_H = 230;

const CSS = `
.ev2p-card{box-sizing:border-box;width:100%;min-width:0;height:auto;min-height:${CARD_MIN}px;flex:none;flex-shrink:0;display:flex;flex-direction:column;gap:10px;
  padding:14px;background:#0d0d0e;border:1px solid #232326;border-radius:14px;color:#e9e9ea;
  font-family:Manrope,"Segoe UI",system-ui,sans-serif;font-size:12px;overflow:hidden}
.ev2p-card *{box-sizing:border-box}
.ev2p-head{display:flex;align-items:center;gap:10px;min-width:0}
.ev2p-logo{font-weight:800;font-size:15px;letter-spacing:-.3px;color:#fff;white-space:nowrap}
.ev2p-logo sup{color:#e8642d;font-size:8px;font-weight:700;margin-left:1px}
.ev2p-title{font-weight:800;font-size:11px;letter-spacing:2px;color:#d6d6d8;white-space:nowrap}
.ev2p-title b{color:#e8642d}
.ev2p-tag{font-size:9px;font-weight:800;letter-spacing:1px;padding:2px 7px;border-radius:5px;background:#2a1a12;color:#ff8a52;white-space:nowrap}
.ev2p-status{margin-left:auto;display:flex;align-items:center;gap:6px;color:#9a9aa0;font-size:11px;min-width:0;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.ev2p-dot{width:7px;height:7px;border-radius:50%;background:#34d399;flex:none}
.ev2p-busy .ev2p-dot{background:#e8642d;animation:ev2p-pulse 1s infinite}
.ev2p-err .ev2p-dot{background:#ef4444}
@keyframes ev2p-pulse{50%{opacity:.25}}
.ev2p-strip{display:flex;gap:4px;height:54px;flex:none;overflow-x:auto;overflow-y:hidden;scrollbar-width:thin}
.ev2p-strip img{height:100%;border-radius:6px;flex:none;border:1px solid #232326}
.ev2p-strip.ev2p-empty{display:none}
.ev2p-parts{display:flex;flex-direction:column;gap:10px}
.ev2p-box{flex:1 0 auto;display:flex;flex-direction:column;background:#151517;border:1px solid #26262a;border-radius:10px}
.ev2p-label{display:flex;justify-content:space-between;padding:8px 12px 0;font-size:9px;font-weight:800;letter-spacing:1.5px;color:#77777d}
.ev2p-right{display:flex;align-items:center;gap:8px}
.ev2p-mini{border:1px solid #3a2416;background:#22150e;color:#ff9a66;border-radius:6px;padding:2px 9px;font:800 9.5px Manrope,system-ui,sans-serif;letter-spacing:1px;cursor:pointer}
.ev2p-mini:hover{background:#2e1b10}
.ev2p-mini:disabled{opacity:.4;cursor:default}
.ev2p-text{flex:none;min-height:72px;width:100%;resize:none;overflow:hidden;display:block;border:0;outline:0;background:transparent;color:#ededee;
  font-family:inherit;font-size:12.5px;line-height:1.55;padding:6px 12px 10px}
.ev2p-text::placeholder{color:#5c5c62}
.ev2p-foot{display:flex;align-items:center;gap:8px}
.ev2p-meta{color:#77777d;font-size:10.5px;line-height:1.45;min-width:0;overflow:hidden;flex:1}
.ev2p-btn{border:0;border-radius:9px;padding:8px 16px;font-family:inherit;font-weight:800;font-size:12px;cursor:pointer;
  background:#e8642d;color:#fff;white-space:nowrap}
.ev2p-btn:hover{background:#ff7a3d}
.ev2p-btn.ev2p-ghost{background:#1d1d20;color:#d6d6d8;border:1px solid #2c2c30}
.ev2p-btn.ev2p-ghost:hover{background:#26262a}
.ev2p-btn:disabled{opacity:.4;cursor:default}
.ev2p-preview{background:#000;border-radius:10px;overflow:hidden}
.ev2p-preview video{width:100%!important;height:100%!important;max-height:${PREVIEW_H}px;object-fit:contain;display:block}
`;

function injectStyle() {
  if (document.getElementById("ev2p-style")) return;
  const font = document.createElement("link");
  font.rel = "stylesheet";
  font.href = "https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap";
  document.head.appendChild(font);
  const st = document.createElement("style");
  st.id = "ev2p-style";
  st.textContent = CSS;
  document.head.appendChild(st);
}

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    return ok;
  }
}

function autoHeight(ta, width) {
  ta.style.height = "0px";
  let h = ta.scrollHeight;
  if (!h) {
    // not laid out yet (off screen): estimate from the text length
    const perLine = Math.max(20, Math.floor((width - 54) / 6.6));
    const lines = ta.value.split("\n").reduce((n, l) => n + Math.max(1, Math.ceil(l.length / perLine)), 0);
    h = lines * 19.5 + 16;
  }
  h = Math.max(72, Math.ceil(h));
  ta.style.height = `${h}px`;
  return h;
}

function wordsOf(text) {
  // H3 format: count the main description only
  const m = /integrated_multimodal_description:\s*([\s\S]*?)(?:\n\noverall_soundscape|$)/.exec(text);
  const body = (m ? m[1] : text).trim();
  return body ? body.split(/\s+/).length : 0;
}

function buildCard(node) {
  const card = el("div", "ev2p-card");
  const head = el("div", "ev2p-head");
  head.append(
    el("span", "ev2p-logo", "enzino<sup>ai</sup>"),
    el("span", "ev2p-title", "VIDEO<b>2</b>PROMPT"),
    el("span", "ev2p-tag", "FOR H3")
  );
  const status = el("div", "ev2p-status");
  const statusText = el("span", "", "Ready");
  status.append(el("span", "ev2p-dot"), statusText);
  head.append(status);

  const strip = el("div", "ev2p-strip ev2p-empty");
  const list = el("div", "ev2p-parts");

  const foot = el("div", "ev2p-foot");
  const meta = el("div", "ev2p-meta", "");
  const clearBtn = el("button", "ev2p-btn ev2p-ghost", "Clear");
  const copyBtn = el("button", "ev2p-btn", "Copy prompt");
  foot.append(meta, clearBtn, copyBtn);
  card.append(head, strip, list, foot);

  let blocks = []; // [{label, ta, words, copy}]
  let height = CARD_MIN;

  const save = () => {
    node.properties.v2p_parts = blocks.map((b) => ({ label: b.labelText, text: b.ta.value }));
  };
  const refreshCounts = () => {
    let any = false;
    for (const b of blocks) {
      const n = wordsOf(b.ta.value);
      b.words.textContent = n ? `${n} words` : "";
      if (b.copy) b.copy.disabled = !n;
      any = any || n > 0;
    }
    copyBtn.disabled = !any;
    copyBtn.textContent = blocks.length > 1 ? "Copy all" : "Copy prompt";
  };

  const fit = () => {
    const width = card.clientWidth || Math.max(200, (node.size?.[0] || 440) - 20);
    let sum = 0;
    for (const b of blocks) sum += autoHeight(b.ta, width) + 40;
    const measured = card.offsetHeight;
    const stripH = strip.classList.contains("ev2p-empty") ? 0 : 64;
    height = Math.max(CARD_MIN, measured || 28 + 2 + 22 + 10 + stripH + sum + 10 * blocks.length + 34);
    node.setSize?.([node.size[0], node.computeSize()[1]]);
    node.setDirtyCanvas?.(true, true);
  };

  const flash = (btn, ok, back) => {
    btn.textContent = ok ? "Copied ✓" : "Copy failed";
    setTimeout(() => (btn.textContent = back), 1400);
  };

  const makeBlock = (labelText, text, multi) => {
    const box = el("div", "ev2p-box");
    const label = el("div", "ev2p-label");
    const right = el("span", "ev2p-right");
    const words = el("span", "", "");
    right.append(words);
    let copy = null;
    if (multi) {
      copy = el("button", "ev2p-mini", "Copy");
      copy.onclick = async () => flash(copy, await copyText(ta.value), "Copy");
      right.append(copy);
    }
    label.append(el("span", "", labelText), right);
    const ta = el("textarea", "ev2p-text");
    ta.spellcheck = false;
    ta.value = text || "";
    ta.placeholder = "Pick or upload a video above, then Run.\nThe prompt for MiniMax H3 appears here.";
    // Typing and selecting stay in the card. The wheel is left to the canvas: the card shows the
    // whole prompt (no scrollbar), so zooming over it behaves like anywhere else.
    for (const ev of ["keydown", "pointerdown"]) ta.addEventListener(ev, (e) => e.stopPropagation());
    ta.addEventListener("input", () => {
      save();
      refreshCounts();
      fit();
    });
    box.append(label, ta);
    list.append(box);
    blocks.push({ labelText, ta, words, copy });
  };

  const setParts = (parts) => {
    list.innerHTML = "";
    blocks = [];
    const items = parts && parts.length ? parts : [{ label: "H3 PROMPT", text: "" }];
    items.forEach((p) => makeBlock(p.label || "H3 PROMPT", p.text, items.length > 1));
    refreshCounts();
  };

  copyBtn.onclick = async () => {
    const all = blocks.length > 1
      ? blocks.map((b) => `===== ${b.labelText} =====\n${b.ta.value}`).join("\n\n")
      : blocks[0]?.ta.value || "";
    flash(copyBtn, await copyText(all), blocks.length > 1 ? "Copy all" : "Copy prompt");
  };
  clearBtn.onclick = () => {
    meta.textContent = "";
    strip.innerHTML = "";
    strip.classList.add("ev2p-empty");
    setParts([]);
    Object.assign(node.properties, { v2p_parts: [], v2p_meta: "", v2p_frames: [], v2p_prompt: "" });
    fit();
  };

  const setStatus = (mode, msg) => {
    card.classList.toggle("ev2p-busy", mode === "busy");
    card.classList.toggle("ev2p-err", mode === "err");
    statusText.textContent = msg;
  };

  const setFrames = (frames) => {
    strip.innerHTML = "";
    (frames || []).forEach((f) => {
      const img = el("img");
      const q = new URLSearchParams({ filename: f.filename, subfolder: f.subfolder || "", type: f.type || "temp" });
      img.src = api.apiURL(`/view?${q}`);
      img.draggable = false;
      img.onerror = () => {
        // temp files are gone after a restart: drop the image, hide the strip when nothing is left
        img.remove();
        if (!strip.querySelector("img")) {
          strip.classList.add("ev2p-empty");
          fit();
        }
      };
      strip.append(img);
    });
    strip.classList.toggle("ev2p-empty", !(frames && frames.length));
  };

  // parts: [{label, text}] (older saves: a single prompt string)
  const show = (parts, metaTxt, frames) => {
    if (typeof parts === "string") parts = parts ? [{ label: "H3 PROMPT", text: parts }] : [];
    setParts(parts);
    meta.textContent = metaTxt || "";
    setFrames(frames);
    requestAnimationFrame(fit);
  };

  // Re-fit whenever the card's width changes (node resized, Nodes 2.0 layout, fonts loaded...).
  let lastW = 0, timer = null;
  try {
    new ResizeObserver(() => {
      const w = card.clientWidth;
      if (w && Math.abs(w - lastW) > 1) {
        lastW = w;
        clearTimeout(timer);
        timer = setTimeout(fit, 30);
      }
    }).observe(card);
  } catch {}
  document.fonts?.ready?.then(() => fit());

  setParts([]);
  return { card, show, setStatus, fit, getHeight: () => height };
}


// The core video preview grows with the video (a 9:16 clip made the node huge and pushed
// the prompt far down). Cap it to a fixed height and keep it under the card.
function capPreview(node, widget, el) {
  el.classList.add("ev2p-preview");
  const orig = widget.computeLayoutSize?.bind(widget);
  widget.computeLayoutSize = () => {
    const base = orig ? orig() : {};
    return { ...base, minHeight: PREVIEW_H, maxHeight: PREVIEW_H, minWidth: 0 };
  };
  widget.computeSize = (width) => [width, PREVIEW_H];
  setTimeout(() => {
    node.setSize([node.size[0], node.computeSize()[1]]);
    node.setDirtyCanvas?.(true, true);
  }, 50);
}

// "✓ Qwen3-VL-4B-Instruct", "↓ Qwen3-VL-4B · 8.9 GB", "Qwen3-VL-4B (recommended)" -> "qwen3-vl-4b"
const modelKey = (v) =>
  String(v || "").replace(/^[\u2713\u2193]\s*/, "").split(" \u00b7 ")[0]
    .replace(/\s*\(.*\)\s*$/, "").replace(/-Instruct$/i, "").trim().toLowerCase();

// Workflows saved with another model label (older version, other PC): pick the matching entry.
function fixModelValue(node) {
  const w = node.widgets?.find((w) => w.name === "model");
  const opts = w?.options?.values;
  if (!w || !Array.isArray(opts) || opts.includes(w.value)) return;
  const k = modelKey(w.value);
  const match = opts.find((o) => o.startsWith("\u2713") && modelKey(o) === k) || opts.find((o) => modelKey(o) === k);
  if (match) w.value = match;
}

// long_clip / window_* only matter for clips over 15 s (H3's maximum): hide them otherwise.
const LONG_WIDGETS = ["long_clip", "window_start", "window_end"];
const durations = new Map(); // video file -> seconds

function videoDuration(name) {
  if (durations.has(name)) return Promise.resolve(durations.get(name));
  return new Promise((resolve) => {
    const parts = String(name).split("/");
    const filename = parts.pop();
    const q = new URLSearchParams({ filename, subfolder: parts.join("/"), type: "input" });
    const v = document.createElement("video");
    v.preload = "metadata";
    v.muted = true;
    const done = (d) => {
      durations.set(name, d);
      v.removeAttribute("src");
      resolve(d);
    };
    v.onloadedmetadata = () => done(Number.isFinite(v.duration) ? v.duration : null);
    v.onerror = () => done(null);
    setTimeout(() => done(null), 8000);
    v.src = api.apiURL(`/view?${q}`);
  });
}

function setHidden(node, name, hidden) {
  const w = node.widgets?.find((x) => x.name === name);
  if (!w || !!w.hidden === hidden) return false;
  w.hidden = hidden;
  w.options = w.options || {};
  w.options.hidden = hidden;
  return true;
}

async function updateLongClip(node) {
  const video = node.widgets?.find((w) => w.name === "video")?.value;
  const wired = node.inputs?.some((i) => (i.name === "video_in" || i.name === "images") && i.link != null);
  let long = true; // unknown length (wired input, unreadable file): show the options
  if (!wired && video && video !== "(none)") {
    const d = await videoDuration(video);
    if (d != null) long = d > 15.05;
  } else if (!wired) {
    long = false;
  }
  const mode = node.widgets?.find((w) => w.name === "long_clip")?.value;
  let changed = setHidden(node, "long_clip", !long);
  changed = setHidden(node, "window_start", !(long && mode === "Window")) || changed;
  changed = setHidden(node, "window_end", !(long && mode === "Window")) || changed;
  if (changed) {
    node.setSize([node.size[0], node.computeSize()[1]]);
    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
  }
}

app.registerExtension({
  name: "enzino.video2prompt",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    injectStyle();

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      this.properties = this.properties || {};
      const ui = buildCard(this);
      this._ev2p = ui;
      const w = this.addDOMWidget("v2p_card", "enzino_v2p_card", ui.card, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => ui.getHeight() + 6,
        getMaxHeight: () => ui.getHeight() + 6,
      });
      // Card follows the node width only; its height never changes, so nothing below can be pushed or overlapped.
      w.computeSize = (width) => [width, ui.getHeight() + 6];
      w.computeLayoutSize = () => ({ minHeight: ui.getHeight() + 6, maxHeight: ui.getHeight() + 6, minWidth: 0 });
      this.color = "#1a1a1c";
      this.bgcolor = "#0f0f10";
      const minW = 440;
      const sz = this.computeSize();
      this.setSize([Math.max(minW, sz[0]), sz[1]]);
      // Show the long-clip options only when they matter.
      for (const name of ["video", "long_clip"]) {
        const w = this.widgets?.find((x) => x.name === name);
        if (!w) continue;
        const cb = w.callback;
        w.callback = (...args) => {
          const out = cb?.apply(w, args);
          setTimeout(() => updateLongClip(this), 0);
          return out;
        };
      }
      const onConn = this.onConnectionsChange;
      this.onConnectionsChange = function () {
        const out = onConn?.apply(this, arguments);
        setTimeout(() => updateLongClip(this), 0);
        return out;
      };
      setTimeout(() => updateLongClip(this), 0);

      // Cap the core video preview when it gets created.
      const addDOM = this.addDOMWidget;
      this.addDOMWidget = function (name, type, element, opts) {
        const wid = addDOM.call(this, name, type, element, opts);
        if (name === "video-preview") capPreview(this, wid, element);
        return wid;
      };
      return r;
    };

    // Text re-wraps when the node gets wider/narrower: refit the card.
    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function () {
      const r = onResize?.apply(this, arguments);
      const w = Math.round(this.size?.[0] || 0);
      if (this._ev2p && w !== this._ev2pW) {
        this._ev2pW = w;
        clearTimeout(this._ev2pT);
        this._ev2pT = setTimeout(() => this._ev2p.fit(), 60);
      }
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      const p = this.properties || {};
      if (this._ev2p && (p.v2p_parts?.length || p.v2p_prompt))
        this._ev2p.show(p.v2p_parts?.length ? p.v2p_parts : p.v2p_prompt, p.v2p_meta, p.v2p_frames);
      fixModelValue(this);
      setTimeout(() => updateLongClip(this), 0);
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (msg) {
      const r = onExecuted?.apply(this, arguments);
      const parts = msg?.v2p_parts?.length ? msg.v2p_parts : [{ label: "H3 PROMPT", text: msg?.text?.[0] ?? "" }];
      const metaTxt = msg?.v2p_meta?.[0] ?? "";
      const frames = msg?.v2p_frames ?? [];
      Object.assign(this.properties, { v2p_parts: parts, v2p_prompt: "", v2p_meta: metaTxt, v2p_frames: frames });
      this._ev2p?.show(parts, metaTxt, frames);
      this._ev2p?.setStatus("ok", "Done");
      return r;
    };
  },

  setup() {
    const find = (id) => app.graph?.getNodeById?.(Number(id));
    let current = null;
    api.addEventListener("executing", ({ detail }) => {
      const id = detail && typeof detail === "object" ? detail.node : detail;
      if (current && current !== id) {
        const n = find(current);
        if (n?._ev2p && n.type === NODE) n._ev2p.setStatus("ok", "Done");
      }
      current = id;
      const n = id != null ? find(id) : null;
      if (n?.type === NODE) n._ev2p?.setStatus("busy", "Watching the video…");
    });
    api.addEventListener("execution_error", ({ detail }) => {
      const n = find(detail?.node_id);
      if (n?.type === NODE) n._ev2p?.setStatus("err", "Error — see console");
    });
  },
});
