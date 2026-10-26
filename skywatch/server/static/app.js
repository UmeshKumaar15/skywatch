/* SkyWatch dashboard - vanilla JS, canvas rendering. */
"use strict";

const CLASS_STYLE = {
  delivery_drone:      { color: "#29b6f6", short: "DLV", shape: "drone", name: "Delivery drone" },
  recreational_drone:  { color: "#66bb6a", short: "REC", shape: "drone", name: "Recreational drone" },
  industrial_drone:    { color: "#ffca28", short: "IND", shape: "drone", name: "Industrial drone" },
  bird:                { color: "#a1887f", short: "BRD", shape: "bird", name: "Bird" },
  helicopter:          { color: "#ab47bc", short: "HEL", shape: "heli", name: "Helicopter" },
  commercial_aircraft: { color: "#cfd8dc", short: "ACFT", shape: "plane", name: "Commercial aircraft" },
  emergency_vehicle:   { color: "#ff5252", short: "EMS", shape: "heli", name: "Emergency vehicle" },
  weather_balloon:     { color: "#f06292", short: "BLN", shape: "balloon", name: "Weather balloon" },
  unknown:             { color: "#9e9e9e", short: "UNK", shape: "unknown", name: "Unknown" },
};
const SENSOR_COLOR = { radar: "#7cb342", lidar: "#ab47bc", camera: "#ffa726", adsb: "#42a5f5", remote_id: "#26c6da", external: "#bdbdbd" };
const HYP_DASH = { cv: [], turn: [6, 3], accel: [2, 3], stop: [1, 3], route: [10, 4], orbit: [8, 3, 2, 3], learned: [] };
const INTENT_COLOR = "#7e9cff";

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : Number(v).toFixed(d);
const nice = (s) => (s || "–").replace(/_/g, " ");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const S = {
  mode: "live",
  meta: null,
  desc: null,
  frame: null,
  liveFrame: null,
  recvWall: 0,
  view: { cx: 0, cy: 0, scale: 0.05 },
  selected: null,
  layers: { obs: true, pred: true, unc: true, trails: true, labels: true, geo: true, coverage: false, truth: false, tentative: false },
  horizon: 60,
  uncMode: "selected", // off | selected | all
  insights: [],
  insightKeys: new Set(),
  unseenInsights: 0,
  pb: { runId: null, times: [], idx: 0, playing: false, speed: 1, pending: false, actual: new Map(), lastWall: 0, t: 0 },
  sort: { key: "id", dir: 1 },
  activeTab: "track",
  lastPanel: 0,
};

/* ------------------------------------------------------------------ canvas */
const canvas = $("map");
const ctx = canvas.getContext("2d");
let W = 0, H = 0, DPR = 1;
function resize() {
  DPR = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  W = r.width; H = r.height;
  canvas.width = Math.round(W * DPR); canvas.height = Math.round(H * DPR);
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
}
window.addEventListener("resize", resize);
const sx = (x) => W / 2 + (x - S.view.cx) * S.view.scale;
const sy = (y) => H / 2 - (y - S.view.cy) * S.view.scale;
const wx = (px) => S.view.cx + (px - W / 2) / S.view.scale;
const wy = (py) => S.view.cy - (py - H / 2) / S.view.scale;

function fitZone() {
  const R = (S.desc && S.desc.radius) || 6000;
  S.view.cx = 0; S.view.cy = 0;
  S.view.scale = Math.min(W, H) / (2 * R * 1.08);
}

function zoomAt(px, py, f) {
  const x = wx(px), y = wy(py);
  S.view.scale = Math.min(Math.max(S.view.scale * f, 0.002), 20);
  S.view.cx = x - (px - W / 2) / S.view.scale;
  S.view.cy = y + (py - H / 2) / S.view.scale;
}

/* ----------------------------------------------------------- interaction */
let drag = null;
const pointers = new Map();
canvas.addEventListener("pointerdown", (e) => {
  canvas.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, { x: e.offsetX, y: e.offsetY });
  drag = { x: e.offsetX, y: e.offsetY, cx: S.view.cx, cy: S.view.cy, moved: 0 };
  canvas.classList.add("dragging");
});
canvas.addEventListener("pointermove", (e) => {
  if (!pointers.has(e.pointerId)) return;
  const prev = pointers.get(e.pointerId);
  pointers.set(e.pointerId, { x: e.offsetX, y: e.offsetY });
  if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    const other = [...pointers.entries()].find(([id]) => id !== e.pointerId)[1];
    const d0 = Math.hypot(prev.x - other.x, prev.y - other.y), d1 = Math.hypot(a.x - b.x, a.y - b.y);
    if (d0 > 0) zoomAt((a.x + b.x) / 2, (a.y + b.y) / 2, d1 / d0);
    drag = null;
    return;
  }
  if (!drag) return;
  const dx = e.offsetX - drag.x, dy = e.offsetY - drag.y;
  drag.moved = Math.max(drag.moved, Math.hypot(dx, dy));
  S.view.cx = drag.cx - dx / S.view.scale;
  S.view.cy = drag.cy + dy / S.view.scale;
});
canvas.addEventListener("pointerup", (e) => {
  pointers.delete(e.pointerId);
  canvas.classList.remove("dragging");
  if (drag && drag.moved < 4) pickAt(e.offsetX, e.offsetY);
  drag = null;
});
canvas.addEventListener("wheel", (e) => { e.preventDefault(); zoomAt(e.offsetX, e.offsetY, Math.exp(-e.deltaY * 0.0015)); }, { passive: false });
$("zoomIn").onclick = () => zoomAt(W / 2, H / 2, 1.4);
$("zoomOut").onclick = () => zoomAt(W / 2, H / 2, 1 / 1.4);
$("zoomFit").onclick = fitZone;

function pickAt(px, py) {
  if (!S.frame) return;
  let best = null, bd = 16;
  for (const t of S.frame.tracks) {
    const p = displayPos(t);
    const d = Math.hypot(sx(p[0]) - px, sy(p[1]) - py);
    if (d < bd) { bd = d; best = t.id; }
  }
  select(best);
}

function select(id, center = false) {
  S.selected = id;
  if (id && center && S.frame) {
    const t = S.frame.tracks.find((x) => x.id === id);
    if (t) { S.view.cx = t.pos[0]; S.view.cy = t.pos[1]; }
  }
  if (id) setTab("track");
  renderPanel(true);
}

/* ---------------------------------------------------------------- layers */
const LAYER_NAMES = { obs: "Detections", pred: "Predictions", trails: "Trails", labels: "Labels", geo: "Map", coverage: "Coverage", truth: "Ground truth", tentative: "Tentative" };
function buildLayers() {
  const el = $("layers");
  el.innerHTML = "";
  for (const [k, name] of Object.entries(LAYER_NAMES)) {
    const b = document.createElement("button");
    b.className = "chip" + (S.layers[k] ? " on" : "");
    b.textContent = name;
    b.onclick = () => { S.layers[k] = !S.layers[k]; buildLayers(); };
    el.appendChild(b);
  }
  const u = document.createElement("button");
  u.className = "chip" + (S.uncMode !== "off" ? " on" : "");
  u.textContent = `Uncertainty: ${S.uncMode}`;
  u.title = "95% prediction regions: off / selected & anomalous tracks / all tracks";
  u.onclick = () => { S.uncMode = { off: "selected", selected: "all", all: "off" }[S.uncMode]; buildLayers(); };
  el.appendChild(u);
  const h = document.createElement("button");
  h.className = "chip on";
  h.textContent = `Horizon ${S.horizon}s`;
  h.title = "Prediction horizon shown";
  h.onclick = () => { S.horizon = S.horizon === 60 ? 15 : S.horizon === 15 ? 30 : 60; buildLayers(); };
  el.appendChild(h);
}
function buildLegend() {
  $("legend").innerHTML = Object.entries(CLASS_STYLE).map(([k, v]) =>
    `<span class="legend-item"><span class="sw" style="background:${v.color}"></span>${v.name}</span>`).join("");
}

/* ------------------------------------------------------------- drawing */
function displayPos(t) {
  if (S.mode !== "live" || !S.meta || S.meta.paused) return t.pos;
  const dt = Math.min(((performance.now() - S.recvWall) / 1000) * (S.meta.time_scale || 1), 1.0);
  return [t.pos[0] + t.vel[0] * dt, t.pos[1] + t.vel[1] * dt, t.pos[2] + t.vel[2] * dt];
}

function ellipse(x, y, sxx, sxy, syy, k) {
  const a = (sxx + syy) / 2, b = Math.sqrt(((sxx - syy) / 2) ** 2 + sxy * sxy);
  const l1 = Math.max(a + b, 0.01), l2 = Math.max(a - b, 0.01);
  const ang = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  const r1 = Math.max(k * Math.sqrt(l1) * S.view.scale, 1), r2 = Math.max(k * Math.sqrt(l2) * S.view.scale, 1);
  ctx.beginPath();
  ctx.ellipse(sx(x), sy(y), r1, r2, -ang, 0, Math.PI * 2);
}

function drawGrid() {
  ctx.fillStyle = css("--mapbg");
  ctx.fillRect(0, 0, W, H);
  const R = (S.desc && S.desc.radius) || 6000;
  // Choose a ring spacing that keeps ~6-12 rings on screen.
  const span = Math.max(W, H) / S.view.scale;
  const steps = [100, 250, 500, 1000, 2000, 5000, 10000];
  const step = steps.find((s) => span / s < 14) || 10000;
  ctx.lineWidth = 1;
  ctx.strokeStyle = css("--grid");
  ctx.fillStyle = css("--muted");
  ctx.font = "10px ui-monospace, Menlo, monospace";
  const maxR = Math.hypot(Math.abs(S.view.cx) + span, Math.abs(S.view.cy) + span);
  for (let r = step; r < Math.min(maxR, R * 3); r += step) {
    ctx.beginPath(); ctx.arc(sx(0), sy(0), r * S.view.scale, 0, Math.PI * 2); ctx.stroke();
    ctx.fillText(r >= 1000 ? `${r / 1000} km` : `${r} m`, sx(0) + r * S.view.scale + 3, sy(0) - 3);
  }
  ctx.strokeStyle = css("--grid");
  ctx.beginPath(); ctx.moveTo(sx(-R * 3), sy(0)); ctx.lineTo(sx(R * 3), sy(0));
  ctx.moveTo(sx(0), sy(-R * 3)); ctx.lineTo(sx(0), sy(R * 3)); ctx.stroke();
  ctx.strokeStyle = css("--grid2");
  ctx.setLineDash([6, 6]);
  ctx.beginPath(); ctx.arc(sx(0), sy(0), R * S.view.scale, 0, Math.PI * 2); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillText("N", sx(0) - 3, sy(R) - 6);
}

function drawGeography() {
  const g = S.desc && S.desc.geography;
  if (!g) return;
  const geo = css("--geo");
  ctx.lineWidth = 1.5;
  ctx.font = "11px system-ui, sans-serif";
  if (S.layers.geo) {
    ctx.strokeStyle = geo; ctx.setLineDash([8, 6]);
    for (const line of g.corridors || []) {
      ctx.beginPath();
      line.forEach((p, i) => (i ? ctx.lineTo(sx(p[0]), sy(p[1])) : ctx.moveTo(sx(p[0]), sy(p[1]))));
      ctx.stroke();
    }
    ctx.setLineDash([3, 4]);
    for (const a of g.mapping_areas || []) ctx.strokeRect(sx(a[0]), sy(a[1] + a[3]), a[2] * S.view.scale, a[3] * S.view.scale);
    for (const p of g.parks || []) { ctx.strokeStyle = "rgba(102,187,106,.45)"; ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), p[2] * S.view.scale, 0, Math.PI * 2); ctx.stroke(); }
    ctx.setLineDash([]);
    for (const z of g.nofly || []) {
      ctx.fillStyle = "rgba(255,82,82,.08)"; ctx.strokeStyle = "rgba(255,82,82,.6)";
      ctx.beginPath(); ctx.arc(sx(z[0]), sy(z[1]), z[2] * S.view.scale, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "rgba(255,120,120,.9)"; ctx.fillText(`${z[3] || "restricted"} (no-fly)`, sx(z[0]) - 30, sy(z[1]) - z[2] * S.view.scale - 4);
    }
    ctx.fillStyle = geo;
    for (const d of g.depots || []) { ctx.fillRect(sx(d[0]) - 4, sy(d[1]) - 4, 8, 8); ctx.fillText("depot", sx(d[0]) + 6, sy(d[1]) + 4); }
    for (const d of g.destinations || []) { ctx.beginPath(); ctx.arc(sx(d[0]), sy(d[1]), 2.5, 0, Math.PI * 2); ctx.fill(); }
    for (const s of g.inspection_sites || []) {
      ctx.beginPath(); ctx.moveTo(sx(s[0]), sy(s[1]) - 6); ctx.lineTo(sx(s[0]) - 4, sy(s[1]) + 4); ctx.lineTo(sx(s[0]) + 4, sy(s[1]) + 4); ctx.closePath(); ctx.fill();
    }
    if (g.hospital) { ctx.fillText("H", sx(g.hospital[0]) - 4, sy(g.hospital[1]) + 4); ctx.beginPath(); ctx.arc(sx(g.hospital[0]), sy(g.hospital[1]), 8, 0, Math.PI * 2); ctx.strokeStyle = geo; ctx.stroke(); }
    if (g.met_station) ctx.fillText("met", sx(g.met_station[0]) - 8, sy(g.met_station[1]) + 4);
  }
  for (const s of (S.desc.sensors || [])) {
    const c = SENSOR_COLOR[s.type] || "#bbb";
    if (S.layers.coverage) {
      ctx.strokeStyle = c + "55"; ctx.setLineDash([2, 5]);
      ctx.beginPath(); ctx.arc(sx(s.position[0]), sy(s.position[1]), Math.min(s.range, 30000) * S.view.scale, 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.fillStyle = c;
    const x = sx(s.position[0]), y = sy(s.position[1]);
    ctx.beginPath(); ctx.moveTo(x, y - 6); ctx.lineTo(x - 5, y + 4); ctx.lineTo(x + 5, y + 4); ctx.closePath(); ctx.fill();
    if (S.layers.coverage || S.view.scale > 0.15) { ctx.fillStyle = css("--muted"); ctx.fillText(s.id, x + 7, y + 4); }
  }
}

function drawSymbol(t, x, y, color) {
  const shape = (CLASS_STYLE[t.class.label] || CLASS_STYLE.unknown).shape;
  const hd = Math.atan2(t.vel[0], t.vel[1]);
  ctx.save();
  ctx.translate(x, y);
  ctx.fillStyle = color; ctx.strokeStyle = color; ctx.lineWidth = 1.6;
  if (shape === "plane") {
    ctx.rotate(hd);
    ctx.beginPath(); ctx.moveTo(0, -9); ctx.lineTo(6, 7); ctx.lineTo(0, 3); ctx.lineTo(-6, 7); ctx.closePath(); ctx.fill();
  } else if (shape === "heli") {
    ctx.beginPath(); ctx.arc(0, 0, 5, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.moveTo(-8, -8); ctx.lineTo(8, 8); ctx.moveTo(8, -8); ctx.lineTo(-8, 8); ctx.stroke();
  } else if (shape === "drone") {
    ctx.rotate(Math.PI / 4); ctx.fillRect(-4.5, -4.5, 9, 9);
  } else if (shape === "bird") {
    ctx.rotate(hd);
    ctx.beginPath(); ctx.moveTo(-6, 2); ctx.lineTo(0, -3); ctx.lineTo(6, 2); ctx.stroke();
  } else if (shape === "balloon") {
    ctx.beginPath(); ctx.arc(0, -1, 5, 0, Math.PI * 2); ctx.stroke(); ctx.beginPath(); ctx.moveTo(0, 4); ctx.lineTo(0, 8); ctx.stroke();
  } else {
    ctx.setLineDash([2, 2]); ctx.beginPath(); ctx.arc(0, 0, 5.5, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
  }
  ctx.restore();
}

function drawPrediction(t, color, isSel) {
  const pred = t.prediction;
  if (!pred || !pred.hypotheses) return;
  const p0 = t.pos;
  const hyps = pred.hypotheses;
  const showUnc = S.uncMode === "all" || (S.uncMode === "selected" && (isSel || t.anomaly.score >= 0.6));
  if (showUnc && pred.mixture) {
    // Uncertainty cone: 2.45-sigma (95 %) ellipses of the hypothesis mixture.
    const strong = isSel || S.uncMode !== "all";
    ctx.fillStyle = color + (strong ? "16" : "08");
    ctx.strokeStyle = color + (strong ? "44" : "18");
    ctx.lineWidth = 1;
    const lim = strong ? S.horizon : Math.min(S.horizon, 30);
    for (const q of pred.mixture) {
      if (q[0] > lim) break;
      ellipse(q[1], q[2], q[4], q[5], q[6], 2.45); ctx.fill(); if (strong) ctx.stroke();
    }
  }
  for (const h of hyps) {
    const a = Math.max(0.18, Math.min(1, h.weight * 1.4));
    ctx.strokeStyle = color;
    ctx.globalAlpha = isSel ? Math.max(a, 0.45) : a * 0.75;
    ctx.lineWidth = 1 + 2.2 * h.weight * (isSel ? 1.4 : 1);
    ctx.setLineDash(HYP_DASH[h.kind] || []);
    ctx.beginPath(); ctx.moveTo(sx(p0[0]), sy(p0[1]));
    for (const q of h.points) { if (q[0] > S.horizon) break; ctx.lineTo(sx(q[1]), sy(q[2])); }
    ctx.stroke();
    if (isSel && S.uncMode !== "off") {
      ctx.setLineDash([]); ctx.lineWidth = 1; ctx.globalAlpha = 0.35 * Math.max(a, 0.4);
      for (const q of h.points) { if (q[0] > S.horizon) break; if (q[0] % 10 === 0) { ellipse(q[1], q[2], q[4], q[5], q[6], 2.45); ctx.stroke(); } }
    }
  }
  ctx.globalAlpha = 1; ctx.setLineDash([]);
}

function drawActual(t) {
  const key = `${S.pb.runId}|${t.id}|${S.frame.t}`;
  const path = S.pb.actual.get(key);
  if (!path || !path.length) return;
  ctx.strokeStyle = css("--text"); ctx.lineWidth = 1.6; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(sx(t.pos[0]), sy(t.pos[1]));
  for (const p of path) { if (p[0] - S.frame.t > S.horizon) break; ctx.lineTo(sx(p[1]), sy(p[2])); }
  ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = css("--text");
  for (const p of path) {
    const dt = p[0] - S.frame.t;
    if (dt > S.horizon) break;
    if (Math.abs(dt - Math.round(dt / 10) * 10) < 0.6 && dt > 5) { ctx.beginPath(); ctx.arc(sx(p[1]), sy(p[2]), 2.5, 0, Math.PI * 2); ctx.fill(); }
  }
}

function draw() {
  requestAnimationFrame(draw);
  if (!W) resize();
  drawGrid();
  drawGeography();
  const f = S.frame;
  if (!f) return;
  ctx.font = "11px ui-monospace, Menlo, monospace";
  if (S.layers.obs && f.observations) {
    for (const o of f.observations) {
      ctx.fillStyle = (SENSOR_COLOR[o[0]] || "#999") + "aa";
      ctx.fillRect(sx(o[1]) - 1.5, sy(o[2]) - 1.5, 3, 3);
    }
  }
  if (S.layers.truth && f.truth) {
    ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1;
    for (const o of f.truth) { ctx.beginPath(); ctx.arc(sx(o.pos[0]), sy(o.pos[1]), 7, 0, Math.PI * 2); ctx.stroke(); }
  }
  if (S.layers.tentative && f.tentative) {
    ctx.strokeStyle = css("--muted");
    for (const o of f.tentative) { const x = sx(o[1]), y = sy(o[2]); ctx.beginPath(); ctx.moveTo(x - 3, y); ctx.lineTo(x + 3, y); ctx.moveTo(x, y - 3); ctx.lineTo(x, y + 3); ctx.stroke(); }
  }
  const sel = f.tracks.find((t) => t.id === S.selected);
  for (const t of f.tracks) {
    const st = CLASS_STYLE[t.class.label] || CLASS_STYLE.unknown;
    const isSel = t.id === S.selected;
    if (S.layers.trails && t.trail && t.trail.length > 1) {
      ctx.strokeStyle = st.color + (isSel ? "cc" : "66"); ctx.lineWidth = isSel ? 2 : 1.2;
      ctx.beginPath();
      t.trail.forEach((p, i) => (i ? ctx.lineTo(sx(p[1]), sy(p[2])) : ctx.moveTo(sx(p[1]), sy(p[2]))));
      ctx.lineTo(sx(t.pos[0]), sy(t.pos[1]));
      ctx.stroke();
    }
    if (S.layers.pred && (!sel || isSel || true)) drawPrediction(t, st.color, isSel);
  }
  if (sel && S.mode === "playback" && $("pbActual").checked) drawActual(sel);
  const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 250);
  for (const t of f.tracks) {
    const st = CLASS_STYLE[t.class.label] || CLASS_STYLE.unknown;
    const p = displayPos(t);
    const x = sx(p[0]), y = sy(p[1]);
    if (x < -50 || y < -50 || x > W + 50 || y > H + 50) continue;
    const isSel = t.id === S.selected;
    ctx.globalAlpha = t.status === "coasting" ? 0.5 : 1;
    drawSymbol(t, x, y, st.color);
    ctx.globalAlpha = 1;
    if (t.anomaly && t.anomaly.score >= 0.6) {
      ctx.strokeStyle = `rgba(255,82,82,${0.4 + 0.5 * pulse})`; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 12 + 3 * pulse, 0, Math.PI * 2); ctx.stroke();
    }
    if (isSel) {
      ctx.strokeStyle = css("--accent"); ctx.lineWidth = 2;
      ctx.strokeRect(x - 13, y - 13, 26, 26);
    }
    if (S.layers.labels && (isSel || S.view.scale > 0.12 || t.anomaly.score >= 0.6)) {
      ctx.fillStyle = css("--text");
      ctx.fillText(`${t.id} ${st.short} ${fmt(t.pos[2])}m`, x + 12, y - 8);
      if (isSel || S.view.scale > 0.3) { ctx.fillStyle = css("--muted"); ctx.fillText(`${nice(t.intent.label)} · ${fmt(t.speed, 1)} m/s`, x + 12, y + 5); }
    }
  }
}

/* ----------------------------------------------------------------- panel */
function bars(obj, color, n = 6) {
  const entries = Object.entries(obj || {}).sort((a, b) => b[1] - a[1]).slice(0, n);
  return `<div class="bars">${entries.map(([k, v]) => {
    const c = typeof color === "function" ? color(k) : color;
    return `<div class="row"><span class="lbl">${esc(nice(k))}</span><span class="track"><span class="fill" style="width:${(v * 100).toFixed(1)}%;background:${c}"></span></span><span class="val">${(v * 100).toFixed(0)}%</span></div>`;
  }).join("")}</div>`;
}

function sparkline(canvasEl, values, color, label) {
  const c = canvasEl, g = c.getContext("2d");
  const r = c.getBoundingClientRect();
  c.width = r.width * DPR; c.height = r.height * DPR;
  g.setTransform(DPR, 0, 0, DPR, 0, 0);
  g.clearRect(0, 0, r.width, r.height);
  if (values.length < 2) return;
  const lo = Math.min(...values), hi = Math.max(...values), span = Math.max(hi - lo, 1e-6);
  g.strokeStyle = color; g.lineWidth = 1.6; g.beginPath();
  values.forEach((v, i) => {
    const x = 4 + (i / (values.length - 1)) * (r.width - 8), y = r.height - 12 - ((v - lo) / span) * (r.height - 22);
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  });
  g.stroke();
  g.fillStyle = css("--muted"); g.font = "10px ui-monospace, Menlo, monospace";
  g.fillText(`${label}  ${fmt(lo)}–${fmt(hi)}`, 6, r.height - 2);
}

function renderTrack() {
  const el = $("tab-track");
  const f = S.frame;
  const t = f && f.tracks.find((x) => x.id === S.selected);
  if (!t) { el.innerHTML = `<div class="empty">${S.selected ? `Track ${esc(S.selected)} is no longer active.` : "Select a track on the map or from the Tracks list."}</div>`; return; }
  const st = CLASS_STYLE[t.class.label] || CLASS_STYLE.unknown;
  const src = Object.entries(t.sources || {}).map(([k, v]) => `${k}:${v}`).join(" · ");
  const pred = t.prediction || { hypotheses: [] };
  let check = "";
  if (S.mode === "playback") {
    const path = S.pb.actual.get(`${S.pb.runId}|${t.id}|${f.t}`);
    if (path && path.length && pred.mixture) {
      const rows = [];
      for (const hz of [10, 20, 30, 60]) {
        const act = path.find((p) => Math.abs(p[0] - f.t - hz) < 0.8);
        const mix = pred.mixture.find((q) => q[0] === hz);
        const top = pred.hypotheses[0] && pred.hypotheses[0].points.find((q) => q[0] === hz);
        if (act && mix) {
          const em = Math.hypot(act[1] - mix[1], act[2] - mix[2], act[3] - mix[3]);
          const et = top ? Math.hypot(act[1] - top[1], act[2] - top[2], act[3] - top[3]) : NaN;
          const sig = Math.sqrt((mix[4] + mix[6]) / 2);
          rows.push(`<tr><td>${hz}s</td><td class="num">${fmt(et)}</td><td class="num">${fmt(em)}</td><td class="num">${fmt(sig)}</td></tr>`);
        }
      }
      if (rows.length) check = `<h3>Prediction check vs. actual path</h3><table><thead><tr><th>Horizon</th><th class="num">Top hyp. err (m)</th><th class="num">Mixture err (m)</th><th class="num">1σ (m)</th></tr></thead><tbody>${rows.join("")}</tbody></table>`;
    }
  }
  el.innerHTML = `
    <div class="trackhead"><span class="sw" style="background:${st.color};width:14px;height:14px"></span><span class="tid">${esc(t.id)}</span>
      <span class="pill ${t.status}">${t.status}</span>${t.identity ? `<span class="pill">${esc(t.identity)}</span>` : ""}</div>
    <div class="note" style="margin-top:4px">${esc(st.name)} · ${esc(nice(t.intent.label))}${t.truth_id ? ` · truth ${esc(t.truth_id)}` : ""}</div>
    <div class="kv">
      <div><div class="k">Altitude</div><div class="v">${fmt(t.pos[2])} m</div></div>
      <div><div class="k">Speed</div><div class="v">${fmt(t.speed, 1)} m/s</div></div>
      <div><div class="k">Heading</div><div class="v">${fmt(t.heading)}°</div></div>
      <div><div class="k">Vertical</div><div class="v">${fmt(t.vel[2], 1)} m/s</div></div>
      <div><div class="k">Pos. σ</div><div class="v">${fmt(Math.sqrt((t.cov[0] + t.cov[2]) / 2), 1)} m</div></div>
      <div><div class="k">Confidence</div><div class="v">${fmt(t.confidence * 100)}%</div></div>
      <div><div class="k">Age</div><div class="v">${fmt(t.age)} s</div></div>
      <div><div class="k">Last update</div><div class="v">${fmt(t.since_update, 1)} s</div></div>
      <div><div class="k">Anomaly</div><div class="v" style="color:${t.anomaly.score >= 0.6 ? "var(--bad)" : "inherit"}">${fmt(t.anomaly.score * 100)}%</div></div>
    </div>
    <div class="note" style="margin-top:6px">Sources: ${esc(src)}</div>
    <h3>Classification <span class="note">(entropy ${fmt(t.class.entropy, 2)})</span></h3>${bars(t.class.probs, (k) => (CLASS_STYLE[k] || CLASS_STYLE.unknown).color)}
    <h3>Inferred intent</h3>${bars(t.intent.probs, INTENT_COLOR)}
    <h3>Motion hypotheses <span class="note">(${pred.discarded || 0} discarded)</span></h3>
    <div class="bars">${pred.hypotheses.map((h) => `<div class="row"><span class="lbl" title="${esc(h.note)}">${esc(h.label)}${h.note ? ` <span class="note">${esc(h.note)}</span>` : ""}</span><span class="track"><span class="fill" style="width:${(h.weight * 100).toFixed(1)}%;background:${st.color}"></span></span><span class="val">${(h.weight * 100).toFixed(0)}%</span></div>`).join("")}</div>
    <h3>Motion model (IMM)</h3>${bars(t.modes, "#8d9db6", 3)}
    ${Object.keys(t.anomaly.scores || {}).length ? `<h3>Anomaly indicators</h3>${bars(t.anomaly.scores, "#ff7961")}` : ""}
    <h3>History</h3><canvas class="spark" id="sparkAlt"></canvas><div style="height:6px"></div><canvas class="spark" id="sparkSpd"></canvas>
    ${check}`;
  const tr = t.trail || [];
  sparkline($("sparkAlt"), tr.map((p) => p[3]), st.color, "altitude m");
  const spd = [];
  for (let i = 1; i < tr.length; i++) spd.push(Math.hypot(tr[i][1] - tr[i - 1][1], tr[i][2] - tr[i - 1][2]) / Math.max(tr[i][0] - tr[i - 1][0], 0.1));
  sparkline($("sparkSpd"), spd, "#8d9db6", "speed m/s");
}

function renderTracks() {
  const f = S.frame;
  if (!f) return;
  const k = S.sort.key, d = S.sort.dir;
  const val = (t) => ({ id: t.id, cls: t.class.label, intent: t.intent.label || "", alt: t.pos[2], speed: t.speed, conf: t.confidence, anom: t.anomaly.score }[k]);
  const rows = [...f.tracks].sort((a, b) => (val(a) > val(b) ? d : val(a) < val(b) ? -d : 0));
  const th = (key, label, num) => `<th data-k="${key}" class="${num ? "num" : ""}">${label}${S.sort.key === key ? (d > 0 ? " ▲" : " ▼") : ""}</th>`;
  $("tab-tracks").innerHTML = `<table><thead><tr>${th("id", "ID")}${th("cls", "Class")}${th("intent", "Intent")}${th("alt", "Alt", 1)}${th("speed", "m/s", 1)}${th("conf", "Conf", 1)}${th("anom", "Anom", 1)}</tr></thead><tbody>
    ${rows.map((t) => `<tr data-id="${esc(t.id)}" class="${t.id === S.selected ? "sel" : ""}"><td>${esc(t.id)}</td><td><span class="sw" style="background:${(CLASS_STYLE[t.class.label] || CLASS_STYLE.unknown).color}"></span> ${(CLASS_STYLE[t.class.label] || CLASS_STYLE.unknown).short}</td><td>${esc(nice(t.intent.label))}</td><td class="num">${fmt(t.pos[2])}</td><td class="num">${fmt(t.speed, 1)}</td><td class="num">${fmt(t.confidence * 100)}%</td><td class="num" style="color:${t.anomaly.score >= 0.6 ? "var(--bad)" : "inherit"}">${fmt(t.anomaly.score * 100)}%</td></tr>`).join("")}
  </tbody></table>`;
}
$("tab-tracks").addEventListener("click", (e) => {
  const h = e.target.closest("th");
  if (h) { const kk = h.dataset.k; S.sort = { key: kk, dir: S.sort.key === kk ? -S.sort.dir : 1 }; renderTracks(); return; }
  const r = e.target.closest("tbody tr");
  if (r) select(r.dataset.id, true);
});

function renderInsights() {
  const list = S.mode === "live" ? S.insights : (S.frame && S.frame.insights) || [];
  $("tab-insights").innerHTML = list.length ? list.slice(0, 150).map((i) =>
    `<div class="ins ${i.severity}" data-id="${esc(i.track_id)}" data-t="${i.t}"><div class="h"><span>${esc(i.track_id)} · ${esc(nice(i.kind))}</span><span>t=${fmt(i.t, 1)} · ${(i.score * 100).toFixed(0)}%</span></div><div class="m">${esc(i.message)}</div></div>`).join("")
    : `<div class="empty">No insights yet. Anomalous behaviour will be summarised here.</div>`;
}
$("tab-insights").addEventListener("click", (e) => {
  const r = e.target.closest(".ins");
  if (r) select(r.dataset.id, true);
});

function renderSystem() {
  const f = S.frame;
  if (!f) return;
  const s = f.stats;
  const acc = s.prediction_accuracy || {};
  const accRows = Object.entries(acc).map(([h, a]) => `<tr><td>${h}</td><td class="num">${fmt(a.top)}</td><td class="num">${fmt(a.mixture)}</td><td class="num">${fmt(a.cv_baseline)}</td><td class="num">${fmt(a.coverage_2sigma * 100)}%</td><td class="num">${a.n}</td></tr>`).join("");
  const tot = Math.max(1, s.tracks);
  const cls = Object.fromEntries(Object.entries(s.classes || {}).map(([k, v]) => [k, v / tot]));
  const ints = Object.fromEntries(Object.entries(s.intents || {}).map(([k, v]) => [k, v / tot]));
  const tk = s.tracker || {};
  $("tab-system").innerHTML = `
    <div class="stats">
      <div class="stat"><div class="k">Confirmed tracks</div><div class="v">${s.tracks}</div></div>
      <div class="stat"><div class="k">Detections / s</div><div class="v">${fmt(s.obs_per_s, 1)}</div></div>
      <div class="stat"><div class="k">Tracking cost / tick</div><div class="v">${fmt(s.track_ms, 1)} ms</div></div>
      <div class="stat"><div class="k">Analytics cost / tick</div><div class="v">${fmt(s.analytics_ms, 1)} ms</div></div>
      <div class="stat"><div class="k">Observation age at use</div><div class="v">${fmt(s.latency_s, 2)} s</div></div>
      <div class="stat"><div class="k">Stale dropped</div><div class="v">${s.stale_dropped}</div></div>
    </div>
    <h3>Prediction accuracy (mean error, m)</h3>
    <table><thead><tr><th>Horizon</th><th class="num">Top</th><th class="num">Mixture</th><th class="num">CV baseline</th><th class="num">In 95% region</th><th class="num">n</th></tr></thead><tbody>${accRows || '<tr><td colspan="6" class="note">Accumulating…</td></tr>'}</tbody></table>
    <div class="note" style="margin-top:4px">Errors measured against the later filtered track position; "CV baseline" is pure constant-velocity extrapolation.</div>
    <h3>Class mix</h3>${bars(cls, (k) => (CLASS_STYLE[k] || CLASS_STYLE.unknown).color, 10)}
    <h3>Intent mix</h3>${bars(ints, INTENT_COLOR, 10)}
    <h3>Tracker</h3>
    <table><tbody>${Object.entries(tk).map(([k, v]) => `<tr><td>${nice(k)}</td><td class="num">${v}</td></tr>`).join("")}</tbody></table>
    <h3>Insights by kind</h3>
    <table><tbody>${Object.entries(s.insight_counts || {}).map(([k, v]) => `<tr><td>${nice(k)}</td><td class="num">${v}</td></tr>`).join("") || '<tr><td class="note">none</td></tr>'}</tbody></table>
    <h3>Models</h3>
    <table><tbody><tr><td>Learned classifier</td><td class="num">${s.models.classifier ? "loaded" : "heuristic fallback"}</td></tr>
    <tr><td>Learned motion model</td><td class="num">${s.models.learned_motion ? "loaded" : "not trained"}</td></tr>
    <tr><td>Flow-map samples</td><td class="num">${s.models.flowmap_samples}</td></tr></tbody></table>`;
}

function renderPanel(force = false) {
  const now = performance.now();
  if (!force && now - S.lastPanel < 500) return;
  S.lastPanel = now;
  // Don't rebuild lists under the pointer: it would swallow clicks.
  const body = $(`tab-${S.activeTab}`);
  const hovering = !force && (S.activeTab === "tracks" || S.activeTab === "insights") && body.matches(":hover");
  if (hovering) { /* keep the list stable while the user is interacting */ }
  else if (S.activeTab === "track") renderTrack();
  else if (S.activeTab === "tracks") renderTracks();
  else if (S.activeTab === "insights") renderInsights();
  else renderSystem();
  $("insCount").textContent = S.unseenInsights ? String(S.unseenInsights) : "";
  if (S.frame) {
    const s = S.frame.stats;
    $("statline").innerHTML = `<span class="mono">${s.tracks} tracks · ${fmt(s.obs_per_s, 0)} det/s · ${Object.values(s.insight_counts || {}).reduce((a, b) => a + b, 0)} insights</span>`;
  }
}

function setTab(name) {
  S.activeTab = name;
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  for (const n of ["track", "tracks", "insights", "system"]) $(`tab-${n}`).classList.toggle("hidden", n !== name);
  if (name === "insights") S.unseenInsights = 0;
  renderPanel(true);
}
document.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => setTab(b.dataset.tab)));

/* ------------------------------------------------------------- live mode */
let ws = null;
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/live`);
  ws.onopen = () => $("conn").classList.add("on");
  ws.onclose = () => { $("conn").classList.remove("on"); setTimeout(connect, 1500); };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "meta") return applyMeta(msg);
    S.liveFrame = msg;
    for (const i of msg.insights || []) {
      const key = `${i.track_id}|${i.kind}|${i.t}`;
      if (!S.insightKeys.has(key)) {
        S.insightKeys.add(key); S.insights.unshift(i);
        if (S.activeTab !== "insights") S.unseenInsights++;
      }
    }
    S.insights.sort((a, b) => b.t - a.t);
    if (S.insights.length > 300) S.insights.length = 300;
    if (S.mode === "live") {
      S.frame = msg; S.recvWall = performance.now();
      $("clock").textContent = `t = ${fmt(msg.t, 1)} s`;
      renderPanel();
    }
  };
}

function applyMeta(m) {
  const first = !S.meta;
  const newRun = !S.meta || S.meta.run_id !== m.run_id;
  S.meta = m;
  if (S.mode === "live") S.desc = m.description;
  const sel = $("scenario");
  if (!sel.options.length || newRun) {
    sel.innerHTML = (m.scenarios || []).map((s) => `<option value="${esc(s.name)}" title="${esc(s.description)}">${esc(s.name)}</option>`).join("");
    if (m.description && m.description.scenario) sel.value = m.description.scenario;
    if (m.description && m.description.seed !== null && m.description.seed !== undefined) $("seed").value = m.description.seed;
  }
  $("pause").textContent = m.paused ? "Resume" : "Pause";
  const sp = $("speed");
  if (![...sp.options].some((o) => Number(o.value) === Number(m.time_scale))) sp.add(new Option(String(m.time_scale), String(m.time_scale)));
  sp.value = [...sp.options].find((o) => Number(o.value) === Number(m.time_scale)).value;
  document.querySelectorAll(".live-only").forEach((e) => e.classList.toggle("hidden", m.mode !== "simulation" || S.mode !== "live"));
  if (newRun) { S.insights = []; S.insightKeys.clear(); S.unseenInsights = 0; }
  if (first || newRun) fitZone();
}

async function control(body) {
  const r = await fetch("/api/control", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (r.ok) applyMeta(await r.json());
}
$("pause").onclick = () => control({ action: S.meta && S.meta.paused ? "resume" : "pause" });
$("speed").onchange = (e) => control({ action: "speed", speed: Number(e.target.value) });
$("restart").onclick = () => {
  const seed = $("seed").value === "" ? null : Number($("seed").value);
  S.selected = null;
  control({ action: "restart", scenario: $("scenario").value, seed });
};

/* --------------------------------------------------------- playback mode */
function setMode(mode) {
  S.mode = mode;
  $("modeLive").classList.toggle("active", mode === "live");
  $("modePlayback").classList.toggle("active", mode === "playback");
  $("playbar").classList.toggle("hidden", mode !== "playback");
  document.querySelectorAll(".live-only").forEach((e) => e.classList.toggle("hidden", mode !== "live" || (S.meta && S.meta.mode !== "simulation")));
  if (mode === "live") {
    S.pb.playing = false;
    S.desc = S.meta && S.meta.description;
    S.frame = S.liveFrame;
    fitZone();
  } else {
    loadRuns();
  }
  renderPanel(true);
  requestAnimationFrame(resize);
}
$("modeLive").onclick = () => setMode("live");
$("modePlayback").onclick = () => setMode("playback");

async function loadRuns() {
  const runs = await (await fetch("/api/runs")).json();
  $("runSel").innerHTML = runs.map((r) => `<option value="${r.id}">#${r.id} ${esc(r.scenario || r.source)}${r.seed !== null ? ` seed ${r.seed}` : ""} · ${fmt((r.t_max || 0) - (r.t_min || 0))} s</option>`).join("");
  if (runs.length) openRun(Number($("runSel").value));
}
$("runSel").onchange = (e) => openRun(Number(e.target.value));

async function openRun(id) {
  const info = await (await fetch(`/api/runs/${id}`)).json();
  S.pb.runId = id; S.pb.times = info.frame_times; S.pb.idx = Math.max(0, info.frame_times.length - 1);
  S.desc = info.meta;
  $("scrub").max = String(Math.max(0, S.pb.times.length - 1));
  fitZone();
  // Open 60 s before the end (or mid-way through a short run) so that the actual
  // future path is available for comparison with the predictions.
  const n = S.pb.times.length;
  S.pb.idx = n > 240 ? n - 121 : Math.floor(n / 2);
  await loadFrame(S.pb.idx);
}

async function loadFrame(idx) {
  if (!S.pb.times.length) return;
  idx = Math.max(0, Math.min(idx, S.pb.times.length - 1));
  S.pb.idx = idx;
  $("scrub").value = String(idx);
  const t = S.pb.times[idx];
  S.pb.pending = true;
  try {
    const fr = await (await fetch(`/api/runs/${S.pb.runId}/frame?t=${t}`)).json();
    if (S.mode !== "playback") return;
    S.frame = fr;
    $("pbTime").textContent = `t = ${fmt(fr.t, 1)} s`;
    $("clock").textContent = `t = ${fmt(fr.t, 1)} s (playback)`;
    await loadActual();
    renderPanel(true);
  } finally { S.pb.pending = false; }
}

async function loadActual() {
  if (!S.selected || !S.frame || !$("pbActual").checked) return;
  const key = `${S.pb.runId}|${S.selected}|${S.frame.t}`;
  if (S.pb.actual.has(key)) return;
  const path = await (await fetch(`/api/runs/${S.pb.runId}/track/${encodeURIComponent(S.selected)}?t0=${S.frame.t}&t1=${S.frame.t + 62}`)).json();
  S.pb.actual.set(key, path);
  if (S.pb.actual.size > 400) S.pb.actual.delete(S.pb.actual.keys().next().value);
}

$("scrub").oninput = (e) => loadFrame(Number(e.target.value));
$("pbBack").onclick = () => loadFrame(S.pb.idx - 1);
$("pbFwd").onclick = () => loadFrame(S.pb.idx + 1);
$("pbSpeed").onchange = (e) => (S.pb.speed = Number(e.target.value));
$("pbActual").onchange = () => loadActual().then(() => renderPanel(true));
$("pbPlay").onclick = () => {
  S.pb.playing = !S.pb.playing;
  $("pbPlay").textContent = S.pb.playing ? "⏸" : "▶";
  S.pb.lastWall = performance.now();
  S.pb.t = S.pb.times[S.pb.idx] || 0;
};
setInterval(() => {
  if (S.mode !== "playback" || !S.pb.playing || S.pb.pending || !S.pb.times.length) return;
  const now = performance.now();
  S.pb.t += ((now - S.pb.lastWall) / 1000) * S.pb.speed;
  S.pb.lastWall = now;
  let i = S.pb.idx;
  while (i + 1 < S.pb.times.length && S.pb.times[i + 1] <= S.pb.t) i++;
  if (i !== S.pb.idx) loadFrame(i);
  if (i >= S.pb.times.length - 1) { S.pb.playing = false; $("pbPlay").textContent = "▶"; }
}, 100);
document.addEventListener("keydown", (e) => {
  if (S.mode !== "playback" || e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "ArrowRight") loadFrame(S.pb.idx + 1);
  if (e.key === "ArrowLeft") loadFrame(S.pb.idx - 1);
  if (e.key === " ") { e.preventDefault(); $("pbPlay").click(); }
});
const _select = select;
select = function (id, center) { _select(id, center); if (S.mode === "playback") loadActual().then(() => renderPanel(true)); };

/* ------------------------------------------------------------------ boot */
buildLayers();
buildLegend();
resize();
fitZone();
connect();
requestAnimationFrame(draw);
