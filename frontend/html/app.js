"use strict";

const REFRESH_MS = 2000;
const css = (name) => getComputedStyle(document.querySelector(".viz-root")).getPropertyValue(name).trim();
const C = {
  s1: css("--series-1"), s2: css("--series-2"), s3: css("--series-3"),
  grid: css("--grid"), muted: css("--text-muted"), secondary: css("--text-secondary"),
  surface: css("--surface-1"),
};
const GiB = 1024 ** 3;
let rangeSec = 300;

/* ---------- Prometheus ---------- */
async function promGet(path, params) {
  const url = `/api/v1/${path}?` + new URLSearchParams(params);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${path}`);
  const body = await res.json();
  if (body.status !== "success") throw new Error(body.error || "query failed");
  return body.data.result;
}
const instant = (query) => promGet("query", { query });
const range = (query, start, end, step) => promGet("query_range", { query, start, end, step });

/** instant vector -> { gpuIndex: value } */
function byGpu(result) {
  const out = {};
  for (const r of result) out[r.metric.gpu] = parseFloat(r.value[1]);
  return out;
}
/** range matrix -> { gpuIndex: [[t, v], ...] } */
function seriesByGpu(result) {
  const out = {};
  for (const r of result) out[r.metric.gpu] = r.values.map(([t, v]) => [t, parseFloat(v)]);
  return out;
}
const scalar = (result) => (result.length ? parseFloat(result[0].value[1]) : NaN);

/* ---------- formatting ---------- */
const fmtGiB = (b) => (isFinite(b) ? (b / GiB).toFixed(2) + " GiB" : "–");
const fmtPct = (v) => (isFinite(v) ? v.toFixed(0) + "%" : "–");
function fmtRate(bps) {
  if (!isFinite(bps)) return "–";
  const u = ["B/s", "KB/s", "MB/s", "GB/s"];
  let i = 0;
  while (bps >= 1000 && i < u.length - 1) { bps /= 1000; i++; }
  return (bps >= 100 ? bps.toFixed(0) : bps.toFixed(1)) + " " + u[i];
}
const fmtNum = (v, d = 1) => (isFinite(v) ? v.toFixed(d) : "–");
const fmtTime = (t) => new Date(t * 1000).toLocaleTimeString([], { hour12: false });

/* ---------- tooltip ---------- */
const tip = document.getElementById("tooltip");
function showTip(x, y, title, rows) {
  tip.innerHTML = (title ? `<div class="t">${title}</div>` : "") + rows.map((r) =>
    `<div class="row"><span class="k"><span class="sw" style="display:inline-block;width:10px;height:${r.line ? 2 : 10}px;border-radius:2px;background:${r.color}"></span>${r.name}</span><span class="v">${r.value}</span></div>`
  ).join("");
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let left = x + 14, top = y + 14;
  if (left + w > window.innerWidth - 8) left = x - w - 14;
  if (top + h > window.innerHeight - 8) top = y - h - 14;
  tip.style.left = left + "px";
  tip.style.top = top + "px";
}
const hideTip = () => { tip.hidden = true; };

/* ---------- canvas line chart (one y-axis, 2px lines, crosshair hover) ---------- */
class LineChart {
  constructor(el, { series, yMax = null, fmt = (v) => fmtNum(v), axisFmt = null }) {
    this.el = el;
    this.series = series; // [{name, color}]
    this.yMax = yMax;
    this.fmt = fmt;
    this.axisFmt = axisFmt || fmt;
    this.data = series.map(() => []);
    this.hoverX = null;
    this.canvas = document.createElement("canvas");
    el.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d");
    new ResizeObserver(() => this.draw()).observe(el);
    this.canvas.addEventListener("mousemove", (e) => this.onMove(e));
    this.canvas.addEventListener("mouseleave", () => { this.hoverX = null; hideTip(); this.draw(); });
  }

  setData(data, start, end) {
    this.data = data;
    this.start = start;
    this.end = end;
    this.draw();
    if (this.hoverX != null && this.lastEvent) this.onMove(this.lastEvent);
  }

  layout() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    return { w, h, l: 58, r: 8, t: 8, b: 20 };
  }

  maxValue() {
    if (this.yMax != null) return this.yMax;
    let m = 0;
    for (const s of this.data) for (const [, v] of s) if (isFinite(v) && v > m) m = v;
    if (m <= 0) return 1;
    const mag = 10 ** Math.floor(Math.log10(m));
    for (const k of [1, 2, 2.5, 5, 10]) if (k * mag >= m * 1.05) return k * mag;
    return 10 * mag;
  }

  draw() {
    const { w, h, l, r, t, b } = this.layout();
    if (!w || !h) return;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.font = "11px -apple-system, Segoe UI, sans-serif";

    const max = this.maxValue();
    this.max = max;
    const pw = w - l - r, ph = h - t - b;
    const X = (ts) => l + ((ts - this.start) / (this.end - this.start)) * pw;
    const Y = (v) => t + ph - (v / max) * ph;
    this.X = X; this.Y = Y;

    // recessive grid + y labels
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.fillStyle = C.muted;
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const f of [0, 0.5, 1]) {
      const y = Math.round(Y(max * f)) + 0.5;
      ctx.beginPath(); ctx.moveTo(l, y); ctx.lineTo(w - r, y); ctx.stroke();
      ctx.fillText(this.axisFmt(max * f), l - 8, y);
    }
    // x labels
    ctx.textBaseline = "alphabetic";
    if (this.start != null) {
      ctx.textAlign = "left"; ctx.fillText(fmtTime(this.start), l, h - 4);
      ctx.textAlign = "right"; ctx.fillText(fmtTime(this.end), w - r, h - 4);
    }

    const hasData = this.data.some((s) => s.some(([, v]) => isFinite(v)));
    if (!hasData) {
      ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText("데이터 없음", l + pw / 2, t + ph / 2);
      return;
    }

    ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.lineCap = "round";
    this.data.forEach((s, i) => {
      ctx.strokeStyle = this.series[i].color;
      ctx.beginPath();
      let pen = false;
      for (const [ts, v] of s) {
        if (!isFinite(v)) { pen = false; continue; }
        const x = X(ts), y = Y(Math.min(v, max));
        pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        pen = true;
      }
      ctx.stroke();
    });

    // crosshair + markers (surface ring)
    if (this.hoverX != null && this.hoverPoints) {
      const x = Math.round(this.hoverX) + 0.5;
      ctx.strokeStyle = C.secondary; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, t); ctx.lineTo(x, t + ph); ctx.stroke();
      this.hoverPoints.forEach((p, i) => {
        if (!p || !isFinite(p[1])) return;
        ctx.beginPath();
        ctx.arc(X(p[0]), Y(Math.min(p[1], max)), 4, 0, Math.PI * 2);
        ctx.fillStyle = this.series[i].color; ctx.fill();
        ctx.lineWidth = 2; ctx.strokeStyle = C.surface; ctx.stroke();
      });
    }
  }

  onMove(e) {
    this.lastEvent = e;
    const rect = this.canvas.getBoundingClientRect();
    const { w, l, r } = this.layout();
    const mx = e.clientX - rect.left;
    if (mx < l || mx > w - r || this.start == null) { this.hoverX = null; hideTip(); this.draw(); return; }
    const ts = this.start + ((mx - l) / (w - l - r)) * (this.end - this.start);
    this.hoverPoints = this.data.map((s) => nearest(s, ts));
    const ref = this.hoverPoints.find(Boolean);
    if (!ref) { hideTip(); return; }
    this.hoverX = this.X(ref[0]);
    this.draw();
    showTip(e.clientX, e.clientY, fmtTime(ref[0]), this.series.map((s, i) => ({
      name: s.name, color: s.color, line: true,
      value: this.hoverPoints[i] ? this.fmt(this.hoverPoints[i][1]) : "–",
    })));
  }
}

function nearest(arr, ts) {
  if (!arr.length) return null;
  let lo = 0, hi = arr.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    arr[mid][0] < ts ? (lo = mid) : (hi = mid);
  }
  return Math.abs(arr[lo][0] - ts) <= Math.abs(arr[hi][0] - ts) ? arr[lo] : arr[hi];
}

/* ---------- legend helper ---------- */
function legendHTML(items) {
  return `<div class="legend">${items.map((it) =>
    `<span class="item"><span class="sw ${it.line ? "line" : ""}" style="background:${it.color}"></span>${it.name}${it.value != null ? ` <span class="v" data-k="${it.key || ""}">${it.value}</span>` : ""}</span>`
  ).join("")}</div>`;
}

/* ---------- request-rate panel ---------- */
const rpsChart = new LineChart(document.getElementById("c-rps"), {
  series: [{ name: "요청/s", color: C.s1 }, { name: "완료/s", color: C.s2 }],
  fmt: (v) => fmtNum(v, 2) + " req/s",
  axisFmt: (v) => fmtNum(v, v < 10 ? 1 : 0),
});
document.getElementById("c-rps").insertAdjacentHTML("beforebegin", legendHTML([
  { name: "들어온 요청/s", color: C.s1, line: true },
  { name: "완료된 요청/s", color: C.s2, line: true },
]));

/* ---------- GPU cards ---------- */
const cards = {};
const SEGMENTS = [
  { key: "model", name: "모델 가중치", cls: "seg-model", color: C.s1 },
  { key: "kvused", name: "KV Cache (사용 중)", cls: "seg-kvused", color: C.s2 },
  { key: "kvfree", name: "KV Cache (예약·미사용)", cls: "seg-kvfree", color: `repeating-linear-gradient(135deg, ${C.s2} 0 2px, transparent 2px 4px)` },
  { key: "other", name: "그 밖의 메모리", cls: "seg-other", color: C.s3 },
  { key: "free", name: "여유", cls: "seg-free", color: css("--free") },
];

function createCard(gpu, name) {
  const el = document.createElement("article");
  el.className = "gpu";
  el.innerHTML = `
    <div class="gpu-head">
      <div class="gpu-title">GPU ${gpu}<span class="name">${name}</span></div>
      <span class="tag" data-k="tag"></span>
    </div>
    <div class="mem-summary"><span>메모리 <b data-k="used">–</b> / <span data-k="total">–</span></span><span data-k="usedpct"></span></div>
    <div class="membar">${SEGMENTS.map((s) => `<div class="${s.cls}" data-seg="${s.key}"></div>`).join("")}</div>
    ${legendHTML(SEGMENTS.map((s) => ({ name: s.name, color: s.color, value: "–", key: s.key })))}
    <div class="minis">
      <div class="mini"><div class="mini-head"><h3>메모리 추이</h3></div>
        ${legendHTML([{ name: "모델", color: C.s1, line: true }, { name: "KV Cache 사용", color: C.s2, line: true }, { name: "그 밖", color: C.s3, line: true }])}
        <div class="chart" data-c="mem"></div></div>
      <div class="mini"><div class="mini-head"><h3>GPU 사용률</h3><span class="now" data-k="util">–</span></div>
        <div class="legend"><span class="item">&nbsp;</span></div>
        <div class="chart" data-c="util"></div></div>
      <div class="mini"><div class="mini-head"><h3>메모리 Read/Write 활동</h3><span class="now" data-k="rw">–</span></div>
        <div class="legend"><span class="item">읽기·쓰기가 일어난 시간 비율</span></div>
        <div class="chart" data-c="rw"></div></div>
      <div class="mini"><div class="mini-head"><h3>PCIe Read/Write</h3><span class="now" data-k="pcie">–</span></div>
        ${legendHTML([{ name: "RX (Host→GPU)", color: C.s1, line: true }, { name: "TX (GPU→Host)", color: C.s2, line: true }])}
        <div class="chart" data-c="pcie"></div></div>
    </div>`;
  document.getElementById("gpus").appendChild(el);

  const q = (sel) => el.querySelector(sel);
  const card = {
    el, q,
    mem: new LineChart(q('[data-c="mem"]'), {
      series: [{ name: "모델", color: C.s1 }, { name: "KV Cache 사용", color: C.s2 }, { name: "그 밖", color: C.s3 }],
      fmt: fmtGiB, axisFmt: (v) => (v / GiB).toFixed(1) + "G",
    }),
    util: new LineChart(q('[data-c="util"]'), { series: [{ name: "GPU 사용률", color: C.s1 }], yMax: 100, fmt: fmtPct }),
    rw: new LineChart(q('[data-c="rw"]'), { series: [{ name: "메모리 R/W", color: C.s1 }], yMax: 100, fmt: fmtPct }),
    pcie: new LineChart(q('[data-c="pcie"]'), {
      series: [{ name: "RX", color: C.s1 }, { name: "TX", color: C.s2 }], fmt: fmtRate,
      axisFmt: (v) => fmtRate(v).replace("/s", ""),
    }),
    values: {},
  };

  // hover tooltip on bar segments
  el.querySelectorAll(".membar > div").forEach((segEl) => {
    segEl.addEventListener("mousemove", (e) => {
      const s = SEGMENTS.find((x) => x.key === segEl.dataset.seg);
      const v = card.values[s.key], total = card.values.total;
      showTip(e.clientX, e.clientY, `GPU ${gpu}`, [{ name: s.name, color: s.color, value: `${fmtGiB(v)} (${fmtPct((v / total) * 100)})` }]);
    });
    segEl.addEventListener("mouseleave", hideTip);
  });
  return card;
}

function updateCard(card, v) {
  const { q } = card;
  const kvfree = Math.max((v.kvalloc || 0) - (v.kvused || 0), 0);
  const free = Math.max(v.total - v.used, 0);
  card.values = { model: v.model || 0, kvused: v.kvused || 0, kvfree, other: v.other || 0, free, total: v.total };
  for (const s of SEGMENTS) {
    const val = card.values[s.key];
    q(`[data-seg="${s.key}"]`).style.flex = `0 0 calc(${(val / v.total) * 100}% - 2px)`;
    q(`.v[data-k="${s.key}"]`).textContent = fmtGiB(val);
  }
  q('[data-k="used"]').textContent = fmtGiB(v.used);
  q('[data-k="total"]').textContent = fmtGiB(v.total);
  q('[data-k="usedpct"]').textContent = `${fmtPct((v.used / v.total) * 100)} 사용`;
  const tag = q('[data-k="tag"]');
  tag.textContent = v.assigned ? "vLLM 사용 중" : "vLLM 미할당";
  tag.classList.toggle("on", !!v.assigned);
  q('[data-k="util"]').textContent = fmtPct(v.util);
  q('[data-k="rw"]').textContent = fmtPct(v.rw);
  q('[data-k="pcie"]').textContent = isFinite(v.rx) || isFinite(v.tx)
    ? `↓${fmtRate(v.rx)}  ↑${fmtRate(v.tx)}` : "미지원";
}

/* ---------- refresh loop ---------- */
const INSTANT = {
  total: "kvmon_gpu_memory_total_bytes",
  used: "kvmon_gpu_memory_used_bytes",
  model: "kvmon_gpu_memory_model_bytes",
  kvalloc: "kvmon_gpu_memory_kv_cache_allocated_bytes",
  kvused: "kvmon_gpu_memory_kv_cache_used_bytes",
  other: "kvmon_gpu_memory_other_bytes",
  assigned: "kvmon_gpu_vllm_assigned",
  util: "kvmon_gpu_utilization_percent",
  rw: "kvmon_gpu_memory_rw_utilization_percent",
  rx: "kvmon_gpu_pcie_rx_bytes_per_second",
  tx: "kvmon_gpu_pcie_tx_bytes_per_second",
};
const RANGE = {
  model: INSTANT.model, kvused: INSTANT.kvused, other: INSTANT.other,
  util: INSTANT.util, rw: INSTANT.rw, rx: INSTANT.rx, tx: INSTANT.tx,
};
const RPS = "sum(rate(kvmon_vllm_requests_received_total[15s])) or vector(0)";
const DONE = "sum(rate(vllm:request_success_total[15s])) or vector(0)";
const GLOBAL = {
  rps: RPS,
  done: DONE,
  running: "sum(vllm:num_requests_running)",
  waiting: "sum(vllm:num_requests_waiting)",
  kv: "max(vllm:kv_cache_usage_perc) or max(vllm:gpu_cache_usage_perc)",
  models: "count by (model_name) (vllm:num_requests_running)",
};

const status = document.getElementById("status");
const statusText = document.getElementById("status-text");

async function refresh() {
  const end = Date.now() / 1000;
  const start = end - rangeSec;
  const step = Math.max(2, Math.round(rangeSec / 200));
  try {
    const instKeys = Object.keys(INSTANT), rangeKeys = Object.keys(RANGE), globKeys = Object.keys(GLOBAL);
    const [inst, rng, glob, rpsR, doneR] = await Promise.all([
      Promise.all(instKeys.map((k) => instant(INSTANT[k]))),
      Promise.all(rangeKeys.map((k) => range(RANGE[k], start, end, step))),
      Promise.all(globKeys.map((k) => instant(GLOBAL[k]))),
      range(RPS, start, end, step),
      range(DONE, start, end, step),
    ]);
    const I = Object.fromEntries(instKeys.map((k, i) => [k, byGpu(inst[i])]));
    const R = Object.fromEntries(rangeKeys.map((k, i) => [k, seriesByGpu(rng[i])]));
    const G = Object.fromEntries(globKeys.map((k, i) => [k, glob[i]]));

    // global tiles
    document.getElementById("t-rps").textContent = fmtNum(scalar(G.rps), 2);
    document.getElementById("t-done").textContent = fmtNum(scalar(G.done), 2);
    document.getElementById("t-running").textContent = fmtNum(scalar(G.running), 0);
    document.getElementById("t-waiting").textContent = fmtNum(scalar(G.waiting), 0);
    document.getElementById("t-kv").textContent = fmtNum(scalar(G.kv) * 100, 1);
    const models = G.models.map((r) => r.metric.model_name).filter(Boolean);
    if (models.length) document.getElementById("model-name").textContent = `모델: ${models.join(", ")}`;

    const toSeries = (res) => (res[0] ? res[0].values.map(([t, v]) => [t, parseFloat(v)]) : []);
    rpsChart.setData([toSeries(rpsR), toSeries(doneR)], start, end);

    // per-GPU cards, ordered by index
    const meta = Object.fromEntries(inst[0].map((r) => [r.metric.gpu, r.metric.name]));
    const gpus = Object.keys(I.total).sort((a, b) => a - b);
    document.getElementById("empty").hidden = gpus.length > 0;
    for (const g of gpus) {
      const card = cards[g] || (cards[g] = createCard(g, meta[g] || ""));
      updateCard(card, Object.fromEntries(instKeys.map((k) => [k, I[k][g]])));
      const s = (k) => R[k][g] || [];
      card.mem.setData([s("model"), s("kvused"), s("other")], start, end);
      card.util.setData([s("util")], start, end);
      card.rw.setData([s("rw")], start, end);
      card.pcie.setData([s("rx"), s("tx")], start, end);
    }

    status.className = "status ok";
    statusText.textContent = `업데이트 ${fmtTime(end)}`;
  } catch (err) {
    status.className = "status err";
    statusText.textContent = `Prometheus 연결 실패 (${err.message})`;
  }
}

document.getElementById("range").addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (!btn) return;
  rangeSec = +btn.dataset.range;
  document.querySelectorAll("#range button").forEach((b) => b.classList.toggle("on", b === btn));
  refresh();
});

refresh();
setInterval(refresh, REFRESH_MS);
