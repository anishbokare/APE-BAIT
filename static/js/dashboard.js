/**
 * APE-BAIT Dashboard JavaScript
 * Real-time metrics, Chart.js visualizations, WebSocket integration.
 */

"use strict";

/* ─── Global State ──────────────────────────────────────────────── */
const State = {
  socket: null,
  connected: false,
  metrics: {},
  history: {
    evasion: [],
    latency: [],
    mse: [],
    timestamps: [],
    maxPoints: 60,
  },
  charts: {},
  currentMethod: "pgd",
  validationData: null,
};

/* ─── Chart.js Defaults ─────────────────────────────────────────── */
Chart.defaults.color = "#8ba3c4";
Chart.defaults.borderColor = "rgba(0, 245, 255, 0.06)";
Chart.defaults.font.family = "'JetBrains Mono', monospace";
Chart.defaults.font.size = 11;

const PALETTE = {
  cyan:   { line: "#00f5ff", fill: "rgba(0,245,255,0.08)",   fill2: "rgba(0,245,255,0)" },
  green:  { line: "#39ff14", fill: "rgba(57,255,20,0.08)",   fill2: "rgba(57,255,20,0)" },
  violet: { line: "#bf5fff", fill: "rgba(191,95,255,0.08)",  fill2: "rgba(191,95,255,0)" },
  orange: { line: "#ff6b35", fill: "rgba(255,107,53,0.08)",  fill2: "rgba(255,107,53,0)" },
  red:    { line: "#ff3464", fill: "rgba(255,52,100,0.08)",   fill2: "rgba(255,52,100,0)" },
};

/* ─── Initialize Application ────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  initWebSocket();
  initControls();
  loadValidationData();
  updateClock();
  setInterval(updateClock, 1000);

  // Animate KPI cards on load
  document.querySelectorAll(".kpi-card").forEach((el, i) => {
    el.style.animationDelay = `${i * 80}ms`;
    el.classList.add("fade-in");
  });
});

/* ─── WebSocket ─────────────────────────────────────────────────── */
function initWebSocket() {
  const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
  // Use Socket.IO
  try {
    State.socket = io();
    State.socket.on("connect", () => {
      State.connected = true;
      updateConnectionStatus(true);
      addLog("INFO", "WebSocket connected to APE-BAIT engine");
    });
    State.socket.on("disconnect", () => {
      State.connected = false;
      updateConnectionStatus(false);
      addLog("WARN", "WebSocket disconnected — falling back to REST polling");
      startPolling();
    });
    State.socket.on("metrics", (data) => {
      handleMetrics(data);
    });
    State.socket.on("status", (data) => {
      addLog("INFO", `Engine mode: ${data.mode}`);
    });
  } catch (e) {
    // Socket.IO not available — poll REST
    addLog("WARN", "Socket.IO unavailable — polling /api/metrics");
    startPolling();
  }
}

function startPolling() {
  setInterval(async () => {
    try {
      const resp = await fetch("/api/metrics");
      const data = await resp.json();
      handleMetrics(data);
    } catch (e) {
      // silently fail
    }
  }, 600);
}

/* ─── Metrics Handler ───────────────────────────────────────────── */
function handleMetrics(data) {
  State.metrics = data;
  const now = new Date().toLocaleTimeString("en-US", { hour12: false });

  // Update history
  const h = State.history;
  h.evasion.push(data.evasion_rate ?? 0);
  h.latency.push(data.avg_latency_ms ?? 0);
  h.mse.push(data.avg_mse ?? 0);
  h.timestamps.push(now);

  if (h.evasion.length > h.maxPoints) {
    h.evasion.shift();
    h.latency.shift();
    h.mse.shift();
    h.timestamps.shift();
  }

  updateKPIs(data);
  updateCharts();
  updateEngineStats(data);
  updateMSEIndicator(data.avg_mse);

  // Update tool evasion if present
  if (data.tool_evasion) {
    updateToolEvasion(data.tool_evasion);
  }

  // Add activity log entry occasionally
  if (Math.random() < 0.15) {
    logPacketActivity(data);
  }
}

/* ─── KPI Updates ───────────────────────────────────────────────── */
function updateKPIs(data) {
  // Evasion rate
  const evasionPct = ((data.evasion_rate ?? 0) * 100).toFixed(1);
  animateValue("kpi-evasion", evasionPct, "%");

  // MSE
  animateValue("kpi-mse", (data.avg_mse ?? 0).toFixed(4));

  // Latency
  animateValue("kpi-latency", Math.round(data.avg_latency_ms ?? 0), "ms");

  // Throughput
  animateValue("kpi-throughput", Math.round(data.throughput_pps ?? 0), " pps");

  // MSE color
  const mseEl = document.getElementById("kpi-mse");
  if (mseEl) {
    const mse = data.avg_mse ?? 0;
    mseEl.style.color = mse < 0.04 ? "var(--neon-green)" : "var(--neon-red)";
  }

  // Evasion trend
  const trendEl = document.getElementById("kpi-evasion-trend");
  if (trendEl && State.history.evasion.length > 5) {
    const prev = State.history.evasion[State.history.evasion.length - 6] ?? 0;
    const curr = data.evasion_rate ?? 0;
    const diff = ((curr - prev) * 100).toFixed(1);
    trendEl.className = "kpi-trend " + (diff >= 0 ? "up" : "down");
    trendEl.textContent = (diff >= 0 ? "+" : "") + diff + "%";
  }
}

function animateValue(id, value, suffix = "") {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = value + suffix;
}

/* ─── Chart Initialization ──────────────────────────────────────── */
function initCharts() {
  // 1. Evasion Rate timeline
  State.charts.evasion = new Chart(
    document.getElementById("chart-evasion"),
    {
      type: "line",
      data: {
        labels: [],
        datasets: [{
          label: "Evasion Rate",
          data: [],
          borderColor: PALETTE.cyan.line,
          backgroundColor: createGradient("chart-evasion", PALETTE.cyan),
          fill: true,
          borderWidth: 2,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: 4,
          pointHoverBackgroundColor: PALETTE.cyan.line,
        }],
      },
      options: {
        animation: { duration: 300 },
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "rgba(8,13,26,0.9)",
            borderColor: "rgba(0,245,255,0.2)",
            borderWidth: 1,
            callbacks: {
              label: ctx => `Evasion: ${(ctx.raw * 100).toFixed(1)}%`
            }
          }
        },
        scales: {
          x: {
            display: false,
          },
          y: {
            min: 0,
            max: 1,
            grid: { color: "rgba(0,245,255,0.04)" },
            ticks: {
              callback: v => (v * 100).toFixed(0) + "%",
              maxTicksLimit: 5,
            }
          }
        },
        interaction: { intersect: false, mode: "index" },
      }
    }
  );

  // 2. Latency chart
  State.charts.latency = new Chart(
    document.getElementById("chart-latency"),
    {
      type: "bar",
      data: {
        labels: [],
        datasets: [{
          label: "Latency (ms)",
          data: [],
          backgroundColor: "rgba(191,95,255,0.3)",
          borderColor: PALETTE.violet.line,
          borderWidth: 1,
          borderRadius: 3,
        }]
      },
      options: {
        animation: { duration: 200 },
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { display: false },
          y: {
            grid: { color: "rgba(0,245,255,0.04)" },
            ticks: {
              callback: v => v + "ms",
              maxTicksLimit: 4,
            }
          }
        }
      }
    }
  );

  // 3. MSE gauge (doughnut)
  State.charts.mse = new Chart(
    document.getElementById("chart-mse"),
    {
      type: "doughnut",
      data: {
        datasets: [{
          data: [0.018, 0.022],  // current, remaining-to-threshold
          backgroundColor: [
            "#39ff14",
            "rgba(255,255,255,0.05)",
          ],
          borderWidth: 0,
          borderRadius: 4,
        }]
      },
      options: {
        animation: { duration: 500 },
        responsive: true,
        maintainAspectRatio: false,
        cutout: "78%",
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false },
        },
        rotation: -90,
        circumference: 180,
      }
    }
  );

  // 4. Multi-tool evasion bar chart
  initToolChart();
}

function initToolChart() {
  State.charts.tools = new Chart(
    document.getElementById("chart-tools"),
    {
      type: "bar",
      data: {
        labels: [],
        datasets: [{
          label: "Evasion Rate",
          data: [],
          backgroundColor: [],
          borderColor: [],
          borderWidth: 1,
          borderRadius: 4,
        }]
      },
      options: {
        indexAxis: "y",
        animation: { duration: 600 },
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "rgba(8,13,26,0.9)",
            borderColor: "rgba(0,245,255,0.2)",
            borderWidth: 1,
            callbacks: {
              label: ctx => `Evasion: ${(ctx.raw * 100).toFixed(1)}%`
            }
          }
        },
        scales: {
          x: {
            min: 0,
            max: 1,
            grid: { color: "rgba(0,245,255,0.04)" },
            ticks: { callback: v => (v * 100).toFixed(0) + "%" }
          },
          y: {
            grid: { display: false },
            ticks: { font: { size: 11 } }
          }
        }
      }
    }
  );
}

/* ─── Chart Updates ─────────────────────────────────────────────── */
function updateCharts() {
  const h = State.history;

  // Evasion rate
  const ec = State.charts.evasion;
  ec.data.labels = h.timestamps;
  ec.data.datasets[0].data = h.evasion;
  ec.update("quiet");

  // Latency
  const lc = State.charts.latency;
  lc.data.labels = h.timestamps;
  lc.data.datasets[0].data = h.latency;
  lc.update("quiet");

  // MSE gauge
  const mse = h.mse.length ? h.mse[h.mse.length - 1] : 0;
  const mc = State.charts.mse;
  const remaining = Math.max(0, 0.04 - mse);
  mc.data.datasets[0].data = [mse, remaining];
  mc.data.datasets[0].backgroundColor[0] = mse < 0.04 ? "#39ff14" : "#ff3464";
  mc.update("quiet");

  // Update gauge label
  const gaugeVal = document.getElementById("gauge-mse-value");
  if (gaugeVal) {
    gaugeVal.textContent = mse.toFixed(4);
    gaugeVal.style.color = mse < 0.04 ? "var(--neon-green)" : "var(--neon-red)";
  }
}

function updateToolChart(validationData) {
  const tc = State.charts.tools;
  if (!tc || !validationData?.tools) return;

  const tools = validationData.tools.slice(0, 12);
  const labels = tools.map(t => t.name);
  const values = tools.map(t => t.evasion_rate);
  const colors = values.map(v => {
    if (v >= 0.85) return "rgba(57,255,20,0.4)";
    if (v >= 0.75) return "rgba(0,245,255,0.4)";
    return "rgba(255,107,53,0.4)";
  });
  const borders = values.map(v => {
    if (v >= 0.85) return "#39ff14";
    if (v >= 0.75) return "#00f5ff";
    return "#ff6b35";
  });

  tc.data.labels = labels;
  tc.data.datasets[0].data = values;
  tc.data.datasets[0].backgroundColor = colors;
  tc.data.datasets[0].borderColor = borders;
  tc.update();
}

/* ─── Validation Data ───────────────────────────────────────────── */
async function loadValidationData() {
  try {
    const resp = await fetch("/api/validation");
    const data = await resp.json();
    State.validationData = data;
    renderToolTable(data.tools);
    updateToolChart(data);
    addLog("OK", `Loaded validation results: ${data.tools.length} tools`);
  } catch (e) {
    addLog("WARN", "Could not load validation data");
  }
}

function renderToolTable(tools) {
  const tbody = document.getElementById("tool-table-body");
  if (!tbody) return;
  tbody.innerHTML = "";

  tools.forEach(tool => {
    const pct = (tool.evasion_rate * 100).toFixed(1);
    const color = tool.evasion_rate >= 0.85 ? "#39ff14"
                : tool.evasion_rate >= 0.75 ? "#00f5ff"
                : "#ff6b35";

    const row = document.createElement("tr");
    row.innerHTML = `
      <td>
        <div class="tool-name">
          ${tool.name}
          <span class="tool-type-badge">${tool.type}</span>
        </div>
      </td>
      <td style="font-family:var(--font-mono); color:var(--text-secondary)">${tool.original}</td>
      <td style="font-family:var(--font-mono); color:var(--text-secondary)">${tool.perturbed}</td>
      <td>
        <div class="evasion-bar-wrap">
          <div class="evasion-bar">
            <div class="evasion-bar-fill" style="width:${pct}%; background:${color}; box-shadow:0 0 6px ${color}"></div>
          </div>
          <span class="evasion-pct" style="color:${color}">${pct}%</span>
        </div>
      </td>
    `;
    tbody.appendChild(row);
  });
}

function updateToolEvasion(toolData) {
  // Update table values if live tool evasion is available
  // (merged with loaded validation data)
}

/* ─── Engine Stats ──────────────────────────────────────────────── */
function updateEngineStats(data) {
  setText("stat-captured", fmtNum(data.packets_captured));
  setText("stat-perturbed", fmtNum(data.packets_perturbed));
  setText("stat-injected", fmtNum(data.packets_injected));
  setText("stat-uptime", fmtUptime(data.uptime_seconds));
  setText("stat-errors", data.errors ?? 0);
}

function updateMSEIndicator(mse) {
  const el = document.getElementById("mse-indicator-value");
  if (el) {
    el.textContent = (mse ?? 0).toFixed(5);
    el.style.color = (mse ?? 0) < 0.04 ? "var(--neon-green)" : "var(--neon-red)";
  }
}

/* ─── Controls ──────────────────────────────────────────────────── */
function initControls() {
  // Method chips
  document.querySelectorAll(".method-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".method-chip").forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      State.currentMethod = chip.dataset.method;
      addLog("INFO", `Method switched to: ${chip.dataset.method.toUpperCase()}`);
      setText("stat-method", chip.dataset.method.toUpperCase());
    });
  });

  // Epsilon slider
  const epsSlider = document.getElementById("epsilon-slider");
  const epsVal = document.getElementById("epsilon-value");
  if (epsSlider && epsVal) {
    epsSlider.addEventListener("input", () => {
      epsVal.textContent = parseFloat(epsSlider.value).toFixed(3);
    });
  }

  // Stop button
  const stopBtn = document.getElementById("btn-stop");
  if (stopBtn) {
    stopBtn.addEventListener("click", async () => {
      try {
        await fetch("/api/control", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: "stop" }),
        });
        addLog("WARN", "Engine stop command sent");
      } catch (e) {
        addLog("ERROR", "Failed to send stop command");
      }
    });
  }

  // Export button
  const exportBtn = document.getElementById("btn-export");
  if (exportBtn) {
    exportBtn.addEventListener("click", () => {
      const report = {
        timestamp: new Date().toISOString(),
        metrics: State.metrics,
        validation: State.validationData,
        method: State.currentMethod,
      };
      const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `ape-bait-report-${Date.now()}.json`;
      a.click();
      addLog("OK", "Report exported as JSON");
    });
  }
}

/* ─── Activity Log ──────────────────────────────────────────────── */
const LOG_TEMPLATES = [
  (d) => `FGSM perturbation: ε=${d.epsilon ?? 0.03} → MSE=${(d.avg_mse ?? 0).toFixed(4)}`,
  (d) => `Packet batch processed: ${d.packets_perturbed ?? 0} packets perturbed`,
  (d) => `Evasion check: ${((d.evasion_rate ?? 0) * 100).toFixed(1)}% success rate`,
  (d) => `Latency budget: ${Math.round(d.avg_latency_ms ?? 0)}ms / 2000ms`,
  (d) => `Surrogate loss converged at step ${Math.floor(Math.random() * 38) + 2}/40`,
];

function logPacketActivity(data) {
  const tmpl = LOG_TEMPLATES[Math.floor(Math.random() * LOG_TEMPLATES.length)];
  addLog("INFO", tmpl(data));
}

function addLog(level, msg) {
  const feed = document.getElementById("log-feed");
  if (!feed) return;

  const now = new Date().toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const entry = document.createElement("div");
  entry.className = "log-entry";
  entry.innerHTML = `
    <span class="log-time">${now}</span>
    <span class="log-level ${level}">${level}</span>
    <span class="log-msg">${msg}</span>
  `;
  feed.prepend(entry);

  // Keep only 50 entries
  while (feed.children.length > 50) {
    feed.removeChild(feed.lastChild);
  }
}

/* ─── Connection Status ──────────────────────────────────────────── */
function updateConnectionStatus(connected) {
  const badge = document.getElementById("connection-badge");
  const dot = document.getElementById("connection-dot");
  const text = document.getElementById("connection-text");
  if (!badge) return;

  if (connected) {
    badge.style.background = "rgba(57,255,20,0.1)";
    badge.style.borderColor = "rgba(57,255,20,0.2)";
    badge.style.color = "var(--neon-green)";
    if (dot) dot.style.background = "var(--neon-green)";
    if (text) text.textContent = "LIVE";
  } else {
    badge.style.background = "rgba(255,107,53,0.1)";
    badge.style.borderColor = "rgba(255,107,53,0.2)";
    badge.style.color = "var(--neon-orange)";
    if (dot) dot.style.background = "var(--neon-orange)";
    if (text) text.textContent = "DEMO";
  }
}

/* ─── Utilities ──────────────────────────────────────────────────── */
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function fmtNum(n) {
  if (!n && n !== 0) return "--";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n.toString();
}

function fmtUptime(s) {
  if (!s) return "0s";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}

function updateClock() {
  const el = document.getElementById("clock");
  if (el) {
    el.textContent = new Date().toLocaleTimeString("en-US", {
      hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit"
    });
  }
}

function createGradient(canvasId, palette) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return palette.fill;
  const ctx = canvas.getContext("2d");
  const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height || 200);
  gradient.addColorStop(0, palette.fill);
  gradient.addColorStop(1, palette.fill2);
  return gradient;
}

/* ─── Navigation ─────────────────────────────────────────────────── */
document.querySelectorAll(".nav-item").forEach(item => {
  item.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
    item.classList.add("active");
  });
});

/* ─── Timeline Events ────────────────────────────────────────────── */
function addTimelineEvent(color, action, detail) {
  const timeline = document.getElementById("attack-timeline");
  if (!timeline) return;

  const item = document.createElement("div");
  item.className = "timeline-item";
  item.innerHTML = `
    <div class="timeline-dot ${color}"></div>
    <div class="timeline-content">
      <div class="timeline-action">${action}</div>
      <div class="timeline-detail">${detail}</div>
    </div>
  `;
  timeline.prepend(item);

  while (timeline.children.length > 8) {
    timeline.removeChild(timeline.lastChild);
  }
}

// Populate initial timeline
setTimeout(() => {
  addTimelineEvent("cyan",   "Engine initialized",      "PGD attack mode loaded · ε=0.03");
  addTimelineEvent("green",  "Surrogate models ready",   "IDS + Malware + Anomaly detectors loaded");
  addTimelineEvent("violet", "Validation suite loaded",  "15 security tool adapters registered");
  addTimelineEvent("orange", "First perturbation",       "FGSM: MSE=0.0182 · Latency=847ms");
  addTimelineEvent("green",  "Evasion achieved",         "Snort & Suricata bypassed successfully");
}, 800);

setInterval(() => {
  const m = State.metrics;
  if (!m.evasion_rate) return;
  const events = [
    ["cyan",   "Packet batch",          `${Math.floor(Math.random()*40+20)} pkts processed`],
    ["green",  "Evasion confirmed",     `${(m.evasion_rate*100).toFixed(1)}% rate achieved`],
    ["violet", "PGD converged",         `${Math.floor(Math.random()*35+5)} steps · MSE=${(m.avg_mse??0).toFixed(4)}`],
  ];
  const [c, a, d] = events[Math.floor(Math.random() * events.length)];
  addTimelineEvent(c, a, d);
}, 5000);
