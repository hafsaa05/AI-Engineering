// Results section (below the race grid): summary table, bar charts, key findings and per puzzle results for one set.

const SET_NAMES = { ollama: "Ollama", openrouter: "OpenRouter" };
const MODE_NAMES = { mock: "mock models", ollama: "local Ollama models", openrouter: "OpenRouter models" };
const charts = {};  // canvas id -> Chart, so we can redraw them

function fixed(value, digits, unit = "") {
  return value === null || value === undefined ? "-" : `${value.toFixed(digits)}${unit}`;
}

function cell(text, className) {
  const td = document.createElement("td");
  td.textContent = text;
  if (className) td.className = className;
  return td;
}

// ---- Summary table ----

function bestValue(summary, key, highest) {
  const values = summary.map((row) => row[key]).filter((v) => v !== null);
  if (new Set(values).size < 2) return null;  // nothing to highlight if every model has the same value
  return highest ? Math.max(...values) : Math.min(...values);
}

function renderSummary(summary) {
  const body = el("summary-body");
  body.replaceChildren();
  if (!summary.length) {
    const td = cell("No results for this set yet. Run a benchmark, or switch the Models filter.", "empty-note");
    td.colSpan = 12;
    body.appendChild(document.createElement("tr")).appendChild(td);
    return;
  }

  // Highlight the best value in the columns that are charted.
  const best = {
    avg_latency: bestValue(summary, "avg_latency", false),
    avg_total_time: bestValue(summary, "avg_total_time", false),
    success_rate: bestValue(summary, "success_rate", true),
    avg_optimality_gap: bestValue(summary, "avg_optimality_gap", false),
  };
  const mark = (row, key) => (row[key] !== null && row[key] === best[key] ? "best" : "");

  summary.forEach((row, lane) => {
    const tr = document.createElement("tr");
    const name = cell("", "model-cell");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = laneColor(lane);
    name.append(swatch, row.model);

    tr.append(
      name,
      cell(row.runs),
      cell(fixed(row.avg_latency, 3, "s"), mark(row, "avg_latency")),
      cell(fixed(row.avg_total_time, 2, "s"), mark(row, "avg_total_time")),
      cell(fixed(row.avg_moves, 1)),
      cell(fixed(row.success_rate, 0, "%"), mark(row, "success_rate")),
      cell(fixed(row.avg_optimality_gap, 2), mark(row, "avg_optimality_gap")),
      cell(row.invalid_outputs),
      cell(row.retries),
      cell(row.api_calls ?? 0),
      cell(row.cache_hits ?? 0),
      cell(`$${(row.total_cost ?? 0).toFixed(4)}`),
    );
    body.appendChild(tr);
  });
}

// ---- Charts ----

// Writes each bar's value above it, so a 0% bar is still visible as "0%".
const percentLabels = {
  id: "percentLabels",
  afterDatasetsDraw(chart) {
    const ctx = chart.ctx;
    ctx.save();
    ctx.font = `bold 13px ${FONT}`;
    ctx.fillStyle = "#1e2a3a";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    chart.getDatasetMeta(0).data.forEach((bar, i) => {
      const value = chart.data.datasets[0].data[i];
      if (value !== null) ctx.fillText(`${Math.round(value)}%`, bar.x, bar.y - 4);
    });
    ctx.restore();
  },
};

function drawChart(canvasId, summary, key, max, plugins = []) {
  if (charts[canvasId]) charts[canvasId].destroy();
  charts[canvasId] = new Chart(el(canvasId), {
    type: "bar",
    data: {
      labels: summary.map((row) => row.model),
      datasets: [{
        data: summary.map((row) => row[key]),
        backgroundColor: summary.map((_, lane) => laneColor(lane)),
        borderRadius: 6,
        maxBarThickness: 56,
      }],
    },
    plugins,
    options: {
      layout: { padding: { top: 20 } },  // room for the value labels
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: "#1f2a37", titleFont: { family: FONT }, bodyFont: { family: FONT }, padding: 10 },
      },
      scales: {
        y: { beginAtZero: true, max, border: { display: false }, grid: { color: "#f0ebe2" },
             ticks: { color: "#7b7468", font: { family: FONT, size: 12 } } },
        x: { border: { color: "#ebe5da" }, grid: { display: false },
             ticks: { color: "#1f2a37", font: { family: FONT, size: 13 } } },
      },
    },
  });
}

function renderCharts(summary) {
  drawChart("chart-latency", summary, "avg_latency");
  drawChart("chart-time", summary, "avg_total_time");
  drawChart("chart-success", summary, "success_rate", 100, [percentLabels]);
  drawChart("chart-gap", summary, "avg_optimality_gap");
  // The gap only exists for solved runs, so say so instead of showing an empty chart.
  el("gap-note").hidden = !summary.length || summary.some((row) => row.avg_optimality_gap !== null);
}

// ---- Key findings ----

function renderFindings(findings) {
  const list = el("findings");
  list.replaceChildren();
  if (!findings.length) {
    list.appendChild(document.createElement("li")).textContent = "Findings appear here once there are results.";
    return;
  }
  for (const finding of findings) {
    const item = document.createElement("li");
    const label = document.createElement("b");
    label.textContent = `${finding.label}: `;
    item.append(label, finding.text);
    list.appendChild(item);
  }
}

// ---- Per puzzle table ----

function renderPuzzleTable(rows, models) {
  const headRow = document.createElement("tr");
  for (const title of ["Puzzle", "Scramble", "Optimal", ...models]) {
    headRow.appendChild(document.createElement("th")).textContent = title;
  }
  el("puzzle-head").replaceChildren(headRow);

  const body = el("puzzle-body");
  body.replaceChildren();
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.append(
      cell(row.puzzle_id, "model-cell"),
      cell(row.scramble.join(" ")),
      cell(row.optimal_distance ?? "-"),
    );
    for (const model of models) {
      const run = row.cells[model];
      if (!run) {
        tr.appendChild(cell("-"));
        continue;
      }
      const td = cell(`${run.moves} moves`, run.solved ? "cell-solved" : "cell-failed");
      const note = document.createElement("span");
      note.className = "cell-small";
      note.textContent = run.solved ? "solved" : run.end_reason.toLowerCase();
      td.appendChild(note);
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
}

// ---- Loading and controls ----

// Shows the results of the race screen's current mode (status and modeSet come from app.js).
async function loadResults() {
  const mode = status.mode;
  const set = modeSet();
  showBenchmarkButton(status);
  el("results-for").textContent = `${SET_NAMES[set]} set, ${MODE_NAMES[mode]}`;
  const data = await getJson(`/api/dashboard?set=${set}&mode=${mode}`);
  el("slide-title").textContent =
    `Results: ${SET_NAMES[set]} set, ${data.puzzles.length} puzzles (${MODE_NAMES[mode]})`;

  renderSummary(data.summary);
  renderCharts(data.summary);
  renderFindings(data.findings);
  renderPuzzleTable(data.puzzles, data.models);
}

// Whole-set benchmarks are switched off in OpenRouter mode: only single confirmed races there.
function showBenchmarkButton(info) {
  const button = el("benchmark-button");
  button.disabled = Boolean(info.busy) || info.mode === "openrouter";
  button.title = info.mode === "openrouter" ? "Switched off in OpenRouter mode (only single live races)" : "";
}

async function followBenchmark() {
  const info = await getJson("/api/status");
  const progress = info.benchmark;
  showBenchmarkButton(info);

  if (progress.running) {
    el("benchmark-text").textContent =
      `Running ${progress.set}: ${progress.done} / ${progress.total} done (now ${progress.current})`;
    setTimeout(followBenchmark, 1000);
  } else if (progress.set) {
    el("benchmark-text").textContent = progress.error
      ? `Benchmark stopped: ${progress.error}`
      : `Finished ${progress.set}: ${progress.done} / ${progress.total} puzzles`;
    loadResults();
  }
}

async function startBenchmark() {
  try {
    await getJson(`/api/benchmark/${modeSet()}`, { method: "POST" });
    followBenchmark();
  } catch (error) {
    el("benchmark-text").textContent = error.message;
  }
}

function setSlideMode(on) {
  document.body.classList.toggle("slide-mode", on);
  window.scrollTo(0, 0);
  if (on && document.documentElement.requestFullscreen) {
    document.documentElement.requestFullscreen().catch(() => {});  // fine if the browser says no
  } else if (!on && document.fullscreenElement) {
    document.exitFullscreen();
  }
}

// The first loadResults() call comes from app.js once it knows the mode.
async function initResults() {
  try {
    const info = await getJson("/api/status");
    if (info.benchmark.running) followBenchmark();
  } catch (error) {
    return;  // app.js already shows "Could not reach the server"
  }

  el("benchmark-button").addEventListener("click", startBenchmark);
  el("slide-button").addEventListener("click", () => setSlideMode(true));
  el("slide-exit").addEventListener("click", () => setSlideMode(false));
  // Pressing Esc leaves full screen; leave slide mode with it.
  document.addEventListener("fullscreenchange", () => {
    if (!document.fullscreenElement) document.body.classList.remove("slide-mode");
  });
}

initResults();
