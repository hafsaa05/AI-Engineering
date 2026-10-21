// Race screen: loads puzzles, runs live races or replays over a WebSocket,
// and draws one panel per model.

const FACES = ["U", "R", "F", "D", "L", "B"];
const KIND_STYLES = { "LLM": "llm", "DECISION MODEL": "decision", "MOCK": "mock" };
const RANK_SUFFIX = { 1: "ST", 2: "ND", 3: "RD" };
// The GUI offers only these modes; mock (and the warmup, primary and extension sets) stay for the tests.
const GUI_MODES = ["ollama", "openrouter"];
// Ollama panels are shown in the order the models take their turns (left to right, top to bottom).
const OLLAMA_PANEL_ORDER = ["Gemma3 1B", "Qwen3.5 0.8B", "DeepSeek-R1 1.5B", "Clef Flash"];

let status = null;       // from /api/status
let allPuzzles = [];
let puzzles = [];        // only the puzzles of the current mode
let panels = new Map();  // model name -> panel object
let socket = null;
let raceSpeed = 1;       // 1 for live races, the chosen speed for replays

// ---- Cube drawing ----

function makeFace(state, face) {
  const div = document.createElement("div");
  div.className = `face face-${face}`;
  div.dataset.face = face;
  const start = FACES.indexOf(face) * 9;
  for (const letter of state.slice(start, start + 9)) {
    const sticker = document.createElement("div");
    sticker.className = `sticker ${letter}`;
    div.appendChild(sticker);
  }
  return div;
}

function drawCube(cubeEl, state) {
  cubeEl.replaceChildren(...["U", "F", "R"].map((face) => makeFace(state, face)));
}

function openNet(panel) {
  el("net-title").textContent = `${panel.name}: all six faces`;
  el("net-view").replaceChildren(...FACES.map((face) => makeFace(panel.state, face)));
  el("net-dialog").hidden = false;
}

// ---- Panels ----

function avatarLetter(name) {
  return name.replace(/^mock-/, "").charAt(0).toUpperCase();
}

function createPanel(model, lane, state) {
  const node = el("panel-template").content.firstElementChild.cloneNode(true);
  const panel = {
    name: model.name,
    kind: model.kind,
    node,
    state,
    running: false,
    base: 0,        // race time (seconds) at the last update from the server
    baseAt: 0,      // performance.now() when that update arrived
    finalTime: null,
    latencies: [],  // seconds per move, for the live average
    cacheHits: 0,
  };
  const find = (selector) => node.querySelector(selector);

  find(".avatar").textContent = avatarLetter(model.name);
  find(".avatar").style.background = laneColor(lane);
  find(".model-name").textContent = model.name;
  find(".kind-badge").textContent = model.kind || "MODEL";
  if (KIND_STYLES[model.kind]) find(".kind-badge").classList.add(KIND_STYLES[model.kind]);
  find(".moves-max").textContent = ` / ${status.max_moves}`;
  find(".net-button").addEventListener("click", () => openNet(panel));

  drawCube(find(".cube"), state);
  return panel;
}

function panelPosition(name) {
  const index = OLLAMA_PANEL_ORDER.indexOf(name);
  return index === -1 ? OLLAMA_PANEL_ORDER.length : index;  // other models keep their order
}

function renderPanels(models, state) {
  // Each model keeps its lane (and its color); only the position in the grid changes.
  const lanes = models.map((m, lane) => ({ m, lane }));
  lanes.sort((a, b) => panelPosition(a.m.name) - panelPosition(b.m.name));
  panels = new Map(lanes.map(({ m, lane }) => [m.name, createPanel(m, lane, state)]));
  el("race-grid").replaceChildren(...[...panels.values()].map((p) => p.node));
}

function setStatus(panel, text, running = false) {
  const box = panel.node.querySelector(".status");
  box.textContent = text;
  box.classList.toggle("running", running);
}

// No move: "failed" if the call itself failed (HTTP error, timeout), "invalid" if the answer was bad.
function missingMove(failed) {
  return failed ? "failed" : "invalid";
}

function addHistory(panel, number, move, cacheHit, failed) {
  const item = document.createElement("li");
  const num = document.createElement("span");
  num.className = "num";
  num.textContent = number;
  item.append(num, move || missingMove(failed));
  if (!move) item.classList.add(missingMove(failed));
  if (cacheHit) {
    item.classList.add("cached");
    item.title = "Answer reused from the cache (no API call)";
  }
  panel.node.querySelector(".history").appendChild(item);
}

// Small stats row at the bottom of the stat card
function setMini(panel, name, text) {
  panel.node.querySelector(`.mini-stats .${name}`).textContent = text;
}

function showAvgLatency(panel, latencies) {
  const avg = latencies.reduce((sum, x) => sum + x, 0) / latencies.length;
  setMini(panel, "avg-latency", latencies.length ? `${avg.toFixed(2)}s` : "-");
}

// The exact API calls and cost come from the saved results once the race is over.
async function showApiStats(raceId) {
  try {
    for (const result of await getJson(`/api/results?race_id=${encodeURIComponent(raceId)}`)) {
      const panel = panels.get(result.model);
      if (!panel) continue;
      setMini(panel, "api-calls", result.api_calls);
      setMini(panel, "cache-hits", result.cache_hits);
      setMini(panel, "cost", `$${result.total_cost.toFixed(4)}`);
      showAvgLatency(panel, result.latencies);
    }
  } catch (error) {
    // the live numbers stay on screen
  }
}

function showFinish(panel, event) {
  // Amber for a call limit (not the model's fault), red for any other way of not solving.
  const style = event.solved ? "solved" : event.end_reason === "CALL LIMIT REACHED" ? "call-limit" : "failed";
  panel.node.querySelector(".stat-card").classList.add(style);
  setStatus(panel, event.solved ? "SOLVED ✓" : "NOT SOLVED");
  const gap = event.optimality_gap;
  setMini(panel, "gap", gap === null || gap === undefined ? "-" : gap > 0 ? `+${gap}` : String(gap));
  if (!event.solved) {
    const pill = panel.node.querySelector(".end-pill");
    pill.textContent = event.end_reason;
    pill.classList.toggle("call-limit", style === "call-limit");
    pill.hidden = false;
  }
}

function showRank(panel, rank) {
  const medal = panel.node.querySelector(".rank-medal");
  medal.querySelector(".rank-num").textContent = rank;
  medal.querySelector(".rank-suffix").textContent = RANK_SUFFIX[rank] || "TH";
  medal.className = `rank-medal rank-${Math.min(rank, 4)}`;
  medal.hidden = false;
  panel.node.classList.toggle("first", rank === 1);
}

// ---- Smooth timers ----
// The server tells us the race time at each event; in between we count up locally.

function tick() {
  const now = performance.now();
  for (const panel of panels.values()) {
    let seconds = panel.finalTime;
    if (seconds === null) {
      seconds = panel.running ? panel.base + ((now - panel.baseAt) / 1000) * raceSpeed : panel.base;
    }
    // The run limit is hard, so the clock never shows more than it (5:00.00)
    panel.node.querySelector(".timer").textContent = formatTime(Math.min(seconds, status.run_time_limit));
  }
  requestAnimationFrame(tick);
}

function syncTimer(panel, raceTime) {
  panel.base = raceTime;
  panel.baseAt = performance.now();
}

// ---- Events from the server ----

function handleEvent(event) {
  const panel = panels.get(event.model);

  switch (event.type) {
    case "race_start":
      renderPanels(event.models, event.state);
      if (event.one_by_one) {
        for (const p of panels.values()) setStatus(p, "waiting for turn...");
      }
      showMessage(event.replay ? `Replaying ${el("replay-select").selectedOptions[0]?.text || "saved race"}`
        : event.one_by_one ? "Race started (local models take turns)" : "Race started");
      break;

    case "start":
      panel.running = true;
      syncTimer(panel, 0);
      setStatus(panel, panel.kind === "DECISION MODEL" ? "deciding..." : "thinking...", true);
      break;

    case "move": {
      panel.state = event.state;
      syncTimer(panel, event.elapsed);
      panel.node.querySelector(".moves-used").textContent = event.number;
      const current = panel.node.querySelector(".current-move");
      current.textContent = event.move || missingMove(event.failed);
      current.classList.toggle("invalid", !event.valid && !event.failed);
      current.classList.toggle("failed", !event.valid && Boolean(event.failed));
      addHistory(panel, event.number, event.move, event.cache_hit, event.failed);
      panel.latencies.push(event.latency);
      showAvgLatency(panel, panel.latencies);
      if (event.cache_hit) setMini(panel, "cache-hits", ++panel.cacheHits);
      drawCube(panel.node.querySelector(".cube"), event.state);
      break;
    }

    case "finish":
      panel.running = false;
      panel.finalTime = event.total_time;
      showFinish(panel, event);
      break;

    case "race_end":
      for (const row of event.ranking) {
        if (panels.has(row.model)) showRank(panels.get(row.model), row.rank);
      }
      showMessage(raceEndMessage(event.api));
      loadReplays();
      showApiStats(event.race_id);
      refreshStatus();
      loadResults();  // the results section below now includes this race
      break;

    case "warning":  // something went wrong, but the race goes on
    case "error":
      showMessage(event.message, true);
      break;
  }
}

function connect(path, speed) {
  raceSpeed = speed;
  setBusy(true);
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${location.host}${path}`);
  let finished = false;

  socket.onmessage = (message) => {
    const event = JSON.parse(message.data);
    if (event.type === "race_end" || event.type === "error") finished = true;
    handleEvent(event);
  };
  socket.onclose = () => {
    if (!finished) showMessage("Connection to the server was lost.", true);
    for (const panel of panels.values()) panel.running = false;
    socket = null;
    setBusy(false);
  };
}

// ---- Controls ----

function selectedPuzzle() {
  return puzzles.find((p) => p.id === el("puzzle-select").value);
}

function showPuzzle() {
  const puzzle = selectedPuzzle();
  el("scramble-text").textContent = puzzle.scramble.join(" ");
  el("optimal-text").textContent =
    puzzle.optimal_distance === null ? "unknown (too deep for BFS)" : `${puzzle.optimal_distance} moves`;
  renderPanels(status.models, puzzle.state);
  showMessage("");
  loadReplays();
}

const MODE_TITLES = { mock: "Mock", ollama: "Ollama", openrouter: "OpenRouter" };

async function loadReplays() {
  const puzzleId = el("puzzle-select").value;
  const results = await getJson("/api/results");
  const races = new Map();
  for (const r of results) {
    if (r.puzzle_id === puzzleId) races.set(r.race_id, r.mode);
  }

  // Readable names like "ollama-1 · Ollama run 2": runs are numbered per mode, oldest first.
  const raceIds = [...races.keys()].sort();  // race ids start with a timestamp, so this is oldest first
  const runCount = {};
  const labels = new Map();
  for (const raceId of raceIds) {
    const mode = races.get(raceId);
    runCount[mode] = (runCount[mode] || 0) + 1;
    labels.set(raceId, `${puzzleId} · ${MODE_TITLES[mode] || mode} run ${runCount[mode]}`);
  }

  const select = el("replay-select");
  select.replaceChildren();
  for (const raceId of raceIds.reverse()) {  // newest first
    select.add(new Option(labels.get(raceId), raceId));
  }
  if (!raceIds.length) select.add(new Option("No saved races yet", ""));
  setBusy(socket !== null);
}

function raceEndMessage(api) {
  if (!api || (!api.calls && !api.cache_hits)) return "Race finished";
  const counts = `${api.calls} API calls, ${api.cache_hits} cache hits`;
  return api.dry_run
    ? `Dry run finished: a real run would use ${counts} (nothing was sent)`
    : `Race finished: ${counts}, cost $${api.cost.toFixed(4)}`;
}

async function refreshStatus() {
  try {
    status = await getJson("/api/status");
    showModeLabel(status.mode);
  } catch (error) {
    // keep the old status; the next race refreshes it again
  }
}

// ---- Mode switch (kept in the server's memory; .env only gives the starting mode) ----

function modeSet() {
  return status.mode === "openrouter" ? status.openrouter_set : "ollama";
}

function setServerMode(mode) {
  return getJson("/api/mode", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
}

// Panels, puzzles, saved races and the results section all follow the mode.
function applyMode() {
  el("mode-select").value = status.mode;
  showModeLabel(status.mode);
  puzzles = allPuzzles.filter((p) => p.set === modeSet());
  el("puzzle-select").replaceChildren(...puzzles.map((p) => new Option(p.id, p.id)));
  showPuzzle();
  loadResults();
}

async function changeMode() {
  try {
    status = await setServerMode(el("mode-select").value);
    applyMode();
  } catch (error) {
    el("mode-select").value = status.mode;  // the server kept the old mode
    showMessage(error.message, true);
  }
}

async function startLiveRace() {
  const puzzleId = el("puzzle-select").value;
  let query = "";
  if (status.mode === "openrouter") {
    await refreshStatus();  // fresh numbers for the question below
    const question = `This will use up to ${status.max_calls_next_race} real API calls. ` +
                     `Calls used so far: ${status.api_calls_used} / ${status.api_call_limit}. Continue?`;
    if (!confirm(question)) return;
    query = "?confirm=yes";
  }
  renderPanels(status.models, selectedPuzzle().state);
  connect(`/ws/race/${encodeURIComponent(puzzleId)}${query}`, 1);
}

function startDryRun() {
  // Always the OpenRouter lineup; the panels are drawn from the race_start event the server sends.
  connect(`/ws/race/${encodeURIComponent(el("puzzle-select").value)}?dry_run=true`, 1);
}

function setBusy(isBusy) {
  el("mode-select").disabled = isBusy;
  el("race-button").disabled = isBusy;
  el("dry-run-button").disabled = isBusy;
  el("puzzle-select").disabled = isBusy;
  el("replay-select").disabled = isBusy;
  el("speed-select").disabled = isBusy;
  el("replay-button").disabled = isBusy || !el("replay-select").value;
}

function showMessage(text, isError = false) {
  const box = el("message");
  box.textContent = text;
  box.classList.toggle("error", isError);
}

async function init() {
  try {
    [status, allPuzzles] = await Promise.all([getJson("/api/status"), getJson("/api/puzzles")]);
    if (!GUI_MODES.includes(status.mode)) status = await setServerMode("ollama");  // .env said mock
  } catch (error) {
    showMessage("Could not reach the server. Is it running?", true);
    return;
  }

  for (const speed of status.replay_speeds) {
    el("speed-select").add(new Option(`${speed}x`, speed));
  }

  el("mode-select").addEventListener("change", changeMode);
  el("puzzle-select").addEventListener("change", showPuzzle);
  el("race-button").addEventListener("click", startLiveRace);
  el("dry-run-button").addEventListener("click", startDryRun);
  el("replay-button").addEventListener("click", () => {
    const speed = Number(el("speed-select").value);
    connect(`/ws/replay/${encodeURIComponent(el("replay-select").value)}?speed=${speed}`, speed);
  });
  el("net-close").addEventListener("click", () => (el("net-dialog").hidden = true));
  el("net-dialog").addEventListener("click", (e) => {
    if (e.target === el("net-dialog")) el("net-dialog").hidden = true;
  });

  applyMode();
  requestAnimationFrame(tick);
}

init();
