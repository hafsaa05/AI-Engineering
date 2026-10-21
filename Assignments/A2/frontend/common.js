// Small helpers used by app.js and dashboard.js.

// One color per lane (model position), for avatars and chart bars.
const LANE_COLORS = ["#ff6b6b", "#3b9eff", "#14b8a6", "#ffa726"];  // coral, sky blue, teal, sunny orange
const FONT = "Georgia, 'Times New Roman', serif";

const el = (id) => document.getElementById(id);

function laneColor(index) {
  return LANE_COLORS[index % LANE_COLORS.length];
}

// 74.5 -> "1:14.50"
function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const rest = (seconds - minutes * 60).toFixed(2).padStart(5, "0");
  return `${minutes}:${rest}`;
}

async function getJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `${url} failed with ${response.status}`);
  return data;
}

// Mode label next to the dropdown. OpenRouter is red so a paid race is never started by mistake.
const MODE_LABELS = { mock: "Mock mode", ollama: "Local models, no API limits", openrouter: "Real API mode" };

function showModeLabel(mode) {
  const label = el("mode-label");
  label.textContent = MODE_LABELS[mode] || MODE_LABELS.mock;
  label.className = `mode-label ${mode}`;
  document.body.classList.toggle("openrouter-mode", mode === "openrouter");
}
