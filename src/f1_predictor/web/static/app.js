/* F1 Predictor — frontend logic + animations */

const $ = (sel) => document.querySelector(sel);

const els = {
  predictBtn: $("#predict-btn"),
  raceInfo: $("#race-info"),
  loader: $("#loader"),
  results: $("#results"),
  raceTitle: $("#race-title"),
  raceMeta: $("#race-meta"),
  podium: $("#podium"),
  cards: $("#cards"),
  leaderboard: $("#leaderboard"),
};

const TEAM_COLORS = {
  red_bull: "#1E3A8A",
  ferrari: "#DC2626",
  mercedes: "#10B981",
  mclaren: "#F59E0B",
  aston_martin: "#065F46",
  alpine: "#3B82F6",
  williams: "#0EA5E9",
  rb: "#1E40AF",
  alphatauri: "#1E40AF",
  haas: "#9CA3AF",
  sauber: "#22C55E",
  alfa: "#7F1D1D",
};

// On load: fetch model status + leaderboard
init();

async function init() {
  try {
    const health = await fetchJSON("/api/health");
    if (!health.model_trained) {
      els.raceInfo.innerHTML =
        '⚠️  Model not trained yet. Run <code>python -m f1_predictor train</code> first.';
      els.predictBtn.disabled = true;
    } else {
      els.raceInfo.textContent = "Click to predict the next Grand Prix.";
    }
    loadLeaderboard();
  } catch (err) {
    els.raceInfo.textContent = "Backend unreachable.";
  }
}

els.predictBtn.addEventListener("click", runPrediction);

async function runPrediction() {
  els.predictBtn.disabled = true;
  els.results.classList.add("hidden");
  els.loader.classList.remove("hidden");
  els.cards.innerHTML = "";
  els.podium.innerHTML = "";

  try {
    const data = await fetchJSON("/api/predict");
    await delay(700); // give the loader time to show
    renderResults(data);
  } catch (err) {
    alert("Prediction failed: " + err.message);
  } finally {
    els.loader.classList.add("hidden");
    els.predictBtn.disabled = false;
  }
}

function renderResults(data) {
  els.raceTitle.textContent = data.race_name.toUpperCase();
  els.raceMeta.textContent =
    `Round ${data.round}  ·  ${data.circuit_id}  ·  ${data.season}`;

  const top = data.predictions.slice(0, 3);
  const rest = data.predictions.slice(3, 12);

  // podium (P2, P1, P3 ordering visually)
  const podiumOrder = [top[1], top[0], top[2]];
  const stepClasses = ["p2", "p1", "p3"];
  podiumOrder.forEach((d, i) => {
    if (!d) return;
    const step = document.createElement("div");
    step.className = `podium-step ${stepClasses[i]}`;
    step.innerHTML = `
      <div class="podium-rank">P${stepClasses[i][1]}</div>
      <div class="podium-driver">${formatName(d.driver_id)}</div>
      <div class="podium-prob">${pct(d.win_prob)} win  ·  ${pct(d.podium_prob)} pod.</div>
    `;
    els.podium.appendChild(step);
  });

  // grid cards
  rest.forEach((d, i) => {
    const card = document.createElement("div");
    card.className = "card";
    card.style.animationDelay = `${0.4 + i * 0.07}s`;
    card.style.borderLeftColor = TEAM_COLORS[d.constructor_id] || "var(--steel)";
    card.innerHTML = `
      <div class="card-rank">P${i + 4}</div>
      <div class="card-body">
        <div class="card-driver">${formatName(d.driver_id)}</div>
        <div class="card-team">${formatTeam(d.constructor_id)}  ·  Grid ${Math.round(d.grid_position) || "?"}</div>
        <div class="card-bars">
          <div class="bar-row">
            <span class="bar-label">WIN</span>
            <div class="bar-track"><div class="bar-fill" data-pct="${d.win_prob * 100}"></div></div>
            <span class="bar-value">${pct(d.win_prob)}</span>
          </div>
          <div class="bar-row">
            <span class="bar-label">POD</span>
            <div class="bar-track"><div class="bar-fill podium-bar" data-pct="${d.podium_prob * 100}"></div></div>
            <span class="bar-value">${pct(d.podium_prob)}</span>
          </div>
        </div>
      </div>
    `;
    els.cards.appendChild(card);
  });

  els.results.classList.remove("hidden");
  els.results.scrollIntoView({ behavior: "smooth", block: "start" });

  // animate bars to fill
  requestAnimationFrame(() => {
    document.querySelectorAll(".bar-fill").forEach((bar) => {
      const pctVal = Math.min(parseFloat(bar.dataset.pct) || 0, 100);
      bar.style.width = pctVal + "%";
    });
  });
}

async function loadLeaderboard() {
  try {
    const data = await fetchJSON("/api/leaderboard");
    if (!data.summary || !data.summary.resolved_races) {
      els.leaderboard.innerHTML =
        '<p class="muted">No past predictions resolved yet — comparison will populate after the first race.</p>';
      return;
    }
    const s = data.summary;
    els.leaderboard.innerHTML = `
      <div class="lb-stat">
        <div class="lb-value">${s.resolved_races}</div>
        <div class="lb-label">Races scored</div>
      </div>
      <div class="lb-stat">
        <div class="lb-value">${pct(s.winner_accuracy)}</div>
        <div class="lb-label">Winner accuracy</div>
      </div>
      <div class="lb-stat">
        <div class="lb-value">${(s.avg_podium_overlap || 0).toFixed(2)} / 3</div>
        <div class="lb-label">Avg podium overlap</div>
      </div>
    `;
  } catch (err) {
    els.leaderboard.innerHTML = '<p class="muted">Could not load leaderboard.</p>';
  }
}

/* ── helpers ───────────────────────────────────────────────── */

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) {
    const body = await r.text();
    throw new Error(body || r.statusText);
  }
  return r.json();
}

function pct(x) {
  if (x === null || x === undefined || isNaN(x)) return "—";
  return (x * 100).toFixed(1) + "%";
}

function formatName(id) {
  return id.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatTeam(id) {
  if (!id) return "";
  return id.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function delay(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
