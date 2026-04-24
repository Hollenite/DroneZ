const ZONE_POSITIONS = {
  hub: { x: 120, y: 260, w: 160, h: 120 },
  Z1: { x: 340, y: 70, w: 160, h: 120 },
  Z2: { x: 540, y: 70, w: 160, h: 120 },
  Z3: { x: 740, y: 70, w: 120, h: 120 },
  Z4: { x: 340, y: 250, w: 160, h: 120 },
  Z5: { x: 540, y: 250, w: 160, h: 120 },
  Z6: { x: 740, y: 250, w: 120, h: 120 },
};

const state = {
  payload: null,
  frames: [],
  currentFrameIndex: 0,
  timerId: null,
  comparison: null,
};

const scenarioSelect = document.getElementById("scenarioSelect");
const policySelect = document.getElementById("policySelect");
const speedRange = document.getElementById("speedRange");
const loadBtn = document.getElementById("loadBtn");
const playBtn = document.getElementById("playBtn");
const pauseBtn = document.getElementById("pauseBtn");
const resetBtn = document.getElementById("resetBtn");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
const mapSvg = document.getElementById("mapSvg");
const metricsGrid = document.getElementById("metricsGrid");
const eventsList = document.getElementById("eventsList");
const frameSummary = document.getElementById("frameSummary");
const comparisonSummary = document.getElementById("comparisonSummary");
const rewardBreakdown = document.getElementById("rewardBreakdown");

async function loadTrace() {
  pauseReplay();
  const tracePath = `../artifacts/traces/${scenarioSelect.value}_${policySelect.value}_trace.json`;
  try {
    const response = await fetch(tracePath);
    if (!response.ok) throw new Error(`Could not load ${tracePath}`);
    state.payload = await response.json();
    state.frames = state.payload.frames || [];
    state.currentFrameIndex = 0;
    if (!state.comparison) {
      const comparisonResponse = await fetch("../artifacts/results/policy_comparison.json");
      if (comparisonResponse.ok) {
        state.comparison = await comparisonResponse.json();
      }
    }
    render();
  } catch (error) {
    mapSvg.innerHTML = "";
    metricsGrid.innerHTML = "";
    eventsList.innerHTML = `<li>${error.message}</li>`;
    frameSummary.textContent = "Generate the requested trace first with scripts/generate_demo_trace.py.";
    comparisonSummary.innerHTML = "";
    rewardBreakdown.innerHTML = "";
  }
}

function currentFrame() {
  if (!state.frames.length) return null;
  return state.frames[state.currentFrameIndex];
}

function currentObservation() {
  const frame = currentFrame();
  return frame?.observation || state.payload?.initial_observation || null;
}

function render() {
  renderMap();
  renderMetrics();
  renderPolicySnapshot();
  renderRewardBreakdown();
  renderEvents();
  renderSummary();
}

function renderMap() {
  const observation = currentObservation();
  if (!observation) {
    mapSvg.innerHTML = "";
    return;
  }

  const sectors = observation.city?.sectors || [];
  const charging = observation.charging || [];
  const orders = observation.orders || [];
  const fleet = observation.fleet || [];

  const sectorMarkup = sectors
    .map((sector) => {
      const zone = ZONE_POSITIONS[sector.zone_id] || { x: 40, y: 40, w: 120, h: 100 };
      const fill = sector.is_no_fly
        ? "#F8D7DA"
        : sector.operations_paused
          ? "#FCE8B2"
          : sector.weather === "storm"
            ? "#F2C6DE"
            : sector.weather === "heavy_rain"
              ? "#D9E6F5"
              : "#F6FBF4";
      return `
        <g>
          <rect x="${zone.x}" y="${zone.y}" width="${zone.w}" height="${zone.h}" rx="20" fill="${fill}" stroke="#95a695" stroke-width="2" />
          <text x="${zone.x + 14}" y="${zone.y + 28}" font-size="18" font-weight="700" fill="#132217">${sector.zone_id}</text>
          <text x="${zone.x + 14}" y="${zone.y + 52}" font-size="12" fill="#587062">weather: ${sector.weather}</text>
          <text x="${zone.x + 14}" y="${zone.y + 70}" font-size="12" fill="#587062">congestion: ${Number(sector.congestion_score).toFixed(2)}</text>
          <text x="${zone.x + 14}" y="${zone.y + 88}" font-size="12" fill="#587062">no-fly: ${sector.is_no_fly ? "yes" : "no"}</text>
          <text x="${zone.x + 14}" y="${zone.y + 106}" font-size="12" fill="#587062">paused: ${sector.operations_paused ? "yes" : "no"}</text>
        </g>
      `;
    })
    .join("");

  const chargingMarkup = charging
    .map((station, index) => {
      const zone = ZONE_POSITIONS[station.zone_id] || ZONE_POSITIONS.hub;
      const x = zone.x + 18 + index * 42;
      const y = zone.y + zone.h - 26;
      return `
        <g>
          <rect x="${x}" y="${y}" width="28" height="18" rx="4" fill="#1d6f5f" opacity="0.85"></rect>
          <text x="${x + 14}" y="${y + 13}" text-anchor="middle" font-size="9" fill="#ffffff">${station.station_id}</text>
        </g>
      `;
    })
    .join("");

  const orderMarkup = orders
    .filter((order) => order.status !== "delivered")
    .map((order, index) => {
      const zone = ZONE_POSITIONS[order.zone_id] || ZONE_POSITIONS.hub;
      const color = order.priority === "medical" ? "#C75146" : order.priority === "urgent" ? "#D98E04" : "#304C89";
      const cx = zone.x + zone.w - 26;
      const cy = zone.y + 24 + index * 14;
      return `
        <g>
          <circle cx="${cx}" cy="${cy}" r="8" fill="${color}" />
          <text x="${cx + 14}" y="${cy + 4}" font-size="11" fill="#132217">${order.order_id}</text>
        </g>
      `;
    })
    .join("");

  const flightPathMarkup = fleet
    .filter((drone) => drone.target_zone)
    .map((drone) => {
      const currentZone = ZONE_POSITIONS[drone.current_zone] || ZONE_POSITIONS.hub;
      const targetZone = ZONE_POSITIONS[drone.target_zone] || ZONE_POSITIONS.hub;
      return `
        <line
          x1="${currentZone.x + currentZone.w / 2}"
          y1="${currentZone.y + currentZone.h / 2}"
          x2="${targetZone.x + targetZone.w / 2}"
          y2="${targetZone.y + targetZone.h / 2}"
          stroke="${drone.active_corridor === "safe" ? "#1d6f5f" : "#304C89"}"
          stroke-width="3"
          stroke-dasharray="8 6"
          opacity="0.8"
        />
      `;
    })
    .join("");

  const droneMarkup = fleet
    .map((drone, index) => {
      const zone = ZONE_POSITIONS[drone.current_zone] || ZONE_POSITIONS.hub;
      const x = zone.x + 38 + (index % 3) * 34;
      const y = zone.y + 36 + Math.floor(index / 3) * 28;
      const fill = drone.health_risk === "critical" ? "#C75146" : drone.health_risk === "high" ? "#D98E04" : "#1d6f5f";
      return `
        <g>
          <circle cx="${x}" cy="${y}" r="12" fill="${fill}" />
          <text x="${x}" y="${y + 4}" text-anchor="middle" font-size="9" fill="#fff">${drone.drone_id}</text>
        </g>
      `;
    })
    .join("");

  mapSvg.innerHTML = `
    <rect x="0" y="0" width="900" height="520" rx="28" fill="transparent" />
    ${sectorMarkup}
    ${flightPathMarkup}
    ${chargingMarkup}
    ${orderMarkup}
    ${droneMarkup}
  `;
}

function renderMetrics() {
  const frame = currentFrame();
  const observation = currentObservation();
  if (!observation) {
    metricsGrid.innerHTML = "";
    return;
  }

  const cumulativeReward = frame?.cumulative_reward ?? 0;
  const pendingOrders = observation.orders.filter((order) => order.status !== "delivered" && order.status !== "canceled");
  const completedDeliveries = observation.orders.filter((order) => order.status === "delivered").length;
  const urgentDelivered = observation.orders.filter((order) => order.status === "delivered" && ["urgent", "medical"].includes(order.priority)).length;
  const safetyViolations = Math.round(Math.abs(frame?.info?.cumulative_reward?.negative?.unsafe_zone_entry || 0) / 8) || 0;
  const invalidActions = frame?.info?.invalid_action_count || 0;
  const minBattery = Math.min(...observation.fleet.map((drone) => drone.battery));
  const urgentPending = observation.orders.filter((order) => order.status !== "delivered" && ["urgent", "medical"].includes(order.priority)).length;
  const safetyBadge = safetyViolations === 0 ? "safe" : "risk";

  const metrics = [
    ["Frame", `${state.currentFrameIndex + 1}/${state.frames.length}`],
    ["Reward", `${Number(cumulativeReward).toFixed(1)}`],
    ["Deliveries", String(completedDeliveries)],
    ["Urgent", String(urgentDelivered)],
    ["Urgent Pending", String(urgentPending)],
    ["Pending", String(pendingOrders.length)],
    ["Invalid", String(invalidActions)],
    ["Safety", `${safetyViolations} (${safetyBadge})`],
    ["Min Battery", `${minBattery}`],
  ];

  metricsGrid.innerHTML = metrics
    .map(
      ([label, value]) => `
        <div class="metric">
          <div class="metric-label">${label}</div>
          <div class="metric-value">${value}</div>
        </div>
      `,
    )
    .join("");
}

function renderPolicySnapshot() {
  if (!state.comparison) {
    comparisonSummary.innerHTML = "<p>Run scripts/evaluate_policies.py to load comparison metrics.</p>";
    return;
  }

  const scenario = scenarioSelect.value;
  const policy = policySelect.value;
  const row = state.comparison.policy_results?.[policy]?.[scenario];
  const improved = state.comparison.policy_results?.improved?.[scenario];
  const heuristic = state.comparison.policy_results?.heuristic?.[scenario];
  if (!row) {
    comparisonSummary.innerHTML = "<p>No comparison data for the selected scenario.</p>";
    return;
  }

  const comparisonLines = [
    `<strong>${policy}</strong> on <strong>${scenario}</strong>`,
    `reward: ${Number(row.total_reward).toFixed(1)}`,
    `normalized score: ${Number(row.normalized_score).toFixed(3)}`,
    `deliveries: ${row.completed_deliveries}`,
    `deadline misses: ${row.deadline_miss_count}`,
    `safety violations: ${row.safety_violations}`,
  ];

  if (scenario === "demo" && improved && heuristic) {
    comparisonLines.push(
      `demo baseline gap: improved ${Number(improved.total_reward).toFixed(1)} vs heuristic ${Number(heuristic.total_reward).toFixed(1)}`,
    );
  }

  comparisonSummary.innerHTML = comparisonLines.map((line) => `<p>${line}</p>`).join("");
}

function renderRewardBreakdown() {
  const frame = currentFrame();
  const breakdown = frame?.info?.cumulative_reward || frame?.reward_breakdown;
  if (!breakdown) {
    rewardBreakdown.innerHTML = "<p>No reward data for this frame.</p>";
    return;
  }

  const positive = Object.entries(breakdown.positive || {})
    .map(([key, value]) => `<li><span>${key}</span><strong>+${Number(value).toFixed(1)}</strong></li>`)
    .join("");
  const negative = Object.entries(breakdown.negative || {})
    .map(([key, value]) => `<li><span>${key}</span><strong>${Number(value).toFixed(1)}</strong></li>`)
    .join("");

  rewardBreakdown.innerHTML = `
    <div class="reward-columns">
      <div>
        <h3>Positive</h3>
        <ul>${positive || "<li><span>none</span><strong>0.0</strong></li>"}</ul>
      </div>
      <div>
        <h3>Negative</h3>
        <ul>${negative || "<li><span>none</span><strong>0.0</strong></li>"}</ul>
      </div>
    </div>
    <p class="reward-total">Total: ${Number(breakdown.total || 0).toFixed(1)}</p>
  `;
}

function renderEvents() {
  const observation = currentObservation();
  if (!observation) {
    eventsList.innerHTML = "";
    return;
  }
  const events = observation.recent_events?.slice(-8) || [];
  eventsList.innerHTML = events.map((event) => `<li>${event}</li>`).join("");
}

function renderSummary() {
  const frame = currentFrame();
  const observation = currentObservation();
  if (!observation) {
    frameSummary.textContent = "";
    return;
  }
  frameSummary.textContent = frame?.summary || observation.summary || "No summary available.";
}

function playReplay() {
  pauseReplay();
  state.timerId = window.setInterval(() => {
    if (state.currentFrameIndex >= state.frames.length - 1) {
      pauseReplay();
      return;
    }
    state.currentFrameIndex += 1;
    render();
  }, Number(speedRange.value));
}

function pauseReplay() {
  if (state.timerId !== null) {
    window.clearInterval(state.timerId);
    state.timerId = null;
  }
}

loadBtn.addEventListener("click", loadTrace);
playBtn.addEventListener("click", playReplay);
pauseBtn.addEventListener("click", pauseReplay);
resetBtn.addEventListener("click", () => {
  state.currentFrameIndex = 0;
  render();
});
prevBtn.addEventListener("click", () => {
  pauseReplay();
  state.currentFrameIndex = Math.max(0, state.currentFrameIndex - 1);
  render();
});
nextBtn.addEventListener("click", () => {
  pauseReplay();
  state.currentFrameIndex = Math.min(state.frames.length - 1, state.currentFrameIndex + 1);
  render();
});
speedRange.addEventListener("change", () => {
  if (state.timerId !== null) playReplay();
});

loadTrace();
