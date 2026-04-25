const ZONES = {
  hub: { x: 410, y: 255, w: 160, h: 120, label: "Hub" },
  Z1: { x: 80, y: 70, w: 170, h: 125, label: "Z1 Downtown" },
  Z2: { x: 405, y: 60, w: 170, h: 125, label: "Z2 Hospital" },
  Z3: { x: 725, y: 80, w: 170, h: 125, label: "Z3 East" },
  Z4: { x: 80, y: 405, w: 170, h: 125, label: "Z4 Market" },
  Z5: { x: 405, y: 430, w: 170, h: 125, label: "Z5 Campus" },
  Z6: { x: 725, y: 410, w: 170, h: 125, label: "Z6 Suburb" },
};

const FALLBACK_FRAME = {
  action_index: 0,
  tick: 0,
  action: { action: "fallback_only", params: { reason: "trace_not_loaded" } },
  step_reward: 0,
  cumulative_reward: 0,
  summary: "Fallback only -- trace not loaded. Check the red error panel for attempted URLs.",
  observation: {
    task_id: "fallback",
    step: 0,
    max_steps: 1,
    fleet: [
      { drone_id: "FA-1", drone_type: "fast_light", status: "idle", battery: 95, current_zone: "hub", target_zone: "Z1", active_corridor: "safe", health_risk: "low" },
      { drone_id: "HE-1", drone_type: "heavy_carrier", status: "idle", battery: 91, current_zone: "hub", target_zone: null, active_corridor: null, health_risk: "low" },
    ],
    orders: [
      { order_id: "O1", priority: "urgent", status: "queued", zone_id: "Z1", deadline: 4 },
      { order_id: "O2", priority: "normal", status: "queued", zone_id: "Z4", deadline: 6 },
    ],
    city: {
      sectors: [
        { zone_id: "hub", weather: "clear", congestion_score: 0, is_no_fly: false, operations_paused: false },
        { zone_id: "Z1", weather: "clear", congestion_score: 0.2, is_no_fly: false, operations_paused: false },
        { zone_id: "Z2", weather: "storm", congestion_score: 0.8, is_no_fly: true, operations_paused: false },
        { zone_id: "Z3", weather: "clear", congestion_score: 0.1, is_no_fly: false, operations_paused: false },
        { zone_id: "Z4", weather: "moderate_wind", congestion_score: 0.3, is_no_fly: false, operations_paused: false },
        { zone_id: "Z5", weather: "clear", congestion_score: 0.4, is_no_fly: false, operations_paused: true },
      ],
      active_no_fly_zones: ["Z2"],
      held_zones: ["Z5"],
    },
    charging: [{ station_id: "C_HUB", zone_id: "hub", capacity: 3, occupied_slots: 1, queue_size: 0 }],
    recent_events: ["Fallback only -- real trace could not be loaded."],
  },
  info: { cumulative_reward: { positive: {}, negative: {}, total: 0 }, invalid_action_count: 0 },
};

const state = {
  payload: null,
  frames: [],
  currentFrameIndex: 0,
  timerId: null,
  comparison: null,
  traceLoaded: false,
};

const $ = (id) => document.getElementById(id);
const scenarioSelect = $("scenarioSelect");
const policySelect = $("policySelect");
const speedRange = $("speedRange");
const loadBtn = $("loadBtn");
const playBtn = $("playBtn");
const pauseBtn = $("pauseBtn");
const resetBtn = $("resetBtn");
const prevBtn = $("prevBtn");
const nextBtn = $("nextBtn");
const mapSvg = $("mapSvg");
const metricsGrid = $("metricsGrid");
const eventsList = $("eventsList");
const frameSummary = $("frameSummary");
const comparisonSummary = $("comparisonSummary");
const rewardBreakdown = $("rewardBreakdown");
const lastActionBox = $("lastActionBox");
const errorPanel = $("errorPanel");
const frameIndicator = $("frameIndicator");
const progressBar = $("progressBar");
const scenarioBadge = $("scenarioBadge");
const policyBadge = $("policyBadge");

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function traceCandidates(scenario, policy) {
  const file = `${scenario}_${policy}_trace.json`;
  return [`../artifacts/traces/${file}`, `/artifacts/traces/${file}`, `artifacts/traces/${file}`];
}

async function fetchFirstJson(urls) {
  const errors = [];
  for (const url of urls) {
    try {
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) {
        errors.push(`${url} -> HTTP ${response.status}`);
        continue;
      }
      return { data: await response.json(), url, errors };
    } catch (error) {
      errors.push(`${url} -> ${error.message}`);
    }
  }
  throw new Error(errors.join("\n"));
}

async function loadTrace() {
  pauseReplay();
  const scenario = scenarioSelect.value;
  const policy = policySelect.value;
  const urls = traceCandidates(scenario, policy);

  try {
    const { data, url } = await fetchFirstJson(urls);
    state.payload = data;
    state.frames = normalizeFrames(data);
    state.currentFrameIndex = 0;
    state.traceLoaded = true;
    errorPanel.hidden = true;
    errorPanel.textContent = "";
    await loadComparison();
    render();
    console.info(`Loaded DroneZ trace from ${url}`);
  } catch (error) {
    state.payload = {
      schema_version: "fallback",
      task_id: "fallback",
      policy_id: "fallback",
      frames: [FALLBACK_FRAME],
      initial_observation: FALLBACK_FRAME.observation,
    };
    state.frames = [FALLBACK_FRAME];
    state.currentFrameIndex = 0;
    state.traceLoaded = false;
    errorPanel.hidden = false;
    errorPanel.innerHTML = `
      <strong>Trace load failed. Showing fallback UI only.</strong>
      <span>Attempted:</span>
      <pre>${escapeHtml(error.message)}</pre>
    `;
    await loadComparison();
    render();
  }
}

async function loadComparison() {
  if (state.comparison) return;
  try {
    const { data } = await fetchFirstJson([
      "../artifacts/results/policy_comparison.json",
      "/artifacts/results/policy_comparison.json",
      "artifacts/results/policy_comparison.json",
    ]);
    state.comparison = data;
  } catch {
    state.comparison = null;
  }
}

function normalizeFrames(payload) {
  const frames = payload?.frames || [];
  if (frames.length) return frames;
  if (payload?.initial_observation) {
    return [{ ...FALLBACK_FRAME, observation: payload.initial_observation, summary: payload.initial_observation.summary }];
  }
  return [FALLBACK_FRAME];
}

function currentFrame() {
  return state.frames[state.currentFrameIndex] || null;
}

function frameObservation(frame = currentFrame()) {
  return (
    frame?.observation ||
    frame?.state?.observation ||
    frame?.state ||
    state.payload?.initial_observation ||
    null
  );
}

function frameInfo(frame = currentFrame()) {
  return frame?.info || frame?.state?.info || {};
}

function render() {
  renderBadges();
  renderMap();
  renderMetrics();
  renderComparison();
  renderRewardBreakdown();
  renderEvents();
  renderSummary();
  renderAction();
}

function renderBadges() {
  const frame = currentFrame();
  scenarioBadge.textContent = `scenario: ${state.payload?.task_id || scenarioSelect.value}`;
  policyBadge.textContent = `policy: ${state.payload?.policy_id || policySelect.value}`;
  frameIndicator.textContent = `Frame ${state.currentFrameIndex + 1} / ${state.frames.length}`;
  const progress = state.frames.length <= 1 ? 100 : (state.currentFrameIndex / (state.frames.length - 1)) * 100;
  progressBar.style.width = `${progress}%`;
  document.body.classList.toggle("fallback-mode", !state.traceLoaded);
}

function zoneCenter(zoneId) {
  const zone = ZONES[zoneId] || ZONES.hub;
  return { x: zone.x + zone.w / 2, y: zone.y + zone.h / 2 };
}

function sectorClass(sector) {
  if (sector.is_no_fly) return "zone no-fly";
  if (sector.operations_paused) return "zone paused";
  if (["storm", "heavy_rain", "heavy_wind"].includes(sector.weather)) return "zone storm";
  if (Number(sector.congestion_score || 0) >= 0.7) return "zone congested";
  if (sector.zone_id === "hub") return "zone hub";
  return "zone clear";
}

function renderMap() {
  const frame = currentFrame();
  const observation = frameObservation(frame);
  if (!observation) {
    mapSvg.innerHTML = `<text x="40" y="80" fill="#c75146">No observation available for this frame.</text>`;
    return;
  }

  const sectorsById = new Map((observation.city?.sectors || []).map((sector) => [sector.zone_id, sector]));
  const sectors = Object.keys(ZONES).map((zoneId) => sectorsById.get(zoneId) || {
    zone_id: zoneId,
    weather: "clear",
    congestion_score: 0,
    is_no_fly: false,
    operations_paused: false,
  });
  const fleet = observation.fleet || [];
  const orders = observation.orders || [];
  const charging = observation.charging || [];
  const activeNoFly = new Set(observation.city?.active_no_fly_zones || []);
  const heldZones = new Set(observation.city?.held_zones || []);

  const roads = [
    ["hub", "Z1"], ["hub", "Z2"], ["hub", "Z3"], ["hub", "Z4"], ["hub", "Z5"], ["hub", "Z6"],
    ["Z1", "Z2"], ["Z2", "Z3"], ["Z4", "Z5"], ["Z5", "Z6"],
  ].map(([a, b]) => {
    const p1 = zoneCenter(a);
    const p2 = zoneCenter(b);
    return `<line class="road" x1="${p1.x}" y1="${p1.y}" x2="${p2.x}" y2="${p2.y}" />`;
  }).join("");

  const zoneMarkup = sectors.map((sector) => {
    const zone = ZONES[sector.zone_id] || ZONES.hub;
    const badges = [
      sector.is_no_fly || activeNoFly.has(sector.zone_id) ? "NO-FLY" : null,
      sector.operations_paused || heldZones.has(sector.zone_id) ? "PAUSED" : null,
      sector.weather && sector.weather !== "clear" ? sector.weather.replace("_", " ") : null,
      Number(sector.congestion_score || 0) >= 0.7 ? "CONGESTED" : null,
    ].filter(Boolean);

    return `
      <g class="${sectorClass(sector)}">
        <rect x="${zone.x}" y="${zone.y}" width="${zone.w}" height="${zone.h}" rx="22" />
        <text class="zone-title" x="${zone.x + 16}" y="${zone.y + 28}">${escapeHtml(zone.label)}</text>
        <text class="zone-detail" x="${zone.x + 16}" y="${zone.y + 52}">weather: ${escapeHtml(sector.weather || "clear")}</text>
        <text class="zone-detail" x="${zone.x + 16}" y="${zone.y + 72}">congestion: ${Number(sector.congestion_score || 0).toFixed(2)}</text>
        <text class="zone-detail" x="${zone.x + 16}" y="${zone.y + 92}">orders: ${orders.filter((order) => order.zone_id === sector.zone_id && order.status !== "delivered").length}</text>
        ${badges.map((badge, index) => `<text class="zone-badge" x="${zone.x + 16}" y="${zone.y + 113 + index * 15}">${escapeHtml(badge)}</text>`).join("")}
      </g>
    `;
  }).join("");

  const pathMarkup = fleet.filter((drone) => drone.target_zone).map((drone, index) => {
    const p1 = zoneCenter(drone.current_zone || "hub");
    const p2 = zoneCenter(drone.target_zone || "hub");
    const safe = drone.active_corridor === "safe";
    return `
      <g>
        <line class="${safe ? "flight-path safe" : "flight-path"}" x1="${p1.x}" y1="${p1.y}" x2="${p2.x}" y2="${p2.y}" />
        <circle class="${safe ? "path-pulse safe" : "path-pulse"}" r="5">
          <animateMotion dur="${2.5 + index * 0.2}s" repeatCount="indefinite" path="M ${p1.x},${p1.y} L ${p2.x},${p2.y}" />
        </circle>
      </g>
    `;
  }).join("");

  const chargingMarkup = charging.map((station, index) => {
    const zone = ZONES[station.zone_id] || ZONES.hub;
    const x = zone.x + zone.w - 58;
    const y = zone.y + zone.h - 38 - (index % 2) * 24;
    return `
      <g class="charger">
        <rect x="${x}" y="${y}" width="46" height="26" rx="8" />
        <text x="${x + 23}" y="${y + 17}" text-anchor="middle">${escapeHtml(station.station_id)}</text>
        <title>${station.occupied_slots}/${station.capacity} occupied, queue ${station.queue_size}</title>
      </g>
    `;
  }).join("");

  const orderMarkup = orders.map((order, index) => {
    const zone = ZONES[order.zone_id] || ZONES.hub;
    const localIndex = orders.filter((item) => item.zone_id === order.zone_id).findIndex((item) => item.order_id === order.order_id);
    const x = zone.x + 20 + (localIndex % 3) * 45;
    const y = zone.y + zone.h - 20 - Math.floor(localIndex / 3) * 22;
    const priorityClass = ["urgent", "medical"].includes(order.priority) ? "priority" : "normal";
    const delivered = order.status === "delivered";
    return `
      <g class="order ${priorityClass} ${delivered ? "delivered" : ""}">
        <rect x="${x}" y="${y - 16}" width="38" height="20" rx="8" />
        <text x="${x + 19}" y="${y - 2}" text-anchor="middle">${escapeHtml(order.order_id)}</text>
        <title>${order.priority} | ${order.status} | deadline ${order.deadline}</title>
      </g>
    `;
  }).join("");

  const droneMarkup = fleet.map((drone, index) => {
    const zone = ZONES[drone.current_zone] || ZONES.hub;
    const offsetX = 36 + (index % 3) * 43;
    const offsetY = 42 + Math.floor(index / 3) * 36;
    const x = zone.x + offsetX;
    const y = zone.y + offsetY;
    const risk = drone.health_risk === "critical" || Number(drone.battery || 0) <= 20 ? "critical" : drone.health_risk === "high" ? "warning" : "ok";
    const batteryWidth = Math.max(4, Math.min(34, Number(drone.battery || 0) * 0.34));
    return `
      <g class="drone ${risk}">
        <circle cx="${x}" cy="${y}" r="17" />
        <text class="drone-label" x="${x}" y="${y + 4}" text-anchor="middle">${escapeHtml(drone.drone_id)}</text>
        <rect class="battery-shell" x="${x - 18}" y="${y + 22}" width="36" height="7" rx="3" />
        <rect class="battery-fill" x="${x - 17}" y="${y + 23}" width="${batteryWidth}" height="5" rx="2" />
        <text class="drone-meta" x="${x}" y="${y + 42}" text-anchor="middle">${escapeHtml(drone.status)} · ${drone.battery}%</text>
        <title>${drone.drone_id}: ${drone.status}, zone ${drone.current_zone}, target ${drone.target_zone || "none"}, assigned ${drone.assigned_order_id || "none"}</title>
      </g>
    `;
  }).join("");

  mapSvg.innerHTML = `
    <defs>
      <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="8" stdDeviation="9" flood-color="#132217" flood-opacity="0.16" />
      </filter>
    </defs>
    <rect class="map-bg" x="0" y="0" width="980" height="640" rx="32" />
    ${roads}
    ${pathMarkup}
    ${zoneMarkup}
    ${chargingMarkup}
    ${orderMarkup}
    ${droneMarkup}
  `;
}

function renderMetrics() {
  const frame = currentFrame();
  const observation = frameObservation(frame);
  if (!observation) {
    metricsGrid.innerHTML = "";
    return;
  }

  const orders = observation.orders || [];
  const fleet = observation.fleet || [];
  const completedDeliveries = orders.filter((order) => order.status === "delivered").length;
  const urgentDelivered = orders.filter((order) => order.status === "delivered" && ["urgent", "medical"].includes(order.priority)).length;
  const pendingOrders = orders.filter((order) => !["delivered", "canceled"].includes(order.status)).length;
  const urgentPending = orders.filter((order) => !["delivered", "canceled"].includes(order.status) && ["urgent", "medical"].includes(order.priority)).length;
  const safetyViolations = state.comparison?.policy_results?.[policySelect.value]?.[scenarioSelect.value]?.safety_violations;
  const invalidActions = frame?.invalid_action_count ?? frameInfo(frame)?.invalid_action_count ?? 0;
  const minBattery = fleet.length ? Math.min(...fleet.map((drone) => Number(drone.battery || 0))) : 0;
  const stepReward = frame?.step_reward ?? frame?.reward ?? frame?.reward_breakdown?.total ?? 0;
  const cumulativeReward = frame?.cumulative_reward ?? frameInfo(frame)?.cumulative_reward?.total ?? 0;

  const metrics = [
    ["Frame", `${state.currentFrameIndex + 1}/${state.frames.length}`],
    ["Step Reward", Number(stepReward).toFixed(1)],
    ["Total Reward", Number(cumulativeReward).toFixed(1)],
    ["Deliveries", String(completedDeliveries)],
    ["Urgent Done", String(urgentDelivered)],
    ["Urgent Pending", String(urgentPending)],
    ["Pending", String(pendingOrders)],
    ["Invalid", String(invalidActions)],
    ["Safety", safetyViolations === undefined ? "trace" : String(safetyViolations)],
    ["Min Battery", `${minBattery}%`],
  ];

  metricsGrid.innerHTML = metrics.map(([label, value]) => `
    <div class="metric">
      <div class="metric-label">${escapeHtml(label)}</div>
      <div class="metric-value">${escapeHtml(value)}</div>
    </div>
  `).join("");
}

function renderComparison() {
  const scenario = scenarioSelect.value;
  const policy = policySelect.value;
  const row = state.comparison?.policy_results?.[policy]?.[scenario];
  const improved = state.comparison?.policy_results?.improved?.demo;
  const heuristic = state.comparison?.policy_results?.heuristic?.demo;

  const headline = improved && heuristic ? `
    <div class="delta-grid">
      <span>Heuristic reward</span><b>${Number(heuristic.total_reward).toFixed(1)}</b>
      <span>Improved reward</span><b>${Number(improved.total_reward).toFixed(1)}</b>
      <span>Safety violations</span><b>${heuristic.safety_violations} → ${improved.safety_violations}</b>
      <span>Invalid actions</span><b>${heuristic.invalid_action_count} → ${improved.invalid_action_count}</b>
      <span>Urgent successes</span><b>${heuristic.urgent_successes} → ${improved.urgent_successes}</b>
    </div>
  ` : "<p>Comparison metrics unavailable.</p>";

  const selected = row ? `
    <p class="selected-policy">Now showing <strong>${escapeHtml(policy)}</strong> on <strong>${escapeHtml(scenario)}</strong>:
    reward ${Number(row.total_reward).toFixed(1)}, score ${Number(row.normalized_score).toFixed(3)}.</p>
  ` : "";

  comparisonSummary.innerHTML = `${headline}${selected}`;
}

function renderRewardBreakdown() {
  const frame = currentFrame();
  const breakdown = frameInfo(frame)?.cumulative_reward || frame?.reward_breakdown || { positive: {}, negative: {}, total: frame?.cumulative_reward || 0 };
  const positive = Object.entries(breakdown.positive || {}).map(([key, value]) => `<li><span>${escapeHtml(key)}</span><strong>+${Number(value).toFixed(1)}</strong></li>`).join("");
  const negative = Object.entries(breakdown.negative || {}).map(([key, value]) => `<li><span>${escapeHtml(key)}</span><strong>${Number(value).toFixed(1)}</strong></li>`).join("");
  rewardBreakdown.innerHTML = `
    <div class="reward-columns">
      <div><h3>Positive</h3><ul>${positive || "<li><span>none</span><strong>0.0</strong></li>"}</ul></div>
      <div><h3>Negative</h3><ul>${negative || "<li><span>none</span><strong>0.0</strong></li>"}</ul></div>
    </div>
    <p class="reward-total">Cumulative: ${Number(breakdown.total || frame?.cumulative_reward || 0).toFixed(1)}</p>
  `;
}

function renderEvents() {
  const observation = frameObservation();
  const events = observation?.recent_events?.slice(-8) || [];
  eventsList.innerHTML = events.length ? events.map((event) => `<li>${escapeHtml(event)}</li>`).join("") : "<li>No recent events.</li>";
}

function renderSummary() {
  const frame = currentFrame();
  const observation = frameObservation(frame);
  frameSummary.textContent = frame?.summary || observation?.summary || "No summary available.";
}

function renderAction() {
  const frame = currentFrame();
  lastActionBox.textContent = JSON.stringify(frame?.action || { action: "initial_state" }, null, 2);
}

function playReplay() {
  pauseReplay();
  if (state.currentFrameIndex >= state.frames.length - 1) state.currentFrameIndex = 0;
  state.timerId = window.setInterval(() => {
    state.currentFrameIndex = Math.min(state.frames.length - 1, state.currentFrameIndex + 1);
    render();
    if (state.currentFrameIndex >= state.frames.length - 1) pauseReplay();
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
  pauseReplay();
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
scenarioSelect.addEventListener("change", loadTrace);
policySelect.addEventListener("change", loadTrace);

loadTrace();
