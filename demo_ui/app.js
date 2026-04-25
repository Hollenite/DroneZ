const ZONES = {
  hub: { x: 455, y: 325, w: 170, h: 120, label: "Central Hub" },
  Z1: { x: 115, y: 105, w: 185, h: 125, label: "Z1 Downtown" },
  Z2: { x: 450, y: 90, w: 190, h: 130, label: "Z2 Hospital" },
  Z3: { x: 780, y: 120, w: 185, h: 125, label: "Z3 East" },
  Z4: { x: 120, y: 500, w: 185, h: 125, label: "Z4 Market" },
  Z5: { x: 455, y: 530, w: 190, h: 125, label: "Z5 Campus" },
  Z6: { x: 785, y: 500, w: 185, h: 125, label: "Z6 Suburb" },
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
const stageModeToggle = $("stageModeToggle");
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
const telemetryPanel = $("telemetryPanel");
const environmentPanel = $("environmentPanel");
const towerPanel = $("towerPanel");
const learnPanel = $("learnPanel");

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function traceCandidates(scenario, policy) {
  const enriched = `${scenario}_${policy}_enriched.json`;
  const raw = `${scenario}_${policy}_trace.json`;
  return [
    `../artifacts/traces/${enriched}`,
    `/artifacts/traces/${enriched}`,
    `artifacts/traces/${enriched}`,
    `../artifacts/traces/${raw}`,
    `/artifacts/traces/${raw}`,
    `artifacts/traces/${raw}`,
  ];
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

function frameVisualization(frame = currentFrame()) {
  return frame?.visualization || null;
}

function render() {
  renderBadges();
  renderMap();
  renderMetrics();
  renderTelemetry();
  renderEnvironment();
  renderTower();
  renderComparison();
  renderRewardBreakdown();
  renderEvents();
  renderSummary();
  renderAction();
  renderLearn();
}

function renderBadges() {
  const frame = currentFrame();
  scenarioBadge.textContent = `scenario: ${state.payload?.task_id || scenarioSelect.value}`;
  policyBadge.textContent = `policy: ${state.payload?.policy_id || policySelect.value}`;
  frameIndicator.textContent = `Frame ${state.currentFrameIndex + 1} / ${state.frames.length}`;
  const progress = state.frames.length <= 1 ? 100 : (state.currentFrameIndex / (state.frames.length - 1)) * 100;
  progressBar.style.width = `${progress}%`;
  document.body.classList.toggle("fallback-mode", !state.traceLoaded);
  document.body.classList.toggle("stage-mode", Boolean(stageModeToggle?.checked));
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
  const visual = frameVisualization(frame);
  if (!observation) {
    mapSvg.innerHTML = `<text x="40" y="80" fill="#c75146">No observation available for this frame.</text>`;
    return;
  }

  const sectorsById = new Map((observation.city?.sectors || []).map((sector) => [sector.zone_id, sector]));
  const sectors = visual?.zone_layout || Object.keys(ZONES).map((zoneId) => sectorsById.get(zoneId) || {
    zone_id: zoneId,
    label: ZONES[zoneId]?.label || zoneId,
    x: ZONES[zoneId]?.x,
    y: ZONES[zoneId]?.y,
    w: ZONES[zoneId]?.w,
    h: ZONES[zoneId]?.h,
    weather: "clear",
    wind_speed_kph: 10,
    risk_score: 0,
    congestion_score: 0,
    is_no_fly: false,
    operations_paused: false,
  });
  const fleet = observation.fleet || [];
  const orders = observation.orders || [];
  const charging = observation.charging || [];
  const activeNoFly = new Set(observation.city?.active_no_fly_zones || []);
  const heldZones = new Set(observation.city?.held_zones || []);
  const telemetry = visual?.drone_telemetry || [];

  const roads = [
    ["hub", "Z1"], ["hub", "Z2"], ["hub", "Z3"], ["hub", "Z4"], ["hub", "Z5"], ["hub", "Z6"],
    ["Z1", "Z2"], ["Z2", "Z3"], ["Z4", "Z5"], ["Z5", "Z6"],
  ].map(([a, b]) => {
    const p1 = zoneCenter(a);
    const p2 = zoneCenter(b);
    return `<line class="road" x1="${p1.x}" y1="${p1.y}" x2="${p2.x}" y2="${p2.y}" />`;
  }).join("");

  const cityBlocks = Object.entries(ZONES).map(([zoneId, zone], index) => {
    const center = zoneCenter(zoneId);
    const height = zoneId === "hub" ? 54 : 20 + (index % 4) * 12;
    return `
      <g class="building">
        <polygon points="${center.x - 42},${center.y + 58} ${center.x + 34},${center.y + 37} ${center.x + 58},${center.y + 52} ${center.x - 16},${center.y + 74}" />
        <polygon points="${center.x + 34},${center.y + 37} ${center.x + 58},${center.y + 52} ${center.x + 58},${center.y + 52 - height} ${center.x + 34},${center.y + 37 - height}" />
        <polygon points="${center.x - 42},${center.y + 58} ${center.x + 34},${center.y + 37} ${center.x + 34},${center.y + 37 - height} ${center.x - 42},${center.y + 58 - height}" />
      </g>
    `;
  }).join("");

  const zoneMarkup = sectors.map((sector) => {
    const zone = {
      ...(ZONES[sector.zone_id] || ZONES.hub),
      ...sector,
    };
    const badges = [
      sector.is_no_fly || activeNoFly.has(sector.zone_id) ? "NO-FLY" : null,
      sector.operations_paused || heldZones.has(sector.zone_id) ? "PAUSED" : null,
      sector.weather && sector.weather !== "clear" ? sector.weather.replace("_", " ") : null,
      Number(sector.congestion_score || 0) >= 0.7 ? "CONGESTED" : null,
    ].filter(Boolean);
    const risk = Math.round(Number(sector.risk_score || 0) * 100);

    return `
      <g class="${sectorClass(sector)}">
        <rect x="${zone.x}" y="${zone.y}" width="${zone.w}" height="${zone.h}" rx="22" />
        <text class="zone-title" x="${Number(zone.x) + 16}" y="${Number(zone.y) + 28}">${escapeHtml(zone.label)}</text>
        <text class="zone-detail" x="${Number(zone.x) + 16}" y="${Number(zone.y) + 52}">weather: ${escapeHtml(sector.weather || "clear")} · wind ${escapeHtml(sector.wind_speed_kph || 0)} kph</text>
        <text class="zone-detail" x="${Number(zone.x) + 16}" y="${Number(zone.y) + 72}">risk: ${risk}% · congestion ${Number(sector.congestion_score || 0).toFixed(2)}</text>
        <text class="zone-detail" x="${Number(zone.x) + 16}" y="${Number(zone.y) + 92}">orders: ${orders.filter((order) => order.zone_id === sector.zone_id && order.status !== "delivered").length}</text>
        ${badges.map((badge, index) => `<text class="zone-badge" x="${zone.x + 16}" y="${zone.y + 113 + index * 15}">${escapeHtml(badge)}</text>`).join("")}
      </g>
    `;
  }).join("");

  const pathMarkup = visual?.route_segments?.length ? visual.route_segments.map((segment, index) => {
    const points = segment.points || [];
    const path = points.length >= 4
      ? `M ${points[0].x},${points[0].y} C ${points[1].x},${points[1].y} ${points[2].x},${points[2].y} ${points[3].x},${points[3].y}`
      : "";
    if (!path) return "";
    const routeClass = `flight-path route-${segment.route_color || "purple"}`;
    return `
      <g class="route-group">
        <path class="${routeClass}" d="${path}" />
        <circle class="path-pulse route-${segment.route_color || "purple"}" r="6">
          <animateMotion dur="${2.8 + index * 0.25}s" repeatCount="indefinite" path="${path}" />
        </circle>
        <text class="route-label" x="${points[1].x}" y="${points[1].y - 12}">${escapeHtml(segment.route_type || "route")}</text>
      </g>
    `;
  }).join("") : fleet.filter((drone) => drone.target_zone).map((drone, index) => {
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
    const droneTelemetry = telemetry.find((item) => item.drone_id === drone.drone_id);
    const zone = ZONES[drone.current_zone] || ZONES.hub;
    const offsetX = 36 + (index % 3) * 43;
    const offsetY = 42 + Math.floor(index / 3) * 36;
    const x = droneTelemetry?.x ?? zone.x + offsetX;
    const y = droneTelemetry?.y ?? zone.y + offsetY;
    const risk = drone.health_risk === "critical" || Number(drone.battery || 0) <= 20 ? "critical" : drone.health_risk === "high" ? "warning" : "ok";
    const batteryWidth = Math.max(4, Math.min(34, Number(drone.battery || 0) * 0.34));
    const altitude = droneTelemetry?.altitude_m ?? 80;
    const speed = droneTelemetry?.speed_kph ?? 0;
    return `
      <g class="drone ${risk}">
        <circle class="drone-aura" cx="${x}" cy="${y}" r="29" />
        <path class="drone-body" d="M ${x - 17},${y} L ${x},${y - 13} L ${x + 17},${y} L ${x},${y + 13} Z" />
        <line class="drone-rotor" x1="${x - 28}" y1="${y - 15}" x2="${x - 8}" y2="${y - 8}" />
        <line class="drone-rotor" x1="${x + 8}" y1="${y - 8}" x2="${x + 28}" y2="${y - 15}" />
        <line class="drone-rotor" x1="${x - 28}" y1="${y + 15}" x2="${x - 8}" y2="${y + 8}" />
        <line class="drone-rotor" x1="${x + 8}" y1="${y + 8}" x2="${x + 28}" y2="${y + 15}" />
        <text class="drone-label" x="${x}" y="${y + 4}" text-anchor="middle">${escapeHtml(drone.drone_id)}</text>
        <rect class="battery-shell" x="${x - 18}" y="${y + 22}" width="36" height="7" rx="3" />
        <rect class="battery-fill" x="${x - 17}" y="${y + 23}" width="${batteryWidth}" height="5" rx="2" />
        <text class="drone-meta" x="${x}" y="${y + 43}" text-anchor="middle">${escapeHtml(drone.status)} · ${drone.battery}% · ${altitude}m · ${speed}kph</text>
        <title>${drone.drone_id}: ${drone.status}, zone ${drone.current_zone}, target ${drone.target_zone || "none"}, assigned ${drone.assigned_order_id || "none"}</title>
      </g>
    `;
  }).join("");

  mapSvg.innerHTML = `
    <defs>
      <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="8" stdDeviation="9" flood-color="#132217" flood-opacity="0.16" />
      </filter>
      <pattern id="cityGrid" width="44" height="44" patternUnits="userSpaceOnUse">
        <path d="M 44 0 L 0 0 0 44" fill="none" stroke="rgba(111, 255, 213, 0.07)" stroke-width="1" />
      </pattern>
    </defs>
    <rect class="map-bg" x="0" y="0" width="980" height="680" rx="32" />
    <rect class="map-grid" x="18" y="18" width="944" height="644" rx="28" />
    ${roads}
    ${cityBlocks}
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

function formatNumber(value, fallback = "0") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return fallback;
  return Number(value).toFixed(Number(value) % 1 === 0 ? 0 : 1);
}

function formatPercent(value, fallback = "0%") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return fallback;
  return `${Math.round(Number(value) * 100)}%`;
}

function statusTone(value) {
  const normalized = String(value || "").toLowerCase();
  if (["critical", "blocked", "high", "lost", "unsafe"].some((term) => normalized.includes(term))) return "danger";
  if (["warning", "caution", "degraded", "queued", "moderate"].some((term) => normalized.includes(term))) return "warn";
  return "ok";
}

function renderTelemetry() {
  if (!telemetryPanel) return;
  const observation = frameObservation();
  const visual = frameVisualization();
  const fleet = observation?.fleet || [];
  const telemetry = visual?.drone_telemetry?.length
    ? visual.drone_telemetry
    : fleet.map((drone, index) => ({
        drone_id: drone.drone_id,
        status: drone.status,
        zone: drone.current_zone,
        altitude_m: drone.status === "idle" ? 0 : 70 + index * 12,
        speed_kph: drone.status === "idle" ? 0 : 32 + index * 5,
        battery: drone.battery,
        wind_exposure: drone.health_risk || "low",
        payload: drone.assigned_order_id ? "assigned" : "none",
        gps_lock: true,
        imu_status: "nominal",
        camera_status: "ready",
        lidar_status: "ready",
        thermal_status: "ready",
        sensor_fusion_confidence: 0.94,
        health_state: drone.health_risk || "low",
        assigned_order: drone.assigned_order_id || "none",
        eta_steps: drone.eta_steps ?? "n/a",
        current_action: currentFrame()?.action?.action || "monitor",
        route_risk: drone.health_risk || "low",
      }));

  telemetryPanel.innerHTML = `
    <p class="panel-note">Simulated visualization telemetry derived from environment traces, not real aircraft sensor streams.</p>
    ${telemetry.map((item) => {
      const battery = Math.max(0, Math.min(100, Number(item.battery || 0)));
      const confidence = Math.max(0, Math.min(1, Number(item.sensor_fusion_confidence || 0)));
      return `
        <article class="telemetry-unit">
          <div class="unit-head">
            <strong>${escapeHtml(item.drone_id)}</strong>
            <span class="status-chip ${statusTone(item.health_state)}">${escapeHtml(item.health_state || item.status || "nominal")}</span>
          </div>
          <div class="unit-grid">
            <span>Zone</span><b>${escapeHtml(item.zone || item.position_zone || "hub")}</b>
            <span>Battery</span><b>${battery}%</b>
            <span>Altitude</span><b>${escapeHtml(formatNumber(item.altitude_m))} m</b>
            <span>Speed</span><b>${escapeHtml(formatNumber(item.speed_kph))} kph</b>
            <span>Wind</span><b>${escapeHtml(item.wind_exposure || "low")}</b>
            <span>ETA</span><b>${escapeHtml(item.eta_steps ?? item.eta ?? "n/a")}</b>
            <span>Payload</span><b>${escapeHtml(item.payload ?? (item.payload_kg === undefined ? "none" : `${item.payload_kg} kg`))}</b>
            <span>Order</span><b>${escapeHtml(item.assigned_order || "none")}</b>
            <span>Action</span><b>${escapeHtml(item.current_action || "monitor")}</b>
            <span>Route risk</span><b>${escapeHtml(item.route_risk || "nominal")}</b>
          </div>
          <div class="bar-row"><span>Battery</span><i style="--w:${battery}%"></i></div>
          <div class="bar-row"><span>Sensor fusion</span><i style="--w:${Math.round(confidence * 100)}%"></i></div>
          <div class="sensor-line">
            <span class="${item.gps_lock ? "ok" : "danger"}">GPS ${item.gps_lock ? "lock" : "lost"}</span>
            <span>${escapeHtml(item.imu_status || "IMU nominal")}</span>
            <span>${escapeHtml(item.camera_status || "camera ready")}</span>
            <span>${escapeHtml(item.lidar_status || "LiDAR ready")}</span>
            <span>${escapeHtml(item.thermal_status || "thermal ready")}</span>
          </div>
        </article>
      `;
    }).join("")}
  `;
}

function renderEnvironment() {
  if (!environmentPanel) return;
  const observation = frameObservation();
  const visual = frameVisualization();
  const environment = visual?.environment || {};
  const zones = visual?.zone_layout || observation?.city?.sectors || [];
  const alerts = environment.alerts || environment.active_alerts || observation?.warnings || observation?.recent_events?.slice(-3) || [];
  const stormZones = environment.storm_zones || zones.filter((zone) => String(zone.weather || "").includes("storm")).map((zone) => zone.zone_id);
  const restrictedZones = environment.restricted_zones || observation?.city?.active_no_fly_zones || [];
  const maxWind = environment.max_wind_kph ?? Math.max(0, ...zones.map((zone) => Number(zone.wind_speed_kph || 0)));
  const congestion = environment.congestion_index ?? (
    zones.length ? zones.reduce((sum, zone) => sum + Number(zone.congestion_score || 0), 0) / zones.length : 0
  );

  environmentPanel.innerHTML = `
    <div class="environment-grid">
      <div><span>Weather</span><b>${escapeHtml(environment.dominant_weather || "mixed urban")}</b></div>
      <div><span>Max wind</span><b>${escapeHtml(formatNumber(maxWind))} kph</b></div>
      <div><span>Storm zones</span><b>${escapeHtml(stormZones.length ? stormZones.join(", ") : "none")}</b></div>
      <div><span>No-fly</span><b>${escapeHtml(restrictedZones.length ? restrictedZones.join(", ") : "none")}</b></div>
      <div><span>Congestion</span><b>${escapeHtml(formatPercent(congestion))}</b></div>
      <div><span>Dynamic restrictions</span><b>${escapeHtml((environment.dynamic_restrictions || []).join(", ") || "monitored")}</b></div>
    </div>
    <div class="alert-stack">
      ${(alerts.length ? alerts : ["No active environment alerts."]).map((alert) => `<span>${escapeHtml(alert)}</span>`).join("")}
    </div>
  `;
}

function renderTower() {
  if (!towerPanel) return;
  const visual = frameVisualization();
  const tower = visual?.tower || {};
  const controlLayers = visual?.control_layers || {};
  const normalizeLayer = (layer, fallback) => {
    if (Array.isArray(layer)) return layer;
    if (layer && typeof layer === "object") {
      return Object.entries(layer).map(([key, value]) => `${key.replaceAll("_", " ")}: ${value}`);
    }
    return fallback;
  };
  const fleetHealth = tower.fleet_health && typeof tower.fleet_health === "object"
    ? Object.entries(tower.fleet_health).map(([key, value]) => `${key.replaceAll("_", " ")} ${value}`).join(", ")
    : tower.fleet_health;
  const layerGroups = [
    ["Low-level drone system", normalizeLayer(controlLayers.low_level_drone, ["PID stability", "sensor fusion", "GPS navigation", "safety rules"])],
    ["High-level RL / AI", normalizeLayer(controlLayers.high_level_rl_ai, ["fleet assignment", "route adaptation", "charging decisions", "reward optimization"])],
    ["Control tower / parent server", normalizeLayer(controlLayers.control_tower_parent_server || controlLayers.control_tower, ["fleet monitoring", "global route planning", "emergency override", "organization policy"])],
  ];

  towerPanel.innerHTML = `
    <div class="tower-status">
      <div><span>Dispatch queue</span><b>${escapeHtml((tower.dispatch_queue || []).join(", ") || "clear")}</b></div>
      <div><span>Urgent queue</span><b>${escapeHtml((tower.urgent_queue || []).join(", ") || "clear")}</b></div>
      <div><span>Override</span><b>${escapeHtml(tower.override_status || "human-on-loop")}</b></div>
      <div><span>RL recommendation</span><b>${escapeHtml(tower.rl_recommendation || currentFrame()?.action?.action || "monitor")}</b></div>
      <div><span>Fleet health</span><b>${escapeHtml(fleetHealth || "nominal")}</b></div>
    </div>
    <div class="architecture-stack">
      ${layerGroups.map(([title, items]) => `
        <article class="architecture-layer">
          <h3>${escapeHtml(title)}</h3>
          <div>${items.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</div>
        </article>
      `).join("")}
    </div>
  `;
}

function renderLearn() {
  if (!learnPanel) return;
  const observation = frameObservation();
  const frame = currentFrame();
  const actionName = frame?.action?.action || "initial_state";
  const reward = frame?.step_reward ?? frame?.reward ?? 0;
  const droneCount = observation?.fleet?.length || 0;
  const orderCount = observation?.orders?.length || 0;

  learnPanel.innerHTML = `
    <div class="learn-step"><b>Observation</b><span>The agent sees ${droneCount} drones, ${orderCount} orders, weather, no-fly zones, chargers, and recent events.</span></div>
    <div class="learn-step"><b>Action</b><span>Current decision: <strong>${escapeHtml(actionName)}</strong>. This is mission-level control, not motor control.</span></div>
    <div class="learn-step"><b>Step</b><span>The environment advances one operation step and updates drones, orders, risks, and events.</span></div>
    <div class="learn-step"><b>Reward</b><span>This step reward is <strong>${escapeHtml(formatNumber(reward))}</strong>. Safety, delivery, battery, deadlines, and invalid actions affect it.</span></div>
    <div class="learn-step"><b>Learn</b><span>The old RTX 5060 GRPO attempt did not improve because invalid actions dominated. Candidate-choice plus SFT warm start teaches valid actions before RL.</span></div>
  `;
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
stageModeToggle?.addEventListener("change", render);
scenarioSelect.addEventListener("change", loadTrace);
policySelect.addEventListener("change", loadTrace);

loadTrace();
