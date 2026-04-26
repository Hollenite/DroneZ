const canvas = document.getElementById("simCanvas");
const ctx = canvas.getContext("2d");

const playBtn = document.getElementById("playBtn");
const pauseBtn = document.getElementById("pauseBtn");
const resetBtn = document.getElementById("resetBtn");
const stageBtn = document.getElementById("stageBtn");
const speedRange = document.getElementById("speedRange");
const frameBadge = document.getElementById("frameBadge");
const rewardMetric = document.getElementById("rewardMetric");
const modeMetric = document.getElementById("modeMetric");
const telemetryList = document.getElementById("telemetryList");
const weatherPanel = document.getElementById("weatherPanel");
const towerPanel = document.getElementById("towerPanel");

const zoneLayout = {
  hub: { x: 570, y: 330, w: 170, h: 120, label: "Central Hub" },
  Z1: { x: 135, y: 110, w: 190, h: 130, label: "Downtown" },
  Z2: { x: 535, y: 92, w: 200, h: 136, label: "Hospital" },
  Z3: { x: 930, y: 128, w: 190, h: 130, label: "East Logistics" },
  Z4: { x: 140, y: 560, w: 190, h: 130, label: "Market" },
  Z5: { x: 560, y: 585, w: 200, h: 130, label: "Campus" },
  Z6: { x: 945, y: 555, w: 190, h: 130, label: "Suburb" },
};

const fallbackFrames = makeFallbackFrames();

const state = {
  payload: null,
  frames: fallbackFrames,
  frameIndex: 0,
  playing: false,
  lastTime: 0,
  accumulator: 0,
  traceMode: "Fallback",
};

function resizeCanvasForDisplay() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(900, Math.floor(rect.width * dpr));
  const height = Math.max(560, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.setTransform(width / 1280, 0, 0, height / 780, 0, 0);
}

async function loadTrace() {
  try {
    const response = await fetch("../artifacts/traces/demo_improved_enriched.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.payload = await response.json();
    state.frames = state.payload.frames || fallbackFrames;
    state.traceMode = "Trace";
  } catch {
    state.payload = { summary: { total_reward: 89.0 } };
    state.frames = fallbackFrames;
    state.traceMode = "Fallback";
  }
  renderPanels();
}

function currentFrame() {
  return state.frames[state.frameIndex] || state.frames[0];
}

function currentVisual() {
  return currentFrame().visualization || fallbackFrames[state.frameIndex % fallbackFrames.length].visualization;
}

function zoneCenter(zoneId) {
  const zone = zoneLayout[zoneId] || zoneLayout.hub;
  return { x: zone.x + zone.w / 2, y: zone.y + zone.h / 2 };
}

function draw(now = 0) {
  resizeCanvasForDisplay();
  ctx.clearRect(0, 0, 1280, 780);
  drawBackground(now);
  drawRoads();
  drawBuildings(now);
  drawZones();
  drawRoutes(now);
  drawOrders();
  drawDrones(now);
  drawHud();
}

function drawBackground(now) {
  const grd = ctx.createLinearGradient(0, 0, 1280, 780);
  grd.addColorStop(0, "#061420");
  grd.addColorStop(0.55, "#020910");
  grd.addColorStop(1, "#071724");
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, 1280, 780);

  ctx.save();
  ctx.globalAlpha = 0.18;
  ctx.strokeStyle = "#64ffe3";
  ctx.lineWidth = 1;
  const shift = (now / 80) % 42;
  for (let x = -80 + shift; x < 1400; x += 42) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x - 280, 780);
    ctx.stroke();
  }
  for (let y = -80 + shift; y < 900; y += 42) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(1280, y - 220);
    ctx.stroke();
  }
  ctx.restore();
}

function drawRoads() {
  ctx.save();
  ctx.strokeStyle = "rgba(91,169,255,0.20)";
  ctx.lineWidth = 22;
  ctx.lineCap = "round";
  const roads = [
    ["Z1", "hub"], ["Z2", "hub"], ["Z3", "hub"], ["Z4", "hub"], ["Z5", "hub"], ["Z6", "hub"],
    ["Z1", "Z2"], ["Z2", "Z3"], ["Z4", "Z5"], ["Z5", "Z6"],
  ];
  for (const [a, b] of roads) {
    const s = zoneCenter(a);
    const e = zoneCenter(b);
    ctx.beginPath();
    ctx.moveTo(s.x, s.y);
    ctx.bezierCurveTo((s.x + e.x) / 2, s.y, (s.x + e.x) / 2, e.y, e.x, e.y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawBuildings(now) {
  ctx.save();
  const t = now / 1000;
  for (let i = 0; i < 46; i += 1) {
    const x = 70 + (i * 137) % 1110;
    const y = 74 + (i * 211) % 620;
    const h = 22 + ((i * 17) % 70);
    const glow = 0.08 + Math.sin(t + i) * 0.02;
    drawIsoBlock(x, y, 24 + (i % 4) * 8, 18 + (i % 3) * 7, h, `rgba(100,255,227,${glow})`);
  }
  ctx.restore();
}

function drawIsoBlock(x, y, w, d, h, color) {
  ctx.fillStyle = "rgba(42,74,94,0.22)";
  ctx.beginPath();
  ctx.moveTo(x, y - h);
  ctx.lineTo(x + w, y - h + d * 0.45);
  ctx.lineTo(x + w, y + d * 0.45);
  ctx.lineTo(x, y);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = color;
  ctx.fillRect(x + 4, y - h + 8, Math.max(2, w - 8), 3);
}

function drawZones() {
  const visual = currentVisual();
  const zones = visual.zone_layout || Object.entries(zoneLayout).map(([zone_id, z]) => ({ zone_id, ...z }));
  for (const zone of zones) {
    const z = zoneLayout[zone.zone_id] || zone;
    const risk = Number(zone.risk_score || 0);
    const color = zone.is_no_fly ? "#ff5870" : zone.weather === "storm" ? "#5ba9ff" : zone.operations_paused ? "#ffd166" : "#64ffe3";
    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = zone.zone_id === "hub" ? 28 : 16;
    ctx.fillStyle = zone.is_no_fly ? "rgba(95,24,36,0.64)" : zone.weather === "storm" ? "rgba(22,48,86,0.66)" : "rgba(8,30,43,0.74)";
    roundRect(z.x, z.y, z.w, z.h, 22);
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#eef9ff";
    ctx.font = "900 18px Avenir Next, sans-serif";
    ctx.fillText(z.label || zone.label || zone.zone_id, z.x + 18, z.y + 34);
    ctx.fillStyle = "#9ab3c4";
    ctx.font = "800 12px Avenir Next, sans-serif";
    ctx.fillText(`wind ${zone.wind_speed_kph || 12} kph | risk ${Math.round(risk * 100)}%`, z.x + 18, z.y + 58);
    if (zone.is_no_fly || zone.weather === "storm" || zone.operations_paused) {
      ctx.fillStyle = color;
      ctx.font = "950 12px Avenir Next, sans-serif";
      ctx.fillText(zone.is_no_fly ? "NO-FLY" : zone.weather === "storm" ? "STORM" : "PAUSED", z.x + 18, z.y + 84);
    }
    ctx.restore();
  }
}

function drawRoutes(now) {
  const visual = currentVisual();
  const routes = visual.route_segments || [];
  const pulse = (Math.sin(now / 220) + 1) / 2;
  for (const route of routes) {
    const pts = route.points || [];
    if (pts.length < 2) continue;
    const color = route.route_color === "green" ? "#65f5a4" : route.route_color === "yellow" ? "#ffd166" : route.route_color === "red" ? "#ff5870" : "#a882ff";
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 5;
    ctx.setLineDash([18, 12]);
    ctx.lineDashOffset = -now / 60;
    ctx.shadowColor = color;
    ctx.shadowBlur = 18;
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length - 2; i += 1) {
      ctx.quadraticCurveTo(pts[i].x, pts[i].y, pts[i + 1].x, pts[i + 1].y);
    }
    const last = pts[pts.length - 1];
    ctx.lineTo(last.x, last.y);
    ctx.stroke();
    ctx.setLineDash([]);
    const p = pointOnPolyline(pts, pulse);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

function drawOrders() {
  const observation = currentFrame().observation || {};
  const orders = observation.orders || [];
  for (const order of orders) {
    if (order.status === "delivered" || order.status === "canceled") continue;
    const c = zoneCenter(order.zone_id);
    const urgent = ["urgent", "medical"].includes(order.priority);
    ctx.save();
    ctx.fillStyle = urgent ? "#ffd166" : "#5ba9ff";
    ctx.shadowColor = ctx.fillStyle;
    ctx.shadowBlur = 18;
    ctx.beginPath();
    ctx.roundRect(c.x + 42, c.y - 34, 48, 24, 8);
    ctx.fill();
    ctx.fillStyle = "#041014";
    ctx.font = "950 11px Avenir Next, sans-serif";
    ctx.fillText(order.order_id, c.x + 53, c.y - 18);
    ctx.restore();
  }
}

function drawDrones(now) {
  const visual = currentVisual();
  const drones = visual.drone_telemetry || [];
  const hover = Math.sin(now / 260) * 6;
  for (const drone of drones) {
    const x = drone.x || zoneCenter(drone.zone).x;
    const y = (drone.y || zoneCenter(drone.zone).y) + hover;
    const statusColor = drone.route_risk === "blocked" ? "#ff5870" : drone.route_risk === "caution" ? "#ffd166" : "#64ffe3";
    ctx.save();
    ctx.translate(x, y);
    ctx.shadowColor = statusColor;
    ctx.shadowBlur = 24;
    ctx.fillStyle = "rgba(100,255,227,0.12)";
    ctx.beginPath();
    ctx.arc(0, 0, 34, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = statusColor;
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(-24, 0);
    ctx.lineTo(24, 0);
    ctx.moveTo(0, -20);
    ctx.lineTo(0, 20);
    ctx.stroke();
    ctx.fillStyle = "#eef9ff";
    ctx.beginPath();
    ctx.roundRect(-15, -10, 30, 20, 8);
    ctx.fill();
    ctx.fillStyle = "#041014";
    ctx.font = "950 9px Avenir Next, sans-serif";
    ctx.fillText(String(drone.drone_id || "DR").slice(0, 4), -12, 4);
    ctx.restore();

    ctx.fillStyle = "#dff6ff";
    ctx.font = "900 11px Avenir Next, sans-serif";
    ctx.fillText(`${drone.drone_id} | ${Math.round(drone.battery || 0)}% | ${drone.altitude_m || 0}m`, x - 42, y + 52);
  }
}

function drawHud() {
  const frame = currentFrame();
  const visual = currentVisual();
  const total = frame.info?.cumulative_reward?.total ?? frame.cumulative_reward ?? state.payload?.summary?.total_reward ?? 0;
  ctx.save();
  ctx.fillStyle = "rgba(2,8,13,0.66)";
  roundRect(24, 24, 330, 110, 18);
  ctx.fill();
  ctx.strokeStyle = "rgba(100,255,227,0.28)";
  ctx.stroke();
  ctx.fillStyle = "#64ffe3";
  ctx.font = "950 13px Avenir Next, sans-serif";
  ctx.fillText("MISSION CONTROL", 46, 56);
  ctx.fillStyle = "#eef9ff";
  ctx.font = "950 28px Avenir Next, sans-serif";
  ctx.fillText(`Reward ${Number(total).toFixed(1)}`, 46, 92);
  ctx.fillStyle = "#9ab3c4";
  ctx.font = "850 13px Avenir Next, sans-serif";
  ctx.fillText(`Weather: ${visual.environment?.dominant_weather || "clear"} | Wind ${visual.environment?.max_wind_kph || 0} kph`, 46, 116);
  ctx.restore();
}

function renderPanels() {
  const frame = currentFrame();
  const visual = currentVisual();
  const summary = state.payload?.summary || {};
  rewardMetric.textContent = Number(summary.total_reward || frame.info?.cumulative_reward?.total || 89).toFixed(1);
  modeMetric.textContent = state.traceMode;
  frameBadge.textContent = `Frame ${state.frameIndex + 1} / ${state.frames.length}`;

  telemetryList.innerHTML = (visual.drone_telemetry || []).map((drone) => `
    <article class="telemetry-card">
      <span>${escapeHtml(drone.drone_id)} | ${escapeHtml(drone.zone)} -> ${escapeHtml(drone.target_zone || drone.zone)}</span>
      <strong>${escapeHtml(drone.current_action || "monitor")} | ${escapeHtml(drone.route_risk || "nominal")}</strong>
      <div class="bar" title="battery"><i style="--value:${Math.max(0, Math.min(100, drone.battery || 0))}%"></i></div>
      <div class="chipline">
        <em>${Math.round(drone.battery || 0)}% battery</em>
        <em>${drone.altitude_m || 0}m altitude</em>
        <em>${drone.speed_kph || 0}kph</em>
        <em>wind ${drone.wind_exposure || 0}kph</em>
        <em>GPS ${drone.gps_lock ? "locked" : "weak"}</em>
        <em>fusion ${Math.round((drone.sensor_fusion_confidence || 0) * 100)}%</em>
      </div>
    </article>
  `).join("");

  const env = visual.environment || {};
  weatherPanel.innerHTML = `
    <div><span>Dominant weather</span><b>${escapeHtml(env.dominant_weather || "clear")}</b></div>
    <div><span>Max wind</span><b>${escapeHtml(env.max_wind_kph || 0)} kph</b></div>
    <div><span>Storm zones</span><b>${escapeHtml((env.storm_zones || []).join(", ") || "none")}</b></div>
    <div><span>Restricted zones</span><b>${escapeHtml((env.restricted_zones || []).join(", ") || "none")}</b></div>
    <div><span>Warnings</span><b>${escapeHtml((env.active_alerts || []).slice(0, 3).join(" | ") || "nominal")}</b></div>
  `;

  const tower = visual.tower || {};
  towerPanel.innerHTML = `
    <div><span>Dispatch queue</span><b>${escapeHtml((tower.dispatch_queue || []).join(", ") || "clear")}</b></div>
    <div><span>Urgent queue</span><b>${escapeHtml((tower.urgent_queue || []).join(", ") || "clear")}</b></div>
    <div><span>Override</span><b>${escapeHtml(tower.override_status || "standby")}</b></div>
    <div><span>RL recommendation</span><b>${escapeHtml(tower.rl_recommendation || "monitor and optimize")}</b></div>
  `;
}

function stepFrame(delta) {
  if (!state.playing) return;
  state.accumulator += delta * Number(speedRange.value);
  if (state.accumulator > 900) {
    state.accumulator = 0;
    state.frameIndex = (state.frameIndex + 1) % state.frames.length;
    renderPanels();
  }
}

function loop(now) {
  const delta = now - (state.lastTime || now);
  state.lastTime = now;
  stepFrame(delta);
  draw(now);
  requestAnimationFrame(loop);
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  if (typeof ctx.roundRect === "function") {
    ctx.roundRect(x, y, w, h, r);
    return;
  }
  const radius = Math.min(r, w / 2, h / 2);
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + w - radius, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
  ctx.lineTo(x + w, y + h - radius);
  ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
  ctx.lineTo(x + radius, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function pointOnPolyline(points, ratio) {
  const index = Math.min(points.length - 2, Math.floor(ratio * (points.length - 1)));
  const local = ratio * (points.length - 1) - index;
  const a = points[index];
  const b = points[index + 1];
  return { x: a.x + (b.x - a.x) * local, y: a.y + (b.y - a.y) * local };
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#039;" })[char]);
}

function makeFallbackFrames() {
  const frames = [];
  const drones = [
    { drone_id: "FA-1", zone: "hub", target_zone: "Z1", battery: 94, altitude_m: 120, speed_kph: 68, current_action: "assign_delivery", route_risk: "nominal" },
    { drone_id: "HE-1", zone: "hub", target_zone: "Z4", battery: 88, altitude_m: 108, speed_kph: 52, current_action: "monitor", route_risk: "caution" },
    { drone_id: "LO-1", zone: "hub", target_zone: "Z2", battery: 76, altitude_m: 140, speed_kph: 62, current_action: "reroute", route_risk: "blocked" },
  ];
  const steps = ["hub", "Z1", "Z2", "Z3", "Z5", "Z4"];
  for (let i = 0; i < 12; i += 1) {
    const telemetry = drones.map((drone, idx) => {
      const zone = steps[(i + idx) % steps.length];
      const c = zoneCenter(zone);
      return {
        ...drone,
        zone,
        target_zone: drone.target_zone,
        x: c.x + idx * 22 - 20,
        y: c.y + (idx % 2) * 18 - 8,
        battery: Math.max(42, drone.battery - i * (idx + 2)),
        wind_exposure: 12 + i * 4 + idx * 5,
        sensor_fusion_confidence: Math.max(0.62, 0.98 - i * 0.02),
        gps_lock: true,
      };
    });
    const route_segments = telemetry.map((drone, idx) => {
      const start = zoneCenter(drone.zone);
      const end = zoneCenter(drone.target_zone);
      return {
        drone_id: drone.drone_id,
        route_color: drone.route_risk === "blocked" ? "red" : drone.route_risk === "caution" ? "yellow" : "purple",
        points: [
          start,
          { x: (start.x + end.x) / 2 + 90 - idx * 30, y: (start.y + end.y) / 2 - 110 + idx * 30 },
          end,
        ],
      };
    });
    frames.push({
      observation: {
        orders: [
          { order_id: "O1", priority: "urgent", status: "assigned", zone_id: "Z1" },
          { order_id: "O2", priority: "medical", status: "queued", zone_id: "Z2" },
          { order_id: "O5", priority: "normal", status: "queued", zone_id: "Z5" },
        ],
      },
      info: { cumulative_reward: { total: 20 + i * 6 } },
      visualization: {
        zone_layout: Object.entries(zoneLayout).map(([zone_id, zone]) => ({
          zone_id,
          ...zone,
          weather: i > 4 && ["Z2", "Z5"].includes(zone_id) ? "storm" : "clear",
          wind_speed_kph: i > 4 && ["Z2", "Z5"].includes(zone_id) ? 58 : 14 + i,
          risk_score: i > 4 && zone_id === "Z2" ? 0.88 : zone_id === "hub" ? 0 : 0.18,
          is_no_fly: i > 6 && zone_id === "Z2",
          operations_paused: i > 8 && zone_id === "Z5",
        })),
        route_segments,
        drone_telemetry: telemetry,
        environment: {
          dominant_weather: i > 4 ? "storm" : "clear",
          max_wind_kph: i > 4 ? 62 : 18,
          storm_zones: i > 4 ? ["Z2", "Z5"] : [],
          restricted_zones: i > 6 ? ["Z2"] : [],
          active_alerts: i > 6 ? ["Z2: no-fly restriction", "Z5: storm reroute", "urgent order O2 inserted"] : ["nominal"],
        },
        tower: {
          dispatch_queue: ["O2", "O5", "O6"],
          urgent_queue: ["O2"],
          override_status: "human-on-loop standby",
          rl_recommendation: i > 6 ? "reroute LO-1 via safe corridor" : "assign FA-1 to urgent order",
        },
      },
    });
  }
  return frames;
}

playBtn.addEventListener("click", () => { state.playing = true; });
pauseBtn.addEventListener("click", () => { state.playing = false; });
resetBtn.addEventListener("click", () => {
  state.frameIndex = 0;
  state.accumulator = 0;
  renderPanels();
});
stageBtn.addEventListener("click", () => document.body.classList.toggle("stage"));

window.addEventListener("resize", resizeCanvasForDisplay);

loadTrace().then(() => {
  renderPanels();
  requestAnimationFrame(loop);
});
