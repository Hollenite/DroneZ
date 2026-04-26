/**
 * Mission Lifecycle Engine
 * Manages end-to-end delivery: order → assign → launch → pickup → deliver → return → dock → charge
 */

// Mission phases (9 total)
export const PHASES = ["ordered","assigned","launching","toPickup","collecting","toDelivery","delivering","returning","docking"];
export const PHASE_LABELS = ["Order Received","Drone Assigned","Launching","To Pickup","Collecting","To Delivery","Delivering","Returning","Docking/Charging"];
export const PHASE_DURATIONS = [2.5, 2.5, 3, 8, 3, 9, 4, 8, 4]; // seconds per phase

// Package types
const PACKAGES = [
  { type: "Medical Kit", icon: "🏥", weight: "1.2 kg", priority: "urgent" },
  { type: "Food Parcel", icon: "🍱", weight: "2.8 kg", priority: "normal" },
  { type: "Electronics", icon: "📦", weight: "1.5 kg", priority: "normal" },
  { type: "Emergency Supply", icon: "🚨", weight: "4.0 kg", priority: "urgent" },
  { type: "Documents", icon: "📄", weight: "0.3 kg", priority: "normal" },
  { type: "Grocery Package", icon: "🛒", weight: "3.2 kg", priority: "normal" },
];

// Zone names for pickup/delivery
const ZONES = [
  { id: "hospital", name: "Hospital District", pos: [-75, -255] },
  { id: "campus", name: "Campus", pos: [150, -245] },
  { id: "suburb", name: "Suburb", pos: [265, 60] },
  { id: "market", name: "Market Zone", pos: [-280, -90] },
  { id: "emergency", name: "Emergency Landing", pos: [255, 245] },
  { id: "coldchain", name: "Cold Chain Storage", pos: [-420, -350] },
  { id: "residential", name: "Residential", pos: [420, -60] },
  { id: "clinic", name: "Rural Clinic", pos: [-470, 260] },
  { id: "lab", name: "Campus Lab", pos: [55, -420] },
  { id: "dock", name: "Port Dock", pos: [430, 320] },
  { id: "tower", name: "Inspect Tower", pos: [10, 360] },
  { id: "locker", name: "Locker Bank", pos: [-360, 20] },
];

const HUB_POS = [-210, 170]; // warehouse position

/**
 * Create a mission definition for a drone
 */
export function createMission(droneIdx, orderId) {
  const pkg = PACKAGES[droneIdx % PACKAGES.length];
  const pickup = ZONES[droneIdx % ZONES.length];
  const delivery = ZONES[(droneIdx + 4) % ZONES.length];
  const altitudeLane = 60 + (droneIdx % 5) * 25; // altitude separation

  return {
    orderId: `O-${String(orderId).padStart(4, "0")}`,
    droneIdx,
    package: pkg,
    pickup,
    delivery,
    altitudeLane,
    phase: -1, // not started
    phaseTimer: 0,
    progress: 0, // 0-1 within current movement phase
    totalTime: 0,
    batteryUsed: 0,
    noFlyAvoided: 1,
    rerouted: false,
    deliveryStyle: droneIdx % 3 === 0 ? "winch" : "landing", // alternate styles
    done: false,

    // Computed route waypoints (set during phase transitions)
    legWaypoints: [], // current leg waypoints
    legIndex: 0,

    // Assignment reason
    assignReason: droneIdx % 2 === 0
      ? "Full battery, correct payload capacity, safe route available"
      : "Closest available drone, priority match, weather-clear corridor",
  };
}

/**
 * Compute waypoints for a route leg, avoiding no-fly zone at (-30, -210)
 */
function computeLeg(fromXZ, toXZ, altitude) {
  const [fx, fz] = fromXZ;
  const [tx, tz] = toXZ;
  const noFlyX = -30, noFlyZ = -210, noFlyR = 80;

  // Check if direct path crosses no-fly zone
  const mx = (fx + tx) / 2, mz = (fz + tz) / 2;
  const distToNoFly = Math.sqrt((mx - noFlyX) ** 2 + (mz - noFlyZ) ** 2);

  const waypoints = [[fx, altitude, fz]];

  if (distToNoFly < noFlyR + 40) {
    // Route around no-fly zone
    const angle = Math.atan2(mz - noFlyZ, mx - noFlyX);
    const avoidX = noFlyX + Math.cos(angle + Math.PI / 2) * (noFlyR + 50);
    const avoidZ = noFlyZ + Math.sin(angle + Math.PI / 2) * (noFlyR + 50);
    waypoints.push([avoidX, altitude + 15, avoidZ]);
  } else {
    // Add a midpoint with slight altitude variation for realism
    waypoints.push([(fx + tx) / 2, altitude + 10, (fz + tz) / 2]);
  }

  waypoints.push([tx, altitude, tz]);
  return waypoints;
}

/**
 * Advance mission by dt seconds. Returns events array.
 */
export function tickMission(mission, dt) {
  if (mission.done || mission.phase < 0) return [];
  const events = [];
  mission.totalTime += dt;
  mission.phaseTimer += dt;
  const dur = PHASE_DURATIONS[mission.phase] || 4;

  // Movement phases: toPickup (3), toDelivery (5), returning (7)
  const isMoving = [3, 5, 7].includes(mission.phase);
  if (isMoving) {
    mission.progress = Math.min(1, mission.phaseTimer / dur);
    mission.batteryUsed += dt * 0.6;
  }

  // Phase transitions
  if (mission.phaseTimer >= dur) {
    mission.phaseTimer = 0;
    mission.progress = 0;
    mission.phase++;

    if (mission.phase >= PHASES.length) {
      mission.done = true;
      mission.phase = PHASES.length - 1;
      events.push({ type: "success", msg: `✅ Mission ${mission.orderId} complete — ${mission.package.type} delivered, drone docked` });
      return events;
    }

    // Phase enter logic
    switch (mission.phase) {
      case 0: // ordered
        events.push({ type: "info", msg: `📦 New order ${mission.orderId}: ${mission.package.icon} ${mission.package.type} (${mission.package.weight})` });
        events.push({ type: "info", msg: `   Pickup: ${mission.pickup.name} → Delivery: ${mission.delivery.name}` });
        break;
      case 1: // assigned
        events.push({ type: "info", msg: `🤖 DZ-${String(mission.droneIdx + 1).padStart(2, "0")} assigned — ${mission.assignReason}` });
        break;
      case 2: // launching
        events.push({ type: "success", msg: `🚀 DZ-${String(mission.droneIdx + 1).padStart(2, "0")} launching from warehouse` });
        mission.legWaypoints = computeLeg(HUB_POS, mission.pickup.pos, mission.altitudeLane);
        break;
      case 3: // toPickup
        events.push({ type: "info", msg: `📍 Heading to pickup: ${mission.pickup.name}` });
        break;
      case 4: // collecting
        events.push({ type: "success", msg: `📦 Package collected at ${mission.pickup.name}` });
        mission.legWaypoints = computeLeg(mission.pickup.pos, mission.delivery.pos, mission.altitudeLane);
        break;
      case 5: // toDelivery
        events.push({ type: "info", msg: `🚁 Delivering to ${mission.delivery.name}` });
        break;
      case 6: // delivering
        const style = mission.deliveryStyle === "winch" ? "Winch/zipline delivery" : "Landing delivery";
        events.push({ type: "success", msg: `📦 ${style} at ${mission.delivery.name} — package delivered!` });
        mission.legWaypoints = computeLeg(mission.delivery.pos, HUB_POS, mission.altitudeLane);
        break;
      case 7: // returning
        events.push({ type: "info", msg: `↩️ Returning to warehouse` });
        break;
      case 8: // docking
        events.push({ type: "info", msg: `🔋 Docked — charging initiated` });
        break;
    }
  }

  return events;
}

/**
 * Get drone world position based on mission state
 * Returns [x, y, z] or null if drone should stay at default position
 */
export function getMissionPosition(mission) {
  if (mission.phase < 0 || mission.done) return null;

  switch (mission.phase) {
    case 0: // ordered — at warehouse
    case 1: // assigned — at warehouse
      return [HUB_POS[0], 8, HUB_POS[1]];
    case 2: { // launching — rise from pad
      const t = Math.min(1, mission.phaseTimer / PHASE_DURATIONS[2]);
      const y = 8 + t * (mission.altitudeLane - 8);
      return [HUB_POS[0], y, HUB_POS[1]];
    }
    case 3: // toPickup — interpolate along waypoints
    case 5: // toDelivery
    case 7: // returning
      return interpolateWaypoints(mission.legWaypoints, mission.progress);
    case 4: { // collecting — hover/descend at pickup
      const t = mission.phaseTimer / PHASE_DURATIONS[4];
      const [px, pz] = mission.pickup.pos;
      const descendY = t < 0.3 ? mission.altitudeLane * (1 - t / 0.3) + 12 * (t / 0.3) :
                        t < 0.7 ? 12 : 12 + (t - 0.7) / 0.3 * (mission.altitudeLane - 12);
      return [px, descendY, pz];
    }
    case 6: { // delivering — descend/winch at delivery
      const t = mission.phaseTimer / PHASE_DURATIONS[6];
      const [dx, dz] = mission.delivery.pos;
      if (mission.deliveryStyle === "winch") {
        return [dx, mission.altitudeLane, dz]; // hover above
      }
      const landY = t < 0.4 ? mission.altitudeLane * (1 - t / 0.4) + 8 * (t / 0.4) :
                     t < 0.7 ? 8 : 8 + (t - 0.7) / 0.3 * (mission.altitudeLane - 8);
      return [dx, landY, dz];
    }
    case 8: { // docking — descend to pad
      const t = Math.min(1, mission.phaseTimer / PHASE_DURATIONS[8]);
      const y = mission.altitudeLane * (1 - t) + 8 * t;
      return [HUB_POS[0], y, HUB_POS[1]];
    }
  }
  return null;
}

/**
 * Get winch line data for delivery phase
 * Returns { from: [x,y,z], to: [x,y,z], progress: 0-1 } or null
 */
export function getWinchData(mission) {
  if (mission.phase !== 6 || mission.deliveryStyle !== "winch") return null;
  const t = mission.phaseTimer / PHASE_DURATIONS[6];
  const [dx, dz] = mission.delivery.pos;
  const lineLen = t < 0.3 ? t / 0.3 : t < 0.7 ? 1 : 1 - (t - 0.7) / 0.3;
  return {
    from: [dx, mission.altitudeLane, dz],
    to: [dx, mission.altitudeLane - lineLen * (mission.altitudeLane - 5), dz],
    hasPackage: t < 0.6,
    progress: t,
  };
}

function interpolateWaypoints(waypoints, t) {
  if (!waypoints || waypoints.length < 2) return null;
  const totalSegs = waypoints.length - 1;
  const segF = t * totalSegs;
  const segIdx = Math.min(Math.floor(segF), totalSegs - 1);
  const segT = segF - segIdx;
  const a = waypoints[segIdx], b = waypoints[segIdx + 1];
  // Smooth easing
  const eased = segT * segT * (3 - 2 * segT);
  return [
    a[0] + (b[0] - a[0]) * eased,
    a[1] + (b[1] - a[1]) * eased,
    a[2] + (b[2] - a[2]) * eased,
  ];
}

/**
 * Generate mission summary text
 */
export function getMissionSummary(mission) {
  return {
    orderId: mission.orderId,
    package: `${mission.package.icon} ${mission.package.type} (${mission.package.weight})`,
    pickup: mission.pickup.name,
    delivery: mission.delivery.name,
    totalTime: `${Math.round(mission.totalTime)}s`,
    batteryUsed: `${Math.round(mission.batteryUsed)}%`,
    noFlyAvoided: mission.noFlyAvoided,
    deliveryStyle: mission.deliveryStyle === "winch" ? "Winch/Zipline" : "Landing",
    status: mission.done ? "✅ Complete" : PHASE_LABELS[mission.phase] || "Pending",
    phase: mission.phase,
    phaseLabel: PHASE_LABELS[mission.phase] || "Pending",
  };
}

export { PACKAGES, ZONES, HUB_POS };
