import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";

const root = document.getElementById("world");
const minimap = document.getElementById("minimap");
const mapCtx = minimap.getContext("2d");
const droneSelect = document.getElementById("droneSelect");
const droneName = document.getElementById("droneName");
const droneClass = document.getElementById("droneClass");
const telemetryGrid = document.getElementById("telemetryGrid");
const modeLabel = document.getElementById("modeLabel");
const weatherLabel = document.getElementById("weatherLabel");
const routeName = document.getElementById("routeName");
const routeStatus = document.getElementById("routeStatus");
const launchBtn = document.getElementById("launchBtn");

const WORLD = 680;
const DRONE_COUNT = 42;
const modeButtons = [...document.querySelectorAll("[data-mode]")];
const clock = new THREE.Clock();
const selected = { id: "DZ-01", index: 0, mode: "operations", launched: false };
const droneTypes = [
  { name: "Light delivery drone", color: 0x69f7ff, speed: 1.0, payload: "2.0 kg" },
  { name: "Urgent medical drone", color: 0xffd36e, speed: 1.28, payload: "1.2 kg cold-chain" },
  { name: "Heavy payload drone", color: 0x7dffb2, speed: 0.78, payload: "8.0 kg" },
  { name: "Long-range rural drone", color: 0xb894ff, speed: 0.92, payload: "3.4 kg" },
  { name: "Surveillance / scout drone", color: 0x4da3ff, speed: 1.45, payload: "sensor pod" },
];

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.7));
renderer.setSize(root.clientWidth, root.clientHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.16;
root.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x060a10);
scene.fog = new THREE.FogExp2(0x07111c, 0.0038);

const camera = new THREE.PerspectiveCamera(58, root.clientWidth / root.clientHeight, 0.1, 2600);
camera.position.set(230, 150, 280);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.maxPolarAngle = Math.PI * 0.48;
controls.minDistance = 50;
controls.maxDistance = 820;
controls.target.set(0, 28, 0);

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
composer.addPass(new UnrealBloomPass(new THREE.Vector2(root.clientWidth, root.clientHeight), 0.42, 0.78, 0.18));

const mats = {
  ground: new THREE.MeshStandardMaterial({ color: 0x1a3023, roughness: 0.92, metalness: 0.02 }),
  runway: new THREE.MeshStandardMaterial({ color: 0x1e2730, roughness: 0.7, metalness: 0.04 }),
  building: new THREE.MeshStandardMaterial({ color: 0x263647, roughness: 0.66, metalness: 0.14 }),
  buildingBlue: new THREE.MeshStandardMaterial({ color: 0x1d4e63, roughness: 0.56, metalness: 0.22, emissive: 0x021018 }),
  glass: new THREE.MeshStandardMaterial({ color: 0x5fcfff, roughness: 0.22, metalness: 0.1, transparent: true, opacity: 0.42 }),
  route: new THREE.LineBasicMaterial({ color: 0xb894ff, transparent: true, opacity: 0.9 }),
  safe: new THREE.LineBasicMaterial({ color: 0x7dffb2, transparent: true, opacity: 0.9 }),
  caution: new THREE.LineBasicMaterial({ color: 0xffd36e, transparent: true, opacity: 0.86 }),
  blocked: new THREE.LineBasicMaterial({ color: 0xff5c78, transparent: true, opacity: 0.72 }),
  noFly: new THREE.MeshStandardMaterial({ color: 0xff5c78, transparent: true, opacity: 0.17, depthWrite: false }),
  weather: new THREE.MeshStandardMaterial({ color: 0x4da3ff, transparent: true, opacity: 0.12, depthWrite: false }),
};

const drones = [];
const routes = [];
const pads = [];
const worldObjects = new THREE.Group();
scene.add(worldObjects);

setupLighting();
buildWorld();
buildFleet();
buildRoutes();
loadTraceHint();
bindUi();
onResize();
window.addEventListener("resize", onResize);
document.body.classList.add("ready");
animate();

function setupLighting() {
  const hemi = new THREE.HemisphereLight(0xb7e6ff, 0x15220e, 1.1);
  scene.add(hemi);

  const sun = new THREE.DirectionalLight(0xffd0a0, 3.2);
  sun.position.set(-240, 340, 180);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 900;
  sun.shadow.camera.left = -480;
  sun.shadow.camera.right = 480;
  sun.shadow.camera.top = 480;
  sun.shadow.camera.bottom = -480;
  scene.add(sun);

  const cityGlow = new THREE.PointLight(0x69f7ff, 1.8, 520, 1.2);
  cityGlow.position.set(90, 90, -120);
  scene.add(cityGlow);
}

function buildWorld() {
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(WORLD, WORLD, 64, 64), mats.ground);
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  worldObjects.add(ground);

  addGridGlow();
  addWarehouse();
  addCity();
  addIndustrialDistrict();
  addTerrain();
  addWeatherVolumes();
  addLandingZones();
}

function addGridGlow() {
  const grid = new THREE.GridHelper(WORLD, 44, 0x19495b, 0x15313f);
  grid.position.y = 0.08;
  worldObjects.add(grid);

  for (let i = -4; i <= 4; i += 1) {
    addRoad(i * 70, 0, 12, WORLD, 0);
    addRoad(0, i * 70, WORLD, 12, 0);
  }
}

function addRoad(x, z, w, d, rot) {
  const road = new THREE.Mesh(new THREE.BoxGeometry(w, 0.16, d), mats.runway);
  road.position.set(x, 0.12, z);
  road.rotation.y = rot;
  road.receiveShadow = true;
  worldObjects.add(road);
}

function addWarehouse() {
  const hangar = new THREE.Group();
  hangar.position.set(-210, 0, 170);
  worldObjects.add(hangar);

  const floor = new THREE.Mesh(new THREE.BoxGeometry(180, 2, 130), mats.runway);
  floor.position.y = 1;
  floor.receiveShadow = true;
  hangar.add(floor);

  const roof = new THREE.Mesh(new THREE.BoxGeometry(190, 12, 138), new THREE.MeshStandardMaterial({
    color: 0x122232,
    roughness: 0.38,
    metalness: 0.42,
    transparent: true,
    opacity: 0.82,
  }));
  roof.position.y = 44;
  roof.castShadow = true;
  hangar.add(roof);

  for (let i = 0; i < 9; i += 1) {
    const beam = new THREE.Mesh(new THREE.BoxGeometry(4, 46, 4), mats.buildingBlue);
    beam.position.set(-86 + i * 21.5, 23, -65);
    hangar.add(beam);
    const beam2 = beam.clone();
    beam2.position.z = 65;
    hangar.add(beam2);
  }

  for (let row = 0; row < 6; row += 1) {
    for (let col = 0; col < 7; col += 1) {
      const pad = createPad();
      pad.position.set(-78 + col * 26, 2.1, -48 + row * 19);
      hangar.add(pad);
      pads.push(pad);
    }
  }
}

function createPad() {
  const pad = new THREE.Group();
  const disc = new THREE.Mesh(new THREE.CylinderGeometry(8, 8, 0.8, 48), new THREE.MeshStandardMaterial({
    color: 0x102733,
    metalness: 0.5,
    roughness: 0.32,
    emissive: 0x021c20,
  }));
  disc.castShadow = true;
  pad.add(disc);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(8.2, 0.35, 8, 48), new THREE.MeshBasicMaterial({ color: 0x69f7ff }));
  ring.rotation.x = Math.PI / 2;
  ring.position.y = 0.8;
  pad.add(ring);
  return pad;
}

function addCity() {
  for (let i = 0; i < 120; i += 1) {
    const x = -40 + (i * 53) % 360;
    const z = -310 + Math.floor(i / 18) * 45;
    if (Math.abs(x) < 45 && Math.abs(z) < 45) continue;
    const h = 18 + ((i * 19) % 110);
    const building = new THREE.Mesh(new THREE.BoxGeometry(18 + (i % 4) * 6, h, 18 + (i % 5) * 4), i % 7 === 0 ? mats.glass : mats.building);
    building.position.set(x, h / 2, z);
    building.castShadow = true;
    building.receiveShadow = true;
    worldObjects.add(building);

    if (i % 5 === 0) {
      const light = new THREE.PointLight(0x69f7ff, 0.25, 58);
      light.position.set(x, h + 4, z);
      worldObjects.add(light);
    }
  }
}

function addIndustrialDistrict() {
  for (let i = 0; i < 28; i += 1) {
    const stack = new THREE.Mesh(new THREE.CylinderGeometry(4, 6, 42 + (i % 5) * 13, 16), mats.building);
    stack.position.set(190 + (i % 7) * 25, 24, 120 + Math.floor(i / 7) * 36);
    stack.castShadow = true;
    worldObjects.add(stack);
  }
}

function addTerrain() {
  for (let i = 0; i < 70; i += 1) {
    const tree = new THREE.Group();
    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(1.2, 1.8, 13, 8), new THREE.MeshStandardMaterial({ color: 0x5a3924 }));
    trunk.position.y = 6.5;
    const crown = new THREE.Mesh(new THREE.ConeGeometry(7 + (i % 3), 21, 8), new THREE.MeshStandardMaterial({ color: 0x1f5a36, roughness: 0.9 }));
    crown.position.y = 22;
    tree.add(trunk, crown);
    tree.position.set(-320 + (i * 41) % 170, 0, -300 + Math.floor(i / 9) * 45);
    worldObjects.add(tree);
  }

  const bridge = new THREE.Mesh(new THREE.BoxGeometry(170, 8, 18), new THREE.MeshStandardMaterial({ color: 0x34414d, metalness: 0.35, roughness: 0.5 }));
  bridge.position.set(160, 18, -12);
  bridge.rotation.y = 0.18;
  bridge.castShadow = true;
  bridge.receiveShadow = true;
  worldObjects.add(bridge);
}

function addWeatherVolumes() {
  const storm = new THREE.Mesh(new THREE.CylinderGeometry(78, 102, 210, 64, 1, true), mats.weather);
  storm.position.set(170, 105, -180);
  worldObjects.add(storm);

  const noFly = new THREE.Mesh(new THREE.CylinderGeometry(58, 70, 160, 6, 1, true), mats.noFly);
  noFly.position.set(-30, 80, -210);
  noFly.rotation.y = Math.PI / 6;
  worldObjects.add(noFly);
}

function addLandingZones() {
  [
    ["Hospital", -75, -255, 0xffd36e],
    ["Campus", 150, -245, 0x7dffb2],
    ["Suburb", 265, 60, 0x69f7ff],
    ["Market", -280, -90, 0xb894ff],
    ["Emergency", 255, 245, 0xff5c78],
  ].forEach(([name, x, z, color]) => {
    const zone = new THREE.Group();
    const disc = new THREE.Mesh(new THREE.CylinderGeometry(18, 18, 1, 48), new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: 0.14,
      transparent: true,
      opacity: 0.78,
    }));
    disc.position.y = 0.8;
    zone.add(disc);
    zone.userData = { label: name };
    zone.position.set(x, 0, z);
    worldObjects.add(zone);
  });
}

function buildFleet() {
  for (let i = 0; i < DRONE_COUNT; i += 1) {
    const type = droneTypes[i % droneTypes.length];
    const drone = createDrone(type.color);
    const pad = pads[i % pads.length];
    const worldPos = new THREE.Vector3();
    pad.getWorldPosition(worldPos);
    drone.group.position.copy(worldPos).add(new THREE.Vector3(0, 7 + (i % 4) * 0.3, 0));
    drone.group.rotation.y = (i % 8) * Math.PI * 0.25;
    drone.group.userData = {
      id: `DZ-${String(i + 1).padStart(2, "0")}`,
      type,
      battery: 92 - (i % 11) * 3,
      altitude: 80 + (i % 8) * 12,
      speed: 0,
      eta: 4 + (i % 9),
      risk: i % 9 === 0 ? "Weather reroute" : i % 13 === 0 ? "No-fly avoidance" : "Nominal",
      status: i < 9 ? "Active mission" : i < 34 ? "Docked / charging" : "Standby",
      phase: i * 0.17,
    };
    worldObjects.add(drone.group);
    drones.push(drone);
    const option = document.createElement("option");
    option.value = String(i);
    option.textContent = `${drone.group.userData.id} — ${type.name}`;
    droneSelect.appendChild(option);
  }
}

function createDrone(color) {
  const group = new THREE.Group();
  const bodyMat = new THREE.MeshStandardMaterial({ color, roughness: 0.28, metalness: 0.62, emissive: color, emissiveIntensity: 0.08 });
  const darkMat = new THREE.MeshStandardMaterial({ color: 0x05070a, roughness: 0.4, metalness: 0.55 });
  const body = new THREE.Mesh(new THREE.SphereGeometry(5.2, 32, 16), bodyMat);
  body.scale.set(1.35, 0.52, 1);
  body.castShadow = true;
  group.add(body);

  const payload = new THREE.Mesh(new THREE.BoxGeometry(5, 4, 5), new THREE.MeshStandardMaterial({ color: 0xe9f7ff, roughness: 0.42, metalness: 0.15 }));
  payload.position.y = -4.2;
  payload.castShadow = true;
  group.add(payload);

  const arms = [
    [13, 0, 0, Math.PI / 2],
    [-13, 0, 0, Math.PI / 2],
    [0, 0, 13, 0],
    [0, 0, -13, 0],
  ];
  arms.forEach(([x, y, z, rz]) => {
    const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.7, 0.7, 26, 12), darkMat);
    arm.position.set(x / 2, y, z / 2);
    arm.rotation.z = rz;
    if (z !== 0) arm.rotation.x = Math.PI / 2;
    arm.castShadow = true;
    group.add(arm);

    const rotor = new THREE.Mesh(new THREE.CylinderGeometry(4.8, 4.8, 0.28, 40), new THREE.MeshBasicMaterial({ color: 0xeafcff, transparent: true, opacity: 0.32 }));
    rotor.position.set(x, 1.2, z);
    rotor.userData.rotor = true;
    group.add(rotor);
  });

  const light = new THREE.PointLight(color, 0.65, 36);
  light.position.y = 3;
  group.add(light);
  return { group };
}

function buildRoutes() {
  [
    { points: [[-210, 18, 170], [-160, 95, 70], [-80, 110, -120], [-75, 55, -255]], mat: mats.route },
    { points: [[-210, 18, 170], [-40, 120, 110], [120, 150, -40], [150, 64, -245]], mat: mats.safe },
    { points: [[-210, 18, 170], [-120, 80, -20], [-20, 112, -210], [170, 70, -180]], mat: mats.caution },
    { points: [[-210, 18, 170], [-80, 60, -120], [-30, 80, -210]], mat: mats.blocked },
  ].forEach((route) => {
    const curve = new THREE.CatmullRomCurve3(route.points.map(([x, y, z]) => new THREE.Vector3(x, y, z)));
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(90)), route.mat);
    line.userData.curve = curve;
    line.userData.progress = Math.random();
    worldObjects.add(line);
    routes.push(line);
  });
}

async function loadTraceHint() {
  try {
    const response = await fetch("../artifacts/traces/demo_improved_enriched.json", { cache: "no-store" });
    if (!response.ok) return;
    const payload = await response.json();
    routeStatus.textContent = `Connected to DroneZ enriched trace: ${payload.frames?.length || 0} replay frames available. 3D geometry is cinematic visualization, not real GPS.`;
  } catch {
    routeStatus.textContent = "Standalone cinematic simulation running without trace file. 3D world uses built-in mission data.";
  }
}

function bindUi() {
  droneSelect.addEventListener("change", () => {
    selected.index = Number(droneSelect.value);
    selected.id = drones[selected.index].group.userData.id;
    updatePanel();
  });

  modeButtons.forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });

  launchBtn.addEventListener("click", () => {
    selected.launched = true;
    drones.forEach((drone, index) => {
      if (index < 18) drone.group.userData.status = "Active mission";
    });
  });

  setMode("operations");
  updatePanel();
}

function setMode(mode) {
  selected.mode = mode;
  document.body.classList.toggle("cinematic", mode === "cinematic");
  modeLabel.textContent = titleCase(mode);
  modeButtons.forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  if (mode === "warehouse") controls.target.set(-210, 20, 170);
}

function updatePanel() {
  const drone = drones[selected.index].group.userData;
  droneName.textContent = drone.id;
  droneClass.textContent = drone.type.name;
  routeName.textContent = `${drone.id}: Hub → ${drone.risk === "No-fly avoidance" ? "Emergency corridor" : "Mission target"}`;
  telemetryGrid.innerHTML = [
    ["Battery", `${Math.max(8, Math.round(drone.battery))}%`],
    ["Altitude", `${Math.round(drone.altitude)} m`],
    ["Speed", `${Math.round(drone.speed)} kph`],
    ["ETA", `${Math.max(1, Math.round(drone.eta))} min`],
    ["Payload", drone.type.payload],
    ["Risk", drone.risk],
    ["GPS", "RTK locked"],
    ["Sensors", "IMU + camera + LiDAR nominal"],
  ].map(([label, value]) => `<div class="metric"><span>${label}</span><b>${value}</b></div>`).join("");
}

function animate() {
  const dt = Math.min(clock.getDelta(), 0.04);
  const t = clock.elapsedTime;
  animateWorld(dt, t);
  moveCamera(dt, t);
  controls.update();
  composer.render();
  drawMinimap();
  requestAnimationFrame(animate);
}

function animateWorld(dt, t) {
  const activeCount = selected.launched ? 24 : 9;
  drones.forEach((drone, index) => {
    const group = drone.group;
    const data = group.userData;
    group.children.forEach((child) => {
      if (child.userData.rotor) child.rotation.y += 1.8 + index * 0.015;
    });

    if (index < activeCount || index === selected.index) {
      const route = routes[index % routes.length].userData.curve;
      const p = ((t * 0.025 * data.type.speed) + data.phase) % 1;
      const pos = route.getPointAt(p);
      group.position.lerp(pos, 0.045);
      const ahead = route.getPointAt((p + 0.01) % 1);
      group.lookAt(ahead);
      group.position.y += Math.sin(t * 4 + index) * 0.03;
      data.speed = 48 + data.type.speed * 42 + Math.sin(t + index) * 8;
      data.altitude = group.position.y;
      data.battery -= dt * (0.08 + index * 0.0008);
      data.eta = 2 + (1 - p) * 9;
    } else {
      group.position.y += Math.sin(t * 2 + index) * 0.004;
      data.speed = 0;
    }
  });

  weatherLabel.textContent = Math.sin(t * 0.08) > 0.35 ? "Storm risk" : "Clear";
  if (Math.floor(t * 2) % 2 === 0) updatePanel();
}

function moveCamera(dt, t) {
  const targetDrone = drones[selected.index].group;
  if (selected.mode === "follow") {
    const follow = targetDrone.position.clone().add(new THREE.Vector3(-46, 28, 58));
    camera.position.lerp(follow, 0.035);
    controls.target.lerp(targetDrone.position, 0.08);
  } else if (selected.mode === "cinematic") {
    const radius = 270 + Math.sin(t * 0.22) * 55;
    const desired = new THREE.Vector3(Math.cos(t * 0.12) * radius, 130 + Math.sin(t * 0.28) * 45, Math.sin(t * 0.12) * radius);
    camera.position.lerp(desired, 0.018);
    controls.target.lerp(targetDrone.position.clone().multiplyScalar(0.35), 0.025);
  } else if (selected.mode === "warehouse") {
    camera.position.lerp(new THREE.Vector3(-110, 120, 280), 0.035);
    controls.target.lerp(new THREE.Vector3(-210, 18, 170), 0.05);
  } else {
    camera.position.lerp(new THREE.Vector3(230, 150, 280), 0.025);
    controls.target.lerp(new THREE.Vector3(0, 28, 0), 0.03);
  }
}

function drawMinimap() {
  const w = minimap.width;
  const h = minimap.height;
  mapCtx.clearRect(0, 0, w, h);
  const grad = mapCtx.createLinearGradient(0, 0, w, h);
  grad.addColorStop(0, "rgba(8,24,34,0.96)");
  grad.addColorStop(1, "rgba(3,8,14,0.96)");
  mapCtx.fillStyle = grad;
  mapCtx.fillRect(0, 0, w, h);

  mapCtx.strokeStyle = "rgba(105,247,255,0.15)";
  for (let x = 0; x < w; x += 28) {
    mapCtx.beginPath();
    mapCtx.moveTo(x, 0);
    mapCtx.lineTo(x, h);
    mapCtx.stroke();
  }
  for (let y = 0; y < h; y += 28) {
    mapCtx.beginPath();
    mapCtx.moveTo(0, y);
    mapCtx.lineTo(w, y);
    mapCtx.stroke();
  }

  drawMiniZone(-30, -210, 52, "rgba(255,92,120,0.38)", "NO-FLY");
  drawMiniZone(170, -180, 72, "rgba(77,163,255,0.28)", "STORM");
  drawMiniZone(-210, 170, 42, "rgba(105,247,255,0.24)", "HUB");

  routes.forEach((route, index) => {
    const pts = route.userData.curve.getPoints(24).map(projectMini);
    mapCtx.strokeStyle = ["#b894ff", "#7dffb2", "#ffd36e", "#ff5c78"][index % 4];
    mapCtx.lineWidth = index === 0 ? 3 : 2;
    mapCtx.beginPath();
    pts.forEach((p, i) => (i ? mapCtx.lineTo(p.x, p.y) : mapCtx.moveTo(p.x, p.y)));
    mapCtx.stroke();
  });

  drones.forEach((drone, index) => {
    const p = projectMini(drone.group.position);
    const isSelected = index === selected.index;
    mapCtx.fillStyle = isSelected ? "#ffffff" : `#${drone.group.userData.type.color.toString(16).padStart(6, "0")}`;
    mapCtx.beginPath();
    mapCtx.arc(p.x, p.y, isSelected ? 5 : 2.6, 0, Math.PI * 2);
    mapCtx.fill();
  });

  mapCtx.fillStyle = "#dff8ff";
  mapCtx.font = "900 12px Avenir Next, sans-serif";
  mapCtx.fillText("TACTICAL MINIMAP", 14, 22);
}

function drawMiniZone(x, z, radius, color, label) {
  const p = projectMini(new THREE.Vector3(x, 0, z));
  mapCtx.fillStyle = color;
  mapCtx.beginPath();
  mapCtx.arc(p.x, p.y, radius * 0.23, 0, Math.PI * 2);
  mapCtx.fill();
  mapCtx.fillStyle = "#f4fbff";
  mapCtx.font = "800 9px Avenir Next, sans-serif";
  mapCtx.fillText(label, p.x - 16, p.y + 3);
}

function projectMini(v) {
  return {
    x: ((v.x + WORLD / 2) / WORLD) * minimap.width,
    y: ((v.z + WORLD / 2) / WORLD) * minimap.height,
  };
}

function onResize() {
  const width = root.clientWidth || window.innerWidth;
  const height = root.clientHeight || window.innerHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
  composer.setSize(width, height);
}

function titleCase(value) {
  return value.replace(/(^|\s|-)\S/g, (letter) => letter.toUpperCase()).replace("-", " ");
}
