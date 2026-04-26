import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {WORLD,buildWorld,buildSelectedMarker,createDrone,setupLighting,pads,mats} from "./world.js";
import {createMission,tickMission,getMissionPosition,getWinchData,getMissionSummary,PHASES,PHASE_LABELS,PHASE_DURATIONS,HUB_POS} from "./missions.js";
import {loadTrace} from "./traces.js";

const $=id=>document.getElementById(id);
const root=$("world"),minimap=$("minimap"),mapCtx=minimap.getContext("2d");
const playBtn=$("playBtn"),pauseBtn=$("pauseBtn"),resetBtn=$("resetBtn"),speedRange=$("speedRange");
const themeBtn=$("themeBtn"),launchBtn=$("launchBtn");
const scenarioSel=$("scenarioSelect"),policySel=$("policySelect");
const droneSel=$("droneSelect"),droneNameEl=$("droneName"),droneClassEl=$("droneClass");
const telGrid=$("telemetryGrid"),modeLabel=$("modeLabel"),modeBadge=$("modeBadge");
const weatherLabel=$("weatherLabel"),traceLabel=$("traceLabel"),traceStatus=$("traceStatus");
const routeNameEl=$("routeName"),alertLog=$("alertLog");
const activeCount=$("activeCount"),dockedCount=$("dockedCount"),standbyCount=$("standbyCount");
const toastBox=$("toastContainer"),fpsEl=$("fpsDisplay");
const lcTracker=$("lifecycleTracker"),missionCard=$("missionCard"),missionSummary=$("missionSummary");
const cameraSel=$("cameraMode"),prevDroneBtn=$("prevDrone"),nextDroneBtn=$("nextDrone");
const modeButtons=[...document.querySelectorAll("[data-mode]")];

const DRONE_COUNT=42, ACTIVE_MISSION_COUNT=6;
const droneTypes=[
  {name:"Light delivery",color:0x69f7ff,speed:1.0,payload:"2.0 kg"},
  {name:"Medical priority",color:0xffd36e,speed:1.28,payload:"1.2 kg"},
  {name:"Heavy payload",color:0x7dffb2,speed:0.78,payload:"8.0 kg"},
  {name:"Long-range",color:0xb894ff,speed:0.92,payload:"3.4 kg"},
  {name:"Surveillance",color:0x4da3ff,speed:1.45,payload:"sensor pod"},
];

const state={mode:"operations",camMode:"tower",selectedIdx:0,launched:false,day:false,playing:false,
  uiAccum:0,minimapAccum:0,alerts:[],fpsFrames:0,fpsTime:0,fps:60};
const clock=new THREE.Clock();
const drones=[],missions=[];

// Renderer — performance first
const renderer=new THREE.WebGLRenderer({antialias:false,powerPreference:"high-performance"});
renderer.setPixelRatio(1);
renderer.setSize(root.clientWidth,root.clientHeight);
renderer.shadowMap.enabled=false;
renderer.outputColorSpace=THREE.SRGBColorSpace;
renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.16;
root.appendChild(renderer.domElement);

const scene=new THREE.Scene();
scene.background=new THREE.Color(0x090d12);
scene.fog=new THREE.FogExp2(0x121820,0.0028);

const camera=new THREE.PerspectiveCamera(58,root.clientWidth/root.clientHeight,1,2200);
camera.position.set(230,150,280);

const controls=new OrbitControls(camera,renderer.domElement);
controls.enableDamping=true;controls.dampingFactor=0.06;
controls.maxPolarAngle=Math.PI*0.48;controls.minDistance=30;controls.maxDistance=820;
controls.target.set(0,28,0);

const worldGroup=new THREE.Group();scene.add(worldGroup);
const lights=setupLighting(scene);
buildWorld(worldGroup);
const selectedMarker=buildSelectedMarker(worldGroup);

// Route lines for active missions (created once, reused)
const routeLines=[];
for(let i=0;i<ACTIVE_MISSION_COUNT;i++){
  const colors=[0xb894ff,0x7dffb2,0xffd36e,0x69f7ff,0xff5c78,0x8be4e7];
  const line=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:colors[i%6],transparent:true,opacity:0.7}));
  line.visible=false;worldGroup.add(line);routeLines.push(line);
}

// Build fleet
for(let i=0;i<DRONE_COUNT;i++){
  const type=droneTypes[i%droneTypes.length];
  const isLow=i>=ACTIVE_MISSION_COUNT;
  const drone=createDrone(type.color,isLow);
  const pad=pads[i%pads.length];
  const wp=new THREE.Vector3();pad.getWorldPosition(wp);
  drone.group.position.copy(wp).add(new THREE.Vector3(0,7+(i%4)*0.3,0));
  drone.group.userData={id:`DZ-${String(i+1).padStart(2,"0")}`,type,battery:98-(i%8)*2,
    altitude:0,speed:0,eta:0,risk:"Nominal",
    status:i<ACTIVE_MISSION_COUNT?"Active mission":"Docked / charging",
    hasPayload:false,missionPhase:"idle"};
  worldGroup.add(drone.group);drones.push(drone);
  const opt=document.createElement("option");
  opt.value=String(i);opt.textContent=`${drone.group.userData.id} — ${type.name}`;
  droneSel.appendChild(opt);
}

// Create missions for first N drones
for(let i=0;i<ACTIVE_MISSION_COUNT;i++){
  const m=createMission(i,1024+i);
  m.phase=0;m.phaseTimer=0;// start immediately
  missions.push(m);
}

// Toast/Alert
function toast(msg,type="info"){
  const el=document.createElement("div");el.className=`toast ${type}`;el.textContent=msg;
  toastBox.appendChild(el);setTimeout(()=>el.remove(),4200);
  state.alerts.unshift({msg,type,time:Date.now()});
  if(state.alerts.length>20)state.alerts.length=20;
}

async function doLoadTrace(){
  const r=await loadTrace(scenarioSel.value,policySel.value);
  traceLabel.textContent=r.mode;traceStatus.textContent=r.message;
  toast(r.message,r.success?"success":"warning");
}

function applyTheme(){
  document.body.dataset.theme=state.day?"day":"night";
  scene.background.set(state.day?0xc7d8e0:0x090d12);
  scene.fog.color.set(state.day?0xc7d8e0:0x121820);
  scene.fog.density=state.day?0.0016:0.0028;
  renderer.toneMappingExposure=state.day?1.3:1.16;
  lights.hemi.intensity=state.day?1.7:1.3;
  lights.sun.intensity=state.day?4.5:3.5;
}

function setMode(mode){
  state.mode=mode;
  document.body.classList.toggle("cinematic-mode",mode==="cinematic");
  document.body.classList.toggle("lifecycle-mode",mode==="lifecycle");
  modeLabel.textContent=mode.charAt(0).toUpperCase()+mode.slice(1);
  modeBadge.textContent=modeLabel.textContent;
  modeButtons.forEach(b=>b.classList.toggle("active",b.dataset.mode===mode));
  // Map mode buttons to camera modes
  const camMap={operations:"tower",follow:"follow",cinematic:"cinematic",warehouse:"tower",lifecycle:"follow"};
  state.camMode=camMap[mode]||"tower";
  cameraSel.value=state.camMode;
  // Warehouse: also move orbit target to warehouse area
  if(mode==="warehouse")controls.target.set(-210,20,170);
}

// === ANIMATION LOOP ===
function animate(){
  const dt=Math.min(clock.getDelta(),0.045);
  const t=clock.elapsedTime;
  const spd=Number(speedRange.value);

  if(state.playing) animateWorld(dt*spd,t);
  moveCamera(dt,t);
  controls.update();
  renderer.render(scene,camera);

  state.uiAccum+=dt;
  if(state.uiAccum>0.35){state.uiAccum=0;updatePanel();updateAlertLog();updateCounts();updateMissionCard();}
  state.minimapAccum+=dt;
  if(state.minimapAccum>0.2){state.minimapAccum=0;drawMinimap();}
  updateFPS(dt);
  requestAnimationFrame(animate);
}

// === WORLD ANIMATION ===
function animateWorld(dt,t){
  // Tick missions
  for(let i=0;i<missions.length;i++){
    const m=missions[i];
    const events=tickMission(m,dt);
    events.forEach(e=>toast(e.msg,e.type));

    // Update drone position from mission
    const pos=getMissionPosition(m);
    const g=drones[m.droneIdx].group;
    const d=g.userData;
    if(pos){
      const target=new THREE.Vector3(pos[0],pos[1],pos[2]);
      g.position.lerp(target,1-Math.pow(0.002,dt)); // ultra smooth
      g.position.y+=Math.sin(t*3+i)*0.015; // gentle hover
      d.altitude=Math.round(g.position.y);
      d.speed=[3,5,7].includes(m.phase)?Math.round(45+d.type.speed*40):0;
      d.eta=m.phase<=5?Math.round((1-m.progress)*(PHASE_DURATIONS[m.phase]||4)):0;
      d.battery=Math.max(10,d.battery-dt*0.15);
      d.missionPhase=PHASE_LABELS[m.phase]||"idle";
      d.risk=m.phase===5&&m.progress>0.3&&m.progress<0.6?"Weather reroute":"Nominal";
    }

    // Payload visibility
    const hasPayload=m.phase>=4&&m.phase<=6&&!(m.phase===6&&m.deliveryStyle==="winch"&&m.phaseTimer/PHASE_DURATIONS[6]>0.6);
    g.children.forEach(c=>{if(c.userData.isPayload)c.visible=hasPayload;});
    d.hasPayload=hasPayload;

    // Winch line
    const winchData=getWinchData(m);
    g.children.forEach(c=>{
      if(c.userData.isWinch){
        if(winchData){
          c.visible=true;
          const len=winchData.from[1]-winchData.to[1];
          c.scale.set(1,Math.max(1,len),1);
          c.position.y=-len/2-4;
        }else{c.visible=false;}
      }
    });

    // Update route line
    if(i<routeLines.length){
      const rl=routeLines[i];
      if(m.legWaypoints&&m.legWaypoints.length>=2&&[3,5,7].includes(m.phase)){
        const pts=m.legWaypoints.map(([x,y,z])=>new THREE.Vector3(x,y,z));
        rl.geometry.dispose();
        rl.geometry=new THREE.BufferGeometry().setFromPoints(pts);
        rl.visible=true;
      }else{rl.visible=false;}
    }

    // Spin rotors for active drones
    if(!drones[m.droneIdx].isLow&&m.phase>=2&&m.phase<=7){
      for(let j=0;j<g.children.length;j++){
        if(g.children[j].userData.rotor)g.children[j].rotation.y+=1.5;
      }
    }
  }

  // Idle drones gentle hover
  for(let i=ACTIVE_MISSION_COUNT;i<drones.length;i++){
    drones[i].group.position.y+=Math.sin(t*2+i)*0.002;
  }

  // Selection marker
  const sel=drones[state.selectedIdx].group;
  selectedMarker.position.copy(sel.position);selectedMarker.position.y=Math.max(2,sel.position.y-9);
  selectedMarker.rotation.y+=dt*0.9;
  weatherLabel.textContent=Math.sin(t*0.08)>0.35?"Storm risk":"Clear";

  // Lifecycle tracker UI
  if(missions[state.selectedIdx]&&state.selectedIdx<missions.length){
    const m=missions[state.selectedIdx];
    lcTracker.querySelectorAll(".lc-stage").forEach((el,idx)=>{
      el.classList.toggle("active",idx===m.phase);
      el.classList.toggle("done",idx<m.phase);
    });
  }
}

// === CAMERA MODES ===
function moveCamera(dt,t){
  const target=drones[state.selectedIdx].group;
  const cm=state.camMode;

  if(cm==="tower"){
    if(state.mode==="warehouse"){
      camera.position.lerp(new THREE.Vector3(-110,120,280),0.03);
      controls.target.lerp(new THREE.Vector3(-210,18,170),0.04);
    }else{
      camera.position.lerp(new THREE.Vector3(230,200,300),0.02);
      controls.target.lerp(new THREE.Vector3(0,28,0),0.025);
    }
  }else if(cm==="follow"){
    camera.position.lerp(target.position.clone().add(new THREE.Vector3(-50,30,60)),0.04);
    controls.target.lerp(target.position,0.08);
  }else if(cm==="chase"){
    const dir=new THREE.Vector3(0,0,-1).applyQuaternion(target.quaternion);
    camera.position.lerp(target.position.clone().add(dir.multiplyScalar(-40)).add(new THREE.Vector3(0,12,0)),0.05);
    controls.target.lerp(target.position,0.1);
  }else if(cm==="fpv"){
    const fwd=new THREE.Vector3(0,0,1).applyQuaternion(target.quaternion);
    camera.position.lerp(target.position.clone().add(new THREE.Vector3(0,2,0)),0.08);
    controls.target.lerp(target.position.clone().add(fwd.multiplyScalar(50)),0.08);
  }else if(cm==="cinematic"){
    const r=280+Math.sin(t*0.2)*60;
    camera.position.lerp(new THREE.Vector3(Math.cos(t*0.1)*r,140+Math.sin(t*0.25)*50,Math.sin(t*0.1)*r),0.015);
    controls.target.lerp(target.position.clone().multiplyScalar(0.3),0.02);
  }else if(cm==="topdown"){
    camera.position.lerp(new THREE.Vector3(0,500,0.1),0.03);
    controls.target.lerp(new THREE.Vector3(0,0,0),0.03);
  }
}

// === UI UPDATES (throttled) ===
let lastPanelKey="";
function updatePanel(){
  const d=drones[state.selectedIdx].group.userData;
  droneNameEl.textContent=d.id;droneClassEl.textContent=d.type.name;
  const bat=Math.round(d.battery),spd=Math.round(d.speed);
  const key=`${d.id}${bat}${spd}${d.altitude}${d.missionPhase}`;
  if(key===lastPanelKey)return;lastPanelKey=key;
  routeNameEl.textContent=`${d.id}: ${d.missionPhase||"Idle"}`;
  telGrid.innerHTML=[
    ["Battery",`${bat}%`,bat],["Altitude",`${d.altitude} m`,Math.min(100,d.altitude/1.5)],
    ["Speed",`${spd} kph`,Math.min(100,spd/1.4)],["ETA",`${d.eta}s`,Math.max(10,100-d.eta*5)],
    ["Payload",d.hasPayload?"📦 Carrying":d.type.payload,d.hasPayload?90:60],
    ["Phase",d.missionPhase||"idle",70],
    ["Risk",d.risk,d.risk==="Nominal"?85:40],["Sensors","IMU+LiDAR ✓",94],
  ].map(([l,v,m])=>`<div class="metric"><div class="metric-label">${l}</div><div class="metric-value">${v}</div><div class="meter-bar"><div class="meter-fill" style="width:${m}%"></div></div></div>`).join("");
}

function updateMissionCard(){
  if(!missionCard)return;
  const mi=state.selectedIdx<missions.length?missions[state.selectedIdx]:null;
  if(!mi){missionCard.innerHTML="<p>No active mission for this drone</p>";return;}
  const s=getMissionSummary(mi);
  missionCard.innerHTML=`
    <div class="mc-row"><span>Order</span><b>${s.orderId}</b></div>
    <div class="mc-row"><span>Package</span><b>${s.package}</b></div>
    <div class="mc-row"><span>Pickup</span><b>${s.pickup}</b></div>
    <div class="mc-row"><span>Delivery</span><b>${s.delivery}</b></div>
    <div class="mc-row"><span>Phase</span><b>${s.phaseLabel}</b></div>
    <div class="mc-row"><span>Style</span><b>${s.deliveryStyle}</b></div>
    <div class="mc-row"><span>Time</span><b>${s.totalTime}</b></div>
    <div class="mc-row"><span>Battery Used</span><b>${s.batteryUsed}</b></div>
    <div class="mc-row"><span>Status</span><b>${s.status}</b></div>`;
}

function updateAlertLog(){
  alertLog.innerHTML=state.alerts.slice(0,10).map(a=>`<div class="alert-entry ${a.type}">${a.msg}</div>`).join("")||'<div class="alert-entry info">System nominal</div>';
}
function updateCounts(){
  let a=0,d2=0,s=0;
  for(let i=0;i<drones.length;i++){
    const st=drones[i].group.userData.status;
    if(st.includes("Active"))a++;else if(st.includes("Dock")||st.includes("charg"))d2++;else s++;
  }
  activeCount.textContent=String(a).padStart(2,"0");
  dockedCount.textContent=String(d2).padStart(2,"0");
  standbyCount.textContent=String(s).padStart(2,"0");
}

// === MINIMAP ===
function drawMinimap(){
  const w=minimap.width,h=minimap.height;
  mapCtx.fillStyle=state.day?"rgba(220,230,240,0.96)":"rgba(8,24,34,0.96)";
  mapCtx.fillRect(0,0,w,h);
  mapCtx.strokeStyle=state.day?"rgba(40,60,80,0.08)":"rgba(105,247,255,0.08)";
  mapCtx.lineWidth=0.5;
  for(let x=0;x<w;x+=30){mapCtx.beginPath();mapCtx.moveTo(x,0);mapCtx.lineTo(x,h);mapCtx.stroke();}
  for(let y=0;y<h;y+=30){mapCtx.beginPath();mapCtx.moveTo(0,y);mapCtx.lineTo(w,y);mapCtx.stroke();}
  [[-30,-210,55,"rgba(255,92,120,0.3)","NO-FLY"],[170,-180,75,"rgba(77,163,255,0.2)","STORM"],[-210,170,44,"rgba(105,247,255,0.2)","HUB"]].forEach(([x,z,r,col,label])=>{
    const p=proj(x,z);mapCtx.fillStyle=col;mapCtx.beginPath();mapCtx.arc(p.x,p.y,r*0.18,0,Math.PI*2);mapCtx.fill();
    mapCtx.fillStyle=state.day?"#1a2830":"#f4fbff";mapCtx.font="800 7px Inter,sans-serif";mapCtx.fillText(label,p.x-12,p.y+3);
  });
  // Route lines on minimap
  missions.forEach((m,i)=>{
    if(!m.legWaypoints||m.legWaypoints.length<2)return;
    const pts=m.legWaypoints.map(([x,,z])=>proj(x,z));
    mapCtx.strokeStyle=["#b894ff","#7dffb2","#ffd36e","#69f7ff","#ff5c78","#8be4e7"][i%6];
    mapCtx.lineWidth=i===state.selectedIdx?2.5:1;
    mapCtx.beginPath();pts.forEach((p,j)=>j?mapCtx.lineTo(p.x,p.y):mapCtx.moveTo(p.x,p.y));mapCtx.stroke();
  });
  for(let i=0;i<Math.min(drones.length,state.launched?24:ACTIVE_MISSION_COUNT+3);i++){
    const p=proj(drones[i].group.position.x,drones[i].group.position.z);
    const isSel=i===state.selectedIdx;
    mapCtx.fillStyle=isSel?"#ffffff":`#${drones[i].group.userData.type.color.toString(16).padStart(6,"0")}`;
    mapCtx.beginPath();mapCtx.arc(p.x,p.y,isSel?4:1.8,0,Math.PI*2);mapCtx.fill();
  }
  mapCtx.fillStyle=state.day?"#1a2830":"#dff8ff";mapCtx.font="900 9px Inter,sans-serif";mapCtx.fillText("TACTICAL MAP",8,14);
}
function proj(x,z){return{x:((x+WORLD/2)/WORLD)*minimap.width,y:((z+WORLD/2)/WORLD)*minimap.height};}

function updateFPS(dt){
  state.fpsFrames++;state.fpsTime+=dt;
  if(state.fpsTime>=1){state.fps=Math.round(state.fpsFrames/state.fpsTime);
    fpsEl.textContent=`${state.fps} FPS`;fpsEl.style.color=state.fps>=45?"#7deca0":state.fps>=25?"#f0b860":"#f06e7e";
    state.fpsFrames=0;state.fpsTime=0;}
}

function onResize(){
  const w=root.clientWidth||window.innerWidth,h=root.clientHeight||window.innerHeight;
  camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h);
}
window.addEventListener("resize",onResize);

// === UI BINDINGS ===
droneSel.addEventListener("change",()=>{state.selectedIdx=Number(droneSel.value);});
cameraSel.addEventListener("change",()=>{state.camMode=cameraSel.value;});
prevDroneBtn?.addEventListener("click",()=>{state.selectedIdx=(state.selectedIdx-1+DRONE_COUNT)%DRONE_COUNT;droneSel.value=String(state.selectedIdx);});
nextDroneBtn?.addEventListener("click",()=>{state.selectedIdx=(state.selectedIdx+1)%DRONE_COUNT;droneSel.value=String(state.selectedIdx);});
playBtn.addEventListener("click",()=>{state.playing=true;toast("▶ Simulation running","info");});
pauseBtn.addEventListener("click",()=>{state.playing=false;});
resetBtn.addEventListener("click",()=>{
  state.playing=false;
  missions.length=0;
  for(let i=0;i<ACTIVE_MISSION_COUNT;i++){
    const m=createMission(i,2000+i);m.phase=0;m.phaseTimer=0;missions.push(m);
  }
  drones.forEach((dr,i)=>{
    const d=dr.group.userData;d.battery=98-(i%8)*2;
    const pad=pads[i%pads.length];const wp=new THREE.Vector3();pad.getWorldPosition(wp);
    dr.group.position.copy(wp).add(new THREE.Vector3(0,7,0));
    d.hasPayload=false;d.missionPhase="idle";
    dr.group.children.forEach(c=>{if(c.userData.isPayload)c.visible=false;if(c.userData.isWinch)c.visible=false;});
  });
  toast("🔄 Reset — new missions created","info");
});
launchBtn.addEventListener("click",()=>{state.launched=true;state.playing=true;toast("🚀 Fleet launched","success");});
themeBtn.addEventListener("click",()=>{state.day=!state.day;themeBtn.textContent=state.day?"🌙 Night":"☀ Day";applyTheme();});
modeButtons.forEach(b=>b.addEventListener("click",()=>setMode(b.dataset.mode)));
scenarioSel.addEventListener("change",()=>doLoadTrace().catch(()=>{}));
policySel.addEventListener("change",()=>doLoadTrace().catch(()=>{}));

// === INIT ===
try{onResize();applyTheme();updatePanel();updateCounts();
  toast("DroneZ Simulator initialized — 42 drones, 6 active missions","info");
  toast("Camera modes: Tower / Follow / Chase / FPV / Cinematic / Top-Down","info");
  updateAlertLog();doLoadTrace().catch(()=>{});
}catch(e){console.error("Init:",e);}
document.body.classList.add("ready");
state.playing=true; // auto-start
requestAnimationFrame(animate);
